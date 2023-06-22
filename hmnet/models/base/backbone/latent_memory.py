# Hierarchical Neural Memory Network
# 
# Copyright (C) 2023 National Institute of Advanced Industrial Science and Technology
# 
# All rights reserved.
# 
# Redistribution and use in source and binary forms, with or without modification,
# are permitted provided that the following conditions are met:
# 
#     * Redistributions of source code must retain the above copyright notice,
#       this list of conditions and the following disclaimer.
#     * Redistributions in binary form must reproduce the above copyright notice,
#       this list of conditions and the following disclaimer in the documentation
#       and/or other materials provided with the distribution.
#     * Neither the name of {{ project }} nor the names of its contributors
#       may be used to endorse or promote products derived from this software
#       without specific prior written permission.
# 
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS
# "AS IS" AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT
# LIMITED TO, THE IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR
# A PARTICULAR PURPOSE ARE DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR
# CONTRIBUTORS BE LIABLE FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL,
# EXEMPLARY, OR CONSEQUENTIAL DAMAGES (INCLUDING, BUT NOT LIMITED TO,
# PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES; LOSS OF USE, DATA, OR
# PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND ON ANY THEORY OF
# LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT (INCLUDING
# NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
# SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

from functools import partial
import numpy as np
import os

import torch.multiprocessing as mp
if mp.get_start_method(True) is None:
    mp.set_start_method('spawn')

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from torch_scatter import scatter

from .vit import Mlp, CrossAttention, PositionEmbedding1D, PositionEmbedding2D, PatchMergingCross, SeqData, TransformerBlock
from .resnet import ResBlock, ResStage, ResEntry
from .convnext import ConvNeXtBlock, ConvNeXtEntry, ConvNeXtStage
from ..layers import ConvBlock, Linear, MV2Block, UpConvBlock, ConvPixelShuffle
from ..blocks import BlockBase

from torch import Tensor
from typing import Tuple, List, Optional, Dict
Meta = Dict[str,Tuple[int,int,int]]

from hmnet.utils.common import Timer
timer = Timer()

class LatentMemory(BlockBase):
    def __init__(self, latent_size: Tuple[int,int], input_dim: int, latent_dim: int, output_dim: int, num_heads: int, update_depth: int,
                       message_gen: bool = True, event_write: bool = False, top_down: bool = True, vector_latent=False, freq=None, start_from_cycle_end=True,
                       cfg_embed: dict = {}, cfg_write: dict = {}, cfg_message: dict = {}, cfg_update: dict = {}, cfg_image_write: Optional[dict] = None) -> None:
        super().__init__()
        self.event_write = event_write
        self.message_gen_enabled = message_gen
        self.top_down_enabled = top_down
        self.image_write_enabled = cfg_image_write is not None
        self.use_multi_process = False
        self.use_cuda_stream = False
        self.cuda_stream_priority = 0
        self.timing_offset = 0 if start_from_cycle_end else 1
        self.fast_inference = False

        self.latent_size = latent_size
        self.latent_dim = latent_dim
        self.output_dim = output_dim

        self.freq = freq
        self.warmup = freq + 1
        latent_stride = 2

        # init latent state
        H, W = latent_size
        if vector_latent:
            self.init_latent = nn.Parameter(torch.zeros(1, 1, latent_dim))
        else:
            self.init_latent = nn.Parameter(torch.zeros(1, H*W, latent_dim))
        trunc_normal_(self.init_latent, std=.02)

        if event_write:
            self.embed = EventEmbedding(latent_size=latent_size, **cfg_embed)
            self.write_bottom_up = EventWrite(input_dim, latent_dim, num_heads, mlp_ratio=4, latent_size=latent_size, **cfg_write)
        else:
            self.write_bottom_up = WriteBottomUp(input_dim, latent_dim, num_heads, latent_stride, mlp_ratio=4, **cfg_write)

        if top_down:
            self.write_top_down = WriteTopDown(latent_dim, mlp_ratio=4)

        self.update_norm = nn.LayerNorm(latent_dim)
        self.update = self._make_update_layer(latent_dim, update_depth, num_heads, cfg_update)

        if message_gen:
            self.message_gen = MessageGen(input_dim, latent_dim, num_heads, latent_stride, **cfg_message)

        if self.image_write_enabled:
            self.write_image = ImageWrite(latent_dim, num_heads, mlp_ratio=4, **cfg_image_write)

        self.norm_out = nn.LayerNorm(latent_dim)
        self.proj_out = Linear(latent_dim, output_dim, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_act=False)

    def _make_update_layer(self, latent_dim: int, update_depth: int, num_heads: int, cfg_update: dict) -> nn.Module:
        if 'drop_path' in cfg_update:
            drop_path = cfg_update.pop('drop_path')
            assert len(drop_path) == update_depth
        else:
            drop_path = [0] * update_depth

        self.layer_type = cfg_update.pop('layer')

        if self.layer_type == 'transformer':
            enable_cyclic_shift = cfg_update.pop('enable_cyclic_shift')
            cyclic_shift = [ (i % 2 == 1 and enable_cyclic_shift) for i in range(update_depth) ]
            update_layer = nn.ModuleList([ TransformerBlock(latent_dim, num_heads, cyclic_shift=cyclic_shift[i], drop_path=drop_path[i], **cfg_update) for i in range(update_depth)])
        elif self.layer_type == 'mv2':
            update_layer = nn.ModuleList([ MV2Block(latent_dim, latent_dim, **cfg_update) for i in range(update_depth) ])
        elif self.layer_type == 'res':
            update_layer = nn.ModuleList([ ResBlock((latent_dim, latent_dim), **cfg_update) for i in range(update_depth) ])
        elif self.layer_type == 'res2':
            update_layer = nn.ModuleList([ ResBlock((latent_dim, latent_dim // 4, latent_dim), **cfg_update) for i in range(update_depth) ])
        elif self.layer_type == 'mvit':
            update_layer = nn.ModuleList([ MViTStage(latent_dim, latent_dim, stride=1, expansion=4, num_layers=update_depth, num_heads=num_heads, **cfg_update) ])
        elif self.layer_type == 'cnext':
            update_layer = nn.ModuleList([ ConvNeXtBlock(latent_dim, drop_path=drop_path[i], **cfg_update) for i in range(update_depth) ])
        else:
            raise RuntimeError

        return update_layer

    @property
    def no_decay_set(self):
        return {'init_latent', 'bias'}

    def init_states(self, batch_size, image_size):
        H, W = self.latent_size
        lat = self.init_latent.expand(batch_size, H*W, -1)
        meta = dict(shape=[batch_size,H,W])
        self.latent = SeqData(lat, meta)
        self.message = None
        self.out_buffer = None
        self.out = None
        self.time_idx = 0
        if self.image_write_enabled:
            self.image_buffer = ImageBuffer(batch_size, image_size, self.init_latent.device)

    def _to_fast_model(self):
        if self.event_write:
            self.fast_inference = True
            self.embed._generate_kv_table(self.latent_dim, self.write_bottom_up)

    def detach(self):
        self.latent = self.latent.detach()
        self.message = self.message.detach() if self.message is not None else None
        self.out_buffer = self.out_buffer.detach() if self.out_buffer is not None else None
        self.out = self.out.detach() if self.out is not None else None

    def _sync_write_update_readout(self):
        cycle_st, cycle_ed, _ = self._timing_flags()
        if cycle_ed:
            if self.use_multi_process and not self.training:
                lat, out = self.return_queue.get()    # sync and get results
                self.latent = lat if lat is not None else self.latent
                self.out_buffer = out
            elif self.use_cuda_stream and not self.training:
                torch.cuda.current_stream().wait_stream(self.cuda_stream)    # sync cuda stream
                self.out_buffer = self.out.clone() if self.out is not None else None
            else: # single thread
                self.out_buffer = self.out    # set output into buffer

    def _sync_message_gen(self):
        cycle_st, cycle_ed, warmup_finished = self._timing_flags()
        if cycle_st:
            if self.use_multi_process and not self.training:
                self.message, _ = self.return_queue.get()
                if not warmup_finished:
                    self.message = None
            elif self.use_cuda_stream and not self.training:
                torch.cuda.current_stream().wait_stream(self.cuda_stream)    # sync
            else: # single thread
                pass # no sync required

    def sync_and_get_state(self):
        if self.freq > 1:
            self._sync_write_update_readout()
        if self.message_gen_enabled:
            self._sync_message_gen()
        return self.latent, self.message

    def place_holder(self, num_output):
        return torch.zeros(num_output, self.output_dim, *self.latent_size, dtype=self.init_latent.dtype, device=self.init_latent.device)

    def _timing_flags(self) -> Tuple[bool]:
        idx = self.time_idx + self.timing_offset
        cycle_st = (idx-1) % self.freq == 0    # start of a cycle
        cycle_ed = (idx+0) % self.freq == 0    # end of a cycle
        warmup_finished = idx > self.warmup    # warmup for message gen is finished or not
        return cycle_st, cycle_ed, warmup_finished

    def prepair_for_inference(self, batch_size, device, input_dim=None, input_size=None, image_size=None):
        if self.use_multi_process and not self.training:
            self.init_states(batch_size, image_size)
            self.inputs_ = SeqData.zeros(batch_size, input_dim, *input_size, dtype=self.init_latent.dtype, device=device)
            self.message_from_top_ = SeqData.zeros(batch_size, self.latent_dim, *self.latent_size, dtype=self.init_latent.dtype, device=device)
            self.inputs_.share_memory_()
            self.message_from_top_.share_memory_()

            # image inputs
            if self.image_write_enabled:
                height, width = image_size
                self.image_input_ = torch.zeros(batch_size, 3, height, width, dtype=self.init_latent.dtype, device=device)
                self.image_input_.share_memory_()
                self.image_valid_batch_ = torch.zeros(batch_size, dtype=bool, device=device)
                self.image_valid_batch_.share_memory_()

            self.queue = mp.Queue()
            self.return_queue = mp.Queue()
            self.proc = mp.Process(target=self._runner, args=(self, batch_size, device))
            self.proc.start()

        elif self.use_cuda_stream and not self.training:
            self.cuda_stream = torch.cuda.Stream(device=device, priority=self.cuda_stream_priority)
            self.to(device)
            self.init_states(batch_size, image_size)

        else:
            self.to(device)
            self.init_states(batch_size, image_size)

    def forward(self, inputs, message_from_top=None, image_input=None, event_metas=None, fast_training=False):
        if self.use_multi_process and not self.training:
            self._inference_mp(inputs, message_from_top, image_input, event_metas)
        elif self.use_cuda_stream and not self.training:
            self._inference_cuda_stream(inputs, message_from_top, image_input, event_metas)
        else:
            self._forward(inputs, message_from_top, image_input, event_metas, fast_training)

        if self.freq == 1:
            # require sync at the end of each time step
            self._sync_write_update_readout()

        self.time_idx += 1

        return self.out_buffer


    def _forward(self, inputs, message_from_top=None, image_input=None, event_metas=None, fast_training=False):
        cycle_st, cycle_ed, warmup_finished = self._timing_flags()

        inputs, message_from_top, image_input = self._to_device([inputs, message_from_top, image_input])

        if self.image_write_enabled:
            self.image_buffer.write(image_input)

        # generate messages
        self.message = None
        if cycle_ed and self.message_gen_enabled:
            self.message = self._forward_message(inputs)
        if not warmup_finished:
            self.message = None

        # top-down write
        if cycle_st and message_from_top is not None and self.top_down_enabled:
            self.latent = self._forward_write_top_down(message_from_top)

        # bottom-up write, readout
        if cycle_st:
            if self.image_write_enabled:
                images, valid_batch = self.image_buffer.read()
                self.latent = self._forward_write_image(images, valid_batch)
            if self.event_write:
                self.latent = self._forward_event_write(inputs, event_metas, fast_training)
            else:
                self.latent = self._forward_write_bottom_up(inputs)
            self.latent = self._forward_update()
            self.out = self._readout()

        return self.out_buffer

    def _inference_cuda_stream(self, inputs, message_from_top=None, image_input=None, event_metas=None):
        cycle_st, cycle_ed, warmup_finished = self._timing_flags()

        inputs, message_from_top, image_input = self._to_device([inputs, message_from_top, image_input])

        if self.image_write_enabled:
            self.image_buffer.write(image_input)

        self.cuda_stream.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(self.cuda_stream):
            # generate messages
            self.message = None
            if cycle_ed and self.message_gen_enabled:
                self.message = self._forward_message(inputs)
            if not warmup_finished:
                self.message = None

            # top-down write
            if cycle_st and message_from_top is not None and self.top_down_enabled:
                self.latent = self._forward_write_top_down(message_from_top)

            # bottom-up write, readout
            if cycle_st:
                if self.image_write_enabled:
                    images, valid_batch = self.image_buffer.read()
                    self.latent = self._forward_write_image(images, valid_batch)
                if self.event_write:
                    self.latent = self._forward_event_write(inputs, event_metas)
                else:
                    self.latent = self._forward_write_bottom_up(inputs)
                self.latent = self._forward_update()
                self.out = self._readout()

    def _inference_mp(self, inputs, message_from_top=None, image_input=None, event_metas=None):
        assert event_metas is None    # multi processing is not supported for event write

        cycle_st, cycle_ed, warmup_finished = self._timing_flags()

        if self.image_write_enabled:
            self.image_buffer.write(image_input)

        # generate messages
        self.message = None
        if cycle_ed and self.message_gen_enabled:
            self.inputs_.copy_(inputs)
            self.queue.put([self.inputs_, 'message_gen'])

        # top-down write
        if cycle_st and message_from_top is not None and self.top_down_enabled:
            self.message_from_top_.copy_(message_from_top)
            self.queue.put([self.message_from_top_, 'write_top_down'])

        # bottom-up write, update, readout
        if (cycle_st and self.top_down_enabled) or (cycle_ed and not self.top_down_enabled):
            # image write
            if self.image_write_enabled:
                images, valid_batch = self.image_buffer.read()
                self.image_input_.copy_(images)
                self.image_valid_batch_.copy_(valid_batch)
                self.queue.put([[self.image_input_, self.image_valid_batch_], 'write_image'])

            self.inputs_.copy_(inputs)
            self.queue.put([self.inputs_, 'write_bottom_up'])

    @staticmethod
    @torch.no_grad()
    def _runner(lat, batch_size, device):
        lat = lat.to(device)
        #lat.return_queue.put([lat.latent,None])
        lat.return_queue.put([None,None])

        while True:
            inputs, task = lat.queue.get()
            if task == 'terminate':
                break

            elif task == 'message_gen':
                msg = lat._forward_message(inputs)
                lat.return_queue.put([msg, None])

            elif task == 'write_top_down':
                lat.latent = lat._forward_write_top_down(inputs)

            elif task == 'write_image':
                image, valid_batch = inputs
                lat.latent = lat._forward_write_image(image, valid_batch)

            elif task == 'write_bottom_up':
                lat.latent = lat._forward_write_bottom_up(inputs)
                lat.latent = lat._forward_update()
                out = lat._readout()
                lat.return_queue.put([lat.latent, out])

    def _to_device(self, data, device=None):
        device = device or self.init_latent.device
        if data is None:
            return data
        elif isinstance(data, (torch.Tensor, SeqData)):
            return data.to(device)
        elif isinstance(data, (list, tuple)):
            return [ self._to_device(d, device) for d in data ]
        elif isinstance(data, dict):
            return { k: self._to_device(v, device) for k, v in data.items() }
        else:
            return data

    def termination(self):
        if self.use_multi_process and not self.training:
            del self.inputs_
            del self.message_from_top_
            self.queue.put([None, 'terminate'])
            self.proc.join()
            del self.queue
            del self.return_queue
            torch.cuda.empty_cache()
        else:
            pass

    def _forward_write_image(self, image: Tensor, valid_batch: Tensor) -> SeqData:
        latent = self.write_image(self.latent, image, valid_batch)
        return latent

    def _forward_event_write(self, inputs: SeqData, event_metas: list, fast_training: bool = False) -> SeqData:
        if fast_training:
            key, value, query_indices = inputs
            if len(key) == 0:
                return self.latent
            latent = SeqData(self.write_bottom_up.forward_fast_train(self.latent.data, key, value, query_indices), self.latent.meta)
        elif self.fast_inference:
            curr_time, duration = event_metas
            key, value, query_indices = self.embed(inputs, curr_time, duration)
            if key is None:
                return self.latent    # no events
            latent = SeqData(self.write_bottom_up.forward_fast_train(self.latent.data, key, value, query_indices), self.latent.meta)
        else:
            curr_time, duration = event_metas
            data_array, query_indices = self.embed(inputs, curr_time, duration)
            if data_array is None:
                return self.latent    # no events
            latent = SeqData(self.write_bottom_up(self.latent.data, data_array, query_indices), self.latent.meta)

        return latent

    def _forward_write_bottom_up(self, inputs: SeqData, fast_training: bool = False) -> SeqData:
        return self.write_bottom_up(self.latent, inputs)

    def _forward_write_top_down(self, message: Tensor) -> SeqData:
        return self.write_top_down(self.latent, message=message)

    def _forward_update(self) -> SeqData:
        latent = SeqData(self.update_norm(self.latent.data), self.latent.meta)
        if self.layer_type == 'transformer':
            for layer in self.update:
                latent = layer(latent)
        else:
            z = latent.to_2D()
            for layer in self.update:
                z = layer(z)
            latent = SeqData.from_2D(z)
        return latent

    def _forward_message(self, inputs: SeqData) -> SeqData:
        return self.message_gen(self.latent, inputs)

    def _readout(self) -> Tensor:
        z = self.proj_out(self.norm_out(self.latent.data))
        latent = SeqData(z, self.latent.meta)
        return latent.to_2D()
        
def scatter_add(values: Tensor, indices: Tensor, dim: int, dim_size: int, use_torch: bool = True, out: Optional[Tensor] = None) -> Tensor:
    if use_torch:
        return scatter(values, indices, reduce='add', dim=dim, dim_size=dim_size, out=out)
    else:
        if out is not None:
            output = out
        else:
            shape = list(values.shape)
            shape[dim] = dim_size
            output = torch.zeros(*shape, device=values.device, dtype=values.dtype)

        if indices.ndim != values.ndim:
            # broadcast
            assert indices.ndim == 1
            view = [1] * values.ndim
            view[dim] = len(indices)
            repeat = list(values.shape)
            repeat[dim] = len(indices)
            indices = indices.view(*view).expand(*repeat)
        output.scatter_add_(dim, indices, values)
        return output

class Downsample(BlockBase):
    def __init__(self, input_dim: int, output_dim: int, stride: int, method: str = 'conv', proj: bool = True, pre_act: bool = True, pre_layer_norm: bool = False) -> None:
        super().__init__()
        self.method = method
        proj = proj or input_dim != output_dim

        self.norm = nn.LayerNorm(input_dim) if pre_layer_norm else nn.Identity()

        if method == 'conv':
            self.downsample = ConvBlock(input_dim, input_dim,  kernel_size=3, padding=1, stride=stride, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act)
            self.proj = ConvBlock(input_dim, output_dim, kernel_size=1, padding=0, stride=1, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act) if proj else nn.Identity()
        elif method == 'mv2':
            proj = True
            self.downsample = ConvBlock(input_dim, input_dim,  kernel_size=3, padding=1, s=stride, groups=input_dim, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act)
            self.proj = ConvBlock(input_dim, output_dim, kernel_size=1, padding=0, stride=1, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act) if proj else nn.Identity()
        elif method == 'pool':
            self.downsample = nn.MaxPool2d(stride, stride=stride)
            self.proj = ConvBlock(input_dim, output_dim, kernel_size=1, padding=0, stride=1, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act) if proj else nn.Identity()
        elif method == 'merge':
            self.downsample = PatchMergingCross(input_dim, stride, norm_layer=nn.LayerNorm, patch_size=[3], out_dim=input_dim)
            self.proj = Linear(input_dim, output_dim, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_act=pre_act) if proj else nn.Identity()
        elif method == 'msmerge':
            self.downsample = PatchMergingCross(input_dim, stride, norm_layer=nn.LayerNorm, patch_size=[3, 5, 7], out_dim=input_dim)
            self.proj = Linear(input_dim, output_dim, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_act=pre_act) if proj else nn.Identity()
        elif method == 'down':
            self.downsample = partial(F.interpolate, scale_factor=1./stride, mode='bilinear')
            self.proj = ConvBlock(input_dim, output_dim, kernel_size=1, padding=0, stride=1, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act) if proj else nn.Identity()
        elif method == 'none':
            self.proj = Linear(input_dim, output_dim, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_act=pre_act) if proj else nn.Identity()
        else:
            raise RuntimeError

    def forward(self, seq_x: SeqData) -> SeqData:
        if self.method in ('conv', 'mv2', 'pool', 'down'):
            seq_x = SeqData(self.norm(seq_x.data), seq_x.meta)
            x = seq_x.to_2D()
            x = self.downsample(x)
            x = self.proj(x)
            seq_x = SeqData.from_2D(x)
        elif self.method in ('merge', 'msmerge'):
            seq_x = self.downsample(seq_x)
            seq_x = SeqData(self.proj(seq_x.data), seq_x.meta)
        elif self.method == 'none':
            seq_x = SeqData(self.proj(seq_x.data), seq_x.meta)
        return seq_x

class Upsample(BlockBase):
    def __init__(self, input_dim: int, output_dim: int, scale_factor: int, method: str = 'bilinear', proj: bool = True, pre_act: bool = True, pre_layer_norm: bool = False) -> None:
        super().__init__()
        self.method = method
        proj = proj or input_dim != output_dim

        self.norm = nn.LayerNorm(input_dim) if pre_layer_norm else nn.Identity()

        if method == 'bilinear':
            self.proj = ConvBlock(input_dim, output_dim, kernel_size=1, padding=0, stride=1, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act) if proj else nn.Identity()
            self.upsample = nn.Upsample(scale_factor=scale_factor, mode='bilinear')
        elif method == 'shuffle':
            self.proj = ConvBlock(input_dim, output_dim, kernel_size=1, padding=0, stride=1, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act) if proj else nn.Identity()
            self.upsample = nn.PixelShuffle(upscale_factor=scale_factor)
        elif method == 'conv-bilinear':
            self.proj = ConvBlock(input_dim, output_dim, kernel_size=1, padding=0, stride=1, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act) if proj else nn.Identity()
            self.upsample = UpConvBlock(output_dim, output_dim, scale_factor=scale_factor, kernel_size=3, padding=1, stride=1, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act)
        elif method == 'conv-shuffle':
            self.proj = ConvBlock(input_dim, output_dim, kernel_size=1, padding=0, stride=1, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act) if proj else nn.Identity()
            self.upsample = ConvPixelShuffle(output_dim, output_dim, scale_factor=scale_factor, kernel_size=3, padding=1, depthwise=False, norm_layer=nn.GroupNorm, act_layer=nn.SiLU)
        elif method == 'dconv-shuffle':
            self.proj = ConvBlock(input_dim, output_dim, kernel_size=1, padding=0, stride=1, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act) if proj else nn.Identity()
            self.upsample = ConvPixelShuffle(output_dim, output_dim, scale_factor=scale_factor, kernel_size=3, padding=1, depthwise=True, norm_layer=nn.GroupNorm, act_layer=nn.SiLU)
        elif method == 'mv2':
            proj = True
            self.proj = ConvBlock(input_dim, output_dim, kernel_size=1, padding=0, stride=1, groups=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act) if proj else nn.Identity()
            self.upsample = UpConvBlock(output_dim, output_dim, scale_factor=sampling_stride, kernel_size=3, padding=1, groups=output_dim, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, gn_dim=32, pre_act=pre_act)
        elif method == 'none':
            self.proj = Linear(input_dim, output_dim, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_act=pre_act) if proj else nn.Identity()
        else:
            raise RuntimeError

    def forward(self, seq_x: SeqData) -> SeqData:
        if self.method == 'none':
            seq_x = SeqData(self.proj(seq_x.data), seq_x.meta)
        else:
            seq_x = SeqData(self.norm(seq_x.data), seq_x.meta)
            x = seq_x.to_2D()
            x = self.proj(x)
            x = self.upsample(x)
            seq_x = SeqData.from_2D(x)
        return seq_x

class EventWrite(BlockBase):
    def __init__(self, input_dim: int, latent_dim: int, num_heads: int,
                 mlp_ratio: int = 4, act_layer: type = nn.GELU, norm_layer: type = nn.LayerNorm, drop: float = 0.,
                 noise_filter: bool = True, filter_init: float = 0., latent_size: Tuple[int, int] = None, linearized_attn: bool = False) -> None:
        super().__init__()
        self.norm_latent = norm_layer(latent_dim)
        self.norm_input = norm_layer(input_dim)
        self.attn = SparseCrossAttention(latent_dim, input_dim, num_heads, qkv_bias=True, proj_drop=0., noise_filter=noise_filter, filter_init=filter_init, latent_size=latent_size, linearized=linearized_attn)
        self.norm_mlp = norm_layer(latent_dim)
        self.mlp = Mlp(in_features=latent_dim, hidden_features=latent_dim * mlp_ratio, act_layer=act_layer, drop=drop)

    def forward_fast_train(self, z: Tensor, key: Tensor, value: Tensor, q_indices: Tensor) -> Tensor:
        lat = z.clone()
        z = self.norm_latent(z)
        z = self.attn.forward_fast_train(z, key, value, q_indices)
        z = lat + z
        z = z + self.mlp(self.norm_mlp(z))
        return z

    def forward(self, z: Tensor, x: Tensor, q_indices: Tensor) -> Tensor:
        lat = z.clone()
        z = self.norm_latent(z)
        x = self.norm_input(x)
        z = self.attn(z, x, q_indices)
        z = lat + z
        z = z + self.mlp(self.norm_mlp(z))
        return z

class WriteBottomUp(BlockBase):
    def __init__(self, input_dim: int, latent_dim: int, num_heads: int, latent_stride: int, downsample: str = 'conv', window_size: Optional[Tuple[int,int]] = None, grouping='intra-window',
                 input_proj: bool = True, mlp_ratio: int = 4, act_layer: type = nn.GELU, norm_layer: type = nn.LayerNorm, drop: float = 0.,
                 pos_dynamic: bool = False, pos_log_scale: bool = False, pre_act: bool = True, pre_layer_norm: bool = False) -> None:
        super().__init__()
        self.downsample = downsample
        self.norm_latent = norm_layer(latent_dim)
        self.norm_input = norm_layer(latent_dim)
        self.norm_mlp = norm_layer(latent_dim)
        self.mlp = Mlp(in_features=latent_dim, hidden_features=latent_dim * mlp_ratio, act_layer=act_layer, drop=drop)

        self.input_resize = Downsample(input_dim, latent_dim, latent_stride, method=downsample, proj=input_proj, pre_act=pre_act, pre_layer_norm=pre_layer_norm)
        self.attn = CrossAttention(latent_dim, latent_dim, wsize=window_size, kv_wsize=window_size, grouping=grouping, num_heads=num_heads, cyclic_shift=False,
                                   pos_dynamic=pos_dynamic, pos_log_scale=pos_log_scale, qkv_bias=True, proj_drop=0.)

    def forward(self, seq_z: SeqData, seq_x: SeqData) -> Tensor:
        seq_x = self.input_resize(seq_x)

        x, x_meta = seq_x.data, seq_x.meta
        z, z_meta = seq_z.data, seq_z.meta

        z1 = z.clone()

        x = self.norm_input(x)
        z = self.norm_latent(z)
        z = self.attn(z, x, z_meta, x_meta)

        z = z1 + z
        z = z + self.mlp(self.norm_mlp(z))

        seq_z = SeqData(z, z_meta)

        return seq_z

class ImageWrite(BlockBase):
    def __init__(self, latent_dim: int, num_heads: int, window_size: Optional[Tuple[int,int]] = None, grouping='intra-window',
                 mlp_ratio: int = 4, act_layer: type = nn.GELU, norm_layer: type = nn.LayerNorm, drop: float = 0.,
                 encoder_type: str = 'ResNet', encoder_layers: Tuple[int,int,int] = [2,2,2], encoder_dims: Tuple[int,int,int] = [64, 128, 256], encoder_drop_path: float = 0.) -> None:
        super().__init__()
        self.image_encoder = self.build_image_encoder(encoder_type, encoder_layers, encoder_dims, encoder_drop_path)
        feature_dim = encoder_dims[-1]
        self.norm_image = norm_layer(feature_dim)
        self.norm_latent = norm_layer(latent_dim)
        self.attn = CrossAttention(latent_dim, feature_dim, wsize=window_size, kv_wsize=window_size, grouping=grouping, num_heads=num_heads)
        self.norm_mlp = norm_layer(latent_dim)
        self.mlp = Mlp(in_features=latent_dim, hidden_features=latent_dim * mlp_ratio, act_layer=act_layer, drop=drop)

    def build_image_encoder(self, type, num_layers, dims, drop_path_rate=0.):
        L1, L2, L3 = num_layers
        D1, D2, D3 = dims

        if type == 'ResNet':
            image_encoder = nn.Sequential(
                ResEntry(inc=3, ent_strided=True, ent_pool=True, deepseg=True),
                ResStage(1, D1, num_layers=L1, bottleneck=False, stride=1, down_act=True, inc=128),
                ResStage(2, D2, num_layers=L2, bottleneck=False, stride=2, down_act=True),
                ResStage(3, D3, num_layers=L3, bottleneck=False, stride=2, down_act=True),
            )
        elif type == 'ConvNeXt':
            dp_rates = [ r.item() for r in torch.linspace(0, drop_path_rate, sum(num_layers)) ]
            dp1, dp2, dp3 = np.split(dp_rates, np.cumsum(num_layers)[:-1])
            image_encoder = nn.Sequential(
                ConvNeXtEntry( 3, D1),
                ConvNeXtStage(D1, D1, L1, dp_rates=dp1, downsample=False),
                ConvNeXtStage(D1, D2, L2, dp_rates=dp2, downsample_kernel_size=3),
                ConvNeXtStage(D2, D3, L3, dp_rates=dp3, downsample_kernel_size=3),
            )

        return image_encoder

    def forward(self, seq_z: SeqData, image: Tensor, valid_batch: Optional[Tensor] = None) -> Tensor:
        #if not valid_batch.any().item():
        #    return seq_z

        image_feature = self.image_encoder(image)
        seq_x = SeqData.from_2D(image_feature)
        x, x_meta = seq_x.data, seq_x.meta

        #valid_seq_z = seq_z.filter_batch(valid_batch)
        #z, z_meta = valid_seq_z.data, valid_seq_z.meta
        z, z_meta = seq_z.data, seq_z.meta

        z1 = z.clone()

        x = self.norm_image(x)
        z = self.norm_latent(z)
        z = self.attn(z, x, z_meta, x_meta)

        #z = z1 + z
        #z = z + self.mlp(self.norm_mlp(z))
        if valid_batch is not None:
            z = z1 + z * valid_batch[:,None,None]
            z = z + self.mlp(self.norm_mlp(z)) * valid_batch[:,None,None]
        else:
            z = z1 + z
            z = z + self.mlp(self.norm_mlp(z))

        #new_z = seq_z.data.clone()
        #new_z[valid_batch] = z
        #seq_z = SeqData(new_z, seq_z.meta)
        seq_z = SeqData(z, z_meta)

        return seq_z


class WriteTopDown(BlockBase):
    def __init__(self, latent_dim: int, mlp_ratio: int = 4, act_layer: type = nn.GELU, norm_layer: type = nn.LayerNorm, drop: float = 0.) -> None:
        super().__init__()
        self.norm_mlp = norm_layer(latent_dim)
        self.mlp = Mlp(in_features=latent_dim, hidden_features=latent_dim * mlp_ratio, act_layer=act_layer, drop=drop)

    def forward(self, seq_z: SeqData, message: Tensor) -> Tensor:
        z, z_meta = seq_z.data, seq_z.meta

        z = z + message.data
        z = z + self.mlp(self.norm_mlp(z))

        seq_z = SeqData(z, z_meta)

        return seq_z

class MessageGen(BlockBase):
    def __init__(self, input_dim: int, latent_dim: int, num_heads: int, latent_stride: int,
                 input_resize: str = 'merge', input_proj: bool = True, latent_resize: str = 'none', latent_proj: bool = False, out_proj : bool = True,
                 norm_layer: type = nn.LayerNorm, drop: float = 0., window_size: Optional[int] = None, grouping='intra-window', pos_dynamic: bool = False, pos_log_scale: bool = False,
                 pre_layer_norm: bool = False, pre_act: bool = True) -> None:
        super().__init__()
        assert input_resize == 'none' or latent_resize == 'none'
        assert input_proj == False or latent_proj == False

        input_attn_dim  = latent_dim if input_proj  else input_dim
        latent_attn_dim = input_dim  if latent_proj else latent_dim

        self.input_resize = Downsample(input_dim, input_attn_dim, latent_stride, method=input_resize, proj=input_proj, pre_act=pre_act, pre_layer_norm=pre_layer_norm)
        self.latent_resize = Upsample(latent_dim, latent_attn_dim, latent_stride, method=latent_resize, proj=latent_proj, pre_act=pre_act, pre_layer_norm=pre_layer_norm)

        self.norm_latent = norm_layer(latent_attn_dim)
        self.norm_input = norm_layer(input_attn_dim)
        self.attn = CrossAttention(input_attn_dim, latent_attn_dim, wsize=window_size, kv_wsize=window_size, grouping=grouping, num_heads=num_heads, cyclic_shift=False,
                                   pos_dynamic=pos_dynamic, pos_log_scale=pos_log_scale, qkv_bias=True, proj_drop=0.)
        self.out_proj = Linear(input_attn_dim, input_dim, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_act=True) if out_proj else nn.Identity()

    def forward(self, seq_z: SeqData, seq_x: SeqData) -> Tensor:
        B, H, W = seq_x.meta['shape']

        seq_x = self.input_resize(seq_x)
        seq_z = self.latent_resize(seq_z)

        x, x_meta = seq_x.data, seq_x.meta
        z, z_meta = seq_z.data, seq_z.meta

        x = self.norm_input(x)
        z = self.norm_latent(z)
        x = self.attn(x, z, x_meta, z_meta)
        x = self.out_proj(x)

        b, h, w = x_meta['shape']
        seq_x = SeqData(x, x_meta)

        if h != H or w != W:
            x = seq_x.to_2D()
            x = F.interpolate(x, size=(H, W), mode='bilinear')
            seq_x = SeqData.from_2D(x)

        return seq_x

class SparseCrossAttention(BlockBase):
    def __init__(self, dim: int, kv_dim: int, num_heads: int, qkv_bias: bool = True, qk_scale: Optional[float] = None, proj_drop: float = 0.,
                        noise_filter: bool = True, filter_init: float = 0., latent_size: Tuple[int, int] = None, linearized: bool = False) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        self.q  = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(kv_dim, dim*2, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.latent_size = latent_size
        self.noise_filter = noise_filter
        self.linearized = linearized
        if noise_filter:
            self.dustbin_weights = nn.Parameter(torch.empty(num_heads).fill_(filter_init))
        self.is_init_w_dust = False

    def _to_fast_model(self):
        if not self.noise_filter:
            return
        # assumes batch_size = 1
        B = 1
        zL = self.latent_size[0] * self.latent_size[1]
        w_dust, q_indices_dust = self._prepair_dust(B, zL)
        self.register_buffer('w_dust', w_dust)
        self.register_buffer('q_indices_dust', q_indices_dust)
        self.is_init_w_dust = True
        print('Prepair dust_bin in advance')

    @property
    def no_decay_set(self):
        return {'dustbin_weights'}

    def _prepair_dust(self, B, zL):
        if self.is_init_w_dust:
            return self.w_dust, self.q_indices_dust
        else:
            w_dust = self.dustbin_weights.view(1,self.num_heads).expand(B*zL, -1)
            q_indices_dust = torch.arange(B*zL, device=w_dust.device)
            return w_dust, q_indices_dust

    def softmax(self, weights, q_indices, z_shape, kv_shape):
        B, zL, _ = z_shape
        L, H, C = kv_shape

        max_weights = scatter(weights.detach(), q_indices, reduce='max', dim=0, dim_size=B*zL)    # (B*zL, H)
        if self.noise_filter:
            w_dust, q_indices_dust = self._prepair_dust(B, zL)
            max_weights = torch.maximum(max_weights, w_dust.detach())
            w_dust = (w_dust - max_weights).exp()
        weights = (weights - max_weights[q_indices]).exp()                               # (L, H)
        accm = scatter_add(weights, q_indices, dim=0, dim_size=B*zL, use_torch=False)    # (B*zL, H)
        if self.noise_filter:
            accm = accm + w_dust
        weights = weights / (accm[q_indices] + 1.0e-7)
        return weights

    def forward(self, z: Tensor, x: Tensor, q_indices: dict, kv_indices: Optional[Tensor] = None) -> Tensor:
        B, zL, C = z.shape
        H = self.num_heads

        # get key value from inputs
        kv = self.kv(x).view(-1, 2, H, C // H).permute(1,0,2,3).contiguous()
        key, value = kv[0], kv[1]    # (L, H, C)

        return self.forward_fast_train(z, key, value, q_indices, kv_indices)

    def forward_fast_train(self, z: Tensor, key: Tensor, value: Tensor, q_indices: dict, kv_indices: Optional[Tensor] = None) -> Tensor:
        device = z.device
        B, zL, C = z.shape
        H = self.num_heads

        # get key value from inputs
        if kv_indices is not None:
            key = key[kv_indices]
            value = value[kv_indices]

        # get query from latents
        query = self.q(z).view(-1, H, C // H)    # (B*zL, H, C)
        query = query * self.scale

        L, H, C = key.shape

        query = query[q_indices, :, :]    # (L, H, C)
        weights = (key * query).sum(-1)    # (L, H)
        weights = self.softmax(weights, q_indices, z.shape, key.shape)

        # get weighted feature
        message = weights[:,:,None] * value      # (L, H, C)
        message = message.view(L, -1)  # (L, HC)

        # aggregate
        message = scatter_add(message, q_indices, dim=0, dim_size=B*zL, use_torch=False)    # (B*zL, HC)

        if self._is_visualize:
            print('vis_attn_map')
            height, width = self.latent_size
            attn_map = scatter_add(weights, q_indices, dim=0, dim_size=B*zL, use_torch=False)    # (B*zL, H)
            attn_map = attn_map.view(B, height, width, H)[0].cpu().numpy()
            for i in range(100):
                fpath_out = self.dpath_out_vis + f'/attn_map_{i}.npy'
                if not os.path.isfile(fpath_out):
                    np.save(fpath_out, attn_map)
                    break

        # projection
        message = message.view(B, zL, H*C)
        message = self.proj(message)
        message = self.proj_drop(message)

        return message    # (B, zL, HC)

class EventEmbedding(BlockBase):
    def __init__(self, input_size: Tuple[int], latent_size: Tuple[int], discrete_time: bool = False, time_bins: int = 100,
                 duration: int = 5000, dynamic: List[bool] = False, dynamic_dim: Optional[List[int]] = None, out_dim: Optional[List[int]] = None) -> None:
        super().__init__()
        self.discrete_time = discrete_time
        self.time_bins = time_bins
        self.time_delta = duration // time_bins

        self.input_h = input_size[0]
        self.input_w = input_size[1]
        self.latent_h = latent_size[0]
        self.latent_w = latent_size[1]

        assert self.input_h % self.latent_h == 0
        assert self.input_w % self.latent_w == 0
        self.window_h = self.input_h // self.latent_h
        self.window_w = self.input_w // self.latent_w

        H, W, T = self.window_h, self.window_w, time_bins
        self.xy   = PositionEmbedding2D(W, H, out_dim[0], dynamic=dynamic[0], dynamic_dim=dynamic_dim[0], shift_normalize=True)
        self.time = PositionEmbedding1D(T   , out_dim[1], dynamic=dynamic[1], dynamic_dim=dynamic_dim[1], shift_normalize=True, scale_normalize=True)
        self.pol  = PositionEmbedding1D(2   , out_dim[2], dynamic=dynamic[2], dynamic_dim=dynamic_dim[2])

    def generate_param_table(self) -> None:
        self.xy.generate_param_table()
        self.time.generate_param_table()
        self.pol.generate_param_table(data=torch.tensor([-1,1], dtype=torch.float).view(-1,1))

    def _forward_single(self, events: Tensor, curr_time: int, duration: int) -> Tuple[Tensor,Tensor]:
        if len(events) == 0:
            return None, None

        t, x, y, p = events[:,0], events[:,1], events[:,2], events[:,3]
        device = x.device

        t0 = curr_time - duration
        dt = t - t0

        # window indices
        #wx = x // self.window_w
        #wy = y // self.window_h
        wx = torch.div(x, self.window_w, rounding_mode='trunc')
        wy = torch.div(y, self.window_h, rounding_mode='trunc')
        indices = (wy * self.latent_w + wx).long()

        # get relative position and discretize time
        x = x % self.window_w
        y = y % self.window_h
        if self.discrete_time:
            #dt = (dt // int(self.time_delta))
            dt = torch.div(dt, int(self.time_delta), rounding_mode='trunc')
        else:
            dt = dt / self.time_delta

        # get embeddings
        xy_embedding = self.xy(x, y)
        time_embedding = self.time(dt)
        pol_embedding = self.pol(p)

        embeddings = torch.cat([xy_embedding, time_embedding, pol_embedding], dim=1)

        return embeddings, indices    # (L, C1+C2+C3), (L,)

    def _forward_single_fast(self, events: Tensor, curr_time: int, duration: int) -> Tuple[Tensor,Tensor]:
        if len(events) == 0:
            return None, None, None

        assert self.discrete_time

        t, x, y, p = events[:,0], events[:,1], events[:,2], events[:,3]
        device = x.device

        t0 = curr_time - duration
        dt = t - t0

        # window indices
        wx = torch.div(x, self.window_w, rounding_mode='trunc')
        wy = torch.div(y, self.window_h, rounding_mode='trunc')
        indices = (wy * self.latent_w + wx).long()

        # get relative position and discretize time
        x = x % self.window_w
        y = y % self.window_h
        dt = torch.div(dt, int(self.time_delta), rounding_mode='trunc')

        # get embeddings
        key = self.key_table[dt,x,y,p]
        value = self.value_table[dt,x,y,p]

        return key, value, indices    # (L, C1+C2+C3), (L,)

    def _generate_kv_table(self, latent_dim, write_bottom_up):
        self._forward_single = self._forward_single_fast
        t_size = self.time.x_size
        x_size = self.xy.x_size
        y_size = self.xy.y_size
        p_size = self.pol.x_size

        device = self.xy.embed[0].linear.weight.device
        t = torch.arange(t_size, device=device, dtype=torch.float)
        x = torch.arange(x_size, device=device, dtype=torch.float)
        y = torch.arange(y_size, device=device, dtype=torch.float)
        p = torch.tensor([-1,1], device=device, dtype=torch.float)
        t, x, y, p = torch.meshgrid(t, x, y, p)

        # get embeddings
        xy_embedding = self.xy(x.reshape(-1), y.reshape(-1))
        time_embedding = self.time(t.reshape(-1))
        pol_embedding = self.pol(p.reshape(-1))
        embeddings = torch.cat([xy_embedding, time_embedding, pol_embedding], dim=1)

        H = write_bottom_up.attn.num_heads
        C = latent_dim
        embeddings = write_bottom_up.norm_input(embeddings)
        kv = write_bottom_up.attn.kv(embeddings).view(-1, 2, H, C // H).permute(1,0,2,3).contiguous()
        key, value = kv[0], kv[1]    # (L, H, C')

        self.key_table = key.view(t_size, x_size, y_size, p_size, H, -1)
        self.value_table = value.view(t_size, x_size, y_size, p_size, H, -1)
        print('Generate Key Value table for events')

    #def _to_fast_model(self):
    #    self.xy.generate_param_table()
    #    self.time.generate_param_table()
    #    self.pol.generate_param_table(data=torch.tensor([-1,1], dtype=torch.float).view(-1,1))
    #    print('Generate table for embedding xy, t, p')

    def forward(self, events: Tensor, curr_time: List[int], duration: List[int]) -> Tuple[Tensor,Tensor]:
        if len(curr_time) == 1:
            events = events[0]
            curr_time = curr_time[0]
            duration = duration[0]
            return self._forward_single(events, curr_time, duration)
        else:
            dt, x, y, p, b = self.preproc_events(events, curr_time, duration)
            return self._forward(dt, x, y, p, b)

    #def forward_fast_train(self, evdata: Tensor, batch_indices: Tensor) -> Tuple[Tensor,Tensor]:
    #    dt, x, y, p = evdata[:,0], evdata[:,1], evdata[:,2], evdata[:,3]
    #    b = batch_indices
    #    return self._forward(dt, x, y, p, b)

    def forward_fast_train(self, lat1, list_events, list_image_metas):
        split_sizes = []
        batch_indices = []
        evdata = []
        for events, image_meta in zip(list_events, list_image_metas):
            curr_time = [ meta['curr_time_crop'] for meta in image_meta ]
            duration = [ meta['delta_t'] for meta in image_meta ]

            dt, x, y, p, b = self.preproc_events(events, curr_time, duration)
            ev = torch.stack([dt, x, y, p], dim=-1)

            evdata.append(ev)
            batch_indices.append(b)
            split_sizes.append(len(dt))

        evdata = torch.cat(evdata, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)

        dt, x, y, p = evdata[:,0], evdata[:,1], evdata[:,2], evdata[:,3]
        b = batch_indices
        ev_tensor, ev_q = self._forward(dt, x, y, p, batch_indices)

        if ev_tensor is None:
            list_evdata = [ [ list(), list(), list() ] for _ in range(len(list_events)) ]
            return list_evdata

        H = lat1.write_bottom_up.attn.num_heads
        C = lat1.latent_dim
        ev_tensor = lat1.write_bottom_up.norm_input(ev_tensor)
        kv = lat1.write_bottom_up.attn.kv(ev_tensor).view(-1, 2, H, C // H).permute(1,0,2,3).contiguous()
        key, value = kv[0], kv[1]    # (L, H, C)

        list_keys = key.split(split_sizes)
        list_values = value.split(split_sizes)
        list_ev_q = ev_q.split(split_sizes)

        list_evdata = list(zip(list_keys, list_values, list_ev_q))

        return list_evdata

    def _forward(self, dt, x, y, p, b):
        if len(dt) == 0:
            return None, None

        if not self.pol.dynamic:
            p = ((p + 1) * 0.5).to(p.dtype)

        # window indices
        wx = torch.div(x, self.window_w, rounding_mode='trunc')
        wy = torch.div(y, self.window_h, rounding_mode='trunc')
        #wx = x // self.window_w
        #wy = y // self.window_h
        indices = (b * self.latent_h * self.latent_w + wy * self.latent_w + wx).long()

        # get relative position and discretize time
        x = x % self.window_w
        y = y % self.window_h
        if self.discrete_time:
            dt = torch.div(dt, self.time_delta, rounding_mode='trunc')
            #dt = (dt // self.time_delta)
        else:
            dt = dt / self.time_delta

        # get embeddings
        xy_embedding = self.xy(x, y)
        time_embedding = self.time(dt)
        pol_embedding = self.pol(p)

        embeddings = torch.cat([xy_embedding, time_embedding, pol_embedding], dim=1)

        return embeddings, indices    # (L, C1+C2+C3), (L,)

    def preproc_events(self, events, curr_time, duration):
        output = []
        for bidx in range(len(events)):
            evt = events[bidx]
            if len(evt) == 0:
                output.append(torch.empty(0,5, dtype=evt.dtype, device=evt.device))
                continue

            t0 = curr_time[bidx] - duration[bidx]
            evt[:,0] = evt[:,0] - t0
            b = torch.tensor([bidx] * len(evt), dtype=evt.dtype, device=evt.device).view(-1,1)
            out = torch.cat([evt, b], dim=1)
            output.append(out)
        output = torch.cat(output, dim=0)

        return output[:,0], output[:,1], output[:,2], output[:,3], output[:,4]

class ImageBuffer:
    def __init__(self, batch_size, image_size, device):
        self.batch_size = batch_size
        self.data = torch.zeros(self.batch_size, 3, *image_size, device=device)
        self.valid_mask = torch.zeros(self.batch_size, dtype=bool, device=device)

    def write(self, images: List[Tensor]):
        if images is None:
            return

        for bidx, image in enumerate(images):
            if image is None:
                continue

            self.data[bidx] = image
            self.valid_mask[bidx] = True

    def read(self, device=None):
        if device is None:
            device = self.data.device

        #out_data = self.data[self.valid_mask].to(device)
        out_data = self.data.clone()
        out_mask = self.valid_mask.clone()
        out_data = out_data.to(device)
        out_mask = out_mask.to(device)
        self.valid_mask.fill_(0)
        return out_data, out_mask



