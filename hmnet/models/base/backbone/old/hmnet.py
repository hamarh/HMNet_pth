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
from collections import OrderedDict, defaultdict

import torch.multiprocessing as mp
if mp.get_start_method(True) is None:
    mp.set_start_method('spawn')

import random
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import trunc_normal_
from torch_scatter import scatter


from torchtools.base_models.backbone.vit import MobileViTAttachment, Mlp, TokenGrouping, QKVTransform, CrossAttention, PositionEmbedding1D, PositionEmbedding2D, PatchMergingCross, SeqData, TransformerBlock, SparseAttentionBlock, Stage
from torchtools.base_models.layers import ConvBlock, Linear, MV2Block, UpConvBlock, ConvPixelShuffle
from torchtools.base_models.backbone.resnet import ResBlock, ResStage, ResEntry
from torchtools.base_models.backbone.convnext import ConvNeXtBlock, ConvNeXtEntry, ConvNeXtStage
from torchtools.base_models.blocks import BlockBase
from torchtools.init import init_transformer, load_state_dict_matched

from torch import Tensor
from typing import Tuple, List, Optional, Dict
Meta = Dict[str,Tuple[int,int,int]]

from common.utils import Timer
timer = Timer()

class HMBackbone(BlockBase):
    def __init__(self, input_size, latent_sizes, latent_dims, embed_dim, num_heads, depth,
                 cfg_embed=None, cfg_lat1=None, cfg_lat2=None, cfg_lat3=None,
                 freq2=3, freq3=9, read2=True, read3=True, warmup=20, warmup2=None, warmup3=None) -> None:
        super().__init__()
        self.freq2 = freq2
        self.freq3 = freq3
        self.warmup2 = warmup2 or freq2 + 1
        self.warmup3 = warmup3 or freq3 + 1
        self.warmup = warmup
        self.read2_enabled = read2
        self.read3_enabled = read3

        L1, L2, L3 = latent_sizes
        D1, D2, D3 = latent_dims
        D0 = sum(embed_dim)
        H1, H2, H3 = num_heads
        N1, N2, N3 = depth

        self.embed = EventEmbedding(input_size=input_size, latent_size=L1, out_dim=embed_dim, **cfg_embed)
        self.lat1 = LatentTrans(latent_size=L1, input_dim=D0, latent_dim=D1, num_heads=H1, update_depth=N1, message_gen=False, event_write=True,  top_down=True,  **cfg_lat1)
        self.lat2 = LatentTrans(latent_size=L2, input_dim=D1, latent_dim=D2, num_heads=H2, update_depth=N2, message_gen=True,  event_write=False, top_down=True,  **cfg_lat2)
        self.lat3 = LatentTrans(latent_size=L3, input_dim=D2, latent_dim=D3, num_heads=H3, update_depth=N3, message_gen=True,  event_write=False, top_down=False, **cfg_lat3)
        self.set_module_names()

    def init_weights(self, pretrained=None):
        init_transformer(self.modules())

    def print_grad_norm(self):
        super().print_grad_norm()
        self.embed.print_grad_norm()
        self.lat1.print_grad_norm()
        self.lat2.print_grad_norm()
        self.lat3.print_grad_norm()

    def clip_grad(self, method='norm', max_norm=1.0, clip_value=0.5):
        self.embed.clip_grad(method, max_norm, clip_value)
        self.lat1.clip_grad(method, max_norm, clip_value)
        self.lat2.clip_grad(method, max_norm, clip_value)
        self.lat3.clip_grad(method, max_norm, clip_value)

    def to_cuda(self, d0, d1, d2):
        self.devices = (d0, d1, d2)
        self.embed = self.embed.to(d0)
        self.lat1 = self.lat1.to(d0)
        self.lat2 = self.lat2.to(d1)
        self.lat3 = self.lat3.to(d2)

    def _timing_flags(self, idx: int) -> Tuple[bool]:
        M2 = (idx+0) % self.freq2 == 0
        M3 = (idx+0) % self.freq3 == 0
        W2 = (idx-1) % self.freq2 == 0
        W3 = (idx-1) % self.freq3 == 0
        readable2 = idx > self.warmup2
        readable3 = idx > self.warmup3
        R1 = W2 and readable2 and self.read2_enabled
        R2 = W3 and readable3 and self.read3_enabled
        return M2, M3, W2, W3, R1, R2, readable2, readable3

    def forward(self, list_events, list_image_metas, gather_indices, states=None, list_images=None) -> Tensor:
        # Init state
        if states is None:
            idx_offset = 0
            batch_size = len(list_events[0])
            z1 = self.lat1.get_initial_latent(batch_size)
            z2 = self.lat2.get_initial_latent(batch_size)
            z3 = self.lat3.get_initial_latent(batch_size)
            states = (idx_offset, z1, z2, z3)

        # Preproc
        evdata, batch_indices, split_sizes = self._preproc_events(list_events, list_image_metas)

        # Embed
        key, value, ev_q = self._forward_embed(evdata, batch_indices)

        # Backbone
        outputs, states = self._forward_backbone(key, value, ev_q, split_sizes, gather_indices, states, list_images)

        idx_offset, z1, z2, z3 = states
        z1 = z1.detach()
        z2 = z2.detach()
        z3 = z3.detach()
        states = [idx_offset, z1, z2, z3]
        return outputs, states

    def _preproc_events(self, list_events, list_image_metas):
        split_sizes = []
        batch_indices = []
        evdata = []
        for events, image_meta in zip(list_events, list_image_metas):
            curr_time = [ meta['curr_time_crop'] for meta in image_meta ]
            duration = [ meta['delta_t'] for meta in image_meta ]

            dt, x, y, p, b = self.embed.preproc_events(events, curr_time, duration)
            ev = torch.stack([dt, x, y, p], dim=-1)

            evdata.append(ev)
            batch_indices.append(b)
            split_sizes.append(len(dt))

        evdata = torch.cat(evdata, dim=0)
        batch_indices = torch.cat(batch_indices, dim=0)

        return evdata, batch_indices, split_sizes

    def _forward_embed(self, evdata, batch_indices):
        ev_tensor, ev_q = self.embed.forward_fast_train(evdata, batch_indices)
        H = self.lat1.write_bottom_up.attn.num_heads
        C = self.lat1.latent_dim
        ev_tensor = self.lat1.write_bottom_up.norm_input(ev_tensor)
        kv = self.lat1.write_bottom_up.attn.kv(ev_tensor).view(-1, 2, H, C // H).permute(1,0,2,3).contiguous()
        key, value = kv[0], kv[1]    # (L, H, C)
        return key, value, ev_q

    def dummy_image(self, device):
        #if random.randint(0,3) == 0:
        #    return None, None
        image = torch.zeros([2,3,260,348], dtype=torch.float, device=device)
        valid_batch = torch.ones([2], dtype=torch.bool, device=device)
        if random.randint(0,1) == 0:
            valid_batch[0] = False
        return image, valid_batch

    def _get_image_shape(self, list_images):
        for images in list_images:
            for image in images:
                if image is not None:
                    return image.shape
        return None

    def _forward_backbone(self, all_keys, all_values, ev_q, split_sizes, gather_indices, states, list_images):
        idx_offset, z1, z2, z3 = states
        batch_size = z1.meta['shape'][0]
        z1_out, z2_out, z3_out = None, None, None

        # output indices
        time_indices = gather_indices['time']
        batch_indices = gather_indices['batch']
        assert len(time_indices) == len(batch_indices)
        destination = torch.arange(len(time_indices))

        # place holder for output
        #out_z1 = torch.zeros_like(z1.data[0])[None,:,:,:].repeat(len(destination),1,1,1)
        #out_z2 = torch.zeros_like(z2.data[0])[None,:,:,:].repeat(len(destination),1,1,1)
        #out_z3 = torch.zeros_like(z3.data[0])[None,:,:,:].repeat(len(destination),1,1,1)

        D1 = self.lat1.output_dim or self.lat1.latent_dim
        D2 = self.lat2.output_dim or self.lat2.latent_dim
        D3 = self.lat3.output_dim or self.lat3.latent_dim
        out_z1 = torch.zeros(len(destination), D1, *self.lat1.latent_size, dtype=z1.data.dtype, device=z1.data.device)
        out_z2 = torch.zeros(len(destination), D2, *self.lat2.latent_size, dtype=z2.data.dtype, device=z2.data.device)
        out_z3 = torch.zeros(len(destination), D3, *self.lat3.latent_size, dtype=z3.data.dtype, device=z3.data.device)

        list_keys = all_keys.split(split_sizes)
        list_values = all_values.split(split_sizes)
        list_ev_q = ev_q.split(split_sizes)

        # init image_buffer
        image_buffer = ImageBuffer(batch_size)
        if list_images is not None:
            for images in list_images:
                image_buffer.write(images)
        image_buffer.read()

        for idx in range(len(list_keys)):
            key = list_keys[idx]
            value = list_values[idx]
            ev_q = list_ev_q[idx]

            if list_images is not None:
                image_buffer.write(list_images[idx])

            # flags
            M2, M3, W2, W3, R1, R2, readable2, readable3 = self._timing_flags(idx_offset + idx)

            # generate messages (require z1, z2, z3)
            if M2:
                if readable2:
                    m2 = self.lat2.forward_message(z2, z1)
                z2_out = self.lat2.readout(z2)
            if M3:
                if readable3:
                    m3 = self.lat3.forward_message(z3, z2)
                z3_out = self.lat3.readout(z3)

            # update z3 (require z2)
            if W3:
                images, valid_batch = image_buffer.read()
                z3 = self.lat3.forward_write_bottom_up(z3, z2, images, valid_batch)
                z3 = self.lat3.forward_update(z3)

            # update z2 (require z1, m3)
            if R2:
                z2 = self.lat2.forward_write_top_down(z2, m3)
            if W2:
                z2 = self.lat2.forward_write_bottom_up(z2, z1)
                z2 = self.lat2.forward_update(z2)

            # update z1 (require events, m2)
            if R1:
                z1 = self.lat1.forward_write_top_down(z1, m2)

            if len(key) > 0:
                z1 = SeqData(self.lat1.write_bottom_up.forward_fast_train(z1.data, key, value, ev_q), z1.meta)
            z1 = self.lat1.forward_update(z1)
            z1_out = self.lat1.readout(z1)

            mask = time_indices == idx
            _batch_indices = batch_indices[mask]
            _destination = destination[mask]
            out_z1[_destination] = z1_out[_batch_indices]
            out_z2[_destination] = z2_out[_batch_indices]
            out_z3[_destination] = z3_out[_batch_indices]

        idx_offset += len(list_keys)

        states = (idx_offset, z1, z2, z3)
        outputs = (out_z1, out_z2, out_z3)

        return outputs, states

    def prepair_for_test(self, batch_size, image_size=None):
        self.z1 = self.lat1.get_initial_latent(batch_size)
        self.z2 = self.lat2.get_initial_latent(batch_size)
        self.z3 = self.lat3.get_initial_latent(batch_size)
        self.z1_out = None
        self.z2_out = None
        self.z3_out = None
        self.image_buffer = ImageBuffer(batch_size)

    def simple_test(self, idx, events, image_metas, images=None) -> Tensor:
        events = events[0]
        image_meta = image_metas[0]
        curr_time = image_meta['curr_time_crop']
        duration = image_meta['delta_t']

        if images is not None:
            self.image_buffer.write(images)

        d0, d1, d2 = self.devices

        # flags
        M2, M3, W2, W3, R1, R2, readable2, readable3 = self._timing_flags(idx)

        if M2:
            self.m2 = self.lat2.forward_message(self.z2, self.z1.to(d1))
            self.z2_out = self.lat2.readout(self.z2).to(d1)
        if M3:
            self.m3 = self.lat3.forward_message(self.z3, self.z2.to(d2))
            self.z3_out = self.lat3.readout(self.z3).to(d2)

        # start writing and updating
        if R2:
            self.z2 = self.lat2.forward_write_top_down(self.z2, self.m3.to(d1))
        if W2:
            self.z2 = self.lat2.forward_write_bottom_up(self.z2, self.z1.to(d1))
            self.z2 = self.lat2.forward_update(self.z2)
        if W3:
            images, valid_batch = self.image_buffer.read(d2)
            self.z3 = self.lat3.forward_write_bottom_up(self.z3, self.z2.to(d2), images, valid_batch)
            self.z3 = self.lat3.forward_update(self.z3)

        if R1:
            self.z1 = self.lat1.forward_write_top_down(self.z1, self.m2.to(d0))

        if len(events) > 0:
            ev_features = self.embed.forward_single(events, curr_time, duration)
            self.z1 = self.lat1.forward_write_bottom_up(self.z1, ev_features)
        self.z1 = self.lat1.forward_update(self.z1)
        self.z1_out = self.lat1.readout(self.z1)

        return self.z1_out, self.z2_out, self.z3_out

    def termination(self):
        pass

class HMBackboneMP(HMBackbone):
    def to_cuda(self, d0, d1, d2):
        self.devices = (d0, d1, d2)

    def prepair_for_test(self, batch_size, image_size=None):
        d0, d1, d2 = self.devices

        # place holders
        self.input_lat3_ = self.lat2.get_initial_latent(batch_size).clone().to(d2)
        self.input_lat2_ = self.lat1.get_initial_latent(batch_size).clone().to(d1)
        self.input_lat3_ = self.lat2.get_initial_latent(batch_size).clone().to(d2)
        self.message_lat2_ = self.lat2.get_initial_latent(batch_size).clone().to(d1) 
        self.input_lat2_.share_memory_()
        self.input_lat3_.share_memory_()
        self.message_lat2_.share_memory_()

        # image inputs
        self.image_buffer = ImageBuffer(batch_size)
        if image_size is not None:
            height, width = image_size
            self.image_lat3_ = torch.zeros(batch_size, 3, height, width, dtype=self.input_lat3_.dtype, device=self.input_lat3_.device)
            self.image_lat3_.share_memory_()

        self.q2 = mp.Queue()
        self.q3 = mp.Queue()
        self.rq2 = mp.Queue()
        self.rq3 = mp.Queue()

        self.z1_out = None
        self.z2_out = None
        self.z3_out = None

        self.proc2 = mp.Process(target=self.forward_lat2, args=(self.lat2, batch_size, d1, self.q2, self.rq2))
        self.proc3 = mp.Process(target=self.forward_lat3, args=(self.lat3, batch_size, d2, self.q3, self.rq3))
        self.proc2.start()
        self.proc3.start()

        self.embed = self.embed.to(d0)
        self.lat1 = self.lat1.to(d0)
        self.z1 = self.lat1.get_initial_latent(batch_size)

        #del self.lat2
        #del self.lat3
        #torch.cuda.empty_cache()

    def simple_test(self, idx, events, image_metas, images=None) -> Tensor:
        events = events[0]
        image_meta = image_metas[0]
        curr_time = image_meta['curr_time_crop']
        duration = image_meta['delta_t']

        if images is not None:
            self.image_buffer.write(images)

        # flags
        M2, M3, W2, W3, R1, R2, readable2, readable3 = self._timing_flags(idx)

        if M2:
            self.input_lat2_.copy_(self.z1)
            self.q2.put([self.input_lat2_, readable2])    # request message gen
            self.z2, self.z2_out = self.rq2.get()         # get current memory state
        if M3:
            images, valid_batch = self.image_buffer.read()
            self.input_lat3_.copy_(self.z2)
            if images is not None:
                self.image_lat3_.copy_(images)
                self.q3.put([self.input_lat3_, self.image_lat3_, readable3])    # request message gen and write
            else:
                self.q3.put([self.input_lat3_, None, readable3])    # request message gen and write
            self.z3, self.z3_out = self.rq3.get()         # get current memory state

        if W2:
            if R2: self.message_lat2_.copy_(self.rq3.get()[0])
            self.q2.put([self.message_lat2_, R2])         # request write from top
            self.input_lat2_.copy_(self.z1)
            self.q2.put([self.input_lat2_, True])         # request write
        if W3:
            pass

        m2 = self.rq2.get()[0].to(self.z1.device) if R1 else None
        self.z1, self.z1_out = self.forward_lat1(self.embed, self.lat1, self.z1, events, m2, curr_time, duration)

        return self.z1_out, self.z2_out, self.z3_out

    def termination(self):
        del self.input_lat2_
        del self.input_lat3_
        del self.message_lat2_
        self.q2.put([None, None])
        self.q3.put([None, None, None])
        self.q2.put([None, None])
        self.q3.put([None, None, None])
        self.proc2.join()
        self.proc3.join()
        del self.q2
        del self.q3
        del self.rq2
        del self.rq3
        torch.cuda.empty_cache()


    @staticmethod
    @torch.no_grad()
    def forward_lat1(embed, lat, z1, events, m2, curr_time, duration):
        if m2 is not None:
            z1 = lat.forward_write_top_down(z1, m2)
        if len(events) > 0:
            ev_features = embed.forward_single(events, curr_time, duration)
            z1 = lat.forward_write_bottom_up(z1, ev_features)
        z1 = lat.forward_update(z1)
        z1_out = lat.readout(z1)

        return z1, z1_out

    @staticmethod
    @torch.no_grad()
    def forward_lat2(lat, batch_size, device, q, rq):
        lat = lat.to(device)
        z = lat.get_initial_latent(batch_size)
        rq.put([z,None])

        while True:
            x, message_gen = q.get()
            if x is None: break

            if message_gen:
                out = lat.forward_message(z, x)
                rq.put([out, None])

            m, message_write = q.get()
            if message_write:
                z = lat.forward_write_top_down(z, m)

            x, _ = q.get()
            if x is None: break

            z = lat.forward_write_bottom_up(z, x)
            z = lat.forward_update(z)
            z_out = lat.readout(z)

            rq.put([z, z_out])

    @staticmethod
    @torch.no_grad()
    def forward_lat3(lat, batch_size, device, q, rq):
        lat = lat.to(device)
        z = lat.get_initial_latent(batch_size)
        rq.put([z,None])

        while True:
            x, im, message_gen = q.get()
            if x is None: break

            if message_gen:
                out = lat.forward_message(z, x)
                rq.put([out, None])

            z = lat.forward_write_bottom_up(z, x, im)
            z = lat.forward_write_bottom_up(z, x)
            z = lat.forward_update(z)
            z_out = lat.readout(z)

            rq.put([z, z_out])


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

class EventSparseWrite(BlockBase):
    def __init__(self, input_dim: int, latent_dim: int, num_heads: int,
                 mlp_ratio: int = 4, act_layer: type = nn.GELU, norm_layer: type = nn.LayerNorm, drop: float = 0.,
                 noise_filter: bool = True, filter_init: float = 0., latent_size: Tuple[int, int] = None, linearized_attn: bool = False) -> None:
        super().__init__()
        self.norm_latent = norm_layer(latent_dim)
        self.norm_input = norm_layer(input_dim)
        self.attn = SpDAttention(latent_dim, input_dim, num_heads, qkv_bias=True, proj_drop=0., noise_filter=noise_filter, filter_init=filter_init, latent_size=latent_size, linearized=linearized_attn)
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

class LatentTrans(BlockBase):
    def __init__(self, latent_size: Tuple[int,int], input_dim: int, latent_dim: int, output_dim: int, num_heads: int, update_depth: int,
                       message_gen: bool = True, event_write: bool = False, top_down: bool = True, vector_latent=False,
                       cfg_write: dict = {}, cfg_message: dict = {}, cfg_update: dict = {}, cfg_image_write: Optional[dict] = None) -> None:
        super().__init__()
        self.event_write = event_write
        self.latent_size = latent_size
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        latent_stride = 2

        # init latent state
        H, W = latent_size
        if vector_latent:
            self.init_latent = nn.Parameter(torch.zeros(1, 1, latent_dim))
        else:
            self.init_latent = nn.Parameter(torch.zeros(1, H*W, latent_dim))
        trunc_normal_(self.init_latent, std=.02)

        if event_write:
            self.write_bottom_up = EventSparseWrite(input_dim, latent_dim, num_heads, mlp_ratio=4, latent_size=latent_size, **cfg_write)
        else:
            self.write_bottom_up = WriteBottomUp(input_dim, latent_dim, num_heads, latent_stride, mlp_ratio=4, **cfg_write)

        if top_down:
            self.write_top_down = WriteTopDown(latent_dim, mlp_ratio=4)

        self.update_norm = nn.LayerNorm(latent_dim)
        self.update = self._make_update_layer(latent_dim, update_depth, cfg_update)

        if message_gen:
            self.message = MessageGen(input_dim, latent_dim, num_heads, latent_stride, **cfg_message)

        if cfg_image_write is not None:
            self.write_image = ImageWrite(latent_dim, num_heads, mlp_ratio=4, **cfg_image_write)

        self.norm_out = nn.LayerNorm(latent_dim)
        self.proj_out = Linear(latent_dim, output_dim, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_act=False)

    def _make_update_layer(self, latent_dim: int, update_depth: int, cfg_update: dict) > nn.Module:
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
            update_layer = nn.ModuleList([ MViTStage(latent_dim, latent_dim, stride=1, expansion=4, num_layers=update_depth, **cfg_update) ])
        elif self.layer_type == 'cnext':
            update_layer = nn.ModuleList([ ConvNeXtBlock(latent_dim, drop_path=drop_path[i], **cfg_update) for i in range(update_depth) ])
        else:
            raise RuntimeError

        return update_layer

    @property
    def no_decay_set(self):
        return {'init_latent', 'bias'}

    def get_initial_latent(self, batch_size):
        H, W = self.latent_size
        lat = self.init_latent.expand(batch_size, H*W, -1)
        meta = dict(shape=[batch_size,H,W])
        return SeqData(lat, meta)

    def forward_write_bottom_up(self, seq_z: SeqData, seq_x: SeqData, image: Optional[Tensor] = None, valid_batch: Optional[Tensor] = None) -> SeqData:
        if image is not None:
            seq_z = self.write_image(seq_z, image, valid_batch)

        if self.event_write:
            spd, qi = seq_x
            seq_z = SeqData(self.write_bottom_up(seq_z.data, spd, qi), seq_z.meta)
        else:
            seq_z = self.write_bottom_up(seq_z, seq_x)
        return seq_z

    def forward_write_top_down(self, seq_z: SeqData, m: Tensor) -> Tensor:
        return self.write_top_down(seq_z, message=m)

    def forward_update(self, seq_z: SeqData) -> SeqData:
        seq_z = SeqData(self.update_norm(seq_z.data), seq_z.meta)
        if self.layer_type == 'transformer':
            for layer in self.update:
                seq_z = layer(seq_z)
        else:
            z = seq_z.to_2D()
            for layer in self.update:
                z = layer(z)
            seq_z = SeqData.from_2D(z)
        return seq_z

    def forward_message(self, seq_z: SeqData, seq_x: SeqData) -> SeqData:
        return self.message(seq_z, seq_x)

    def readout(self, seq_z: SeqData) -> Tensor:
        z_data = self.proj_out(self.norm_out(seq_z.data))
        seq_z = SeqData(z_data, seq_z.meta)
        return seq_z.to_2D()
        
class SpDAttention(BlockBase):
    def __init__(self, dim: int, kv_dim: int, num_heads: int, qkv_bias: bool = True, qk_scale: Optional[float] = None, proj_drop: float = 0.,
                        noise_filter: bool = True, filter_init: float = 0., latent_size: Tuple[int, int] = None, linearized: bool = False) -> None:
        super().__init__()
        assert dim % num_heads == 0
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5
        #self.qkv = QKVTransform(dim, num_heads, qkv_bias, cross=True, kv_dim=kv_dim)
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

    def softmax0(self, weights, q_indices, z_shape, kv_shape):
        B, zL, _ = z_shape
        L, H, C = kv_shape

        max_weights = scatter(weights.detach(), q_indices, reduce='max', dim=0, dim_size=B*zL)    # (B*zL, H)
        if self.noise_filter:
            if len(self.dustbin_weights) == H:
                w_dust = self.dustbin_weights.view(1,H).expand(B*zL, -1)
            else:
                w_dust = self.dustbin_weights.expand(B*zL, H)
            max_weights = torch.maximum(max_weights, w_dust.detach())
            w_dust = (w_dust - max_weights).exp()
        weights = (weights - max_weights[q_indices]).exp()                               # (L, H)
        accm = scatter_add(weights, q_indices, dim=0, dim_size=B*zL, use_torch=False)    # (B*zL, H)
        if self.noise_filter:
            accm = accm + w_dust
        weights = weights / (accm[q_indices] + 1.0e-7)
        return weights

    def softmax1(self, weights, q_indices, z_shape, kv_shape):
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

    def softmax2(self, weights, q_indices, z_shape, kv_shape):
        B, zL, _ = z_shape
        L, H, C = kv_shape

        if self.noise_filter:
            w_dust, q_indices_dust = self._prepair_dust(B, zL)
            weights = torch.cat([weights, w_dust], dim=0)
            q_indices = torch.cat([q_indices, q_indices_dust])

        max_weights = scatter(weights.detach(), q_indices, reduce='max', dim=0, dim_size=B*zL)    # (B*zL, H)
        weights = (weights - max_weights[q_indices]).exp()                               # (L, H)
        accm = scatter_add(weights, q_indices, dim=0, dim_size=B*zL, use_torch=False)    # (B*zL, H)
        weights = weights / (accm[q_indices] + 1.0e-7)

        if self.noise_filter:
            weights = weights[:L]

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
        weights = self.softmax1(weights, q_indices, z.shape, key.shape)

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
                 duration: int = 5000, dynamic: List[bool] = False, dynamic_dim: Optional[List[int]] = None, out_dim: Optional[List[int]] = None,
                 local_spa: bool = False, num_spa_heads: int = 4, spa_mlp_ratio: int = 1) -> None:
        super().__init__()
        self.discrete_time = discrete_time
        self.time_bins = time_bins
        self.time_delta = duration // time_bins
        self.local_spa = local_spa

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

        if local_spa:
            self.spa = SparseAttentionBlock(sum(out_dim), num_heads=num_spa_heads, mlp_ratio=spa_mlp_ratio)

    def generate_param_table(self) -> None:
        self.xy.generate_param_table()
        self.time.generate_param_table()
        self.pol.generate_param_table(data=torch.tensor([-1,1], dtype=torch.float).view(-1,1))

    def attention(self, embeddings, indices):
        adj = indices[:,None] == indices[None,:]      # (E, E)
        embeddings = self.spa(embeddings, adj)
        return embeddings

    def forward_single(self, events: Tensor, curr_time: int, duration: int) -> Tuple[Tensor,Tensor]:
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

        if self.local_spa:
            embeddings = self.attention(embeddings, indices)

        return embeddings, indices    # (L, C1+C2+C3), (L,)

    def _to_fast_model(self):
        self.xy.generate_param_table()
        self.time.generate_param_table()
        self.pol.generate_param_table(data=torch.tensor([-1,1], dtype=torch.float).view(-1,1))
        print('Generate table for embedding xy, t, p')

    def forward(self, events: Tensor, curr_time: List[int], duration: List[int]) -> Tuple[Tensor,Tensor]:
        device = events[0].device
        dt, x, y, p, b = self.preproc_events(events, curr_time, duration)
        return self._forward(dt, x, y, p, b)

    def forward_fast_train(self, evdata: Tensor, batch_indices: Tensor) -> Tuple[Tensor,Tensor]:
        dt, x, y, p = evdata[:,0], evdata[:,1], evdata[:,2], evdata[:,3]
        b = batch_indices
        return self._forward(dt, x, y, p, b)

    def _forward(self, dt, x, y, p, b):
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

        if self.local_spa:
            embeddings = self.attention(embeddings, indices)

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
    def __init__(self, batch_size):
        self.batch_size = batch_size
        self.data = None
        self.valid_mask = None

    def write(self, images: List[Tensor]):
        for bidx, image in enumerate(images):
            if image is None:
                continue

            if self.data is None:
                self.data = image.new_zeros(self.batch_size, *image.shape)
                self.valid_mask = image.new_zeros(self.batch_size).bool()

            self.data[bidx] = image
            self.valid_mask[bidx] = True

    def read(self, device=None):
        if self.data is None:
            return self.data, self.valid_mask

        if device is None:
            device = self.data.device

        #out_data = self.data[self.valid_mask].to(device)
        out_data = self.data.clone()
        out_mask = self.valid_mask.clone()
        out_data = out_data.to(device)
        out_mask = out_mask.to(device)
        self.valid_mask.fill_(0)
        return out_data, out_mask



