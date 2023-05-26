# Hierarchical Neural Memory Network
# 
# Copyright (C) 2023 National Institute of Advanced Industrial Science and Technology
# 
# This program is free software: you can redistribute it and/or modify
# it under the terms of the GNU General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
# 
# This program is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# GNU General Public License for more details.
# 
# You should have received a copy of the GNU General Public License
# along with this program.  If not, see <http://www.gnu.org/licenses/>.

from functools import partial
import numpy as np
from collections import OrderedDict, defaultdict

import torch.multiprocessing as mp
if mp.get_start_method(True) is None:
    mp.set_start_method('spawn')

import random
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from timm.models.layers import trunc_normal_
from torch_scatter import scatter


from torchtools.base_models.backbone.vit import MobileViTAttachment, Mlp, TokenGrouping, QKVTransform, CrossAttention, PositionEmbedding1D, PositionEmbedding2D, PatchMergingCross, SeqData, TransformerBlock, SparseAttentionBlock, Stage
from torchtools.base_models.layers import ConvBlock, Linear, MV2Block, UpConvBlock, ConvPixelShuffle
from torchtools.base_models.backbone.resnet import ResBlock, ResStage, ResEntry
from torchtools.base_models.backbone.convnext import ConvNeXtBlock, ConvNeXtEntry, ConvNeXtStage
from torchtools.base_models.blocks import BlockBase
from torchtools.init import init_transformer, load_state_dict_matched
from torchtools.utils import adapt_bn_momentum_for_checkpointing

from torch import Tensor
from typing import Tuple, List, Optional, Dict
Meta = Dict[str,Tuple[int,int,int]]

from common.utils import Timer
timer = Timer()

class HEMBackboneL1(BlockBase):
    def __init__(self, input_size, latent_sizes, latent_dims, embed_dim, num_heads, depth,
                 cfg_embed=None, cfg_lat1=None, cfg_res=None, warmup=20, max_num_events=-1, use_resnet=False) -> None:
        super().__init__()
        self.warmup = warmup
        self.use_resnet = use_resnet

        L1 = latent_sizes
        D1 = latent_dims
        D0 = sum(embed_dim)
        H1 = num_heads
        N1 = depth

        self.embed = EventEmbedding(input_size=input_size, latent_size=L1, out_dim=embed_dim, **cfg_embed)
        self.lat1 = LatentTrans(latent_size=L1, latent_dim=D1, write_dim=D0, read_dim=None, num_heads=H1, update_depth=N1, readable=False, sparse_write=True, from_top=False, **cfg_lat1)

        if self.use_resnet:
            self.layer2 = ResStage(2, 128, num_layers=4, bottleneck=True, stride=2, inc=D1, **cfg_res)
            self.layer3 = ResStage(3, 256, num_layers=6, bottleneck=True, stride=2, **cfg_res)
            self.layer4 = ResStage(4, 512, num_layers=3, bottleneck=True, stride=2, **cfg_res)

        self.set_module_names()

    def init_weights(self, pretrained=None):
        init_transformer(self.modules())

    def print_grad_norm(self):
        super().print_grad_norm()
        self.embed.print_grad_norm()
        self.lat1.print_grad_norm()
        if self.use_resnet:
            self.layer2.print_grad_norm()
            self.layer3.print_grad_norm()
            self.layer4.print_grad_norm()

    def clip_grad(self, method='norm', max_norm=1.0, clip_value=0.5):
        self.embed.clip_grad(method, max_norm, clip_value)
        self.lat1.clip_grad(method, max_norm, clip_value)
        if self.use_resnet:
            self.layer2.clip_grad(method, max_norm, clip_value)
            self.layer3.clip_grad(method, max_norm, clip_value)
            self.layer4.clip_grad(method, max_norm, clip_value)

    def forward(self, list_events, list_image_metas, gather_indices, states=None, list_images=None) -> Tensor:
        # Init state
        if states is None:
            idx_offset = 0
            batch_size = len(list_events[0])
            z1 = self.lat1.get_initial_latent(batch_size)
            states = (idx_offset, z1)

        # Preproc
        evdata, batch_indices, split_sizes = self._preproc_events(list_events, list_image_metas)

        # Embed
        key, value, ev_q = self._forward_embed(evdata, batch_indices)

        # Backbone
        z1_out, states = self._forward_backbone(key, value, ev_q, split_sizes, gather_indices, states, list_images)

        if self.use_resnet:
            z2_out = self.layer2(z1_out)
            z3_out = self.layer3(z2_out)
            z4_out = self.layer4(z3_out)
            outputs = [z1_out, z2_out, z3_out, z4_out]
        else:
            outputs = [z1_out]

        idx_offset, z1 = states
        z1 = z1.detach()
        states = [idx_offset, z1]
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
        ev_tensor, ev_q = self.embed.forward_for_checkpointing(evdata, batch_indices)
        H = self.lat1.write.attn.num_heads
        C = self.lat1.latent_dim
        ev_tensor = self.lat1.write.norm_input(ev_tensor)
        kv = self.lat1.write.attn.kv(ev_tensor).view(-1, 2, H, C // H).permute(1,0,2,3).contiguous()
        key, value = kv[0], kv[1]    # (L, H, C)
        return key, value, ev_q

    def _forward_backbone(self, all_keys, all_values, ev_q, split_sizes, gather_indices, states, list_images):
        idx_offset, z1 = states
        batch_size = z1.meta['shape'][0]
        z1_out = None

        # output indices
        time_indices = gather_indices['time']
        batch_indices = gather_indices['batch']
        assert len(time_indices) == len(batch_indices)
        destination = torch.arange(len(time_indices))

        D1 = self.lat1.output_dim or self.lat1.latent_dim
        out_z1 = torch.zeros(len(destination), D1, *self.lat1.latent_size, dtype=z1.data.dtype, device=z1.data.device)

        list_keys = all_keys.split(split_sizes)
        list_values = all_values.split(split_sizes)
        list_ev_q = ev_q.split(split_sizes)

        for idx in range(len(list_keys)):
            key = list_keys[idx]
            value = list_values[idx]
            ev_q = list_ev_q[idx]

            if len(key) > 0:
                z1 = SeqData(self.lat1.write.forward_for_checkpointing(z1.data, key, value, ev_q), z1.meta)
            z1 = self.lat1.forward_update(z1)
            z1_out = self.lat1.readout(z1)

            mask = time_indices == idx
            _batch_indices = batch_indices[mask]
            _destination = destination[mask]
            out_z1[_destination] = z1_out[_batch_indices]

        idx_offset += len(list_keys)

        states = (idx_offset, z1)

        return out_z1, states

    def prepair_for_test(self, batch_size, image_size=None):
        self.z1 = self.lat1.get_initial_latent(batch_size)

    def simple_test(self, idx, events, image_metas, images=None) -> Tensor:
        events = events[0]
        image_meta = image_metas[0]
        curr_time = image_meta['curr_time_crop']
        duration = image_meta['delta_t']

        if len(events) > 0:
            ev_features = self.embed.forward_single(events, curr_time, duration)
            self.z1 = self.lat1.forward_write(self.z1, ev_features)
        self.z1 = self.lat1.forward_update(self.z1)
        z1_out = self.lat1.readout(self.z1)

        if self.use_resnet:
            z2_out = self.layer2(z1_out)
            z3_out = self.layer3(z2_out)
            z4_out = self.layer4(z3_out)
            outputs = [z1_out, z2_out, z3_out, z4_out]
        else:
            outputs = [z1_out]

        return outputs

    def termination(self):
        pass

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

class SparseWrite(BlockBase):
    def __init__(self, input_dim: int, latent_dim: int, num_heads: int,
                 mlp_ratio: int = 4, act_layer: type = nn.GELU, norm_layer: type = nn.LayerNorm, drop: float = 0.,
                 noise_filter: bool = True, filter_init: float = 0., latent_size: Tuple[int, int] = None, linearized_attn: bool = False) -> None:
        super().__init__()
        self.norm_latent = norm_layer(latent_dim)
        self.norm_input = norm_layer(input_dim)
        self.attn = SpDAttention(latent_dim, input_dim, num_heads, qkv_bias=True, proj_drop=0., noise_filter=noise_filter, filter_init=filter_init, latent_size=latent_size, linearized=linearized_attn)
        self.norm_mlp = norm_layer(latent_dim)
        self.mlp = Mlp(in_features=latent_dim, hidden_features=latent_dim * mlp_ratio, act_layer=act_layer, drop=drop)

    def forward_for_checkpointing(self, z: Tensor, key: Tensor, value: Tensor, q_indices: Tensor) -> Tensor:
        lat = z.clone()
        z = self.norm_latent(z)
        z = self.attn.forward_for_checkpointing(z, key, value, q_indices)
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

class LatentWrite(BlockBase):
    def __init__(self, input_dim: int, latent_dim: int, num_heads: int, latent_stride: int, downsample: str = 'conv', window_size: Optional[Tuple[int,int]] = None, grouping='intra-window',
                 input_proj: bool = True, mlp_ratio: int = 4, act_layer: type = nn.GELU, norm_layer: type = nn.LayerNorm, drop: float = 0.,
                 pos_dynamic: bool = False, pos_log_scale: bool = False, mvit: bool = False, pre_act: bool = True, pre_layer_norm: bool = False) -> None:
        super().__init__()
        self.downsample = downsample
        self.mvit = MobileViTAttachment(latent_dim, gn_dim=32, pre_act=True, dw=False, fuse_method='cat') if mvit else None
        self.norm_latent = norm_layer(latent_dim)
        self.norm_input = norm_layer(latent_dim) if downsample != 'none' else norm_layer(input_dim)
        self.norm_mlp = norm_layer(latent_dim)
        self.mlp = Mlp(in_features=latent_dim, hidden_features=latent_dim * mlp_ratio, act_layer=act_layer, drop=drop)

        if downsample == 'none':
            s = latent_stride
            #self.attn = CrossAttention(latent_dim, input_dim, wsize=(1, 1), kv_wsize=(s, s), grouping=grouping, num_heads=num_heads, cyclic_shift=False,
            #                           pos_dynamic=pos_dynamic, pos_log_scale=pos_log_scale, qkv_bias=True, proj_drop=0.)
            qWh, qWw = window_size
            kvWh = qWh * s
            kvWw = qWw * s
            self.attn = CrossAttention(latent_dim, input_dim, wsize=(qWh, qWw), kv_wsize=(kvWh, kvWw), grouping=grouping, num_heads=num_heads, cyclic_shift=False,
                                       pos_dynamic=pos_dynamic, pos_log_scale=pos_log_scale, qkv_bias=True, proj_drop=0.)
        else:
            self.input_resize = Downsample(input_dim, latent_dim, latent_stride, method=downsample, proj=input_proj, pre_act=pre_act, pre_layer_norm=pre_layer_norm)
            self.attn = CrossAttention(latent_dim, latent_dim, wsize=window_size, kv_wsize=window_size, grouping=grouping, num_heads=num_heads, cyclic_shift=False,
                                       pos_dynamic=pos_dynamic, pos_log_scale=pos_log_scale, qkv_bias=True, proj_drop=0.)

    def forward(self, seq_z: SeqData, seq_x: SeqData) -> Tensor:
        if self.mvit is not None:
            seq_in = seq_z.clone()    # skip
            seq_z = self.mvit.local_rep(seq_z)

        if self.downsample != 'none':
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

        if self.mvit is not None:
            seq_z = self.mvit.fuse(seq_z, seq_in)

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


class MessageWrite(BlockBase):
    def __init__(self, latent_dim: int, mlp_ratio: int = 4, act_layer: type = nn.GELU, norm_layer: type = nn.LayerNorm, drop: float = 0., mvit: bool = False) -> None:
        super().__init__()
        self.mvit = MobileViTAttachment(latent_dim, gn_dim=32, pre_act=True, dw=False, fuse_method='cat') if mvit else None
        self.norm_mlp = norm_layer(latent_dim)
        self.mlp = Mlp(in_features=latent_dim, hidden_features=latent_dim * mlp_ratio, act_layer=act_layer, drop=drop)

    def forward(self, seq_z: SeqData, message: Tensor) -> Tensor:
        if self.mvit is not None:
            seq_in = seq_z.clone()    # skip
            seq_z = self.mvit.local_rep(seq_z)

        z, z_meta = seq_z.data, seq_z.meta

        z = z + message.data
        z = z + self.mlp(self.norm_mlp(z))

        seq_z = SeqData(z, z_meta)

        if self.mvit is not None:
            seq_z = self.mvit.fuse(seq_z, seq_in)

        return seq_z

class LatentRead(BlockBase):
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
    def __init__(self, latent_size: Tuple[int,int], latent_dim: int, write_dim: int, read_dim: int, num_heads: int, update_depth: int,
                       readable: bool = True, sparse_write: bool = False, from_top: bool = True, mvit: bool = False, vector_latent=False,
                       output_proj: bool = False, output_dim: int = None,
                       cfg_write: dict = {}, cfg_read: dict = {}, cfg_update: dict = {}, cfg_image_write: Optional[dict] = None) -> None:
        super().__init__()
        self.sparse_write = sparse_write
        self.latent_size = latent_size
        self.latent_dim = latent_dim
        self.output_dim = output_dim
        latent_stride = 2
        H, W = latent_size
        if vector_latent:
            self.init_latent = nn.Parameter(torch.zeros(1, 1, latent_dim))
        else:
            self.init_latent = nn.Parameter(torch.zeros(1, H*W, latent_dim))
        trunc_normal_(self.init_latent, std=.02)

        if sparse_write:
            self.write = SparseWrite(write_dim, latent_dim, num_heads, mlp_ratio=4, latent_size=latent_size, **cfg_write)
        else:
            self.write = LatentWrite(write_dim, latent_dim, num_heads, latent_stride, mvit=mvit, mlp_ratio=4, **cfg_write)

        if from_top:
            self.write_from_top = MessageWrite(latent_dim, mlp_ratio=4, mvit=mvit)

        self.layer_type = cfg_update.pop('layer')
        if 'norm_before_update' in cfg_update and cfg_update.pop('norm_before_update'):
            self.update_norm = nn.LayerNorm(latent_dim)
        else:
            self.update_norm = nn.Identity()

        if 'drop_path' in cfg_update:
            drop_path = cfg_update.pop('drop_path')
            assert len(drop_path) == update_depth
        else:
            drop_path = [0] * update_depth

        if self.layer_type == 'transformer':
            enable_cyclic_shift = cfg_update.pop('enable_cyclic_shift')
            cyclic_shift = [ (i % 2 == 1 and enable_cyclic_shift) for i in range(update_depth) ]
            self.update = nn.ModuleList([ TransformerBlock(latent_dim, num_heads, cyclic_shift=cyclic_shift[i], drop_path=drop_path[i], **cfg_update) for i in range(update_depth)])
        elif self.layer_type == 'mv2':
            self.update = nn.ModuleList([ MV2Block(latent_dim, latent_dim, **cfg_update) for i in range(update_depth) ])
        elif self.layer_type == 'res':
            self.update = nn.ModuleList([ ResBlock((latent_dim, latent_dim), **cfg_update) for i in range(update_depth) ])
        elif self.layer_type == 'res2':
            self.update = nn.ModuleList([ ResBlock((latent_dim, latent_dim // 4, latent_dim), **cfg_update) for i in range(update_depth) ])
        elif self.layer_type == 'mvit':
            self.update = nn.ModuleList([ MViTStage(latent_dim, latent_dim, stride=1, expansion=4, num_layers=update_depth, num_heads=num_heads, **cfg_update) ])
        elif self.layer_type == 'cnext':
            self.update = nn.ModuleList([ ConvNeXtBlock(latent_dim, drop_path=drop_path[i], **cfg_update) for i in range(update_depth) ])
        else:
            raise RuntimeError

        if readable:
            self.read = LatentRead(read_dim, latent_dim, num_heads, latent_stride, **cfg_read)

        if cfg_image_write is not None:
            self.write_image = ImageWrite(latent_dim, num_heads, mlp_ratio=4, **cfg_image_write)

        self.norm_out = nn.LayerNorm(latent_dim)
        self.proj_out = Linear(latent_dim, output_dim, norm_layer=nn.LayerNorm, act_layer=nn.GELU, pre_act=False) if output_proj else nn.Identity()

    @torch.jit.ignore
    def optim_settings(self, lr, lr_from_top, weight_decay):
        if hasattr(self, 'no_weight_decay'):
            names_no_decay = self.no_weight_decay()
        else:
            names_no_decay = []

        params_decay, params_no_decay = [], []
        params_decay_m, params_no_decay_m = [], []

        for name, param in self.named_parameters():
            module_name = name.split('.')[0]
            if name in names_no_decay and module_name != 'write_from_top':
                #print(f'no decay: {name}')
                print(f'{name}: lr={lr:.2e}, decay={0:.2e}')
                params_no_decay.append(param)
            elif name in names_no_decay and module_name == 'write_from_top':
                #print(f'no decay: {name}')
                print(f'{name}: lr={lr_from_top:.2e}, decay={0:.2e}')
                params_no_decay_m.append(param)
            elif module_name != 'write_from_top':
                print(f'{name}: lr={lr:.2e}, decay={weight_decay:.2e}')
                params_decay.append(param)
            else:
                print(f'{name}: lr={lr_from_top:.2e}, decay={weight_decay:.2e}')
                params_decay_m.append(param)

        settings = [
            {'params': params_no_decay,   'lr': lr, 'weight_decay': 0.},
            {'params': params_decay,      'lr': lr, 'weight_decay': weight_decay},
        ]
        if len(params_decay_m) > 0:
            settings.append({'params': params_decay_m, 'lr': lr_from_top, 'weight_decay': weight_decay})
        if len(params_no_decay_m) > 0:
            settings.append({'params': params_no_decay_m, 'lr': lr_from_top, 'weight_decay': 0.})

        return settings

    @property
    def no_decay_set(self):
        return {'init_latent', 'bias'}

    def get_initial_latent(self, batch_size):
        H, W = self.latent_size
        lat = self.init_latent.expand(batch_size, H*W, -1)
        meta = dict(shape=[batch_size,H,W])
        return SeqData(lat, meta)

    def forward_write(self, seq_z: SeqData, seq_x: SeqData, image: Optional[Tensor] = None, valid_batch: Optional[Tensor] = None) -> SeqData:
        if image is not None:
            seq_z = self.write_image(seq_z, image, valid_batch)

        if self.sparse_write:
            spd, qi = seq_x
            seq_z = SeqData(self.write(seq_z.data, spd, qi), seq_z.meta)
        else:
            seq_z = self.write(seq_z, seq_x)
        return seq_z

    def forward_write_from_top(self, seq_z: SeqData, m: Tensor) -> Tensor:
        return self.write_from_top(seq_z, message=m)

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
        return self.read(seq_z, seq_x)

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

        return self.forward_for_checkpointing(z, key, value, q_indices, kv_indices)

    def forward_for_checkpointing(self, z: Tensor, key: Tensor, value: Tensor, q_indices: dict, kv_indices: Optional[Tensor] = None) -> Tensor:
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


        #if self.linearized:
        #    key = F.relu(key)        # (L, H, C)
        #    query = F.relu(query)    # (B*zL, H, C)

        #    message = (key[:,:,:,None] * value[:,:,None,:]).view(L, -1)    # (L, H*C*C)
        #    message = scatter_add(message, q_indices, dim=0, dim_size=B*zL, use_torch=False)       # (B*zL, H*C*C)
        #    message = (message.view(B*zL, H, C, C) * query.view(B*zL, H, C, 1)).sum(-1)    # (B*zL, H, C)

        #    accm = scatter_add(key.view(L, H*C), q_indices, dim=0, dim_size=B*zL, use_torch=False)       # (B*zL, H*C)
        #    accm = (accm.view(B*zL, H, C) * query).sum(-1)    # (B*zL, H)
        #    if self.noise_filter:
        #        w_dust, _ = self._prepair_dust(B, zL)    # (B*zL, H)
        #        accm = accm + w_dust

        #    message = message / (accm[:,:,None] + 1.0e-7)

        #    # projection
        #    message = message.view(B, zL, H*C)
        #    message = self.proj(message)
        #    message = self.proj_drop(message)

        if self.linearized:
            query = query[q_indices, :, :]    # (L, H, C)
            key = F.relu(key)
            query = F.relu(query)
            weights = (key * query).sum(-1)    # (L, H)

            # get weighted feature
            message = weights[:,:,None] * value      # (L, H, C)
            message = message.view(L, -1)  # (L, HC)

            # aggregate
            accm = scatter_add(weights, q_indices, dim=0, dim_size=B*zL, use_torch=False)       # (B*zL, H)
            if self.noise_filter:
                w_dust, _ = self._prepair_dust(B, zL)    # (B*zL, H)
                accm = accm + w_dust
            message = scatter_add(message, q_indices, dim=0, dim_size=B*zL, use_torch=False)    # (B*zL, HC)
            message = (message.view(B*zL, H, C) / (accm[:,:,None] + 1.0e-7)).view(B, zL, H*C)

            # projection
            message = message.view(B, zL, H*C)
            message = self.proj(message)
            message = self.proj_drop(message)

        else:
            query = query[q_indices, :, :]    # (L, H, C)
            weights = (key * query).sum(-1)    # (L, H)
            weights = self.softmax1(weights, q_indices, z.shape, key.shape)

            # get weighted feature
            message = weights[:,:,None] * value      # (L, H, C)
            message = message.view(L, -1)  # (L, HC)

            # aggregate
            message = scatter_add(message, q_indices, dim=0, dim_size=B*zL, use_torch=False)    # (B*zL, HC)

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

    def forward_for_checkpointing(self, evdata: Tensor, batch_indices: Tensor) -> Tuple[Tensor,Tensor]:
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



