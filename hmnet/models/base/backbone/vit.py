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

import numpy as np
from copy import deepcopy
from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from torch_scatter import scatter

from ..init import init_transformer, load_state_dict_flexible
from ..blocks import BlockBase
from ..layers import ConvBlock, Linear

from torch import Tensor
from typing import Tuple, List, Optional, Dict
Meta = Dict[str,Tuple[int,int,int]]

cfg_swin_tiny = dict(
    inc = 3,
    embed_force_stride = None,      # ------------------------------
    embed_dim  = 96,                # 128                192
    num_layers = [2,  2,  6,  2],   # [ 2, 2, 18,  2 ]   [ 2,  2, 18,  2 ]
    num_heads  = [3,  6, 12, 24],   # [ 4, 8, 16, 32 ]   [ 6, 12, 24, 48 ]
    drop_path_rate = 0.2,           # 0.5                0.2
    stride     = [2,  2,  2,  1],
    norm_layer = nn.LayerNorm,
    cfg_shared = dict(
        window_size = (7,7),
        enable_cyclic_shift = True,
        token_grouping = 'intra-window',
        pos_dynamic    = False,
        pos_log_scale  = False,
        use_checkpoint = False,
        mvit           = False,
    ),
)

cfg_cross_small = dict(
    inc = 3,
    patch_size = [4, 8, 16, 32],    # Base               Large
    embed_force_stride = None,      # ------------------------------
    embed_dim  = 96,                # 128                192
    num_layers = [2,  2,  6,  2],   # [ 2, 2, 18,  2 ]   [ 2,  2, 18,  2 ]
    num_heads  = [3,  6, 12, 24],   # [ 4, 8, 16, 32 ]   [ 6, 12, 24, 48 ]
    drop_path_rate = 0.2,           # 0.5                0.2
    stride     = [2,  2,  2,  1],
    norm_layer = nn.LayerNorm,
    downsample = 'Cross',
    cfg_shared = dict(
        window_size = (7,7),
        enable_cyclic_shift = False,
        token_grouping = 'dilated',
        pos_dynamic    = True,
        pos_log_scale  = False,
        use_checkpoint = False,
        mvit           = False,
        down_kernel_size = [2, 4],
    ),
)

class SeqData:
    def __init__(self, data: Tensor, meta: Meta) -> None:
        self.data = data
        self.meta = meta

    def clone(self):
        data = self.data.clone()
        meta = deepcopy(self.meta)
        return SeqData(data, meta)

    def to_2D(self):
        return self._seq_to_2D(self.data, self.meta)

    @staticmethod
    def zeros(B, C, H, W, dtype=float, device='cpu'):
        data = torch.zeros(B, H*W, C, dtype=dtype, device=device)
        meta = {'shape': [B, H, W]}
        return SeqData(data, meta)

    @staticmethod
    def from_2D(x):
        B, _, H, W = x.shape
        data, meta = SeqData._2D_to_seq(x)
        return SeqData(data, meta)

    @staticmethod
    def _seq_to_2D(x: Tensor, x_meta: Meta) -> Tensor:
        B, H, W = x_meta['shape']
        return x.transpose(1,2).contiguous().view(B, -1, H, W)

    @staticmethod
    def _2D_to_seq(x: Tensor) -> Tensor:
        B, C, H, W = x.shape
        data = x.view(B, C, H*W).transpose(1,2).contiguous()
        meta = {'shape': (B,H,W)}
        return data, meta

    @property
    def device(self):
        return self.data.device

    @property
    def dtype(self):
        return self.data.dtype

    def to(self, device):
        return SeqData(self.data.to(device), self.meta)

    def copy_(self, seq_data):
        self.data.copy_(seq_data.data)
        return self

    def share_memory_(self):
        self.data.share_memory_()

    def detach_(self):
        self.data = self.data.detach()

    def detach(self):
        return SeqData(self.data.detach(), self.meta)

    def filter_batch(self, mask):
        new_data = self.data[mask]
        B, H, W = self.meta['shape']
        B = mask.sum()
        new_meta = {'shape': (B,H,W)}
        return SeqData(new_data, new_meta)

    def __getitem__(self, index):
        new_data = self.data[index][None,:,:]
        B, H, W = self.meta['shape']
        B = len(new_data)
        new_meta = {'shape': (B,H,W)}
        return SeqData(new_data, new_meta)

    def __len__(self):
        return len(self.data)

    @staticmethod
    def concat_batch(list_seq_data):
        if len(list_seq_data) == 0:
            return list_seq_data

        # check input shape
        shape = list_seq_data[0].meta['shape']
        dshape = list_seq_data[0].data.shape
        for seq in list_seq_data:
            assert seq.meta['shape'] == shape
            assert seq.data.shape == dshape

        new_data = torch.cat([seq.data for seq in list_seq_data], dim=0)
        B, H, W = shape
        B = len(new_data)
        new_meta = {'shape': (B,H,W)}
        return SeqData(new_data, new_meta)

class VisionTransformer(BlockBase):
    def __init__(self, inc: int = 3, out_indices: Optional[List[int]] = None, seq_out: bool = False, embed_force_stride: Optional[int] = None, patch_size: Tuple[int] = [4], drop_path_rate=0.2, downsample: str = 'Swin',
                       embed_dim=96, num_layers: List[int] = [2,2,6,2], num_heads: List[int] = [3,6,12,24], stride=[2,2,2,1], cfg_shared: Optional[dict] = None,
                       norm_layer: type = nn.LayerNorm) -> None:
        super().__init__()
        self.seq_out = seq_out
        self.out_indices = out_indices

        n1,n2,n3,n4 = num_layers
        h1,h2,h3,h4 = num_heads
        s1,s2,s3,s4 = stride

        # stochastic depth
        dprs = self.stochastic_depth(drop_path_rate, num_layers)

        # split image into non-overlapping patches
        self.patch_embed = PatchEmbed(inc=inc, embed_dim=embed_dim, patch_size=patch_size, force_stride=embed_force_stride, norm_layer=norm_layer, ape=False)
        self.layer1 = Stage(dim=embed_dim*1, depth=n1, num_heads=h1, skip=True, downsample=downsample, down_stride=s1, drop_path=dprs[0], norm_layer=norm_layer, **cfg_shared)
        self.layer2 = Stage(dim=embed_dim*2, depth=n2, num_heads=h2, skip=True, downsample=downsample, down_stride=s2, drop_path=dprs[1], norm_layer=norm_layer, **cfg_shared)
        self.layer3 = Stage(dim=embed_dim*4, depth=n3, num_heads=h3, skip=True, downsample=downsample, down_stride=s3, drop_path=dprs[2], norm_layer=norm_layer, **cfg_shared)
        self.layer4 = Stage(dim=embed_dim*8, depth=n4, num_heads=h4, skip=True, downsample=None      , down_stride=s4, drop_path=dprs[3], norm_layer=norm_layer, **cfg_shared)

        if seq_out == False:
            self.norm_s1 = norm_layer(embed_dim*1)
            self.norm_s2 = norm_layer(embed_dim*2)
            self.norm_s3 = norm_layer(embed_dim*4)
            self.norm_s4 = norm_layer(embed_dim*8)
            self.norm_x4 = norm_layer(embed_dim*8)
        self.init_weights()

    def stochastic_depth(self, drop_path_rate: float, depths: int) -> List[float]:
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]  # stochastic depth decay rule
        return [ dpr[sum(depths[:i]):sum(depths[:i+1])] for i in range(len(depths)) ]

    @torch.jit.ignore
    def init_weights(self):
        init_transformer(self.modules())

    @property
    def no_decay_set(self):
        #return {'absolute_pos_embed', 'relative_position_bias_table', 'position_embedding_table', 'bias'}
        return {'bias'}

    def load_from_pretrained(self, fpath_pretrained):
        src_dict = {}
        org_dict = torch.load(fpath_pretrained)
        for key, value in org_dict.items():
            key = key.replace('layers.3', 'layers.4')
            key = key.replace('layers.2', 'layers.3')
            key = key.replace('layers.1', 'layers.2')
            key = key.replace('layers.0', 'layers.1')
            src_dict[key] = value
        load_state_dict_flexible(self, src_dict)

    def forward(self, x: Tensor) -> Tensor:
        seq_x0 = self.patch_embed(x)
        seq_x1, seq_s1 = self.layer1(seq_x0)
        seq_x2, seq_s2 = self.layer2(seq_x1)
        seq_x3, seq_s3 = self.layer3(seq_x2)
        seq_x4, seq_s4 = self.layer4(seq_x3)    # Note: seq_x4 and seq_s4 are identical

        features = self._to_out(seq_s1, seq_s2, seq_s3, seq_s4)

        if self.out_indices is not None:
            output = [ x for i, x in enumerate(features) if i in self.out_indices ]
            return output
        else:
            return features[-1]

    def _to_out(self, seq_s1, seq_s2, seq_s3, seq_s4):
        if self.seq_out:
            s1 = seq_s1
            s2 = seq_s2
            s3 = seq_s3
            s4 = seq_s4
        else:
            seq_s1 = SeqData(self.norm_s1(seq_s1.data), seq_s1.meta)
            seq_s2 = SeqData(self.norm_s2(seq_s2.data), seq_s2.meta)
            seq_s3 = SeqData(self.norm_s3(seq_s3.data), seq_s3.meta)
            seq_s4 = SeqData(self.norm_s4(seq_s4.data), seq_s4.meta)
            s1 = seq_s1.to_2D()
            s2 = seq_s2.to_2D()
            s3 = seq_s3.to_2D()
            s4 = seq_s4.to_2D()

        return s1, s2, s3, s4

class Stage(BlockBase):
    def __init__(self, dim: int, depth: int, num_heads: int, window_size: Tuple[int,int], skip=False,
                 enable_cyclic_shift: bool = False, token_grouping: str = 'none', relative_pos_bias: bool = True, pos_dynamic: bool = False, pos_log_scale: bool = False,
                 drop_path: float = 0., norm_layer: type = nn.LayerNorm, mvit: bool = False, mvit_norm_layer: type = nn.BatchNorm2d,
                 downsample: Optional[str] = None, use_checkpoint: bool = False, down_kernel_size: int = 2, down_stride: int = 2) -> None:

        super().__init__()
        assert downsample is None or downsample in ('Swin', 'Cross')
        if not isinstance(drop_path, (list, tuple)):
            drop_path = [drop_path] * depth
        assert len(drop_path) == depth

        self.dim = dim
        self.depth = depth
        self.use_checkpoint = use_checkpoint
        self.skip = skip
        self.mvit = MobileViTAttachment(dim, pre_act=False, dw=False, fuse_method='cat', norm_layer=mvit_norm_layer, gn_dim=32) if mvit else None

        # build blocks
        self.blocks = nn.ModuleList()
        for i in range(depth):
            cyclic_shift = (i % 2 == 1) and enable_cyclic_shift
            block = TransformerBlock(dim=dim, num_heads=num_heads, window_size=window_size, cyclic_shift=cyclic_shift, grouping=token_grouping,
                                     drop_path=drop_path[i], relative_pos_bias=relative_pos_bias, pos_dynamic=pos_dynamic, pos_log_scale=pos_log_scale, norm_layer=norm_layer)
            self.blocks.append(block)

        # patch merging layer
        if downsample is None:
            self.downsample = nn.Identity()
        elif downsample == 'Swin':
            self.downsample = PatchMergingSwin(dim=dim, stride=down_stride, norm_layer=norm_layer)
        elif downsample == 'Cross':
            self.downsample = PatchMergingCross(dim=dim, patch_size=down_kernel_size, stride=down_stride, norm_layer=norm_layer)

    def forward(self, seq_x: SeqData) -> Tensor:
        if self.mvit is not None:
            seq_in = seq_x.clone()    # skip
            seq_x = self.mvit.local_rep(seq_x)

        for block in self.blocks:
            seq_x = block(seq_x)

        if self.mvit is not None:
            seq_x = self.mvit.fuse(seq_x, seq_in)

        seq_s = seq_x
        seq_x = self.downsample(seq_x)

        if self.skip:
            return seq_x, seq_s
        else:
            return seq_x

class TransformerBlock(BlockBase):
    def __init__(self, dim: int, num_heads: int, window_size: Tuple[int,int] = (7,7),
                 cyclic_shift: bool = False, grouping: str = 'intra-window', drop_path=0,
                 mvit: bool = False, mvit_norm_layer: type = nn.GroupNorm, default_input_size: Optional[Tuple[int, int]] = None,
                 relative_pos_bias: bool = True, pos_dynamic: bool = False, pos_log_scale: bool = False, norm_layer: type = nn.LayerNorm) -> None:
        super().__init__()
        # minor params
        qkv_bias  = True
        qk_scale  = None
        attn_drop = 0.
        mlp_ratio = 4
        drop      = 0
        act_layer = nn.GELU

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size

        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = Attention(
            dim, window_size=self.window_size, grouping=grouping, num_heads=num_heads, relative_pos_bias=relative_pos_bias, pos_dynamic=pos_dynamic, pos_log_scale=pos_log_scale,
            qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, cyclic_shift=cyclic_shift, default_input_size=default_input_size)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)

        self.mvit = MobileViTAttachment(dim, pre_act=True, dw=False, fuse_method='add', norm_layer=mvit_norm_layer, gn_dim=32) if mvit else None

    def forward(self, seq_x: SeqData) -> Tensor:
        if self.mvit is not None:
            seq_in = seq_x.clone()    # skip
            seq_x = self.mvit.local_rep(seq_x)

        x, x_meta = seq_x.data, seq_x.meta
        x1 = x.clone()

        # attention
        x = self.norm1(x)
        x = self.attn(x, x_meta)

        # FFN
        x = x1 + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        seq_x = SeqData(x, x_meta)

        if self.mvit is not None:
            seq_x = self.mvit.fuse(seq_x, seq_in)

        return seq_x

class CrossAttentionBlock(BlockBase):
    def __init__(self, dim: int, kv_dim: int, num_heads: int, window_size: Tuple[int,int] = (7,7), kv_window_size: Tuple[int,int] = (7,7),
                 cyclic_shift: bool = False, grouping: str = 'intra-window', mvit: bool = False, mvit_norm_layer: type = nn.BatchNorm2d, drop_path=0,
                 use_relative_pos_bias: bool = True, pos_dynamic: bool = False, pos_log_scale: bool = False, norm_layer: type = nn.LayerNorm) -> None:
        super().__init__()
        # minor params
        qkv_bias  = True
        qk_scale  = None
        mlp_ratio = 4
        drop      = 0
        act_layer = nn.GELU

        self.dim = dim
        self.num_heads = num_heads
        self.window_size = window_size
        self.kv_window_size = kv_window_size

        self.mlp_ratio = mlp_ratio
        self.mvit = MobileViTAttachment(dim, pre_act=True, dw=False, fuse_method='cat', gn_dim=32, norm_layer=mvit_norm_layer) if mvit else None
        self.kv_mvit = MobileViTAttachment(kv_dim, pre_act=True, dw=False, fuse_method='cat', gn_dim=32, norm_layer=mvit_norm_layer) if mvit else None

        self.attn_norm1 = norm_layer(dim)
        self.attn_norm2 = norm_layer(kv_dim)
        self.attn = CrossAttention(
            dim, kv_dim, wsize=window_size, kv_wsize=kv_window_size, grouping=grouping, num_heads=num_heads, cyclic_shift=cyclic_shift,
            pos_dynamic=pos_dynamic, pos_log_scale=pos_log_scale,
            qkv_bias=qkv_bias, qk_scale=qk_scale, proj_drop=drop, use_relative_pos_bias=use_relative_pos_bias)

        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.mlp_norm = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=drop)

    def forward(self, seq_x: SeqData, seq_y: SeqData) -> Tensor:
        if self.mvit is not None:
            seq_s = seq_x.clone()    # skip
            seq_x = self.mvit.local_rep(seq_x)
            seq_y = self.kv_mvit.local_rep(seq_y)

        x, x_meta = seq_x.data, seq_x.meta
        y, y_meta = seq_y.data, seq_y.meta
        x1 = x.clone()

        x = self.attn_norm1(x)
        y = self.attn_norm2(y)
        x = self.attn(x, y, x_meta, y_meta)

        x = x1 + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.mlp_norm(x)))

        seq_x = SeqData(x, x_meta)

        if self.mvit is not None:
            seq_x = self.mvit.fuse(seq_x, seq_s)

        return seq_x

class Attention(BlockBase):
    def __init__(self, dim: int, window_size: Tuple[int,int], grouping: str, num_heads: int,
                       relative_pos_bias: bool = True, pos_dynamic: bool = False, pos_log_scale: bool = False, qkv_bias: bool = True,
                       qk_scale: Optional[float] = None, attn_drop: float = 0., proj_drop: float = 0.,
                       cyclic_shift: bool = False, default_input_size: Optional[Tuple[int, int]] = None) -> None:

        super().__init__()
        self.dim = dim
        self.window_size = window_size  # Wh, Ww
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.grouping = TokenGrouping(window_size, cyclic_shift, grouping, default_input_size)
        self.qkv = QKVTransform(dim, num_heads, qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.relative_pos_bias = relative_pos_bias
        if relative_pos_bias:
            self.position_bias = PositionBias(window_size, num_heads, dynamic=pos_dynamic, dynamic_dim=dim // 4, log_scale=pos_log_scale)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, x: Tensor, x_meta: dict, pos: Optional[Tensor] = None) -> Tensor:
        # grouping by local windows
        x, attn_mask = self.grouping.build(x, x_meta)

        B_, N, C = x.shape
        q, k, v = self.qkv(x)

        if pos is not None:
            q = q + pos
            k = k + pos

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))

        if self.relative_pos_bias:
            bias = self.position_bias()
            attn = attn + bias.unsqueeze(0)

        if attn_mask is not None:
            nW = attn_mask.shape[0]
            attn = attn.view(B_ // nW, nW, self.num_heads, N, N) + attn_mask.unsqueeze(1).unsqueeze(0)
            attn = attn.view(-1, self.num_heads, N, N)
        attn = self.softmax(attn)

        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B_, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        # ungrouping
        x = self.grouping.resolve(x)

        return x

class SparseAttentionBlock(BlockBase):
    def __init__(self, dim: int, num_heads: int, use_edge_weight: bool = False, drop_path=0., mlp_ratio=4, norm_layer: type = nn.LayerNorm, act_layer: type = nn.GELU) -> None:
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        self.mlp_ratio = mlp_ratio

        self.norm1 = norm_layer(dim)
        self.attn = SparseAttention(dim, num_heads, use_edge_weight)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        self.mlp = Mlp(in_features=dim, hidden_features=dim*mlp_ratio, act_layer=act_layer, drop=0)

    def forward(self, x: Tensor, adj: Tensor) -> Tensor:
        x1 = x.clone()

        # attention
        x = self.norm1(x)
        x = self.attn(x, adj)

        # FFN
        x = x1 + self.drop_path(x)
        x = x + self.drop_path(self.mlp(self.norm2(x)))

        return x


# TODO add relative_pos_bias
class SparseAttention(BlockBase):
    def __init__(self, dim, num_heads, use_edge_weight=False):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv  = nn.Linear(dim, dim*3)
        self.proj = nn.Linear(dim, dim)
        self.use_edge_weight = use_edge_weight

    def forward(self, x, adj):
        L, C = x.shape
        H = self.num_heads

        qkv = self.qkv(x).view(L, 3, H, C // H).permute(1,0,2,3)
        query, key, value = qkv[0], qkv[1], qkv[2]    # (L, H, C')
        query = query * self.scale

        row, col = adj.nonzero().T                    # (E,)
        attn = (query[row] * key[col]).sum(dim=-1)    # (E, H)
        if self.use_edge_weight:
            attn = attn + adj[row, col][:,None]

        # softmax
        max_attn = scatter(attn.detach(), row, reduce='max', dim=0, dim_size=L)    # (L, H)
        attn = attn - max_attn[row]
        attn = attn.exp()

        acc = scatter_add(attn, row, dim=0, dim_size=L, use_torch=False)    # (L, H)
        attn = attn / (acc[row] + 1e-7)

        # get weighted feature
        message = attn[:,:,None] * value[col]    # (E, H, C')
        message = message.view(-1, C)          # (E, C)
        message = scatter_add(message, row, dim=0, dim_size=L, use_torch=False)    # (L, C)
        message = self.proj(message)

        return message


class MobileViTAttachment(BlockBase):
    def __init__(self, dim: int, gn_dim: int = 32, pre_act: bool = False, dw: bool = False, fuse_method: str = 'cat', norm_layer: type = nn.BatchNorm2d, act_layer: type = nn.SiLU) -> None:
        super().__init__()
        assert fuse_method in ('cat', 'add')
        self.fuse_method = fuse_method
        g = dim if dw else 1
        fdim = dim * 2 if fuse_method == 'cat' else dim

        self.conv1 = ConvBlock( dim, dim, kernel_size=3, padding=1, groups=g, norm_layer=norm_layer, act_layer=act_layer, gn_dim=gn_dim, pre_act=pre_act)
        self.conv2 = ConvBlock( dim, dim, kernel_size=1, padding=0, groups=1, norm_layer=norm_layer, act_layer=act_layer, gn_dim=gn_dim, pre_act=pre_act)
        self.fuse1 = ConvBlock( dim, dim, kernel_size=1, padding=0, groups=1, norm_layer=norm_layer, act_layer=act_layer, gn_dim=gn_dim, pre_act=pre_act)
        self.fuse2 = ConvBlock(fdim, dim, kernel_size=3, padding=1, groups=1, norm_layer=norm_layer, act_layer=act_layer, gn_dim=gn_dim, pre_act=pre_act)
        self.norm = nn.LayerNorm(dim)

    def local_rep(self, seq_x: SeqData) -> Tensor:
        if isinstance(seq_x, SeqData):
            x = seq_x.to_2D()
        else:
            x = seq_x
        x = self.conv1(x)
        x = self.conv2(x)
        seq_x = SeqData.from_2D(x)
        return seq_x

    def fuse(self, seq_x: SeqData, seq_s: SeqData) -> Tensor:
        seq_x = SeqData(self.norm(seq_x.data), seq_x.meta)

        x = seq_x.to_2D()
        if isinstance(seq_s, SeqData):
            s = seq_s.to_2D()
        else:
            s = seq_s
        x = self.fuse1(x)
        if self.fuse_method == 'cat':
            x = torch.cat([x, s], dim=1)
        elif self.fuse_method == 'add':
            x = x + s
        x = self.fuse2(x)
        x = SeqData.from_2D(x)
        return x

class TokenGrouping(BlockBase):
    def __init__(self, window_size: Tuple[int,int], cyclic_shift: bool = False, grouping: str = 'intra-window', default_input_size: Optional[Tuple[int, int]] = None):
        # intra-window: SwinTransformer
        # inter-window: MobileViT
        # dilated     : CrossFormer
        super().__init__()
        assert grouping in ('intra-window', 'inter-window', 'dilated', 'none')
        self.window_size = window_size
        self.cyclic_shift = cyclic_shift
        self.shift_size = (window_size[0] // 2, window_size[1] // 2)
        self.grouping = grouping
        self.default_input_size = default_input_size
        self.org_size    = None    # temporary holder
        self.padded_size = None    # temporary holder

        self.attn_mask = None
        if self.cyclic_shift and default_input_size is not None:
            self.attn_mask = self._make_attn_mask(default_input_size, torch.device('cpu'))

    def build(self, x: Tensor, x_meta: dict, input_shape: str = 'seq', output_shape: str = 'seq') -> Tuple[Tensor,Tensor]:
        B, H, W = x_meta['shape']
        self.org_size = (B, H, W)

        if self.grouping == 'none':
            return x, None
        if H <= self.window_size[0] and W <= self.window_size[1]:
            return x, None

        x = self._to_in(x, (H,W), input_shape)

        x = self._padding(x)

        attn_mask = self._make_attn_mask(self.padded_size[-2:], x.device) if self.cyclic_shift else None

        if self.cyclic_shift:
            x = self._cyclic_shift(x)

        x = self._window_partition(x)

        x = self._to_out(x, output_shape)

        return x, attn_mask

    def resolve(self, x: Tensor, input_shape: str = 'seq', output_shape: str = 'seq') -> Tensor:
        B, H, W = self.org_size
        B, padH, padW = self.padded_size

        if self.grouping == 'none':
            return x
        if H <= self.window_size[0] and W <= self.window_size[1]:
            return x

        if self.grouping in ('intra-window', 'dilated'):
            tile_size = self.window_size
        elif self.grouping == 'inter-window':
            h, w = self.window_size
            tile_size = (padH // h, padW // w)

        x = self._to_in(x, tile_size, input_shape)

        x = self._window_reverse(x)

        if self.cyclic_shift is True:
            x = self._cyclic_shift(x, reverse=True)

        x = self._crop(x)

        x = self._to_out(x, output_shape)

        return x

    def _to_in(self, x: Tensor, size: Tuple[int,int], input_shape: str) -> Tensor:
        if input_shape == 'seq':
            B, L, C = x.shape
            H, W = size
            return x.view(B, H, W, C)
        elif input_shape == '2D':
            B, C, H, W = x.shape
            return x.permute(0, 2, 3, 1).contiguous()
        else:
            raise RuntimeError

    def _to_out(self, x: Tensor, output_shape: str) -> Tensor:
        if output_shape == 'seq':
            B, H, W, C = x.shape
            return x.view(B, H*W, C)
        elif output_shape == '2D':
            return x.permute(0,3,1,2).contiguous()
        else:
            raise RuntimeError

    def _cyclic_shift(self, x: Tensor, reverse: bool = False) -> Tensor:
        shift_h = self.shift_size[0] * (reverse * 2 - 1)
        shift_w = self.shift_size[1] * (reverse * 2 - 1)
        return torch.roll(x, shifts=(shift_h, shift_w), dims=(1, 2))

    def _padding(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        h, w = self.window_size

        padH = int(np.ceil(float(H) / h)) * h
        padW = int(np.ceil(float(W) / w)) * w

        self.padded_size = (B, padH, padW)

        padding = (0, 0, 0, padW - W, 0, padH - H)

        return F.pad(x, padding, mode='constant', value=0)

    def _window_partition(self, x: Tensor) -> Tensor:
        B, H, W, C = x.shape
        h, w = self.window_size

        if h == 1 and w == 1:
            if self.grouping == 'intra-window':
                x = x.view(-1, 1, 1, C)
            elif self.grouping == 'inter-window':
                pass
            elif self.grouping == 'dilated':
                x = x.view(-1, 1, 1, C)
        else:
            if self.grouping == 'intra-window':
                x = x.view(B, H // h, h, W // w, w, C)
                x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(-1, h, w, C)
            elif self.grouping == 'inter-window':
                x = x.view(B, H // h, h, W // w, w, C)
                x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, H // h, W // w, C)
            elif self.grouping == 'dilated':
                x = x.view(B, h, H // h, w, W // w, C)
                x = x.permute(0, 2, 4, 1, 3, 5).contiguous().view(-1, h, w, C)

        return x

    def _window_reverse(self, x: Tensor) -> Tensor:
        B, H, W = self.padded_size
        h, w = self.window_size

        if h == 1 and w == 1:
            if self.grouping == 'intra-window':
                x = x.view(B, H, W, -1)
            elif self.grouping == 'inter-window':
                pass
            elif self.grouping == 'dilated':
                x = x.view(B, H, W, -1)
        else:
            if self.grouping == 'intra-window':
                x = x.view(B, H // h, W // w, h, w, -1)
                x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H, W, -1)
            elif self.grouping == 'inter-window':
                x = x.view(B, h, w, H // h, W // w, -1)
                x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, -1)
            elif self.grouping == 'dilated':
                x = x.view(B, H // h, W // w, h, w, -1)
                x = x.permute(0, 3, 1, 4, 2, 5).contiguous().view(B, H, W, -1)
        return x

    def _crop(self, x: Tensor) -> Tensor:
        B, padH, padW, C = x.shape
        B, H, W = self.org_size
        x = x[:,:H,:W,:].contiguous()
        return x

    def _make_attn_mask(self, input_size: int, device: torch.device) -> Tensor:
        H, W = input_size
        if self.attn_mask is not None and self.default_input_size is not None and H == self.default_input_size[0] and W == self.default_input_size[1]:
            return self.attn_mask

        # calculate attention mask for SW-MSA
        img_mask = torch.zeros((1, H, W, 1), device=device)  # 1 H W 1
        h_slices = (slice(0, -self.window_size[0]),
                    slice(-self.window_size[0], -self.shift_size[0]),
                    slice(-self.shift_size[0], None))
        w_slices = (slice(0, -self.window_size[1]),
                    slice(-self.window_size[1], -self.shift_size[1]),
                    slice(-self.shift_size[1], None))
        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w, :] = cnt
                cnt += 1

        mask_windows = self._window_partition(img_mask)
        mask_windows = mask_windows.view(-1, self.window_size[0] * self.window_size[1])
        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, float(-100.0)).masked_fill(attn_mask == 0, float(0.0))
        return attn_mask

class PositionBias(BlockBase):
    def __init__(self, window_size: Tuple[int,int], num_heads: int, dynamic: bool = False, dynamic_dim: Optional[int] = None, log_scale: bool = False, sparse_input: bool = False) -> None:
        super().__init__()
        h, w = window_size
        self.window_size = window_size
        self.dynamic = dynamic
        self.sparse_input = sparse_input
        self.log_scale = log_scale
        self.num_heads = num_heads
        assert sparse_input is False or (sparse_input * dynamic)

        if dynamic is True:
            self.pos = DynamicPosBias(dynamic_dim, num_heads, residual=False, log_scale=log_scale)
            # generate mother-set
            position_bias_h = torch.arange(1 - h, h)
            position_bias_w = torch.arange(1 - w, w)
            inputs_for_table = torch.stack(torch.meshgrid([position_bias_h, position_bias_w]))
            inputs_for_table = inputs_for_table.flatten(1).transpose(0, 1).float()    # (2h-1 * 2w-1, 2)
            self.register_buffer("inputs_for_table", inputs_for_table)
        else:
            # define a parameter table of relative position bias
            table_h = 2 * h - 1
            table_w = 2 * w - 1
            self.relative_position_bias_table = nn.Parameter(torch.zeros(table_h * table_w, num_heads))  # 2h-1 * 2w-1, nH
            trunc_normal_(self.relative_position_bias_table, std=.02)

        # get pair-wise relative position index for each token inside the window
        coords = self._grid_coords(h, w)
        relative_coords = self.relative_coords(coords)
        relative_position_index = self._to_table_indices(relative_coords, h, w)
        self.register_buffer("relative_position_index", relative_position_index)

    @property
    def no_decay_set(self):
        return {'relative_position_bias_table'}

    def forward(self, coords: Optional[Tensor] = None) -> Tensor:
        h, w = self.window_size

        if self.sparse_input is True:
            relative_coords = self.relative_coords(coords)    # (L, L, 2)
            relative_position_bias = self.pos(relative_coords.view(-1, 2))
        else:
            table = self.pos(self.inputs_for_table) if self.dynamic else self.relative_position_bias_table    # (2h-1 * 2w-1, nH)
            relative_position_bias = table[self.relative_position_index.view(-1)]

        relative_position_bias = relative_position_bias.view(h*w, h*w, -1)  # (L, L, nH)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()  # (nH, L, L)

        return relative_position_bias

    def relative_coords(self, coords: Tensor) -> Tensor:
        return coords[:, None, :] - coords[None, :, :]  # (L, L, 2)

    def _grid_coords(self, h: int, w: int) -> Tensor:
        coords_h = torch.arange(h)
        coords_w = torch.arange(w)
        return torch.stack(torch.meshgrid([coords_h, coords_w])).view(2, h*w).transpose(1, 0).contiguous()    # (L, 2)

    def _to_table_indices(self, relative_coords: Tensor, h: int, w: int) -> Tensor:
        relative_coords[:, :, 0] += h - 1  # shift to start from 0
        relative_coords[:, :, 1] += w - 1
        relative_coords[:, :, 0] *= 2 * w - 1
        return relative_coords.sum(-1)  # convert to indices of flattened table

class DynamicPosBias(BlockBase):
    def __init__(self, dim: int, num_heads: int, residual: bool, log_scale: bool) -> None:
        super().__init__()
        self.residual = residual
        self.log_scale = log_scale
        self.num_heads = num_heads
        self.pos_dim = dim // 4
        self.pos_proj = nn.Linear(2, self.pos_dim)
        self.pos1 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim),
        )
        self.pos2 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.pos_dim)
        )
        self.pos3 = nn.Sequential(
            nn.LayerNorm(self.pos_dim),
            nn.ReLU(inplace=True),
            nn.Linear(self.pos_dim, self.num_heads)
        )

    def forward(self, relative_coords: Tensor) -> Tensor:
        if self.log_scale:
            relative_coords = self._to_log_scale(relative_coords)

        if self.residual:
            pos = self.pos_proj(relative_coords) # (2h-1 * 2w-1, pos_dim)
            pos = pos + self.pos1(pos)
            pos = pos + self.pos2(pos)
            pos = self.pos3(pos)    # (2h-1 * 2w-1, nH)
        else:
            pos = self.pos3(self.pos2(self.pos1(self.pos_proj(relative_coords))))
        return pos

    def _to_log_scale(self, relative_coords: Tensor) -> Tensor:
        return relative_coords.sign() * (1 + relative_coords.abs()).log()

class QKVTransform(BlockBase):
    def __init__(self, dim: int, num_heads: int, qkv_bias: bool = True, cross: bool = False, kv_dim: Optional[int] = None) -> None:
        super().__init__()
        self.num_heads = num_heads
        self.cross = cross
        assert dim % num_heads == 0

        if self.cross:
            self.q = nn.Linear(dim, dim, bias=qkv_bias)
            self.kv = nn.Linear(kv_dim or dim, dim * 2, bias=qkv_bias)
        else:
            self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)

    def forward(self, x: Tensor, context: Optional[Tensor] = None) -> Tuple[Tensor,Tensor,Tensor]:
        B, _, C = x.shape
        if self.cross:
            assert context is not None
            q = self.q(x).reshape(B, -1, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
            kv = self.kv(context).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            k, v = kv[0], kv[1]  # make torchscript happy (cannot use tensor as tuple)
        else:
            qkv = self.qkv(x).reshape(B, -1, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
            q, k, v = qkv[0], qkv[1], qkv[2]  # make torchscript happy (cannot use tensor as tuple)

        return q, k, v

class PatchMergingSwin(BlockBase):
    def __init__(self, dim: int, stride: int, norm_layer: type = nn.LayerNorm) -> None:
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.norm = norm_layer(4 * dim)
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        
    def forward(self, seq_x: SeqData) -> SeqData:
        x, x_meta = seq_x.data, seq_x.meta
        C = x.shape[-1]
        B, H, W = x_meta['shape']

        x = x.view(B, H, W, C)
        if self.stride == 2:
            x0 = x[:, 0::2, 0::2, :]  # B, H/2, W/2, C
            x1 = x[:, 1::2, 0::2, :]  # B, H/2, W/2, C
            x2 = x[:, 0::2, 1::2, :]  # B, H/2, W/2, C
            x3 = x[:, 1::2, 1::2, :]  # B, H/2, W/2, C
            x = torch.cat([x0, x1, x2, x3], -1)  # B, H/2, W/2, 4*C
        elif self.stride == 1:
            x = F.pad(x, (0,1,0,1))
            x0 = x[:, 0:-1, 0:-1, :]  # B, H, W, C
            x1 = x[:, 1:  , 0:-1, :]  # B, H, W, C
            x2 = x[:, 0:-1, 1:  , :]  # B, H, W, C
            x3 = x[:, 1:  , 1:  , :]  # B, H, W, C
            x = torch.cat([x0, x1, x2, x3], -1)  # B, H, W, 4*C

        x = x.view(B, -1, 4*C)
        x = self.norm(x)
        x = self.reduction(x)

        new_H = int(H / self.stride)
        new_W = int(W / self.stride)
        x_meta = {'shape': (B, new_H, new_W)}

        return SeqData(x, x_meta)

class PatchMergingCross(BlockBase):
    def __init__(self, dim: int, stride: int, norm_layer: type = nn.LayerNorm, patch_size: List[int] = [2], out_dim: Optional[int] = None) -> None:
        super().__init__()
        self.dim = dim
        self.stride = stride
        self.reductions = nn.ModuleList()
        self.patch_size = patch_size
        self.norm = norm_layer(dim) if norm_layer else nn.Identity()
        self.out_dim = out_dim or 2 * dim

        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                _outc = self.out_dim // 2 ** i
            else:
                _outc = self.out_dim // 2 ** (i + 1)
            padding = (ps - 1) // 2 if stride > 1 else 'same'
            self.reductions.append(nn.Conv2d(dim, _outc, kernel_size=ps, stride=stride, padding=padding))

    def forward(self, seq_x: SeqData) -> SeqData:
        seq_x = SeqData(self.norm(seq_x.data), seq_x.meta)
        x = seq_x.to_2D()
        xs = [ reduction(x) for reduction in self.reductions ]
        x = torch.cat(xs, dim=1)
        return SeqData.from_2D(x)

def padding_same(data, kernel_size, stride):
    pad = max(0, kernel_size - stride)
    pad_l = pad // 2
    pad_r = pad - pad_l
    pad_t = pad // 2
    pad_b = pad - pad_t
    return F.pad(data, (pad_l, pad_r, pad_t, pad_b))

class PatchEmbed(BlockBase):
    def __init__(self, patch_size: List[int] = [4], inc: int = 3, embed_dim: int = 96, force_stride: Optional[int] = None,
                       norm_layer: Optional[type] = None, ape: bool = False, ape_seq_length: Optional[int] = None, drop_rate: float = 0) -> None:
        super().__init__()
        if not isinstance(patch_size, (tuple, list)):
            patch_size = [patch_size]

        self.patch_size = patch_size
        self.inc = inc
        self.embed_dim = embed_dim
        self.ape = ape

        self.projs = nn.ModuleList()
        self.paddings = []
        for i, ps in enumerate(patch_size):
            if i == len(patch_size) - 1:
                dim = embed_dim // 2 ** i
            else:
                dim = embed_dim // 2 ** (i + 1)
            stride = force_stride or patch_size[0]
            if stride > 1:
                padding = (ps - stride) // 2
            else:
                #padding = 'same'
                padding = 0
                padfunc = partial(padding_same, kernel_size=ps, stride=stride)
                self.paddings.append(padfunc)
            self.projs.append(nn.Conv2d(inc, dim, kernel_size=ps, stride=stride, padding=padding))

        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()

        # absolute position embedding
        if self.ape:
            self.absolute_pos_embed = nn.Parameter(torch.zeros(1, ape_seq_length, embed_dim))
            trunc_normal_(self.absolute_pos_embed, std=.02)
        self.pos_drop = nn.Dropout(p=drop_rate)

    @property
    def no_decay_set(self):
        return {'absolute_pos_embed'}

    def forward(self, x: Tensor) -> SeqData:

        if len(self.paddings) > 0:
            xs = []
            for padding, proj in zip(self.paddings, self.projs):
                xs.append(proj(padding(x)))
        else:
            xs = [ proj(x) for proj in self.projs ]
        x = torch.cat(xs, dim=1)
        B, C, H, W = x.shape
        x = x.flatten(2).transpose(1,2)

        #xs = []
        #for proj in self.projs:
        #    tx = proj(x)
        #    B, C, H, W = tx.shape
        #    tx = tx.flatten(2).transpose(1, 2)
        #    xs.append(tx)  # B Ph*Pw C
        #x = torch.cat(xs, dim=2)

        x = self.norm(x)

        if self.ape:
            x = x + self.absolute_pos_embed

        x = self.pos_drop(x)
        x_meta = {'shape': (B, H, W)}
        return SeqData(x, x_meta)

class Mlp(BlockBase):
    def __init__(self, in_features: int, hidden_features: Optional[int] = None, out_features: Optional[int] = None, act_layer: type = nn.GELU, drop: float = 0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x: Tensor) -> Tensor:
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class CrossAttention(nn.Module):
    def __init__(self, dim: int, kv_dim: int, wsize: Tuple[int,int], kv_wsize: Tuple[int,int], num_heads: int, grouping: str = 'intra-window', cyclic_shift: bool = False,
                       pos_dynamic: bool = False, pos_log_scale: bool = False, qkv_bias: bool = True, qk_scale: Optional[float] = None, proj_drop: float = 0., use_relative_pos_bias: bool = True) -> None:

        super().__init__()
        self.num_heads = num_heads
        self.scale = qk_scale or (dim // num_heads) ** -0.5

        self.grouping = TokenGrouping(wsize, cyclic_shift, grouping)
        self.kv_grouping = TokenGrouping(kv_wsize, cyclic_shift, grouping)
        self.qkv = QKVTransform(dim, num_heads, qkv_bias, cross=True, kv_dim=kv_dim)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.use_relative_pos_bias = use_relative_pos_bias
        if self.use_relative_pos_bias:
            rcoords, table_size = get_relative_position_indices(wsize, kv_wsize)
            self.register_buffer("relative_position_indices", rcoords)
            self.relative_position_bias = PositionEmbedding2D(table_size[0], table_size[1], num_heads, dynamic=pos_dynamic, dynamic_dim=dim // 4, shift_normalize=True, log_scale=pos_log_scale)
        self.softmax = nn.Softmax(dim=-1)

    def forward(self, z: Tensor, x: Tensor, z_meta: dict, x_meta: dict, z_pos: Optional[Tensor] = None, x_pos: Optional[Tensor] = None) -> Tensor:
        z, _ = self.grouping.build(z, z_meta)
        x, _ = self.kv_grouping.build(x, x_meta)

        B, N0, C = z.shape
        B, N1, _ = x.shape

        q, k, v = self.qkv(z, context=x)                      # q = (B, H, N0, C)
                                                              # k, v = (B, H, N1, C)
        if z_pos is not None:
            q = q + z_pos
        if x_pos is not None:
            k = k + x_pos

        q = q * self.scale
        attn = (q @ k.transpose(-2, -1))                      # attn = (B, H, N0, N1)

        if self.use_relative_pos_bias:
            rpi = self.relative_position_indices
            bias = self.relative_position_bias(rpi[0], rpi[1])  # bias = (N0*N1, H)
            bias = bias.T.view(1, -1, N0, N1)                   # bias = (1, H, N0, N1)
            attn = attn + bias
        attn = self.softmax(attn)

        m = (attn @ v).transpose(1, 2).reshape(B, N0, C)
        m = self.proj(m)
        m = self.proj_drop(m)

        m = self.grouping.resolve(m)

        return m

def get_relative_position_indices(window1: Tuple[int], window2: Tuple[int]) -> Tuple[Tensor,list]:
    assert len(window1) == len(window2)
    ndim = len(window1)
    w1 = torch.LongTensor(window1)
    w2 = torch.LongTensor(window2)
    s1 = (w2 / w1).clip(min=1)
    s2 = (w1 / w2).clip(min=1)

    coords1 = torch.stack(torch.meshgrid([ torch.arange(n) * int(s) for s, n in zip(s1, w1) ])).view(ndim, -1)
    coords2 = torch.stack(torch.meshgrid([ torch.arange(n) * int(s) for s, n in zip(s2, w2) ])).view(ndim, -1)
    relative_coords = (coords2[:, None, :] - coords1[:, :, None]).view(ndim, -1)
    relative_coords -= relative_coords.amin(dim=1, keepdims=True)
    table_size = (relative_coords.amax(dim=1) + 1).tolist()
    return relative_coords, table_size

class PositionEmbedding1D(nn.Module):
    def __init__(self, x_size: int, embed_dim: int, dynamic: bool = False, dynamic_dim: Optional[int] = None,
                       shift_normalize: bool = False, scale_normalize: bool = False, log_scale: bool = False) -> None:
        super().__init__()
        self.x_size = x_size
        self.dynamic = dynamic
        self.shift_normalize = shift_normalize
        self.scale_normalize = scale_normalize
        self.log_scale = log_scale

        if dynamic:
            self.embed = nn.Sequential(
                Linear(1, dynamic_dim, norm_layer=nn.LayerNorm, act_layer=nn.ReLU),
                Linear(dynamic_dim, embed_dim, norm_layer=False, act_layer=False),
            )
            self.has_table = False
        else:
            self.embed = nn.Identity()
            self.position_embedding_table = nn.Parameter(torch.zeros(x_size, embed_dim))
            trunc_normal_(self.position_embedding_table, std=.02)
            self.has_table = True

    @property
    def no_decay_set(self):
        return {'position_embedding_table'}

    def generate_param_table(self, data=None) -> None:
        if not self.has_table and self.dynamic:
            if data is None:
                data = torch.arange(self.x_size).view(-1,1).float()
            data = data.to(self.embed[0].linear.weight.device)
            if self.shift_normalize:
                data = data - self.x_size * 0.5
            if self.scale_normalize:
                data = data / self.x_size
            if self.log_scale:
                data = self._to_log_scale(data)
            table = self.embed(data)
            self.position_embedding_table = nn.Parameter(table.detach())
            self.has_table = True

    def forward(self, x: Tensor) -> Tensor:
        if self.has_table:
            assert self.x_size > x.max()
            embedding = self.position_embedding_table[x.long()]
        else:
            data = x.view(-1,1).float()
            if self.shift_normalize:
                data = data - self.x_size * 0.5
            if self.scale_normalize:
                data = data / self.x_size
            if self.log_scale:
                data = self._to_log_scale(data)
            embedding = self.embed(data)
        return embedding

    def _to_log_scale(self, relative_coords: Tensor) -> Tensor:
        return relative_coords.sign() * (1 + relative_coords.abs()).log()

class PositionEmbedding2D(nn.Module):
    def __init__(self, x_size: int, y_size: int, embed_dim: int, dynamic: bool = False, dynamic_dim: Optional[int] = None,
                       shift_normalize: bool = False, scale_normalize: bool = False, log_scale: bool = False) -> None:
        super().__init__()
        self.x_size = x_size
        self.y_size = y_size
        self.dynamic = dynamic
        self.shift_normalize = shift_normalize
        self.scale_normalize = scale_normalize
        self.log_scale = log_scale

        if dynamic:
            self.embed = nn.Sequential(
                Linear(2, dynamic_dim, norm_layer=nn.LayerNorm, act_layer=nn.ReLU),
                Linear(dynamic_dim, embed_dim, norm_layer=False, act_layer=False),
            )
            self.has_table = False
        else:
            self.embed = nn.Identity()
            self.position_embedding_table = nn.Parameter(torch.zeros(x_size, y_size, embed_dim))
            trunc_normal_(self.position_embedding_table, std=.02)
            self.has_table = True

    @property
    def no_decay_set(self):
        return {'position_embedding_table'}

    def generate_param_table(self) -> None:
        if not self.has_table and self.dynamic:
            x, y = torch.meshgrid([ torch.arange(self.x_size), torch.arange(self.y_size) ])
            data = torch.stack([x, y], dim=-1).view(-1,2).float()
            data = data.to(self.embed[0].linear.weight.device)
            if self.shift_normalize:
                data = data - torch.tensor([self.x_size, self.y_size], device=data.device).float() * 0.5
            if self.scale_normalize:
                data = data / torch.tensor([self.x_size, self.y_size], device=data.device).float()
            if self.log_scale:
                data = self._to_log_scale(data)
            table = self.embed(data)
            table = table.view(self.x_size, self.y_size, -1)
            self.position_embedding_table = nn.Parameter(table.detach())
            self.has_table = True

    def forward(self, x: Tensor, y: Tensor) -> Tensor:
        if self.has_table:
            embedding = self.position_embedding_table[x, y]
        else:
            data = torch.stack([x, y], dim=-1)
            if self.shift_normalize:
                data = data - torch.tensor([self.x_size, self.y_size], device=data.device).float() * 0.5
            if self.scale_normalize:
                data = data / torch.tensor([self.x_size, self.y_size], device=data.device).float()
            if self.log_scale:
                data = self._to_log_scale(data)
            embedding = self.embed(data)
        return embedding

    def _to_log_scale(self, relative_coords: Tensor) -> Tensor:
        return relative_coords.sign() * (1 + relative_coords.abs()).log()


class PositionEmbedding(nn.Module):
    def __init__(self, table_size: Tuple[int], embed_dim: int, dynamic: bool = False, dynamic_dim: Optional[int] = None,
                       shift_normalize: bool = False, scale_normalize: bool = False, log_scale: bool = False) -> None:
        super().__init__()
        assert isinstance(table_size, (list, tuple))
        self.table_size = table_size
        self.dynamic = dynamic
        self.shift_normalize = shift_normalize
        self.scale_normalize = scale_normalize
        self.log_scale = log_scale

        if dynamic:
            #self.embed = DynamicEmbedding(len(table_size), dynamic_dim, embed_dim)
            self.embed = nn.Sequential(
                Linear(len(table_size), dynamic_dim, norm_layer=nn.LayerNorm, act_layer=nn.ReLU),
                Linear(dynamic_dim, embed_dim, norm_layer=False, act_layer=False),
            )
            self.has_table = False
        else:
            self.embed = nn.Identity()
            self.position_embedding_table = nn.Parameter(torch.zeros(*table_size, embed_dim))
            trunc_normal_(self.position_embedding_table, std=.02)
            self.has_table = True

    @property
    def no_decay_set(self):
        return {'position_embedding_table'}

    def generate_param_table(self) -> None:
        if not self.has_table and self.dynamic:
            data = torch.stack(torch.meshgrid([ torch.arange(s) for s in self.table_size ]), dim=-1).view(-1,len(table_size)).float()
            data = data.to(self.embed[0].linear.weight.device)
            if self.shift_normalize:
                data = data - torch.FloatTensor(self.table_size, device=data.device) * 0.5
            if self.scale_normalize:
                data = data / torch.FloatTensor(self.table_size, device=data.device)
            if self.log_scale:
                data = self._to_log_scale(data)
            table = self.embed(data)
            self.position_embedding_table = nn.Parameter(table.detach())
            self.has_table = True

    def forward(self, list_data: Tuple[Tensor]) -> Tensor:
        assert len(list_data) == len(self.table_size)

        if self.has_table:
            embedding = self.position_embedding_table[list_data]
        else:
            data = torch.stack(list_data, dim=-1)
            if self.shift_normalize:
                data = data - torch.tensor(self.table_size, device=data.device).float() * 0.5
            if self.scale_normalize:
                data = data / torch.tensor(self.table_size, device=data.device).float()
            if self.log_scale:
                data = self._to_log_scale(data)
            embedding = self.embed(data)
        return embedding

    def _to_log_scale(self, relative_coords: Tensor) -> Tensor:
        return relative_coords.sign() * (1 + relative_coords.abs()).log()

def _seq_to_2D(x: Tensor, x_meta: dict) -> Tensor:
    B, H, W = x_meta['shape']
    return x.transpose(1,2).contiguous().view(B, -1, H, W)

def _2D_to_seq(x: Tensor, x_meta: dict) -> Tensor:
    B, C, H, W = x.shape
    return x.view(B, C, H*W).transpose(1,2).contiguous()

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

