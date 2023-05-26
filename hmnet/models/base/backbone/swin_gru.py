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

import numpy as np
from copy import deepcopy
from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from torch_scatter import scatter

from .vit import PatchEmbed, Stage, SeqData
from ..init import init_transformer
from ..blocks import BlockBase
from ..layers import ConvBlock, Linear, ConvGRUCell, LayerNorm2d

from torch import Tensor
from typing import Tuple, List, Optional, Dict
Meta = Dict[str,Tuple[int,int,int]]



class SwinTransformerGRU(BlockBase):
    def __init__(self, inc: int = 3, out_indices: Optional[List[int]] = None, seq_out: bool = False, embed_force_stride: Optional[int] = None, patch_size: Tuple[int] = [4], drop_path_rate=0.2, downsample: str = 'Swin',
                       embed_dim=96, num_layers: List[int] = [2,2,6,2], num_heads: List[int] = [3,6,12,24], stride=[2,2,2,1], cfg_shared: Optional[dict] = None,
                       norm_layer: type = nn.LayerNorm, is_norm_gru_out: bool = True) -> None:
        super().__init__()
        self.seq_out = seq_out
        self.out_indices = out_indices
        self.is_norm_gru_out = is_norm_gru_out

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

        gru_norm_out = nn.GroupNorm if is_norm_gru_out else None
        self.gru1 = ConvGRUCell(embed_dim*1, embed_dim*1, kernel_size=3, bias=False, norm_out=gru_norm_out, norm_in=nn.GroupNorm)
        self.gru2 = ConvGRUCell(embed_dim*2, embed_dim*2, kernel_size=3, bias=False, norm_out=gru_norm_out, norm_in=nn.GroupNorm)
        self.gru3 = ConvGRUCell(embed_dim*4, embed_dim*4, kernel_size=3, bias=False, norm_out=gru_norm_out, norm_in=nn.GroupNorm)
        self.gru4 = ConvGRUCell(embed_dim*8, embed_dim*8, kernel_size=3, bias=False, norm_out=gru_norm_out, norm_in=nn.GroupNorm)

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

    def forward_sequence(self, list_x, states):
        list_feats = []
        for x in list_x:
            features, states = self.forward(x, states)
            list_feats.append(features)

        list_feats = list(zip(*list_feats))

        return list_feats, states

    def forward(self, x: Tensor, states: list) -> Tensor:
        if states is None:
            z1, z2, z3, z4 = None, None, None, None
        else:
            z1, z2, z3, z4 = states

        seq_x = self.patch_embed(x)
        seq_x, z1 = self.forward_stage(seq_x, z1, self.layer1, self.gru1)
        seq_x, z2 = self.forward_stage(seq_x, z2, self.layer2, self.gru2)
        seq_x, z3 = self.forward_stage(seq_x, z3, self.layer3, self.gru3)
        seq_x, z4 = self.forward_stage(seq_x, z4, self.layer4, self.gru4)

        states = (z1, z2, z3, z4)
        features = (z1, z2, z3, z4)

        if self.out_indices is not None:
            output = [ x for i, x in enumerate(features) if i in self.out_indices ]
            return output, states
        else:
            return z4, states

    def forward_stage(self, seq_x, z, stage, gru):
        for block in stage.blocks:
            seq_x = block(seq_x)

        x = seq_x.to_2D()
        z = gru(x, z)
        seq_x = SeqData.from_2D(z)

        seq_x = stage.downsample(seq_x)

        return seq_x, z

    def prepair_for_test(self, batch_size, image_size=None):
        pass

    def termination(self):
        pass

