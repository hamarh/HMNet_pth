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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..blocks import BlockBase
from ..layers import ConvBlock, UpConvBlock
from ..backbone.resnet import ResStage
from .task_head.builder import build_task_head
from ..init import xavier_uniform_relu

from torch import Tensor
from typing import Tuple, List, Optional, Dict

class ResUHead(BlockBase):
    def __init__(self, bottleneck=False, in_channels=[64,128,256,512], dims=[64,128,256,512], num_layers=[3,4,6,3],
                       dilation=[1,1,1,1], fuse_layers=['cat','cat','cat'], num_heads=[None,None,None,None],
                       norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, cfg_task_head=None):
        super().__init__()
        cfg_norm_act = {'norm_layer': norm_layer, 'act_layer': act_layer}
        ex = 4 if bottleneck else 1

        num_stages = len(in_channels)

        self.dec_layers = nn.ModuleList()
        self.fuse_layers = nn.ModuleList()

        outc = dims[0]
        for i in range(num_stages-1):
            inc = dims[i]*ex
            dim = dims[i]
            skc = in_channels[i]

            fuse = self._make_fuse(fuse_layers[i], inc, skc, inc, num_heads=num_heads[i], **cfg_norm_act)
            dec  = ResStage(i+1, dim, num_layers[i], bottleneck, 1, dilation[i], inc=inc, outc=outc, **cfg_norm_act)

            self.dec_layers.append(dec)
            self.fuse_layers.append(fuse)
            outc = inc

        entry = ResStage(num_stages, dims[-1], num_layers[-1], bottleneck, 1, dilation[-1], inc=in_channels[-1], outc=outc, **cfg_norm_act)
        self.dec_layers.append(entry)

        #self.head = SegHead(dims[0], num_classes, drop=0.1, **cfg_norm_act)
        cfg_task_head.update(cfg_norm_act)
        self.head = build_task_head(cfg_task_head)
        self.init_weights()

    def _make_fuse(self, fuse: str, inc: int, skc: int, outc: int, num_heads: int, norm_layer: nn.Module, act_layer: nn.Module) -> nn.Module:
        if fuse in ('cat', 'add', 'max'):
            return FuseLayer(inc, skc, outc, fuse, norm_layer, act_layer)
        elif fuse == 'cross':
            return FuseCross(inc, skc, outc, num_heads, norm_layer, act_layer)
        elif fuse == 'none':
            return FuseDummy()

    @torch.jit.ignore
    def init_weights(self):
        xavier_uniform_relu(self.modules())

    def forward(self, inputs: List[Tensor], image_meta: List[dict]) -> Tensor:
        x = self._decode(inputs)
        out = self.head(x, image_meta)
        return out

    def inference(self, inputs: List[Tensor], image_meta: List[dict]) -> Tensor:
        x = self._decode(inputs)
        out = self.head.inference(x, image_meta)
        return out

    def loss(self, *args, **kargs):
        return self.head.loss(*args, **kargs)

    def _decode(self, inputs: List[Tensor]) -> Tensor:
        assert len(inputs) == len(self.dec_layers)

        x = self.dec_layers[-1](inputs[-1])
        for i in reversed(range(len(inputs)-1)):
            x = self.fuse_layers[i](x, inputs[i])
            x = self.dec_layers[i](x)
        return x

class FuseLayer(BlockBase):
    def __init__(self, inc: int, skc: int, outc: int, fuse_method: str, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU) -> None:
        super().__init__()
        assert fuse_method in ('cat', 'add', 'max')
        cfg_norm_act = {'norm_layer': norm_layer, 'act_layer': act_layer}
        self.fuse_method = fuse_method

        self.in_conv1 = UpConvBlock(inc, outc, kernel_size=1, padding=0, **cfg_norm_act)
        self.in_conv2 = ConvBlock(skc, outc, kernel_size=3, padding=1, **cfg_norm_act)

        fuse_dim = 2 * outc if fuse_method == 'cat' else outc
        self.out_conv = ConvBlock(fuse_dim, outc, kernel_size=3, padding=1, **cfg_norm_act)

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        x = self.in_conv1(x, s.shape[-2:])
        s = self.in_conv2(s)
        x = self.fuse_feature(x, s)
        x = self.out_conv(x)

        return x

    def fuse_feature(self, x, s):
        if self.fuse_method == 'add':
            return x + s
        elif self.fuse_method == 'max':
            return torch.amax(torch.stack([x,s]), dim=0)
        elif self.fuse_method == 'cat':
            return torch.cat([s, x], dim=1)

class FuseCross(BlockBase):
    def __init__(self, inc: int, skc: int, outc: int, num_heads: int, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU) -> None:
        super().__init__()
        cfg_norm_act = {'norm_layer': norm_layer, 'act_layer': act_layer}
        self.in_conv1 = UpConvBlock(inc, outc, kernel_size=1, padding=0, **cfg_norm_act)
        self.in_conv2 = ConvBlock(skc, outc, kernel_size=3, padding=1, **cfg_norm_act)
        self.cross_attn_block = CrossAttentionBlock(outc, outc, window_size=(7,7), kv_window_size=(7,7), grouping='intra-window', num_heads=num_heads, pos_dynamic=False, pos_log_scale=False)

    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        x = self.in_conv1(x, s.shape)
        s = self.in_conv2(s)
        seq_x = SeqData.from_2D(x)
        seq_s = SeqData.from_2D(s)
        seq_x = self.cross_attn_block(seq_x, seq_s)

        return seq_x.to_2D()

def FuseDummy(BlockBase):
    def __init__(self) -> None:
        super().__init__()
    def forward(self, x: Tensor, s: Tensor) -> Tensor:
        return x




