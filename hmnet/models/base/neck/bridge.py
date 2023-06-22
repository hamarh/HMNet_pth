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

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..init import init_transformer
from ..blocks import BlockBase
from ..layers import ConvBlock, UpConvBlock

from torch import Tensor
from typing import Tuple, List, Optional, Dict


# cfg_sample = [
#     dict(type='ConvBridge', inc=512, dim=512, num_convs=4, groups=1, dilation=1, fuse_input=True, fuse_method='cat', norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU),
#     dict(type='ConvBridge', inc=512, dim=512, num_convs=4, groups=1, dilation=1, fuse_input=True, fuse_method='cat', norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU),
# ]

class Bridge(BlockBase):
    def __init__(self, list_cfgs):
        super().__init__()
        self.layers = nn.ModuleList()
        for cfg in list_cfgs:
            self.layers.append(self._make_layer(cfg))

    def _make_layer(self, cfg):
        type = cfg.pop('type')
        if type == 'ConvBridge':
            return ConvBridge(**cfg)
        elif type == 'Identity':
            return nn.Identity()

    def init_weights(self):
        init_transformer(self.modules())

    def forward(self, inputs):
        if not isinstance(inputs, (list, tuple)):
            inputs = [inputs]
            list_out = False
        else:
            list_out = True

        outputs = []
        for x, layer in zip(inputs, self.layers):
            x = layer(x)
            outputs.append(x)

        if list_out is False:
            return outputs[0]
        else:
            return outputs

class ConvBridge(BlockBase):
    def __init__(self, inc, dim, outc=None, num_convs=4, dilation=1, groups=1, fuse_input=False, fuse_method='add', norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()
        cfg_norm_act = {'norm_layer': norm_layer, 'act_layer': act_layer}
        self.fuse_input = fuse_input

        outc = outc or inc
        bridge = []
        for i in range(num_convs):
            idim = inc if i == 0 else dim
            odim = outc if i == (num_convs - 1) else dim
            conv = ConvBlock(idim, odim, kernel_size=3, padding=dilation, stride=1, dilation=dilation, groups=groups, **cfg_norm_act)
            bridge.append(conv)
        self.bridge = nn.Sequential(*bridge)

        if fuse_input:
            self.fuse = FuseLayer(outc, inc, outc, fuse_method, **cfg_norm_act)

    def forward(self, x):
        z = self.bridge(x)
        if self.fuse_input:
            z = self.fuse(z, x)
        return z

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
        x = torch.cat([x, s], dim=1)
        x = self.out_conv(x)

        return x

    def fuse_feature(self, x, s):
        if self.fuse_method == 'add':
            return x + s
        elif self.fuse_method == 'max':
            return torch.amax(torch.stack([x,s]), dim=0)
        elif self.fuse_method == 'cat':
            return torch.cat([s, x], dim=1)



