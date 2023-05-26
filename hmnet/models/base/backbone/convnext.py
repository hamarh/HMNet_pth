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

# Copyright (c) Meta Platforms, Inc. and affiliates.

# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from ..blocks import BlockBase
from ..init import init_transformer

CFGS = {
    'tiny'   : dict(depths=[3, 3,  9, 3], dims=[ 96, 192,  384,  768]),
    'small'  : dict(depths=[3, 3, 27, 3], dims=[ 96, 192,  384,  768]),
    'base'   : dict(depths=[3, 3, 27, 3], dims=[128, 256,  512, 1024]),
    'large'  : dict(depths=[3, 3, 27, 3], dims=[192, 384,  768, 1536]),
    'xlarge' : dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048]),
}


class ConvNeXt(BlockBase):
    def __init__(self, inc=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=None):
        super().__init__()
        self.out_indices = out_indices

        L1, L2, L3, L4 = depths
        D1, D2, D3, D4 = dims
        dp_rates = [ r.item() for r in torch.linspace(0, drop_path_rate, sum(depths)) ]
        dp1, dp2, dp3, dp4 = np.split(dp_rates, np.cumsum(depths)[:-1])

        self.entry  = ConvNeXtEntry(inc, D1)
        self.stage1 = ConvNeXtStage(D1, D1, L1, dp_rates=dp1, downsample=False)
        self.stage2 = ConvNeXtStage(D1, D2, L2, dp_rates=dp2)
        self.stage3 = ConvNeXtStage(D2, D3, L3, dp_rates=dp3)
        self.stage4 = ConvNeXtStage(D3, D4, L4, dp_rates=dp4)
        self.init_weights()
        self.set_module_names()

    def init_weights(self):
        init_transformer(self.modules())
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.entry(x)
        x1 = self.stage1(x)
        x2 = self.stage2(x1)
        x3 = self.stage3(x2)
        x4 = self.stage4(x3)

        if self.out_indices is not None:
            features = [x1, x2, x3, x4]
            output = [ x for i, x in enumerate(features) if i in self.out_indices ]
            return output
        else:
            return x4

class ConvNeXtBlock(nn.Module):
    r""" ConvNeXt Block. There are two equivalent implementations:
    (1) DwConv -> LayerNorm (channels_first) -> 1x1 Conv -> GELU -> 1x1 Conv; all in (N, C, H, W)
    (2) DwConv -> Permute to (N, H, W, C); LayerNorm (channels_last) -> Linear -> GELU -> Linear; Permute back
    We use (2) as we find it slightly faster in PyTorch
    
    Args:
        dim (int): Number of input channels.
        drop_path (float): Stochastic depth rate. Default: 0.0
        layer_scale_init_value (float): Init value for Layer Scale. Default: 1e-6.
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim) # depthwise conv
        self.norm = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        input = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

    def init_weights(self, m):
        for m in self.moddules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                nn.init.constant_(m.bias, 0)

class ConvNeXtEntry(BlockBase):
    def __init__(self, inc, dim):
        super().__init__()
        self.conv = nn.Conv2d(inc, dim, kernel_size=4, stride=4)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_first")

class ConvNeXtStage(BlockBase):
    def __init__(self, inc, dim, depth, dp_rates, layer_scale_init_value=1e-6, downsample=True, downsample_kernel_size=2):
        super().__init__()
        assert downsample_kernel_size in (2, 3)
        if downsample and downsample_kernel_size == 2:
            self.downsample = nn.Sequential(
                LayerNorm(inc, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(inc, dim, kernel_size=2, stride=2),
            )
        elif downsample and downsample_kernel_size == 3:
            self.downsample = nn.Sequential(
                LayerNorm(inc, eps=1e-6, data_format="channels_first"),
                nn.Conv2d(inc, dim, kernel_size=3, stride=2, padding=1),
            )
        else:
            self.downsample = nn.Identity()

        layers = [ ConvNeXtBlock(dim=dim, drop_path=dp_rates[j], layer_scale_init_value=layer_scale_init_value) for j in range(depth) ]
        self.stage = nn.Sequential(*layers)

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


