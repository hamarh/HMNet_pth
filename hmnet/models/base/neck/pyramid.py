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

from ..init import kaiming_uniform_relu
from ..blocks import BlockBase
from ..layers import conv_bn_act, MV2Block
from ..backbone.resnet import ResBlock
from ..backbone.vit import TransformerBlock

class Pyramid(BlockBase):
    def __init__(self, in_channels, dim, out_indices, input_start_index=0, num_inputs=None, attatch_extra='on_input', input_proj=True,
                 cfg_pyramid_fuse=[ dict(direction='top-down', pre_trans=False, fuse_method='cat', post_trans=True, post_convs=True) ], act_layer=nn.ReLU):
        super().__init__()
        assert attatch_extra in ('on_input', 'on_output', 'before_fpn')
        self.out_indices = out_indices
        self.attatch_extra = attatch_extra
        self.input_start_index = input_start_index
        self.num_inputs = num_inputs or (len(in_channels) - input_start_index)

        in_channels = in_channels[self.input_start_index:self.input_start_index + self.num_inputs]

        if input_proj:
            self.input_convs  = nn.ModuleList([ conv_bn_act(inc, dim, kernel_size=1, padding=0, act_layer=act_layer) for inc in in_channels ])
        else:
            self.input_convs  = nn.ModuleList([ nn.Identity() for inc in in_channels ])

        num_levels = len(out_indices) if attatch_extra == 'before_fpn' else len(in_channels)
        self.pyramid_fuse = nn.ModuleList([ PyramidFuse(dim=dim, num_levels=num_levels, act_layer=act_layer, **cfg) for cfg in cfg_pyramid_fuse ])

        self.extra_convs  = nn.ModuleList()
        fdim = in_channels[-1] if attatch_extra == 'on_input' else dim
        num_extra_conv = max(out_indices) - (self.num_inputs - 1)
        for _ in range(num_extra_conv):
            conv_layer = conv_bn_act(fdim, dim, kernel_size=3, padding=1, stride=2, act_layer=act_layer)
            self.extra_convs.append(conv_layer)
            fdim = dim
        self.init_weights()

    @torch.jit.ignore
    def init_weights(self):
        kaiming_uniform_relu(self.modules())

    def forward(self, inputs):
        inputs = inputs[self.input_start_index:self.input_start_index + self.num_inputs]
        assert len(inputs) == len(self.input_convs)

        features = [ conv(x) for conv, x in zip(self.input_convs, inputs) ]

        if self.attatch_extra == 'before_fpn':
            x = features[-1]
            for ex_conv in self.extra_convs:
                x = ex_conv(x)
                features.append(x)

        for pfuse in self.pyramid_fuse:
            features = pfuse(features)

        if self.attatch_extra == 'on_input':
            x = inputs[-1]
            for ex_conv in self.extra_convs:
                x = ex_conv(x)
                features.append(x)

        elif self.attatch_extra == 'on_output':
            x = features[-1]
            for ex_conv in self.extra_convs:
                x = ex_conv(x)
                features.append(x)

        return [ features[i] for i in self.out_indices ]

class PyramidFuse(BlockBase):
    def __init__(self, direction, dim, num_levels, pre_trans=False, fuse_method='cat', post_trans=True, post_convs=True, resampling='bilinear', act_layer=nn.ReLU):
        super().__init__()
        assert direction in ('bottom-up', 'top-down')
        self.direction = direction
        self.post_convs = post_convs

        self.fuse = nn.ModuleList([ FuseLayer(dim, dim, pre_trans=pre_trans, fuse_method=fuse_method, post_trans=post_trans, resampling=resampling, act_layer=act_layer) for _ in range(num_levels-1) ])
        if post_convs:
            self.convs = nn.ModuleList([ conv_bn_act(dim, dim, kernel_size=3, padding=1, act_layer=act_layer) for _ in range(num_levels) ])

    def forward(self, features):
        if self.direction == 'bottom-up':
            features = self.bottom_up_fuse(features)
        elif self.direction == 'top-down':
            features = self.top_down_fuse(features)

        if self.post_convs:
            features = [ conv(f) for conv, f in zip(self.convs, features) ]

        return features

    def bottom_up_fuse(self, features):
        for i in range(len(features)-1):
            fuse = self.fuse[i]
            features[i+1] = fuse(features[i+1], features[i])
        return features

    def top_down_fuse(self, features):
        for i in reversed(range(len(features)-1)):
            fuse = self.fuse[i]
            features[i] = fuse(features[i], features[i+1])
        return features

class FuseLayer(BlockBase):
    def __init__(self, inc, skc, outc=None, pre_trans=None, fuse_method='cat', post_trans=None, resampling='bilinear', act_layer=nn.ReLU):
        super(FuseLayer, self).__init__()
        assert fuse_method in ['cat', 'add', 'max']

        if outc is None:
            outc = inc

        self.pre_trans = pre_trans
        self.fuse_method = fuse_method
        self.post_trans = post_trans
        self.resampling = resampling

        dim = skc
        if self.pre_trans is not None:
            self.skip_conv = self._make_trans_module(self.pre_trans, dim, inc, act_layer)
            dim = inc

        if self.fuse_method == 'cat':
            dim = dim + inc

        if self.post_trans is not None:
            self.out_conv  = self._make_trans_module(self.post_trans, dim, outc, act_layer)

    def _make_trans_module(self, trans_type, inc, outc, act_layer):
        if isinstance(trans_type, bool):
            if trans_type:
                return conv_bn_act(inc, outc, kernel_size=3, padding=1, act_layer=act_layer)
            else:
                return nn.Identity()

        if trans_type == 'conv':
            return conv_bn_act(inc, outc, kernel_size=3, padding=1, act_layer=act_layer)
        elif trans_type == 'strided-conv':
            return conv_bn_act(inc, outc, kernel_size=3, padding=1, act_layer=act_layer, stride=2)
        elif trans_type == 'res':
            return ResBlock((inc, outc), stride=2, act_layer=act_layer)
        elif trans_type == 'mv2':
            return MV2Block(inc, outc, stride=2, act_layer=act_layer)
        elif trans_type == 'swin':
            assert inc == outc
            return TransformerBlock(outc, num_heads=8, window_size=(7,7), grouping='intra-window')
        elif trans_type == 'mvit':
            assert inc == outc
            return TransformerBlock(outc, num_heads=8, window_size=(7,7), grouping='inter-window', mvit=True)
        else:
            raise RuntimeError

    def forward(self, x, xs):
        if self.pre_trans is True:
            xs = self.skip_conv(xs)

        x = self.fuse_feature(x, xs)

        if self.post_trans is True:
            x  = self.out_conv(x)

        return x

    def fuse_feature(self, x, xs):
        xs = self._resize(xs, x.shape[-2:], self.resampling)

        if self.fuse_method == 'add':
            x = x + xs
        elif self.fuse_method == 'max':
            x  = torch.max(torch.stack([x,xs]), dim=0)[0]
        elif self.fuse_method == 'cat':
            x  = torch.cat([xs, x], dim=1)
        else:
            raise RuntimeError
        return x

    def _resize(self, data, size, resampling='bilinear'):
        H, W = size
        h, w = data.shape[-2:]
        if h == H and w == W:
            return data
        else:
            return F.interpolate(data, size=(H,W), mode=resampling)




