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

import copy
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from hmnet.utils.common import CDict as cdict

def conv_params_as_dict(conv: nn.Conv2d):
    return cdict(
        in_channels    = conv.in_channels,
        out_channels   = conv.out_channels,
        kernel_size    = conv.kernel_size,
        stride         = conv.stride,
        groups         = conv.groups,
        bias           = conv.bias is not None,
        dilation       = conv.dilation,
        padding        = conv.padding,
        padding_mode   = conv.padding_mode)

def bn_params_as_dict(bn: nn.BatchNorm2d):
    return cdict(
        num_features = bn.num_features,
        eps          = bn.eps,
        momentum     = bn.momentum,
        affine       = bn.affine,
        track_running_stats = bn.track_running_stats,
    )

def merge_bn(bn1: nn.BatchNorm2d, bn2: nn.BatchNorm2d) -> nn.BatchNorm2d:
    assert bn1.eps == bn2.eps
    assert bn1.momentum == bn2.momentum
    assert bn1.affine == bn2.affine
    assert bn1.track_running_stats == bn2.track_running_stats
    
    p1 = bn_params_as_dict(bn1)
    p2 = bn_params_as_dict(bn2)

    w1 = bn1.weight.data
    b1 = bn1.bias.data
    m1 = bn1.running_mean
    v1 = bn1.running_var

    w2 = bn2.weight.data
    b2 = bn2.bias.data
    m2 = bn2.running_mean
    v2 = bn2.running_var

    p1.num_features = p1.num_features + p2.num_features

    new_bn = nn.BatchNorm2d(**p1).to(bn1.weight.device)
    new_w = torch.cat([w1, w2])
    new_b = torch.cat([b1, b2])
    new_m = torch.cat([m1, m2])
    new_v = torch.cat([v1, v2])

    new_bn.weight.data.copy_(new_w)
    new_bn.bias.data.copy_(new_b)
    new_bn.running_mean.copy_(new_m)
    new_bn.running_var.copy_(new_v)

    return new_bn

def merge_conv2d(conv1: nn.Conv2d, conv2: nn.Conv2d, group: bool = False) -> nn.Conv2d:
    assert conv1.kernel_size == conv2.kernel_size
    assert conv1.stride == conv2.stride
    assert conv1.groups == conv2.groups
    assert conv1.in_channels == conv2.in_channels
    assert type(conv1.bias) == type(conv2.bias)
    assert conv1.dilation == conv2.dilation
    assert conv1.padding == conv2.padding
    assert conv1.padding_mode == conv2.padding_mode

    p1 = conv_params_as_dict(conv1)
    p2 = conv_params_as_dict(conv2)

    if group:
        p1.in_channels = p1.in_channels + p2.in_channels
        p1.out_channels = p1.out_channels + p2.out_channels
        p1.groups = p1.groups + p2.groups
    else:
        p1.out_channels = p1.out_channels + p2.out_channels

    new_conv = nn.Conv2d(**p1).to(conv1.weight.device)

    w1 = conv1.weight.data
    w2 = conv2.weight.data
    new_w = torch.cat([w1, w2], dim=0)
    new_conv.weight.data.copy_(new_w)

    if conv1.bias is not None:
        b1 = conv1.bias.data
        b2 = conv2.bias.data
        new_b = torch.cat([b1, b2])
        new_conv.bias.data.copy_(new_b)

    return new_conv

def merge_conv_block(block1: nn.Module, block2: nn.Module, group: bool):
    from hmnet.models.base.layers import ConvBlock
    assert isinstance(block1, ConvBlock)
    assert isinstance(block2, ConvBlock)
    assert type(block1.norm) == type(block2.norm)
    assert type(block1.act) == type(block2.act)
    assert block1.pre_act == block2.pre_act
    assert block1.pre_norm == block2.pre_norm

    new_block = copy.deepcopy(block1).to(block1.norm.weight.device)
    new_block.conv = merge_conv2d(block1.conv, block2.conv, group)

    if (block1.pre_act or block1.pre_norm) and not group:
        new_block.norm = block1.norm
    else:
        new_block.norm = merge_bn(block1.norm, block2.norm)

    return new_block

