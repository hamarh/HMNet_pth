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

from ..init import kaiming_uniform_relu
from .ppm import PPM
from ..blocks import BlockBase
from ..layers import ConvBlock
from ..neck.pyramid import Pyramid
from .task_head.builder import build_task_head

class UPerHead(BlockBase):
    def __init__(self, in_channels, dim, input_proj=True, ppm_sizes=(1, 2, 3, 6), drop=0.1,
                       norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, ppm_norm_layer=nn.BatchNorm2d, cfg_pyramid_fuse=None, cfg_task_head=None):
        super().__init__()
        # PPM Module
        self.ppm = PPM(in_channels[-1], dim, ppm_sizes, ppm_norm_layer)
        ppm_dim = in_channels[-1] + len(ppm_sizes) * dim
        self.ppm_out_conv = ConvBlock(ppm_dim, dim, kernel_size=3, padding=1, norm_layer=norm_layer, act_layer=act_layer)
        in_channels[-1] = dim

        # FPN Module
        num_levels = len(in_channels)
        self.fpn = Pyramid(in_channels, dim, out_indices=list(range(num_levels)), input_proj=input_proj, cfg_pyramid_fuse=cfg_pyramid_fuse, act_layer=act_layer)
        self.fpn_out_conv = ConvBlock(dim*num_levels, dim, kernel_size=3, padding=1, norm_layer=norm_layer, act_layer=act_layer)

        # seg head
        self.dropout = nn.Dropout2d(drop) if drop > 0 else nn.Identity()
        self.head = build_task_head(cfg_task_head)
        self.init_weights()

    @torch.jit.ignore
    def init_weights(self):
        kaiming_uniform_relu(self.modules())

    def forward(self, inputs, image_meta):
        output = self._decode(inputs)
        output = self.head(output, image_meta)
        return output

    def inference(self, inputs, image_meta):
        output = self._decode(inputs)
        output = self.head.inference(output, image_meta)
        return output

    def loss(self, *args, **kargs):
        return self.head.loss(*args, **kargs)

    def _decode(self, inputs):
        # PPM on top level feature
        ppm_out = self.ppm(inputs[-1])
        ppm_out = self.ppm_out_conv(ppm_out)
        inputs[-1] = ppm_out

        # FPN
        fpn_outs = self.fpn(inputs)

        target_size = fpn_outs[0].shape[-2:]
        fpn_outs = [ self._resize(feat, target_size) for feat in fpn_outs ]
        fpn_outs = torch.cat(fpn_outs, dim=1)
        fpn_outs = self.fpn_out_conv(fpn_outs)

        output = self.dropout(fpn_outs)
        return output

    def _resize(self, data, size):
        H, W = size
        h, w = data.shape[-2:]
        if h == H and w == W:
            return data
        else:
            return F.interpolate(data, size=(H,W), mode='bilinear')



