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

from ...init import kaiming_uniform_relu
from ...blocks import BlockBase
from ...layers import ConvBlock
from ...criterion.builder import build_loss

class SegHead(BlockBase):
    def __init__(self, inc, num_classes, num_extra_conv=1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, drop=0.1, cfg_loss=None):
        super().__init__()
        head = []
        head += [ ConvBlock(inc, inc, kernel_size=3, padding=1, norm_layer=norm_layer, act_layer=act_layer, drop=drop) for _ in range(num_extra_conv) ]
        head.append( ConvBlock(inc, num_classes, kernel_size=1, bias=True, norm_layer=None, act_layer=None) )
        self.head = nn.Sequential(*head)
        self.criterion = build_loss(cfg_loss)

    @torch.jit.ignore
    def init_weights(self):
        kaiming_uniform_relu(self.modules())

    def forward(self, features, image_metas):
        W = image_metas[0]['width']
        H = image_metas[0]['height']

        x = features[0] if isinstance(features, (list, tuple)) else features

        x = self.head(x)
        if x.shape[-2:] != (H, W):
            x = F.interpolate(x, size=(H, W), mode='bilinear')
        return x

    def inference(self, x, image_metas):
        x = self.forward(x, image_metas)
        return F.softmax(x, dim=1)

    def loss(self, pred, gt, image_metas):
        loss, log_vars = self.criterion(pred, gt)
        return loss, log_vars
        


