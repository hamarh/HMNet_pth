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
        


