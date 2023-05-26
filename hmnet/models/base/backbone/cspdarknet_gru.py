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

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath

from ..blocks import BlockBase
from ..init import init_transformer
from ..layers import ConvGRUCell, LayerNorm2d
from .cspdarknet import CSPDarknet


DIMS = {
    1.0 : [128, 256, 512, 1024],
    0.75: [ 96, 192, 384,  768],
    0.5 : [ 64, 128, 256,  512],
}

class CSPDarkNetGRU(BlockBase):
    def __init__(self, backbone, out_indices=(0,1,2,3), is_norm_gru_out=True):
        super().__init__()
        self.out_indices = out_indices

        darknet = CSPDarknet(**backbone)
        gru_norm_out = LayerNorm2d if is_norm_gru_out else None
        D1, D2, D3, D4 = DIMS[backbone['wid_mul']]

        self.entry  = darknet.stem
        self.stage1 = darknet.dark2
        self.gru1  = ConvGRUCell(D1, D1, kernel_size=3, bias=False, norm_out=gru_norm_out)
        self.stage2 = darknet.dark3
        self.gru2  = ConvGRUCell(D2, D2, kernel_size=3, bias=False, norm_out=gru_norm_out)
        self.stage3 = darknet.dark4
        self.gru3  = ConvGRUCell(D3, D3, kernel_size=3, bias=False, norm_out=gru_norm_out)
        self.stage4 = darknet.dark5
        self.gru4  = ConvGRUCell(D4, D4, kernel_size=3, bias=False, norm_out=gru_norm_out)
        self.init_weights()
        self.set_module_names()

    def init_weights(self):
        init_transformer(self.modules())
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward_sequence(self, list_x, states):
        if states is None:
            s1, s2, s3, s4 = None, None, None, None
        else:
            s1, s2, s3, s4 = states

        list_feats = []
        for x in list_x:
            features, states = self.forward(x, states)
            list_feats.append(features)

        list_feats = list(zip(*list_feats))

        return list_feats, states

    def forward(self, x, states):
        if states is None:
            s1, s2, s3, s4 = None, None, None, None
        else:
            s1, s2, s3, s4 = states

        x = self.entry(x)
        x1 = self.stage1(x)
        s1 = self.gru1(x1, s1)
        x2 = self.stage2(x1)
        s2 = self.gru2(x2, s2)
        x3 = self.stage3(x2)
        s3 = self.gru3(x3, s3)
        x4 = self.stage4(x3)
        s4 = self.gru4(x4, s4)

        states = (s1, s2, s3, s4)

        if self.out_indices is not None:
            features = [s1, s2, s3, s4]
            output = [ x for i, x in enumerate(features) if i in self.out_indices ]
            return output, states
        else:
            return s4, states

    def prepair_for_test(self, batch_size, image_size=None):
        pass

    def termination(self):
        pass



