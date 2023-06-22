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
from ..layers import ConvGRUCell, LayerNorm2d
from .convnext import ConvNeXtEntry, ConvNeXtStage

CFGS = {
    'tiny'   : dict(depths=[3, 3,  9, 3], dims=[ 96, 192,  384,  768]),
    'small'  : dict(depths=[3, 3, 27, 3], dims=[ 96, 192,  384,  768]),
    'base'   : dict(depths=[3, 3, 27, 3], dims=[128, 256,  512, 1024]),
    'large'  : dict(depths=[3, 3, 27, 3], dims=[192, 384,  768, 1536]),
    'xlarge' : dict(depths=[3, 3, 27, 3], dims=[256, 512, 1024, 2048]),
}


class ConvNeXtGRU(BlockBase):
    def __init__(self, inc=3, depths=[3, 3, 9, 3], dims=[96, 192, 384, 768],
                 drop_path_rate=0., layer_scale_init_value=1e-6, out_indices=None, is_norm_gru_out=True):
        super().__init__()
        self.out_indices = out_indices

        L1, L2, L3, L4 = depths
        D1, D2, D3, D4 = dims
        dp_rates = [ r.item() for r in torch.linspace(0, drop_path_rate, sum(depths)) ]
        dp1, dp2, dp3, dp4 = np.split(dp_rates, np.cumsum(depths)[:-1])

        gru_norm_out = LayerNorm2d if is_norm_gru_out else None

        self.entry  = ConvNeXtEntry(inc, D1)
        self.stage1 = ConvNeXtStage(D1, D1, L1, dp_rates=dp1, downsample=False)
        self.gru1  = ConvGRUCell(D1, D1, kernel_size=3, bias=False, norm_out=gru_norm_out)
        self.stage2 = ConvNeXtStage(D1, D2, L2, dp_rates=dp2)
        self.gru2  = ConvGRUCell(D2, D2, kernel_size=3, bias=False, norm_out=gru_norm_out)
        self.stage3 = ConvNeXtStage(D2, D3, L3, dp_rates=dp3)
        self.gru3  = ConvGRUCell(D3, D3, kernel_size=3, bias=False, norm_out=gru_norm_out)
        self.stage4 = ConvNeXtStage(D3, D4, L4, dp_rates=dp4)
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



