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
from ..blocks import BlockBase
from ..layers import ConvBlock, ConvGRUCell, LayerNorm2d
from .resnet import ResEntry, ResStage

from torch import Tensor
from typing import Tuple, List, Optional, Dict

from hmnet.utils.common import Timer


class ResNetGRU(BlockBase):
    def __init__(self, inc, ent_strided=True, ent_pool=True, deepseg=False, bottleneck=False,
                       num_layers=[3,4,6,3], stride=[1,2,2,2], dilation=[1,1,1,1],
                       out_indices=None, no_decay_norm_bias=False, down_act=False, skip_layers=None, is_norm_gru_out=True):
        super().__init__()
        self.no_decay_norm_bias = no_decay_norm_bias
        if skip_layers is not None:
            self.out_indices = [ int(name.replace('layer', ''))-1 for name in skip_layers ]
        else:
            self.out_indices = out_indices

        norm_act = dict(
            norm_layer = nn.GroupNorm,
            act_layer  = nn.SiLU,
        )

        self.entry = ResEntry(inc, ent_strided=ent_strided, ent_pool=ent_pool, deepseg=deepseg, **norm_act)
        fdim = 128 if deepseg else 64

        gru_norm_out = nn.GroupNorm if is_norm_gru_out else None

        r = 4 if bottleneck else 1

        self.layer1 = ResStage(1,  64, num_layers[0], bottleneck, stride[0], dilation[0], down_act=down_act, **norm_act, inc=fdim)
        self.gru1 = ConvGRUCell( 64*r,  64*r, kernel_size=3, bias=False, norm_out=gru_norm_out)
        self.layer2 = ResStage(2, 128, num_layers[1], bottleneck, stride[1], dilation[1], down_act=down_act, **norm_act)
        self.gru2 = ConvGRUCell(128*r, 128*r, kernel_size=3, bias=False, norm_out=gru_norm_out)
        self.layer3 = ResStage(3, 256, num_layers[2], bottleneck, stride[2], dilation[2], down_act=down_act, **norm_act)
        self.gru3 = ConvGRUCell(256*r, 256*r, kernel_size=3, bias=False, norm_out=gru_norm_out)
        self.layer4 = ResStage(4, 512, num_layers[3], bottleneck, stride[3], dilation[3], down_act=down_act, **norm_act)
        self.gru4 = ConvGRUCell(512*r, 512*r, kernel_size=3, bias=False, norm_out=gru_norm_out)

        self.init_weights()
        self.set_module_names()

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

    def forward(self, x: Tensor, states: list) -> Tensor:
        if states is None:
            s1, s2, s3, s4 = None, None, None, None
        else:
            s1, s2, s3, s4 = states

        x = self.entry(x)
        x1 = self.layer1(x)
        s1 = self.gru1(x1, s1)
        x2 = self.layer2(x1)
        s2 = self.gru2(x2, s2)
        x3 = self.layer3(x2)
        s3 = self.gru3(x3, s3)
        x4 = self.layer4(x3)
        s4 = self.gru4(x4, s4)

        states = (s1, s2, s3, s4)

        if self.out_indices is not None:
            features = [s1, s2, s3, s4]
            output = [ x for i, x in enumerate(features) if i in self.out_indices ]
            return output, states
        else:
            return s4, states

    @torch.jit.ignore
    def init_weights(self):
        kaiming_uniform_relu(self.modules())

    @torch.jit.ignore
    def load_from_imagenet(self, model_path, adjust_inc=False):
        resnet_dict = torch.load(model_path)
        self.entry.load_from_resnet(resnet_dict, adjust_inc)
        self.layer1.load_from_resnet(resnet_dict)
        self.layer2.load_from_resnet(resnet_dict)
        self.layer3.load_from_resnet(resnet_dict)
        self.layer4.load_from_resnet(resnet_dict)

    @torch.jit.ignore
    def no_weight_decay(self):
        names_no_decay = set()
        no_decay_set = set()
        for name, param in self.named_parameters():
            if name.split('.')[-1] in no_decay_set:
                names_no_decay.update({name})
            if self.no_decay_norm_bias and (len(param.shape) == 1 or name.endswith(".bias")):
                names_no_decay.update({name})
        return names_no_decay


    def prepair_for_test(self, batch_size, image_size=None):
        pass

    def termination(self):
        pass

