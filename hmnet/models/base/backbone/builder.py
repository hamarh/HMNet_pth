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


from .hmnet import HMNet
from .hmnet1 import HMNet1

from .resnet import ResNet
from .convnext import ConvNeXt
from .cspdarknet import CSPDarknet
from .vit import VisionTransformer

from .resnet_gru import ResNetGRU
from .convnext_gru import ConvNeXtGRU
from .cspdarknet_gru import CSPDarkNetGRU
from .swin_gru import SwinTransformerGRU


BACKBONES = {
    'HMNet': HMNet,
    'HMNet1': HMNet1,
    'ConvNeXt': ConvNeXt,
    'ConvNeXtGRU': ConvNeXtGRU,
    'ResNet': ResNet,
    'ResNetGRU': ResNetGRU,
    'CSPDarknet': CSPDarknet,
    'CSPDarkNetGRU': CSPDarkNetGRU,
    'SwinTransformer': VisionTransformer,
    'SwinTransformerGRU': SwinTransformerGRU,
}

def build_backbone(param):
    param = param.copy()
    name = param.pop('type')
    cls = BACKBONES[name]
    return cls(**param)
