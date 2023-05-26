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
