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
from ..layers import ConvBlock

from torch import Tensor
from typing import Tuple, List, Optional, Dict

from hmnet.utils.common import Timer
from functools import partial
Timer = partial(Timer, enabled=False)

class ResNet(BlockBase):
    def __init__(self, inc, ent_strided=True, ent_pool=True, deepseg=False, bottleneck=False,
                       num_layers=[3,4,6,3], stride=[1,2,2,2], dilation=[1,1,1,1],
                       out_indices=None, no_decay_norm_bias=False, down_act=False, skip_layers=None):
        super().__init__()
        self.no_decay_norm_bias = no_decay_norm_bias
        if skip_layers is not None:
            self.out_indices = [ int(name.replace('layer', ''))-1 for name in skip_layers ]
        else:
            self.out_indices = out_indices

        self.entry = ResEntry(inc, ent_strided=ent_strided, ent_pool=ent_pool, deepseg=deepseg)
        fdim = 128 if deepseg else 64

        self.layer1 = ResStage(1,  64, num_layers[0], bottleneck, stride[0], dilation[0], down_act=down_act, inc=fdim)
        self.layer2 = ResStage(2, 128, num_layers[1], bottleneck, stride[1], dilation[1], down_act=down_act)
        self.layer3 = ResStage(3, 256, num_layers[2], bottleneck, stride[2], dilation[2], down_act=down_act)
        self.layer4 = ResStage(4, 512, num_layers[3], bottleneck, stride[3], dilation[3], down_act=down_act)
        self.init_weights()
        self.set_module_names()

    def forward(self, x: Tensor) -> Tensor:
        with Timer('entry'):
            x = self.entry(x)
        with Timer('layer1'):
            x1 = self.layer1(x)
        with Timer('layer2'):
            x2 = self.layer2(x1)
        with Timer('layer3'):
            x3 = self.layer3(x2)
        with Timer('layer4'):
            x4 = self.layer4(x3)

        if self.out_indices is not None:
            features = [x1, x2, x3, x4]
            output = [ x for i, x in enumerate(features) if i in self.out_indices ]
            return output
        else:
            return x4

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

class ResStage(BlockBase):
    def __init__(self, stage_id, dim, num_layers, bottleneck, stride=1, dilation=1, inc=None, outc=None, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, down_act=False):
        super().__init__()

        shared_kargs = dict(dilation=dilation, norm_layer=norm_layer, act_layer=act_layer, down_act=down_act)
        s = stride
        N = num_layers
        layers = []

        if bottleneck is True:
            inc = inc or dim*2
            outc = outc or dim*4
            if N == 1:
                layers  = [ ( 'res%d_%d' % (stage_id, 1), ResBlock((  inc, dim,  outc), stride=s, **shared_kargs) ) ]
            else:
                layers  = [ ( 'res%d_%d' % (stage_id, 1), ResBlock((  inc, dim, dim*4), stride=s, **shared_kargs) ) ]
                layers += [ ( 'res%d_%d' % (stage_id, i), ResBlock((dim*4, dim, dim*4), stride=1, **shared_kargs) ) for i in range(2, N) ]
                layers += [ ( 'res%d_%d' % (stage_id, N), ResBlock((dim*4, dim,  outc), stride=1, **shared_kargs) ) ]
        else:
            inc = inc or int(dim / 2)
            outc = outc or dim
            if N == 1:
                layers  = [ ( 'res%d_%d' % (stage_id, 1), ResBlock((inc, outc), stride=s, **shared_kargs) ) ]
            else:
                layers  = [ ( 'res%d_%d' % (stage_id, 1), ResBlock((inc,  dim), stride=s, **shared_kargs) ) ]
                layers += [ ( 'res%d_%d' % (stage_id, i), ResBlock((dim,  dim), stride=1, **shared_kargs) ) for i in range(2, N) ]
                layers += [ ( 'res%d_%d' % (stage_id, N), ResBlock((dim, outc), stride=1, **shared_kargs) ) ]

        for name, module in layers:
            setattr(self, name, module)

    def load_from_resnet(self, state_dict):
        for name, module in self.named_children():
            if 'res' in name:
                elems = name.replace('res', '').split('_')
                i_stage = int(elems[0])
                i_layer = int(elems[1])-1
                key = 'layer%d.%d' % (i_stage, i_layer)
                module._load_from_resnet(state_dict, key)
            else:
                print('%s skip loading' % name)

class ResEntry(BlockBase):
    def __init__(self, inc, ent_strided=True, ent_pool=True, deepseg=True, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()
        self.ent_strided = ent_strided
        self.ent_pool = ent_pool
        self.deepseg = deepseg
        stride = 2 if ent_strided is True else 1

        if deepseg is not True:
            self.conv1 = ConvBlock(inc, 64, kernel_size=7, padding=3, stride=stride, norm_layer=norm_layer, act_layer=act_layer)
        else:
            self.conv1 = ConvBlock(inc, 64, kernel_size=3, padding=1, stride=stride, norm_layer=norm_layer, act_layer=act_layer)
            self.conv2 = ConvBlock( 64, 64, kernel_size=3, padding=1, norm_layer=norm_layer, act_layer=act_layer)
            self.conv3 = ConvBlock( 64,128, kernel_size=3, padding=1, norm_layer=norm_layer, act_layer=act_layer)

        if ent_pool is True:
            self.pool1 = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

    def load_from_resnet(self, state_dict, adjust_inc):
        self.conv1._load(state_dict, 'conv1', 'bn1', adjust_inc)
        if self.deepseg is True:
            self.conv2._load(state_dict, 'conv2', 'bn2')
            self.conv3._load(state_dict, 'conv3', 'bn3')

class ResBlock(BlockBase):
    def __init__(self, channels, stride=1, dilation=1, padding=-1, act=True, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, down_act=False):
        super().__init__()

        self.residual = ResidualPath(channels, stride, dilation, padding, norm_layer=norm_layer, act_layer=act_layer)
        self.downsample = None
        inc, outc = channels[0], channels[-1]
        if inc != outc or stride > 1:
            act = act_layer if down_act else None
            self.downsample = ConvBlock( inc, outc, kernel_size=1, padding=0, stride=stride, bias=False, norm_layer=norm_layer, act_layer=act)
        self.act = act_layer() if act else nn.Identity()

    def _center_crop(self, layer, tar_h, tar_w):
        src_n, src_c, src_h, src_w = layer.size()
        st_y = (src_h - tar_h) // 2
        st_x = (src_w - tar_w) // 2
        return layer[:, :, st_y:(st_y + tar_h), st_x:(st_x + tar_w)]

    def forward(self, x):
        # residual path
        res = self.residual(x)

        # identity path
        id = x
        if self.downsample is not None:
            id = self.downsample(x)

        # crop identity feature (might be needed when padding are set manually)
        if res.size() != id.size():
            id = self._center_crop(id, res.size(2), res.size(3))

        # add residual, then activate
        res += id

        out = self.act(res)

        return out

    def _load_from_resnet(self, state_dict, module_name):
        self.residual.conv1._load(state_dict, module_name+'.conv1', module_name+'.bn1')
        self.residual.conv2._load(state_dict, module_name+'.conv2', module_name+'.bn2')
        if 'conv3' in dir(self.residual):
            self.residual.conv3._load(state_dict, module_name+'.conv3', module_name+'.bn3')
        if self.downsample is not None:
            self.downsample._load(state_dict, module_name+'.downsample.0', module_name+'.downsample.1')

    def _load_from_regular(self, state_dict, module_name):
        self.residual.conv1._load(state_dict, module_name+'.residual.conv1.conv', module_name+'.residual.conv1.bn')
        self.residual.conv2._load(state_dict, module_name+'.residual.conv2.conv', module_name+'.residual.conv2.bn')
        if 'conv3' in dir(self.residual):
            self.residual.conv3._load(state_dict, module_name+'.residual.conv3.conv', module_name+'.residual.conv3.bn')
        if self.downsample is not None:
            self.downsample._load(state_dict, module_name+'.downsample.conv', module_name+'.downsample.bn')


class ResidualPath(BlockBase):
    def __init__(self, channels, stride, dilation=1, padding=-1, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU):
        super().__init__()

        s = stride
        d = dilation
        p = dilation if padding == -1 else padding
        if len(channels) == 2:
            inc, outc = channels
            self.conv1 = ConvBlock(  inc, outc, kernel_size=3, padding=p, stride=s, dilation=d, norm_layer=norm_layer, act_layer=None)
            self.relu  = act_layer()
            self.conv2 = ConvBlock( outc, outc, kernel_size=3, padding=p, stride=1, dilation=d, norm_layer=norm_layer, act_layer=None)
        elif len(channels) == 3:
            inc, midc, outc = channels
            self.conv1 = ConvBlock(  inc, midc, kernel_size=1, padding=0, stride=1, dilation=1, norm_layer=norm_layer, act_layer=None)
            self.relu1 = act_layer()
            self.conv2 = ConvBlock( midc, midc, kernel_size=3, padding=p, stride=s, dilation=d, norm_layer=norm_layer, act_layer=None)
            self.relu2 = act_layer()
            self.conv3 = ConvBlock( midc, outc, kernel_size=1, padding=0, stride=1, dilation=1, norm_layer=norm_layer, act_layer=None)


