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

from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F

from .blocks import BlockBase

from torch import Tensor
from typing import Tuple, List, Optional

def bn_conv_relu(inc, outc, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return ConvBlock(inc, outc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                     bias=False, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU, pre_norm=True)

def conv_bn_relu(inc, outc, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return ConvBlock(inc, outc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                     bias=False, norm_layer=nn.BatchNorm2d, act_layer=nn.ReLU)

def conv_bn_silu(inc, outc, kernel_size, stride=1, padding=0, dilation=1, groups=1):
    return ConvBlock(inc, outc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                     bias=False, norm_layer=nn.BatchNorm2d, act_layer=nn.SiLU)

def conv_bn_act(inc, outc, kernel_size, stride=1, padding=0, dilation=1, groups=1, act_layer=nn.ReLU):
    return ConvBlock(inc, outc, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups,
                     bias=False, norm_layer=nn.BatchNorm2d, act_layer=act_layer)

class ConvBlock(BlockBase):
    def __init__(self, inc: int, outc: int, kernel_size: int, padding: int = 0, stride: int = 1, groups: int = 1, dilation: int = 1, bias: bool = None,
                       norm_layer: type = nn.BatchNorm2d, act_layer: type = nn.ReLU, drop: float = 0.0,
                       gn_dim: int = 32, pre_act: bool = False, pre_norm: bool = False) -> None:
        super().__init__()
        self.pre_norm = pre_norm
        self.pre_act = pre_act

        if bias is None:
            bias = norm_layer is None

        norm_dim = inc if pre_act or pre_norm else outc

        self.conv = nn.Conv2d(inc, outc, kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=bias)
        self.norm = self._make_norm_layer(norm_layer, norm_dim, gn_dim)
        self.act  = act_layer() if act_layer else nn.Identity()
        self.drop = nn.Dropout(p=drop, inplace=False) if drop > 0 else nn.Identity()

    def _make_norm_layer(self, norm_layer, dim, gn_dim):
        if norm_layer is None:
            return nn.Identity()

        if norm_layer is nn.GroupNorm:
            assert dim % gn_dim == 0
            norm_layer = partial(norm_layer, dim // gn_dim)

        return norm_layer(dim)

    def forward(self, x: Tensor) -> Tensor:
        if self.pre_norm:
            x = self.norm(x)
            x = self.conv(x)
            x = self.act(x)
            x = self.drop(x)
        elif self.pre_act:
            x = self.norm(x)
            x = self.act(x)
            x = self.conv(x)
            x = self.drop(x)
        else:
            x = self.conv(x)
            x = self.norm(x)
            x = self.act(x)
            x = self.drop(x)
        return x

    @torch.jit.ignore
    def _load(self, src_dict, conv_name=None, norm_name=None, adjust=False):
        if conv_name is not None:
            if adjust:
                src_dict[conv_name + '.weight'] = self._adjust_conv_weight(src_dict[conv_name + '.weight'], self.conv.weight.shape[1])
            load_weight_bias(self.conv, src_dict, conv_name, adjust)

        if norm_name is not None and self.norm:
            if has_running_stat(self.norm):
                load_running_stat(self.norm, src_dict, norm_name)
            load_weight_bias(self.norm, src_dict, norm_name)

    @torch.jit.ignore
    def _adjust_conv_weight(self, src, dst_inc):
        src_inc = src.shape[1]
        if src_inc == dst_inc:
            return src
        elif src_inc > dst_inc:
            return src[:,:dst_inc,:,:]
        else:
            n_repeat = (dst_inc - 1) // src_inc + 1
            return src.repeat(1,n_repeat,1,1)[:,:dst_inc,:,:]

class Linear(BlockBase):
    def __init__(self, inc: int, outc: int, norm_layer: type = nn.LayerNorm, act_layer: type = nn.ReLU, drop: float = 0.0, pre_act: bool = False) -> None:
        super().__init__()
        if pre_act:
            self.norm = norm_layer(inc) if norm_layer else None
            self.act = act_layer() if act_layer else None
            self.linear = nn.Linear(inc, outc)
            self.drop = nn.Dropout(p=drop, inplace=False) if drop > 0 else None
        else:
            self.linear = nn.Linear(inc, outc)
            self.norm = norm_layer(outc) if norm_layer else None
            self.act = act_layer() if act_layer else None
            self.drop = nn.Dropout(p=drop, inplace=False) if drop > 0 else None

    @torch.jit.ignore
    def _load(self, src_dict, linear_name=None, norm_name=None):
        if linear_name is not None:
            load_weight_bias(self.linear, src_dict, linear_name)

        if norm_name is not None and self.norm:
            if has_running_stat(self.norm):
                load_running_stat(self.norm, src_dict, norm_name)
            load_weight_bias(self.norm, src_dict, norm_name)

class UpConvBlock(ConvBlock):
    def upsample(self, x, size, scale_factor):
        if size is None:
            size = (int(x.shape[-2] * scale_factor), int(x.shape[-1] * scale_factor))
        if scale_factor is None:
            scale_factor = float(size[0]) / x.shape[-2]
        if x.shape[-2] != size[0] or x.shape[-1] != size[1]:
            x = F.interpolate(x, size=size, mode='bilinear')
        return x

    def forward(self, x: Tensor, size: Optional[Tuple[int]] = None, scale_factor: Optional[float] = None) -> Tensor:
        assert size is not None or scale_factor is not None

        if self.pre_act:
            x = self.norm(x)
            x = self.act(x)
            x = self.conv(x)
            x = self.upsample(x, size, scale_factor)
            x = self.drop(x)
        elif self.pre_norm:
            x = self.norm(x)
            x = self.conv(x)
            x = self.upsample(x, size, scale_factor)
            x = self.act(x)
            x = self.drop(x)
        else:
            x = self.conv(x)
            x = self.upsample(x, size, scale_factor)
            x = self.norm(x)
            x = self.act(x)
            x = self.drop(x)
        return x

class ConvPixelShuffle(BlockBase):
    def __init__(self, inc: int, outc: int, kernel_size: int, scale_factor: int, padding: int = 0, depthwise: bool = False,
                       norm_layer: type = nn.BatchNorm2d, act_layer: type = nn.ReLU, drop: float = 0, pre_act: bool = False) -> None:
        super().__init__()
        groups = inc if depthwise else 1

        expansion = scale_factor ** 2

        if norm_layer is nn.GroupNorm:
            gn_dim = 32
            assert outc % gn_dim == 0
            norm_layer = partial(norm_layer, outc // gn_dim)

        self.conv = ConvBlock(inc, inc, kernel_size, padding=padding, groups=groups, norm_layer=norm_layer, act_layer=act_layer, pre_act=pre_act)
        if pre_act:
            self.pixel_shuffle = nn.Sequential(
                norm_layer(inc),
                act_layer(),
                nn.Conv2d(inc, outc * expansion, kernel_size=1),
                nn.PixelShuffle(upscale_factor=scale_factor),
            )
        else:
            self.pixel_shuffle = nn.Sequential(
                nn.Conv2d(inc, outc * expansion, kernel_size=1),
                nn.PixelShuffle(upscale_factor=scale_factor),
                norm_layer(outc),
                act_layer(),
            )
        self.drop  = nn.Dropout(p=drop) if drop > 0 else nn.Identity()

    def forward(self, x: Tensor) -> Tensor:
        x = self.conv(x)
        x = self.pixel_shuffle(x)
        x = self.drop(x)
        return x

class MV2Block(BlockBase):
    def __init__(self, inc: int, outc: int, stride: int = 1, expansion: int = 1, pre_act: bool = False, norm_layer: nn.Module = nn.GroupNorm, act_layer: nn.Module = nn.SiLU) -> None:
        super().__init__()
        self.residual = (inc == outc and stride == 1)
        if expansion == 1:
            self.conv = nn.Sequential(
                ConvBlock(inc, inc,  kernel_size=3, padding=1, stride=stride, groups=inc, norm_layer=norm_layer, act_layer=act_layer, gn_dim=32, pre_act=pre_act),
                ConvBlock(inc, outc, kernel_size=1, padding=0, stride=1,      groups=1  , norm_layer=norm_layer, act_layer=act_layer, gn_dim=32, pre_act=pre_act),
            )
        else:
            midc = inc * expansion
            self.conv = nn.Sequential(
                ConvBlock( inc, midc, kernel_size=1, padding=0, stride=1,      groups=1   , norm_layer=norm_layer, act_layer=act_layer, gn_dim=32, pre_act=pre_act),
                ConvBlock(midc, midc, kernel_size=3, padding=1, stride=stride, groups=midc, norm_layer=norm_layer, act_layer=act_layer, gn_dim=32, pre_act=pre_act),
                ConvBlock(midc, outc, kernel_size=1, padding=0, stride=1,      groups=1   , norm_layer=norm_layer, act_layer=act_layer, gn_dim=32, pre_act=pre_act),
            )

    def forward(self, x: Tensor) -> Tensor:
        if self.residual:
            return x + self.conv(x)
        else:
            return self.conv(x)


class ConvLSTMCell(BlockBase):
    def __init__(self, input_dim, hidden_dim, kernel_size, stride=1, bias=True, norm_input=None, norm_hidden=None, norm_out=None, norm_state=None):
        super().__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size // 2, kernel_size // 2
        self.bias = bias

        self.input_conv  = ConvBlock( input_dim, 4*hidden_dim, kernel_size=kernel_size, padding=self.padding, stride=stride, norm_layer=norm_input, act_layer=None, bias=bias)
        self.hidden_conv = ConvBlock(hidden_dim, 4*hidden_dim, kernel_size=kernel_size, padding=self.padding, norm_layer=norm_hidden, act_layer=None, bias=False)
        self.norm_out = norm_out(hidden_dim) if norm_out is not None else nn.Identity()
        self.norm_state = norm_state(hidden_dim) if norm_state is not None else nn.Identity()

    def forward(self, x, cur_state):
        z_in = self.input_conv(x)

        if cur_state is None:
            B, C, H, W = z_in.shape
            h_cur, c_cur = self.init_hidden(B, (H, W), z_in.device)
        else:
            h_cur, c_cur = cur_state

        z_h = self.hidden_conv(h_cur)
        z = z_in + z_h

        cc_i, cc_f, cc_o, cc_g = torch.split(z, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        c_next = self.norm_state(c_next)
        h_next = self.norm_out(h_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class ConvGRUCell(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, bias, norm_out=None, norm_in=None, gn_dim=32):
        """
        Initialize the ConvGRU cell
        :param input_dim: int
            Number of channels of input tensor.
        :param hidden_dim: int
            Number of channels of hidden state.
        :param kernel_size: (int, int)
            Size of the convolutional kernel.
        :param bias: bool
            Whether or not to add the bias.
        :param dtype: torch.cuda.FloatTensor or torch.FloatTensor
            Whether or not to use cuda.
        """
        super(ConvGRUCell, self).__init__()
        self.padding = kernel_size // 2, kernel_size // 2
        self.hidden_dim = hidden_dim

        self.conv_gates = nn.Conv2d(in_channels=input_dim + hidden_dim,
                                    out_channels=2*self.hidden_dim,
                                    kernel_size=kernel_size,
                                    padding=self.padding,
                                    bias=bias)

        self.conv = nn.Conv2d(in_channels=input_dim+hidden_dim,
                              out_channels=self.hidden_dim,
                              kernel_size=kernel_size,
                              padding=self.padding,
                              bias=bias)

        if norm_in is nn.GroupNorm:
            assert input_dim % gn_dim == 0
            norm_in = partial(norm_in, input_dim // gn_dim)

        if norm_out is nn.GroupNorm:
            assert hidden_dim % gn_dim == 0
            norm_out = partial(norm_out, hidden_dim // gn_dim)

        self.norm_in = norm_in(input_dim) if norm_in is not None else nn.Identity()
        self.norm_out = norm_out(hidden_dim) if norm_out is not None else nn.Identity()

    def init_hidden(self, batch_size, image_size, device):
        height, width = image_size
        return torch.zeros(batch_size, self.hidden_dim, height, width, device=device)

    def forward(self, x, cell_state):
        """
        :param self:
        :param x: (b, c, h, w)
            input is actually the target_model
        :param cell_state: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        if cell_state is None:
            B, C, H, W = x.shape
            cell_state = self.init_hidden(B, (H, W), x.device)

        x = self.norm_in(x)

        z = torch.cat([x, cell_state], dim=1)
        z = self.conv_gates(z)

        gamma, beta = torch.split(z, self.hidden_dim, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        z = torch.cat([x, reset_gate*cell_state], dim=1)
        z = self.conv(z)
        z = torch.tanh(z)

        cell_state = (1 - update_gate) * cell_state + update_gate * z

        cell_state = self.norm_out(cell_state)

        return cell_state


class LayerNorm2d(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.normalized_shape = (normalized_shape, )

    def forward(self, x):
        u = x.mean(1, keepdim=True)
        s = (x - u).pow(2).mean(1, keepdim=True)
        x = (x - u) / torch.sqrt(s + self.eps)
        x = self.weight[:, None, None] * x + self.bias[:, None, None]
        return x


@torch.jit.ignore
def has_running_stat(module):
    return isinstance(module, (nn.BatchNorm3d, nn.BatchNorm2d, nn.BatchNorm1d, nn.GroupNorm))

@torch.jit.ignore
def load_weight_bias(module, src_dict, key, adjust=False):
    sdic = module.state_dict()
    sdic['weight'] = src_dict[key + '.weight']
    if key + '.bias' in src_dict:
        sdic['bias'] = src_dict[key + '.bias']
    module.load_state_dict(sdic)

@torch.jit.ignore
def load_running_stat(module, src_dict, key):
    sdic = module.state_dict()
    sdic['running_mean'] = src_dict[key + '.running_mean']
    sdic['running_var' ] = src_dict[key + '.running_var' ]
    module.load_state_dict(sdic)



