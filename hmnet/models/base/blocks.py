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

import itertools
import torch
import torch.nn as nn
from torch.nn.utils import clip_grad_norm_, clip_grad_value_
from torch.cuda.amp import autocast
from collections import OrderedDict

from torch import Tensor
from typing import Tuple, List, Optional, Dict

class BlockBase(nn.Module):
    def __init__(self):
        super().__init__()
        self.__module_name = ''
        self._is_visualize = False
        self._is_debug = False

    @torch.jit.ignore
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.children():
            x = layer(x)
        return x

    @torch.jit.ignore
    def _expand_param(self, type, param):
         if type == 'conv':
             return dict(stride=param.get('s',1), padding=param.get('p',0), dilation=param.get('d',1), groups=param.get('groups',1), bias=param.get('bias',True))

    @torch.jit.ignore
    def no_weight_decay(self, target_module=None, path=''):
        if target_module is None:
            target_module = self
        names_no_decay = set()

        # register no_decay params in the target_module
        if hasattr(target_module, 'no_decay_set'):
            for name, param in target_module.named_parameters():
                if name.split('.')[-1] in target_module.no_decay_set:
                    names_no_decay.update({path+name})
                if 'bias' in target_module.no_decay_set and (len(param.shape) == 1 or name.endswith(".bias")):
                    names_no_decay.update({path+name})

        # search for child modules
        for name, module in target_module.named_children():
            path_child = path + name + '.'
            no_decay = self.no_weight_decay(module, path_child)
            names_no_decay.update(no_decay)

        return names_no_decay


    @torch.jit.ignore
    def optim_settings(self, lr, weight_decay):
        if hasattr(self, 'no_weight_decay'):
            names_no_decay = self.no_weight_decay()
        else:
            names_no_decay = []

        params_decay, params_no_decay = [], []
        for name, param in self.named_parameters():
            if name in names_no_decay:
                #print(f'{name}: lr={lr:.2e}, decay={0:.2e}')
                params_no_decay.append(param)
            else:
                #print(f'{name}: lr={lr:.2e}, decay={weight_decay:.2e}')
                params_decay.append(param)

        settings = [
            {'params': params_no_decay, 'lr': lr, 'weight_decay': 0.},
            {'params': params_decay,    'lr': lr, 'weight_decay': weight_decay},
        ]

        return settings

    @torch.jit.ignore
    def enable_debug(self):
        self._is_debug = True

    @torch.jit.ignore
    def disable_debug(self):
        self._is_debug = False

    @torch.jit.ignore
    def enable_visualization(self, dpath_out_vis):
        self.dpath_out_vis = dpath_out_vis
        self._is_visualize = True
        for m in self.modules():
            if isinstance(m, BlockBase):
                m.dpath_out_vis = dpath_out_vis
                m._is_visualize = True

    @torch.jit.ignore
    def disable_visualization(self):
        self._is_visualize = False
        for m in self.modules():
            if isinstance(m, BlockBase):
                m._is_visualize = False

    @torch.jit.ignore
    def set_module_names(self, name='model'):
        self.__module_name = name
        for child_name, module in self.named_children():
            if isinstance(module, BlockBase):
                module.set_module_names(name + '.' + child_name)
            elif isinstance(module, nn.Sequential):
                self._handle_sequential(module, name + '.' + child_name)
            else:
                self._handle_sequential(module, name + '.' + child_name)

    @torch.jit.ignore
    def _handle_sequential(self, seq_module, name):
        for child_name, module in seq_module.named_children():
            if isinstance(module, BlockBase):
                module.set_module_names(name + '.' + child_name)
            elif isinstance(module, nn.Sequential):
                self._handle_sequential(module, name + '.' + child_name)
            else:
                self._handle_sequential(module, name + '.' + child_name)

    @torch.jit.ignore
    def get_module_name(self):
        return self.__module_name

    @torch.jit.ignore
    def _optim_settings(self, settings):
        """
            deprecated
        """
        for set in settings:
            target = set.pop('target')
            set['params'] = self._children_params(target)
        return settings

    @torch.jit.ignore
    def _children_params(self, children):
        """
            deprecated
        """
        list_gen = [child.parameters() for child in children]
        return itertools.chain(*list_gen)

    @torch.jit.ignore
    def print_grad_norm(self):
        self._print_grad_norm(self, self.get_module_name())

    @torch.jit.ignore
    def clip_grad(self, method='norm', max_norm=1.0, clip_value=0.5):
        self._clip_grad(self, method, max_norm, clip_value)

    @staticmethod
    @torch.jit.ignore
    def _print_grad_norm(module, name):
        total_norm = 0
        for param in module.parameters():
            total_norm += param.grad.norm(2.0).item() ** 2
        total_norm = total_norm ** 0.5
        print(f'{name}: {total_norm}')

    @staticmethod
    @torch.jit.ignore
    def _clip_grad(module, method='norm', max_norm=1.0, clip_value=0.5):
        assert method in ('norm', 'value')
        if method == 'norm':
            clip_grad_norm_(module.parameters(), max_norm=max_norm)
        elif method == 'value':
            clip_grad_value_(module.parameters(), clip_value=clip_value)

    @torch.jit.ignore
    def to_fast_model(self) -> nn.Module:
        for n, m in self.named_modules():
            if hasattr(m, '_to_fast_model'):
                m._to_fast_model()
                print(f'Convert to fast model at {n}')
        return self

    @torch.jit.ignore
    def compile(self, backend, fp16=False, input_shapes=None):
        assert backend in ('jit', 'trt', 'inductor', 'aot_ts_nvfuser', 'onnx', 'tensorrt')
        PTH2 = torch.__version__.split('.')[0] == '2'
        dtype = torch.half if fp16 else torch.float

        device = None
        for m in self.parameters():
            device = m.device
            break

        def get_inputs(input_shapes, dtype, device):
            if isinstance(input_shapes[0], (list, tuple)):
                return [ torch.randn(*shape, dtype=dtype, device=device) for shape in input_shapes ]
            else:
                return [ torch.randn(*input_shapes, dtype=dtype, device=device) ]

        if PTH2:
            if backend == 'jit':
                #model = torch.jit.script(self)
                #with autocast(enabled=fp16):
                #    model = torch.jit.trace(self, get_inputs(input_shapes, dtype, device))
                model = torch.jit.trace(self, get_inputs(input_shapes, torch.float, device))
                return model
            elif backend == 'trt':
                import torch_tensorrt
                model = torch.jit.trace(self, get_inputs(input_shapes, torch.float, device))
                #model = torch_tensorrt.compile(model, inputs=get_inputs(input_shapes, torch.float, device), enabled_precisions={torch.float, torch.half}, truncate_long_and_double=True)
                with autocast(enabled=fp16):
                    #model = torch_tensorrt.compile(model, inputs=get_inputs(input_shapes, dtype, device), enabled_precisions={dtype}, truncate_long_and_double=True)
                    model = torch_tensorrt.compile(model, inputs=get_inputs(input_shapes, dtype, device), enabled_precisions={torch.float, torch.half}, truncate_long_and_double=True)
                return model
            elif backend == 'inductor':
                return torch.compile(self, backend='inductor')
            elif backend == 'aot_ts_nvfuser':
                return torch.compile(self, backend='aot_ts_nvfuser')
            elif backend == 'onnx':
                return torch.compile(self, backend='onnxrt')
            elif backend == 'tensorrt':
                return torch.compile(self, backend='tensorrt')
        else:
            assert backend in ('jit', 'trt')
            if backend == 'jit':
                return torch.jit.script(self)
            elif backend == 'trt':
                import torch_tensorrt
                model = torch.jit.script(self)
                return torch_tensorrt.compile(model, inputs=[torch.randn(*input_shape, dtype=dtype, device=device)], enabled_precisions={dtype})

        


