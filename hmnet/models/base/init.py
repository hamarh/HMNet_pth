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
import torch.nn.init as init
import numpy as np
import math

def init_transformer(modules):
    for m in modules:
        if isinstance(m, nn.Linear):
            init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm)):
            init.constant_(m.bias, 0)
            init.constant_(m.weight, 1.0)
        elif isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        if hasattr(m, 'custom_init'):
            m.custom_init()

def kaiming_uniform_silu(modules):
    for m in modules:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight, a=math.sqrt(5), nonlinearity='leaky_relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        if hasattr(m, 'custom_init'):
            m.custom_init()

def kaiming_uniform_relu(modules):
    for m in modules:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.kaiming_uniform_(m.weight, nonlinearity='relu')
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        if hasattr(m, 'custom_init'):
            m.custom_init()

def xavier_uniform_relu(modules):
    for m in modules:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.xavier_uniform_(m.weight, gain=init.calculate_gain('relu'))
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.normal_(m.weight, std=0.01)
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        if hasattr(m, 'custom_init'):
            m.custom_init()

def xavier_uniform_sigmoid(modules):
    for m in modules:
        if isinstance(m, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            init.xavier_uniform_(m.weight, gain=init.calculate_gain('sigmoid'))
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, nn.Linear):
            init.xavier_uniform_(m.weight, gain=init.calculate_gain('sigmoid'))
            if m.bias is not None:
                init.constant_(m.bias, 0)
        elif isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
            init.constant_(m.weight, 1)
            init.constant_(m.bias, 0)
        if hasattr(m, 'custom_init'):
            m.custom_init()

def load_state_dict_matched(model, src_dict, device=torch.device('cpu')):
    no_matching = []
    tar_dict = model.state_dict()
    if isinstance(src_dict, str):
        src_dict = torch.load(src_dict, map_location=device)
        if 'state_dict' in src_dict:
            src_dict = src_dict['state_dict']

    for tar_key in tar_dict.keys():
        if tar_key in src_dict:
            tar_dict[tar_key] = src_dict[tar_key].clone()
        else:
            no_matching.append(tar_key)
    model.load_state_dict(tar_dict)
    return no_matching

def load_state_dict_flexible(model, src_dict, key_matching='gestalt', device=torch.device('cpu')):
    assert key_matching in ('gestalt', 'strict')
    dst_dict = model.state_dict()
    if isinstance(src_dict, str):
        src_dict = torch.load(src_dict, map_location=device)
        if 'state_dict' in src_dict:
            src_dict = src_dict['state_dict']

    keys_src = list(src_dict.keys())
    keys_dst = list(dst_dict.keys())

    if key_matching == 'gestalt':
        strcomp = lambda x, y: difflib.SequenceMatcher(None, x, y).ratio()
    elif key_matching == 'strict':
        strcomp = lambda x, y: x == y

    scores = [ [ strcomp(skey, dkey) for skey in keys_src ] for dkey in keys_dst]
    scores = np.array(scores)
    matching = scores.argmax(axis=1)
    uniq_s_indices, counts = np.unique(matching, return_counts=True)
    for s_index in uniq_s_indices[counts > 1]:
        d_indices = np.flatnonzero(matching == s_index)
        matching[d_indices] = _resolve_duplication(s_index, d_indices, scores, src_dict, dst_dict)

    dkeys_not_matched = [ keys_dst[idx] for idx in np.flatnonzero(matching == -1) ]
    values, counts = np.unique(np.concatenate([np.arange(len(keys_src)), uniq_s_indices]), return_counts=True)
    skeys_not_matched = [ keys_src[idx] for idx in values[counts == 1] ]

    for didx, sidx in enumerate(matching):
        if sidx == -1: continue
        dkey = keys_dst[didx]
        skey = keys_src[sidx]
        dst_dict[dkey] = src_dict[skey]

    model.load_state_dict(dst_dict)

def _resolve_duplication(s_index, d_indices, scores, src_dict, dst_dict):
    s_indices = -1 * np.ones(len(d_indices))
    _scores = scores[d_indices, s_index]
    _shapes = np.array([ _shape_matching(s_index, d_index, src_dict, dst_dict) for d_index in d_indices ])
    _scores = _scores * _shapes

    if np.any(_scores > 0):
        s_indices[_scores.argmax()] = s_index

    return s_indices

def _shape_matching(s_index, d_index, src_dict, dst_dict):
    svalue = list(src_dict.values())[s_index]
    dvalue = list(dst_dict.values())[d_index]
    return svalue.shape == dvalue.shape
