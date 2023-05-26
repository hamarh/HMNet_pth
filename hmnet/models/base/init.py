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
