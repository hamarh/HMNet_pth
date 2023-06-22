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

import sys
import torch
import torch.nn as nn
import numpy as np
from torch_scatter import scatter

class VoxelGrid(nn.Module):
    def __init__(self, keep_polarity=True, num_bins=5, dynamic_duration=False, dt_offset=1, measurement='time'):
        super().__init__()
        self.keep_polarity = keep_polarity
        self.num_bins = num_bins
        self.dynamic_duration = dynamic_duration
        self.dt_offset = dt_offset
        self.measurement = measurement
        self.num_channels = (1 + keep_polarity) * num_bins

    def forward(self, events, image_meta):
        curr_time = image_meta['curr_time_crop']
        delta_t = image_meta['delta_t']
        width = image_meta['width']
        height = image_meta['height']

        if len(events) > 0:
            t, x, y, p = events[:,0], events[:,1], events[:,2], events[:,3]
            image = voxel_grid(t, x, y, p, curr_time=curr_time, duration=delta_t, num_bins=self.num_bins, width=width, height=height,
                               keep_polarity=self.keep_polarity, dynamic_duration=self.dynamic_duration, dt_offset=self.dt_offset, measurement=self.measurement)
        else:
            image = torch.zeros([self.num_channels, height, width], device=events.device, dtype=torch.float)
        C, H, W = image.shape
        image_meta['img_shape'] = [W, H, C]
        image_meta['pad_shape'] = [W, H, C]

        return image, image_meta

    def forward_batch(self, list_events, list_image_meta):
        out_image, out_image_meta = [], []

        for events, image_meta in zip(list_events, list_image_meta):
            image_meta = image_meta.copy()
            curr_time = image_meta['curr_time_crop']
            delta_t = image_meta['delta_t']
            width = image_meta['width']
            height = image_meta['height']

            if len(events) > 0:
                t, x, y, p = events[:,0], events[:,1], events[:,2], events[:,3]
                image = voxel_grid(t, x, y, p, curr_time=curr_time, duration=delta_t, num_bins=self.num_bins, width=width, height=height,
                                   keep_polarity=self.keep_polarity, dynamic_duration=self.dynamic_duration, dt_offset=self.dt_offset, measurement=self.measurement)
            else:
                image = torch.zeros([self.num_channels, height, width], device=events.device, dtype=torch.float)
            C, H, W = image.shape
            image_meta['img_shape'] = [W, H, C]
            image_meta['pad_shape'] = [W, H, C]

            out_image.append(image)
            out_image_meta.append(image_meta)

        return torch.stack(out_image), out_image_meta

class TimeSurface(nn.Module):
    def __init__(self, keep_polarity=True, tau=[10e3, 100e3], return_time_stamp=False):
        super().__init__()
        self.keep_polarity = keep_polarity
        self.tau = tau
        self.return_time_stamp = return_time_stamp

    def forward(self, events, image_meta, time_stamp=None):
        curr_time = image_meta['curr_time_crop']
        width = image_meta['width']
        height = image_meta['height']

        if len(events) > 0:
            t, x, y, p = events[:,0], events[:,1], events[:,2], events[:,3]
            image, time_stamp, _ = time_surface(t, x, y, p, time_stamp, curr_time, width=width, height=height, keep_polarity=self.keep_polarity, tau=self.tau, update_curr_time=False)
        elif time_stamp is None:
            num_channels = len(self.tau) * (self.keep_polarity + 1)
            image = torch.zeros([num_channels, height, width], device=events.device, dtype=torch.float)
        else:
            image = _decay(time_stamp, curr_time, self.tau)
        C, H, W = image.shape
        image_meta['img_shape'] = [W, H, C]
        image_meta['pad_shape'] = [W, H, C]

        if self.return_time_stamp:
            return image, image_meta, time_stamp
        else:
            return image, image_meta

    def forward_batch(self, list_events, list_image_meta, list_time_stamps=None):
        out_image, out_image_meta, out_time_stamps = [], [], []

        if list_time_stamps is None:
            list_time_stamps = [None] * len(list_events)

        for events, image_meta, time_stamp in zip(list_events, list_image_meta, list_time_stamps):
            image_meta = image_meta.copy()
            curr_time = image_meta['curr_time_crop']
            delta_t = image_meta['delta_t']
            width = image_meta['width']
            height = image_meta['height']

            if len(events) > 0:
                t, x, y, p = events[:,0], events[:,1], events[:,2], events[:,3]
                image, time_stamp, _ = time_surface(t, x, y, p, time_stamp, curr_time, width=width, height=height, keep_polarity=self.keep_polarity, tau=self.tau, update_curr_time=False)
            elif time_stamp is None:
                C = len(self.tau) * (1 + self.keep_polarity)
                image = torch.zeros([C, height, width], device=events.device)
            else:
                image = _decay(time_stamp, curr_time, self.tau)
            C, H, W = image.shape
            image_meta['img_shape'] = [W, H, C]
            image_meta['pad_shape'] = [W, H, C]

            out_image.append(image)
            out_image_meta.append(image_meta)
            out_time_stamps.append(time_stamp)

        if self.return_time_stamp:
            return torch.stack(out_image), out_image_meta, torch.stack(out_time_stamps)
        else:
            return torch.stack(out_image), out_image_meta

class Histogram(nn.Module):
    def __init__(self, keep_polarity=True, mean=0, std=20, clip_max=1, clip_min=None):
        super().__init__()
        self.keep_polarity = keep_polarity
        self.mean = mean
        self.std = std
        self.clip_max = clip_max
        self.clip_min = clip_min

    def forward(self, events, image_meta):
        curr_time = image_meta['curr_time_crop']
        delta_t = image_meta['delta_t']
        width = image_meta['width']
        height = image_meta['height']

        if len(events > 0):
            t, x, y, p = events[:,0], events[:,1], events[:,2], events[:,3]
            image = histogram(x, y, p, width, height, self.keep_polarity, self.mean, self.std, self.clip_max, self.clip_min)
        else:
            image = torch.zeros([1 + self.keep_polarity, height, width], device=events.device, dtype=torch.float)
        C, H, W = image.shape
        image_meta['img_shape'] = [W, H, C]
        image_meta['pad_shape'] = [W, H, C]

        return image, image_meta

    def forward_batch(self, list_events, list_image_meta):
        out_image, out_image_meta = [], []

        for events, image_meta in zip(list_events, list_image_meta):
            image_meta = image_meta.copy()
            curr_time = image_meta['curr_time_crop']
            delta_t = image_meta['delta_t']
            width = image_meta['width']
            height = image_meta['height']

            if len(events > 0):
                t, x, y, p = events[:,0], events[:,1], events[:,2], events[:,3]
                image = histogram(x, y, p, width, height, self.keep_polarity, self.mean, self.std, self.clip_max, self.clip_min)
            else:
                image = torch.zeros([1 + self.keep_polarity, height, width], device=events.device, dtype=torch.float)
            C, H, W = image.shape
            image_meta['img_shape'] = [W, H, C]
            image_meta['pad_shape'] = [W, H, C]

            out_image.append(image)
            out_image_meta.append(image_meta)

        return torch.stack(out_image), out_image_meta

    def forward_batch2(self, list_events, list_image_meta):
        events = self._concat(list_events)

        t, x, y, p, b = events[:,0], events[:,1], events[:,2], events[:,3], events[:,4]

        width = list_image_meta[0]['width']
        height = list_image_meta[0]['height']
        image = histogram_batch(x, y, p, b, width, height, self.keep_polarity, len(list_image_meta), self.mean, self.std, self.clip_max, self.clip_min)

        for image_meta in list_image_meta:
            B, C, H, W = image.shape
            image_meta['img_shape'] = [W, H, C]
            image_meta['pad_shape'] = [W, H, C]

        return image, list_image_meta

    def _concat(self, list_events):
        output = []
        for bidx, events in enumerate(list_events):
            b = torch.full([len(events),1], bidx, dtype=events.dtype, device=events.device)
            events = torch.cat([events, b], dim=1)
            output.append(events)
        return torch.cat(output, dim=0)



def histogram(x, y, p, width, height, keep_polarity, mean=0, std=20, clip_max=1, clip_min=None, out=None):
    if not keep_polarity:
        positive = torch.zeros_like(p)
        pdim = 1
    else:
        positive = p > 0
        pdim = 2

    list_indices = [positive.long(), y.long(), x.long()]
    shape = [pdim, height, width]
    values = (p.abs() - mean) / std

    img = _event_scatter(list_indices, values, shape, reduce='add', out=out)

    if clip_min is not None or clip_max is not None:
        img = img.clip_(min=clip_min, max=clip_max)

    return img

def time_surface(t, x, y, p, time_stamp, curr_time, width=1280, height=720, keep_polarity=True, update_curr_time=True, tau=[10e3, 100e3]):
    if len(t) > 0 and update_curr_time:
        curr_time = t[-1]

    if not keep_polarity:
        positive = torch.zeros_like(p)
    else:
        positive = p > 0

    list_indices = [positive.long(), y.long(), x.long()]
    shape = [1+keep_polarity, height, width]
    values = t

    time_stamp = _event_scatter(list_indices, values, shape, reduce='max', out=time_stamp)
    surface = _decay(time_stamp, curr_time, tau)
    return surface, time_stamp, curr_time

def voxel_grid(t, x, y, p, curr_time, duration, num_bins, width, height, keep_polarity, dynamic_duration=False, dt_offset=1, measurement='time', out=None):
    t0 = curr_time - duration
    tE = curr_time
    t1 = t[0]
    tN = t[-1]

    if torch.any(t > curr_time).item() == True:
        print(t.max(), curr_time)
        raise RuntimeError

    if dynamic_duration:
        bt = t0
        dt = tN - t1
        dt += dt_offset
    else:
        bt = t0
        dt = tE - t0
        dt += dt_offset

    # get kernel weight and corresponding bin indices
    bin_indices_float = float(num_bins - 1) * (t - bt) / dt
    backward_bin_indices = bin_indices_float.long()
    forward_bin_indices = backward_bin_indices + 1

    forward_weight = bin_indices_float - backward_bin_indices.float()
    backward_weight = 1. - forward_weight

    bin_indices = torch.cat([backward_bin_indices, forward_bin_indices])
    kernel_weight = torch.cat([backward_weight, forward_weight])

    x = x.repeat(2)
    y = y.repeat(2)
    bin = bin_indices
    p = p.repeat(2)

    if measurement == 'count':
        values = kernel_weight * p.abs()
    elif measurement == 'time':
        values = kernel_weight * ((t - bt) / dt).repeat(2)
    else:
        raise RuntimeError

    if not keep_polarity:
        positive = torch.zeros_like(p)
        pdim = 1
    else:
        positive = p > 0
        pdim = 2

    list_indices = [bin, positive.long(), y.long(), x.long()]
    shape = [num_bins, pdim, height, width]

    img = _event_scatter(list_indices, values, shape, reduce='add', out=out)

    return img.view(-1, height, width)

def _event_scatter(list_indices, values, shape, reduce, out=None):
    assert len(list_indices) == len(shape)

    shape += [1]
    indices = torch.zeros_like(list_indices[0])
    for i, inds in enumerate(list_indices):
        indices += inds * np.prod(shape[i+1:])

    if out is None:
        img = scatter(values, indices, reduce=reduce, dim_size=np.prod(shape))
    else:
        img = scatter(values, indices, reduce=reduce, out=out.view(-1))

    return img.view(*shape[:-1])

def _decay(time_stamp, curr_time, tau):
    if not isinstance(tau, torch.Tensor):
        tau = torch.Tensor(tau).to(time_stamp.device)
    height, width = time_stamp.shape[-2:]
    surface = ((time_stamp - curr_time)[:,None,:,:] / tau[None,:,None,None]).exp()    # [N_pol, N_tau, H, W]
    mask = time_stamp > 0
    surface = surface * mask[:,None,:,:]
    return surface.view(-1, height, width)



# ======================== batch method (not used) ========================================
def histogram_batch(x, y, p, b, width, height, keep_polarity, batch_size, mean=0, std=20, clip_max=1, clip_min=None, out=None):
    if not keep_polarity:
        positive = torch.zeros_like(p)
        pdim = 1
    else:
        positive = p > 0
        pdim = 2

    list_indices = [b, positive.long(), y.long(), x.long()]
    shape = [batch_size, pdim, height, width]
    values = (p.abs() - mean) / std
    #values = (1. - mean) / std
    #values = torch.tensor([values]*len(x), dtype=torch.float, device=x.device)

    img = _event_scatter(list_indices, values, shape, reduce='add', out=out)
    img = img.clip_(min=clip_min, max=clip_max)

    return img

def time_surface_batch(t, x, y, p, b, time_stamp, curr_time, width, height, keep_polarity, batch_size, update_curr_time=True, tau=[10e3, 100e3]):
    if len(t) > 0 and update_curr_time:
        curr_time = t[-1]

    if not keep_polarity:
        positive = torch.zeros_like(p)
    else:
        positive = p > 0

    list_indices = [b, positive.long(), y.long(), x.long()]
    shape = list(time_stamp.shape)
    values = t

    time_stamp = _event_scatter(list_indices, values, shape, reduce='max', out=time_stamp)
    surface = _decay(time_stamp, curr_time, tau)
    return surface, time_stamp, curr_time

def voxel_grid_batch(t, x, y, p, b, curr_time, duration, num_bins, width, height, keep_polarity, batch_size, dynamic_duration=False, dt_offset=1, measurement='time', out=None):
    t0 = curr_time - duration
    tE = curr_time
    t1 = t[0]
    tN = t[-1]

    if dynamic_duration:
        bt = t0
        dt = tN - t1
        dt += dt_offset
    else:
        bt = t0
        dt = tE - t0
        dt += dt_offset

    # get kernel weight and corresponding bin indices
    bin_indices_float = float(num_bins - 1) * (t - bt) / dt
    backward_bin_indices = bin_indices_float.long()
    forward_bin_indices = backward_bin_indices + 1

    forward_weight = bin_indices_float - backward_bin_indices
    backward_weight = 1 - forward_weight

    bin_indices = torch.cat([backward_bin_indices, forward_bin_indices])
    kernel_weight = torch.cat([backward_weight, forward_weight])

    x = x.repeat(2)
    y = y.repeat(2)
    bin = bin_indices
    p = p.repeat(2)

    if measurement == 'count':
        values = kernel_weight * p.abs()
    elif measurement == 'time':
        values = kernel_weight * ((t - bt) / dt).repeat(2)
    else:
        raise RuntimeError

    if not keep_polarity:
        positive = torch.zeros_like(p)
        pdim = 1
    else:
        positive = p > 0
        pdim = 2

    list_indices = [b, bin, positive.long(), y.long(), x.long()]
    shape = [batch_size, num_bins, pdim, height, width]

    img = _event_scatter(list_indices, values, shape, reduce='add', out=out)
    return img.view(batch_size, -1, height, width)

