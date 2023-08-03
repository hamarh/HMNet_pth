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

import os
import numpy as np
import random
import collections
import torch
import torchvision
import pandas as pd
import math
import json
import copy
import pickle as pkl
import h5py

from hmnet.utils.common import get_list
from hmnet.models.base.event_repr.builder import build_eventrepr

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

WIDTH = 346
HEIGHT = 260

DEPTH_MAX = 80
ALPHA = 3.7

MEAN_STD = {
    'eventscape' : [ 94.386783, 52.621100 ],
    'day1'       : [46.02644024, 66.58477104],
    'day2'       : [44.88074027, 75.6648636 ],
    'night1'     : [23.50456365, 51.03250885],
    'simple'     : [0, 255],
}

def depth_to_nlogdepth(depth):
    ndepth = np.log(depth / DEPTH_MAX) / ALPHA + 1
    return ndepth

class EventPacket(torch.utils.data.Dataset):
    def __init__(self, fpath_data=None, fpath_gt=None, fpath_meta=None, image_mean_std='eventscape', train_duration=5e4,
                 fpath_sampling=None, sampling='random', start_index_aug_method='center', start_index_aug_ratio=1., sampling_stride=-1,
                 random_time_scaling=False, min_time_scale=0.5, max_time_scale=2.0,
                 event_transform=None, output_type=None, skip_image_loading=False,
                 max_events_per_packet=-1, downsample_packet_length=None):

        assert sampling in ('random', 'file', 'label', 'regular')

        self.sampling = sampling
        self.ev_meta = np.load(fpath_meta)
        self.skip_ts = 0
        self.event_transform = event_transform
        self.output_type = output_type

        mean, std = MEAN_STD[image_mean_std]
        self.gray_mean = np.array(mean).repeat(3,0)[:,None,None]
        self.gray_std = np.array(std).repeat(3,0)[:,None,None]
        self.skip_image_loading = skip_image_loading
        self.max_events_per_packet = max_events_per_packet
        self.downsample_packet_length = downsample_packet_length or train_duration
        self.start_index_aug_method = start_index_aug_method
        self.start_index_aug_ratio = start_index_aug_ratio
        self.train_duration = int(train_duration)
        self.random_time_scaling = random_time_scaling
        self.min_time_scale = min_time_scale
        self.max_time_scale = max_time_scale

        self.fpath_data = fpath_data
        self.fpath_gt = fpath_gt

        data = h5py.File(fpath_data)
        gt = h5py.File(fpath_gt)

        events   = data['davis']['left']['events']    # x, y, t, p (unit t: sec)
        self.start_t  = events[0,2] * 1e6
        self.end_t    = events[-1,2] * 1e6
        self.images_t = data['davis']['left']['image_raw_ts'][...] * 1e6 - self.start_t
        self.label_t  = gt['davis']['left']['depth_image_raw_ts'][...] * 1e6 - self.start_t

        if sampling == 'file':
            sampling_schedule = pkl.load(open(fpath_sampling, 'rb'))
            self.sampling_timings = sampling_schedule['timings']
            self.total_seq = len(self.sampling_timings)
        elif sampling == 'label':
            seg_indices = np.unique((self.label_t // 1000).astype(int)).tolist()
            self.sampling_timings = [ seg_index for seg_index in seg_indices ]
            self.total_seq = len(self.sampling_timings)
        elif sampling == 'regular':
            sampling_stride = sampling_stride if sampling_stride > 0 else train_duration
            seg_stride = int(sampling_stride // 1000)
            end = int((self.end_t - self.start_t) // 1000)
            self.sampling_timings = [ seg_index for seg_index in range(0, end, seg_stride) ]
            self.total_seq = len(self.sampling_timings)
        elif sampling == 'random':
            sampling_stride = sampling_stride if sampling_stride > 0 else train_duration
            self.total_seq = int((self.end_t - self.start_t) // sampling_stride)

        self._image_meta = {
            'width': WIDTH,
            'height': HEIGHT,
            'max_depth': 80,
            'min_depth': 1.978,
        }

    def _read_h5py(self):
        data = h5py.File(self.fpath_data)
        gt = h5py.File(self.fpath_gt)
        events = data['davis']['left']['events']    # x, y, t, p (unit t: sec)
        images = data['davis']['left']['image_raw']
        label  = gt['davis']['left']['depth_image_raw']
        return events, images, label

    def _get_path(self, filename):
        if filename[:2] == './':
            filename = filename[2:]
        return self.base_path + '/' + filename

    def __len__(self):
        return self.total_seq

    def __getitem__(self, index):
        events, _, image, _, label, meta_data, image_indices, label_indices = self.getdata(index)

        events = torch.from_numpy(events)
        image = torch.from_numpy(image).float() if len(image) > 0 else None
        label = torch.from_numpy(label).float() if len(label) > 0 else None

        data = { 'events': events, 'images': image }
        target = { 'depth': label }
        meta_data['image_meta']['image_indices'] = image_indices
        meta_data['image_meta']['depth_indices'] = label_indices

        return data, target, meta_data

    def getdata(self, index, keep_latest_labels=True, keep_latest_images=True, skip_ts=0):
        if self.random_time_scaling:
            time_scaling = random.uniform(self.min_time_scale, self.max_time_scale)
        else:
            time_scaling = 1

        train_duration = int(self.train_duration * time_scaling)

        base_time, ev_range = self._choose_time(index, train_duration)
        events, image_t, image, label_t, label, image_indices, label_indices = self._load(base_time, train_duration, ev_range)

        events = self._bind(events)
        events, image_t, label_t = self._subtract_base_time(events, image_t, label_t, base_time)

        if self.max_events_per_packet > 0:
            events = self._event_downsample(events, self.max_events_per_packet, self.downsample_packet_length)

        curr_time_org = base_time + train_duration
        curr_time_crop = train_duration
        image_meta = self._update_image_meta(self._image_meta, self.fpath_data, curr_time_org, curr_time_crop, train_duration, time_scaling)

        # keep latest labels/images and discard others
        if keep_latest_labels:
            label_t, label, label_indices = self._keep_latest_labels(label_t, label, label_indices)
        if keep_latest_images:
            image_t, image, image_indices = self._keep_latest_images(image_t, image, image_indices)

        if self.event_transform is not None:
            events, label, image, image_meta = self.event_transform(events, label, image, image_meta, types=['event', 'image', 'image', 'meta'])

        # preprocess rgb image
        if len(image) > 0:
            image = (image - self.gray_mean) / self.gray_std

        if self.output_type is None:
            pass
        elif self.output_type == 'long':
            events[:,-1] = ((events[:,-1] + 1) * 0.5)    # polarity in {0, 1}
            events = events.astype(np.int64)
        elif self.output_type == 'float':
            events = events.astype(np.float)             # polarity in {-1.0, 1.0}
        else:
            raise RuntimeError

        meta_data = {
            'image_meta': image_meta,
            'label_meta': None,
        }

        return events, image_t, image, label_t, label, meta_data, image_indices, label_indices

    def _load(self, time, duration, ev_range):
        events, images, label = self._read_h5py()
        st, ed = ev_range
        events = np.array(events[st:ed])
        events[:,2] = events[:,2] * 1e6 - self.start_t

        label, label_t, label_indices = self._load_delta_t(label, self.label_t, time + self.skip_ts, duration - self.skip_ts)
        label = label[:,None,:,:]

        # load image
        if self.skip_image_loading:
            image_t = np.array([], dtype=label_t.dtype)
            image = []
            image_indices = []
        else:
            image, image_t, image_indices = self._load_delta_t(images, self.images_t, time, duration)
            image = image[:,None,:,:].repeat(3,1)

        return events, image_t, image, label_t, label, image_indices, label_indices

    def _load_delta_t(self, data, data_t, base_time, duration):
        mask = np.logical_and( data_t >= base_time, data_t < base_time + duration )
        indices = np.flatnonzero(mask)
        return np.array(data[mask]), data_t[mask], indices

    def _augment_index(self, seg_index, method, aug_ratio, nseg_per_packet, start_seg_index, end_seg_index):
        whole_range = nseg_per_packet - 1
        aug_range = int(whole_range * aug_ratio)
        if method == 'none':
            return seg_index, True
        elif method == 'center':
            min_seg_idx = seg_index - int(whole_range * 0.5 + aug_range * 0.5)
            max_seg_idx = seg_index - int(whole_range * 0.5 - aug_range * 0.5)
        elif method == 'start':
            min_seg_idx = seg_index - aug_range
            max_seg_idx = seg_index
        elif method == 'end':
            min_seg_idx = seg_index - whole_range
            max_seg_idx = seg_index - whole_range + aug_range

        min_seg_idx = max(start_seg_index, min_seg_idx)
        max_seg_idx = min(end_seg_index - whole_range, max_seg_idx)

        if min_seg_idx > max_seg_idx:
            if min_seg_idx == start_seg_index:
                return min_seg_idx, False
            else:
                return max_seg_idx, False

        seg_index = random.randint(min_seg_idx, max_seg_idx)
        return seg_index, True

    def _choose_time(self, index, train_duration):
        nseg_per_packet = int(train_duration / 1000)
        start_seg_index = 0
        end_seg_index   = int((self.end_t - self.start_t) // 1000)

        if self.sampling == 'random':
            seg_index = random.randint(start_seg_index, end_seg_index - nseg_per_packet)
        elif self.sampling in ('file', 'label', 'regular'):
            seg_index = self.sampling_timings[index]

        seg_index, is_success = self._augment_index(seg_index, self.start_index_aug_method, self.start_index_aug_ratio, nseg_per_packet, start_seg_index, end_seg_index)
        if not is_success and self.sampling == 'file':
            print('Desired segment cannnot be cropped. Retrying with another index.')
            index = random.randrange(len(self.sampling_timings))
            return self._choose_time(index, train_duration)
        elif not is_success:
            print('Desired segment cannnot be cropped. Cropping nearest segment.')

        ev_index, ev_count = self.ev_meta[:,0], self.ev_meta[:,1]
        ev_i = int(ev_index[seg_index])
        ev_c = int(ev_count[seg_index: seg_index + nseg_per_packet].sum())
        ev_range = (ev_i, ev_i + ev_c)
        time = seg_index * 1000

        return time, ev_range

    def _update_image_meta(self, image_meta, fpath_evt, curr_time_org, curr_time_crop, delta_t, time_scaling):
        image_meta = image_meta.copy()
        image_meta['filename'] = fpath_evt
        image_meta['ori_filename'] = fpath_evt.split('/')[-1]
        image_meta['curr_time_org'] = curr_time_org
        image_meta['curr_time_crop'] = curr_time_crop
        image_meta['delta_t'] = delta_t
        image_meta['stride_t'] = delta_t
        image_meta['time_scaling'] = time_scaling
        image_meta['image_indices'] = None
        image_meta['depth_indices'] = None
        return image_meta

    def _keep_latest_labels(self, label_t, label, label_path):
        if len(label) == 0:
            return label_t, label, None
        return label_t[-1], label[-1], label_path[-1]

    def _keep_latest_images(self, image_t, image, image_path):
        if len(image) == 0:
            return image_t, image, None
        return image_t[-1], image[-1], image_path[-1]

    def _bind(self, events):
        x_evt, y_evt, t_evt, p_evt = events[:,0], events[:,1], events[:,2], events[:,3]
        events = np.stack([t_evt, x_evt, y_evt, p_evt], axis=-1).astype(int)
        return events

    def _subtract_base_time(self, events, image_t, label_t, base_time):
        events[:,0] = events[:,0] - base_time
        image_t = image_t - base_time
        label_t = label_t - base_time
        return events, image_t, label_t

    def _split_by_indices(self, data_array, indices):
        split_indices = np.flatnonzero(indices[1:] - indices[:-1]) + 1
        data_splits = np.split(data_array, split_indices)
        new_indices = np.unique(indices)
        return data_splits, new_indices

    def _event_downsample(self, events, num_samples_per_duration, duration):
        times_evt = events[:,0].astype(int)
        segment_indices_evt = times_evt // duration
        event_splits, segment_indices_evt = self._split_by_indices(events, segment_indices_evt)
        event_splits = [ self._randchoice(evt, num_samples_per_duration) for evt in event_splits ]
        output = np.concatenate(event_splits, axis=0)
        return output

    def _randchoice(self, data, num_sample):
        if len(data) <= num_sample:
            return data
        indices = np.random.choice(len(data), size=num_sample, replace=False)
        indices = np.sort(indices)
        return data[indices]


class EventFrame(EventPacket):
    def __init__(self, *args, event_repr=None, frame_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert event_repr is not None
        self.event_repr = build_eventrepr(event_repr)
        self.frame_transform = frame_transform

    def __getitem__(self, index):
        events, _, image, _, label, meta_data, image_indices, label_indices = self.getdata(index)

        events = torch.from_numpy(events)
        image = torch.from_numpy(image).float() if len(image) > 0 else None
        label = torch.from_numpy(label).float() if len(label) > 0 else None

        image_meta = meta_data['image_meta']
        ev_image, image_meta = self.event_repr(events, image_meta)
        if self.frame_transform is not None:
            ev_image, image, label, image_meta = self.frame_transform(ev_image, image, label, image_meta, types=['image', 'image', 'image', 'meta'])
        assert ev_image.shape[1] == image_meta['height'] and ev_image.shape[2] == image_meta['width']

        data = {'events': ev_image, 'images': image}
        target = {'depth': label}
        image_meta['image_indices'] = image_indices
        image_meta['depth_indices'] = label_indices
        meta_data['image_meta'] = image_meta

        return data, target, meta_data


class EventPacketStream(EventPacket):
    def __init__(self, *args, skip_ts=0, delta_t=1000, stream_stride=None, use_nearest_label=False, use_nearest_image=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.delta_t = delta_t
        self.stride_t = stream_stride or delta_t
        self.skip_ts = skip_ts
        self.use_nearest_label = use_nearest_label
        self.use_nearest_image = use_nearest_image

    def __getitem__(self, index):
        events, image_t, image, label_t, label, meta_data, image_indices, label_indices = self.getdata(index, keep_latest_labels=False, keep_latest_images=False, skip_ts=self.skip_ts)

        label_meta   = meta_data['label_meta']
        image_meta   = meta_data['image_meta']
        time_scaling = image_meta['time_scaling']
        fpath_evt    = image_meta['filename']

        train_duration = int(self.train_duration * time_scaling)

        curr  = int(image_meta['curr_time_org'])
        discard = max(0, curr - self.end_t)
        train_duration = train_duration - discard

        delta_t        = int(self.delta_t * time_scaling)
        stride_t       = int(self.stride_t * time_scaling)
        num_frames     = int(math.ceil(train_duration / stride_t))

        # split data into sub-packets
        times_evt = events[:,0].astype(int)
        times_img = image_t.astype(int)
        times_lbl = label_t.astype(int)
        segment_indices_evt = times_evt // stride_t
        segment_indices_img = times_img // stride_t
        segment_indices_lbl = times_lbl // stride_t

        event_splits, segment_indices_evt = self._split_by_indices(events, segment_indices_evt)

        backet_evt = DataBacket(num=num_frames)
        for event_data, seg_idx in zip(event_splits, segment_indices_evt):
            if seg_idx < 0 or seg_idx >= num_frames:
                continue
            backet_evt.append(seg_idx, event_data)

        backet_img = DataBacket(num=num_frames)
        for img, seg_idx in zip(image, segment_indices_img):
            backet_img.append(seg_idx, img)

        backet_lbl = DataBacket(num=num_frames)
        for lbl, seg_idx in zip(label, segment_indices_lbl):
            backet_lbl.append(seg_idx, lbl)

        if self.use_nearest_label:
            backet_lbl.pad_blank_backet()
        if self.use_nearest_image:
            backet_img.pad_blank_backet(direction='forward')

        event_streams = backet_evt.concat(axis=0)    # Tensor (L, 4)
        image_streams = backet_img.latest()
        label_streams = backet_lbl.latest()

        meta_streams = []
        for i in range(len(backet_evt)):
            curr_time_org = (image_meta['curr_time_org'] - image_meta['curr_time_crop']) + stride_t * (i + 1)
            curr_time_crop = stride_t * (i + 1)
            image_meta = self._update_image_meta(image_meta, fpath_evt, curr_time_org, curr_time_crop, stride_t, time_scaling)
            meta = {
                'image_meta': image_meta,
                'label_meta': label_meta,
            }
            meta_streams.append(meta)

        for i, seg_idx in enumerate(segment_indices_img):
            meta_streams[seg_idx]['image_meta']['image_indices'] = image_indices[i]
        for i, seg_idx in enumerate(segment_indices_lbl):
            meta_streams[seg_idx]['image_meta']['depth_indices'] = label_indices[i]

        if delta_t != stride_t:
            event_streams, image_streams, label_streams, meta_streams = self.merge_streams(event_streams, image_streams, label_streams, meta_streams, delta_t, stride_t)

        data_streams, target_streams = [], []
        for evt, img, lbl in zip(event_streams, image_streams, label_streams):
            evt = torch.from_numpy(evt)
            img = torch.from_numpy(img).float() if len(img) > 0 else None
            lbl = torch.from_numpy(lbl).float() if len(lbl) > 0 else None

            data_streams.append({'events': evt, 'images': img})
            target_streams.append({'depth': lbl})

        assert len(data_streams) == len(target_streams)
        assert len(data_streams) == len(meta_streams)

        return data_streams, target_streams, meta_streams

    def merge_streams(self, event_streams, image_streams, label_streams, meta_streams, delta_t, stride_t):
        N = len(event_streams)
        l = int(delta_t // stride_t)
        event_streams = [ self._merge_events(event_streams[i-l:i]) for i in range(l, N) ]
        image_streams = [ image_streams[i] for i in range(l, N)]
        label_streams = [ label_streams[i] for i in range(l, N)]
        meta_streams  = [  meta_streams[i] for i in range(l, N)]
        for meta in meta_streams:
            meta['image_meta']['delta_t'] = stride_t * l
            meta['image_meta']['stride_t'] = stride_t
        return event_streams, image_streams, label_streams, meta_streams

    def _merge_events(self, list_events):
        list_events = [ events for events in list_events if len(events) > 0 ]
        if len(list_events) == 0:
            return np.array([], dtype=np.float32)
        else:
            return np.concatenate(list_events, axis=0)

class EventFrameStream(EventPacketStream):
    def __init__(self, *args, event_repr=None, frame_transform=None, **kwargs):
        super().__init__(*args, **kwargs)
        assert event_repr is not None
        assert frame_transform is not None
        self.event_repr = build_eventrepr(event_repr)
        self.frame_transform = frame_transform

    def __getitem__(self, index):
        data_streams, target_streams, meta_streams = super().__getitem__(index)

        out_data_streams, out_target_streams, out_meta_streams = [], [], []
        for idx, (data, target, meta_data) in enumerate(zip(data_streams, target_streams, meta_streams)):
            repeat = idx != 0
            events = data['events']
            image = data['images']
            label = target['depth']
            image_meta = meta_data['image_meta']

            ev_image, image_meta = self.event_repr(events, image_meta)
            if self.frame_transform is not None:
                ev_image, image, label, image_meta = self.frame_transform(ev_image, image, label, image_meta, types=['image', 'image', 'image', 'meta'], repeat=repeat)

            assert ev_image.shape[1] == image_meta['height'] and ev_image.shape[2] == image_meta['width']

            meta_data['image_meta'] = image_meta

            out_data_streams.append({'events': ev_image, 'images': image})
            out_target_streams.append({'depth': label})
            out_meta_streams.append(meta_data)

        return out_data_streams, out_target_streams, out_meta_streams

class DataBacket(object):
    def __init__(self, num=1):
        self._backet = [ list() for _ in range(num) ]

    def append(self, idx, data):
        if idx >= len(self._backet):
            num_append = idx - len(self._backet) + 1
            blank_backets = [ list() for _ in range(num_append) ]
            self._backet += blank_backets
        self._backet[idx].append(data)

    def _backet_dtype(self):
        for contents in self._backet:
            if len(contents) > 0:
                return contents[0].dtype

    def concat(self, axis):
        dtype = self._backet_dtype()
        output = []
        for contents in self._backet:
            if len(contents) == 0:
                output.append(np.array([], dtype=dtype))
            else:
                output.append(np.concatenate(contents, axis=axis))
        return output

    def stack(self, axis=0):
        dtype = self._backet_dtype()
        output = []
        for contents in self._backet:
            if len(contents) == 0:
                output.append(np.array([], dtype=dtype))
            else:
                output.append(np.stack(contents, axis=axis))
        return output

    def latest(self):
        dtype = self._backet_dtype()
        output = []
        for contents in self._backet:
            if len(contents) == 0:
                output.append(np.array([], dtype=dtype))
            else:
                output.append(contents[-1])
        return output

    def expand(self, length):
        if length > len(self._backet):
            num_append = length - len(self._backet)
            blank_backets = [ list() for _ in range(num_append) ]
            self._backet += blank_backets

    def pad_blank_backet(self, direction='forward'):
        if direction == 'forward':
            self._pad_forward()
            self._pad_backward()
        elif direction == 'backward':
            self._pad_backward()
            self._pad_forward()

    def blank_backet_as_none(self):
        for i in range(len(self._backet)):
            if len(self._backet[i]) == 0:
                self._backet[i] = None

    def _pad_forward(self):
        pad = None
        for i, backet in enumerate(self._backet):
            if len(backet) > 0:
                pad = backet
            elif pad is not None:
                self._backet[i] = copy.deepcopy(pad)
            else:
                pass

    def _pad_backward(self):
        pad = None
        for i, backet in enumerate(reversed(self._backet)):
            if len(backet) > 0:
                pad = backet
            elif pad is not None:
                self._backet[len(self._backet)-1-i] = copy.deepcopy(pad)
            else:
                pass

    @property
    def data(self):
        return self._backet

    def __len__(self):
        return len(self._backet)


