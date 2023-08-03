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
import hdf5plugin
from torch.utils import data

from hmnet.utils.common import get_list
from hmnet.models.base.event_repr.builder import build_eventrepr

#import torch.multiprocessing
#torch.multiprocessing.set_sharing_strategy('file_system')

WIDTH = 640
HEIGHT = 440    # 480 original

RGB_MEAN = [78.32271813, 82.63940533, 86.36723003]
RGB_STD  = [68.91888093, 73.94419327, 79.91293315]

class EventPacket(data.Dataset):
    def __init__(self, fpath_evt_lst=None, fpath_image_lst=None, fpath_label_lst=None, base_path='', fpath_meta=None, fpath_video_duration=None, train_duration=5e4,
                 fpath_sampling=None, sampling='random', start_index_aug_method='center', start_index_aug_ratio=1., sampling_stride=-1,
                 random_time_scaling=False, min_time_scale=0.5, max_time_scale=2.0,
                 event_transform=None, output_type=None,
                 rgb_mean=RGB_MEAN, rgb_std=RGB_STD, skip_image_loading=False,
                 max_events_per_packet=-1, downsample_packet_length=None, ignore_index=255):

        assert sampling in ('random', 'file', 'label', 'regular')

        self.base_path = base_path
        self.sampling = sampling
        self.ev_meta = pkl.load(open(fpath_meta, 'rb'))
        self.video_durations = pd.read_csv(fpath_video_duration, index_col=0)
        self.skip_ts = 0
        self.ignore_index = ignore_index
        self.event_transform = event_transform
        self.output_type = output_type
        self.rgb_mean = np.array(rgb_mean)[:,None,None]
        self.rgb_std = np.array(rgb_std)[:,None,None]
        self.skip_image_loading = skip_image_loading
        self.max_events_per_packet = max_events_per_packet
        self.downsample_packet_length = downsample_packet_length or train_duration
        self.start_index_aug_method = start_index_aug_method
        self.start_index_aug_ratio = start_index_aug_ratio
        self.train_duration = int(train_duration)
        self.random_time_scaling = random_time_scaling
        self.min_time_scale = min_time_scale
        self.max_time_scale = max_time_scale

        self.list_fpath_evt = get_list(fpath_evt_lst, ext=None)
        self.list_fpath_image = get_list(fpath_image_lst, ext=None)
        self.list_fpath_label = get_list(fpath_label_lst, ext=None)
        assert len(self.list_fpath_evt) == len(self.list_fpath_image)
        assert len(self.list_fpath_evt) == len(self.list_fpath_label)

        self.segment_ranges = []
        for fpath_evt in self.list_fpath_evt:
            name = fpath_evt.split('/')[-1]
            start = int(self.video_durations.loc[name, 'start'] // 1000)
            end = int(self.video_durations.loc[name, 'end'] // 1000)
            self.segment_ranges.append([start, end])

        if sampling == 'file':
            sampling_schedule = pkl.load(open(fpath_sampling, 'rb'))
            self.sampling_timings = sampling_schedule['timings']
            self.list_fpath_evt   = sampling_schedule['fpath_evt']
            self.list_fpath_image = sampling_schedule['fpath_image']
            self.list_fpath_label = sampling_schedule['fpath_label']
            self.total_seq = len(self.sampling_timings)
        elif sampling == 'label':
            self.sampling_timings = []
            for ifile, fpath_lbl in enumerate(self.list_fpath_label):
                fpath_lbl = self._get_path(fpath_lbl)
                with h5py.File(fpath_lbl) as fp:
                    ts = fp['data_ts'][...]
                seg_indices = np.unique(ts // 1000).tolist()
                self.sampling_timings += [ (ifile, seg_index) for seg_index in seg_indices ]
            self.total_seq = len(self.sampling_timings)
        elif sampling == 'regular':
            self.sampling_timings = []
            sampling_stride = sampling_stride if sampling_stride > 0 else train_duration
            seg_stride = int(sampling_stride // 1000)
            for ifile, (start, end) in enumerate(self.segment_ranges):
                self.sampling_timings += [ (ifile, seg_index) for seg_index in range(start,end,seg_stride) ]
            self.total_seq = len(self.sampling_timings)
        elif sampling == 'random':
            self.total_seq = 0
            sampling_stride = sampling_stride if sampling_stride > 0 else train_duration
            seg_stride = int(sampling_stride // 1000)
            for start, end in self.segment_ranges:
                self.total_seq += int((end - start) // seg_stride)

        self._image_meta = {
            'width': WIDTH,
            'height': HEIGHT,
        }

    def _get_path(self, filename):
        if filename[:2] == './':
            filename = filename[2:]
        return self.base_path + '/' + filename

    def __len__(self):
        return self.total_seq

    def __getitem__(self, index):
        events, _, image, _, label, meta_data, image_path, label_path = self.getdata(index)

        events = torch.from_numpy(events)
        image = torch.from_numpy(image).float() if len(image) > 0 else None
        label = torch.from_numpy(label).long() if len(label) > 0 else None

        data = { 'events': events, 'images': image }
        target = { 'labels': label }
        meta_data['image_meta']['image_path'] = image_path
        meta_data['image_meta']['label_path'] = label_path

        return data, target, meta_data

    def getdata(self, index, keep_latest_labels=True, keep_latest_images=True, skip_ts=0):
        if self.random_time_scaling:
            time_scaling = random.uniform(self.min_time_scale, self.max_time_scale)
        else:
            time_scaling = 1

        train_duration = int(self.train_duration * time_scaling)

        fpath_evt, fpath_image, fpath_label, base_time, ev_range = self._choose_file_and_time(index, train_duration)
        events, image_t, image, label_t, label, image_path, label_path = self._load(fpath_evt, fpath_image, fpath_label, base_time, train_duration, ev_range)

        if self.max_events_per_packet > 0:
            events = self._event_downsample(events, self.max_events_per_packet, self.downsample_packet_length)

        curr_time_org = base_time + train_duration
        curr_time_crop = train_duration
        image_meta = self._update_image_meta(self._image_meta, fpath_evt, curr_time_org, curr_time_crop, train_duration, time_scaling)

        # keep latest labels/images and discard others
        if keep_latest_labels:
            label_t, label, label_path = self._keep_latest_labels(label_t, label, label_path)
        if keep_latest_images:
            image_t, image, image_path = self._keep_latest_images(image_t, image, image_path)

        if self.event_transform is not None:
            event_dict = { 'events': events }
            event_dict, label, image, image_meta = self.event_transform(event_dict, label, image, image_meta, types=['event', 'mask', 'image', 'meta'])
            events = event_dict['events']

        # preprocess rgb image
        if len(image) > 0:
            image = (image - self.rgb_mean) / self.rgb_std

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
            'label_meta': dict(ignore_index=self.ignore_index),
        }

        return events, image_t, image, label_t, label, meta_data, image_path, label_path


    def _load(self, fpath_evt, fpath_image, fpath_label, time, duration, ev_range):
        # load events
        with h5py.File(fpath_evt) as fp:
            data = fp['events']
            st, ed = ev_range
            events = data[st:ed]

        # load label
        label, label_t, label_indices = self._load_images_from_hdf5(fpath_label, time + self.skip_ts, duration - self.skip_ts)

        # load image
        if self.skip_image_loading:
            image_t = np.array([], dtype=label_t.dtype)
            image, image_indices = [], []
        else:
            image, image_t, image_indices = self._load_images_from_hdf5(fpath_image, time, duration)

        events, image_t, label_t = self._subtract_base_time(events, image_t, label_t, time)

        return events, image_t, image, label_t, label, image_indices, label_indices

    def _load_images_from_hdf5(self, fpath_image_h5, time, duration):
        with h5py.File(fpath_image_h5) as fp:
            ts = fp['data_ts'][...]
            mask = (ts >= time) & (ts < time + duration )
            image = fp['data'][mask]
        image_t = ts[mask]
        indices = np.flatnonzero(mask)
        return image, image_t, indices

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

    def _choose_file_and_time(self, index, train_duration):
        nseg_per_packet = int(train_duration / 1000)

        if self.sampling == 'random':
            ifile = random.randint(1, len(self.list_fpath_evt)-1)
            start_seg_index, end_seg_index = self.segment_ranges[ifile]
            seg_index = random.randint(start_seg_index, end_seg_index - nseg_per_packet)
        elif self.sampling in ('file', 'label', 'regular'):
            ifile, seg_index = self.sampling_timings[index]

        start_seg_index, end_seg_index = self.segment_ranges[ifile]
        seg_index, is_success = self._augment_index(seg_index, self.start_index_aug_method, self.start_index_aug_ratio, nseg_per_packet, start_seg_index, end_seg_index)
        if not is_success and self.sampling == 'file':
            print('Desired segment cannnot be cropped. Retrying with another index.')
            index = random.randrange(len(self.sampling_timings))
            return self._choose_file_and_time(index, train_duration)
        elif not is_success:
            print('Desired segment cannnot be cropped. Cropping nearest segment.')

        fpath_evt = self._get_path(self.list_fpath_evt[ifile])
        fpath_image = self._get_path(self.list_fpath_image[ifile])
        fpath_label = self._get_path(self.list_fpath_label[ifile])

        ev_index, ev_count = self.ev_meta[fpath_evt.split('/')[-1]]
        ev_i = ev_index[seg_index]
        ev_c = ev_count[seg_index: seg_index + nseg_per_packet].sum()
        ev_range = (ev_i, ev_i + ev_c)
        time = seg_index * 1000

        return fpath_evt, fpath_image, fpath_label, time, ev_range

    def _update_image_meta(self, image_meta, fpath_evt, curr_time_org, curr_time_crop, delta_t, time_scaling):
        image_meta = image_meta.copy()
        image_meta['filename'] = fpath_evt
        image_meta['ori_filename'] = fpath_evt.split('/')[-1]
        image_meta['curr_time_org'] = curr_time_org
        image_meta['curr_time_crop'] = curr_time_crop
        image_meta['delta_t'] = delta_t
        image_meta['stride_t'] = delta_t
        image_meta['time_scaling'] = time_scaling
        image_meta['image_path'] = None
        image_meta['label_path'] = None
        return image_meta

    def _keep_latest_labels(self, label_t, label, label_path):
        if len(label) == 0:
            return label_t, label, None, None
        return label_t[-1], label[-1], label_path[-1]

    def _keep_latest_images(self, image_t, image, image_path):
        if len(image) == 0:
            return image_t, image, None
        return image_t[-1], image[-1], image_path[-1]

    def _bind(self, events):
        t_evt, x_evt, y_evt, p_evt = events['t'], events['x'], events['y'], events['p']
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
        events, _, image, _, label, meta_data, image_path, label_path = self.getdata(index)

        events = torch.from_numpy(events)
        image = torch.from_numpy(image).float() if len(image) > 0 else None
        label = torch.from_numpy(label).long() if len(label) > 0 else None

        image_meta = meta_data['image_meta']
        ev_image, image_meta = self.event_repr(events, image_meta)
        if self.frame_transform is not None:
            ev_image, image, label, image_meta = self.frame_transform(ev_image, image, label, image_meta, types=['image', 'image', 'mask', 'meta'])
        assert ev_image.shape[1] == image_meta['height'] and ev_image.shape[2] == image_meta['width']

        data = {'events': ev_image, 'images': image}
        target = {'labels': label}
        image_meta['image_path'] = image_path
        image_meta['label_path'] = label_path
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
        events, image_t, image, label_t, label, meta_data, image_path, label_path = self.getdata(index, keep_latest_labels=False, keep_latest_images=False, skip_ts=self.skip_ts)

        label_meta   = meta_data['label_meta']
        image_meta   = meta_data['image_meta']
        time_scaling = image_meta['time_scaling']
        fpath_evt    = image_meta['filename']

        train_duration = int(self.train_duration * time_scaling)

        filename = image_meta['ori_filename']
        end   = int(self.video_durations.loc[filename, 'end'])
        curr  = int(image_meta['curr_time_org'])
        discard = max(0, curr - end)
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
            backet_img.pad_blank_backet(direction='backward')

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
            meta_streams[seg_idx]['image_meta']['image_path'] = image_path[i]
        for i, seg_idx in enumerate(segment_indices_lbl):
            meta_streams[seg_idx]['image_meta']['label_path'] = label_path[i]

        if delta_t != stride_t:
            event_streams, image_streams, label_streams, meta_streams = self.merge_streams(event_streams, image_streams, label_streams, meta_streams, delta_t, stride_t)

        data_streams, target_streams = [], []
        for evt, img, lbl in zip(event_streams, image_streams, label_streams):
            evt = torch.from_numpy(evt)
            img = torch.from_numpy(img).float() if len(img) > 0 else None
            lbl = torch.from_numpy(lbl).long() if len(lbl) > 0 else None

            data_streams.append({'events': evt, 'images': img})
            target_streams.append({'labels': lbl})

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
            label = target['labels']
            image_meta = meta_data['image_meta']

            ev_image, image_meta = self.event_repr(events, image_meta)
            if self.frame_transform is not None:
                ev_image, image, label, image_meta = self.frame_transform(ev_image, image, label, image_meta, types=['image', 'image', 'mask', 'meta'], repeat=repeat)

            assert ev_image.shape[1] == image_meta['height'] and ev_image.shape[2] == image_meta['width']

            meta_data['image_meta'] = image_meta

            out_data_streams.append({'events': ev_image, 'images': image})
            out_target_streams.append({'labels': label})
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

    def concat(self, axis, dtype=None):
        if dtype is None:
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


