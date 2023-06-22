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

import random
import math
import numpy as np
import numbers
import collections
from PIL import Image, ImageOps
from operator import xor
#import faiss
try:
    import cv2
except:
    print('Warning: cv2 cannot be found at transform. Using transform with OpenCV backend will raise an error.')
    # dummy class
    class cv2:
        INTER_NEAREST = 0
        INTER_LINEAR = 0
        INTER_CUBIC = 0
        INTER_LANCZOS4 = 0

import torch
from torchvision import transforms
import torchvision.transforms.functional as TF

class TransformBase(object):
    def __init__(self):
        self.forward_params = []
        self.backward_params = []

    def _input_check(self, datas, types, need_image_size):
        assert ('image' in types) or ('mask' in types) or ('meta' in types) or (not need_image_size)
        assert len(datas) == len(types)
        for type in types:
            assert type in ('image', 'mask', 'bbox', 'point', 'event', 'meta')

    def _transform(self, datas, types, trans):
        self._input_check(datas, types, need_image_size=True)

        # get params
        image_size = self._get_image_size(datas, types)
        if hasattr(trans, 'get_param_using_data'):
            param, back_param = trans.get_param_using_data(image_size, datas, types)
        else:
            param, back_param = trans.get_param(image_size)

        # set input image_size
        param['image_size'] = image_size

        # apply transform
        datas = self._transform_one_step(datas, types, param)

        # set output image_size for backward
        image_size = self._get_image_size(datas, types)
        back_param['image_size'] = image_size

        # record params
        self.forward_params.append(param)
        self.backward_params.append(back_param)

        return datas

    def backward(self, *datas, types=[]):
        self._input_check(datas, types, need_image_size=False)
        datas, is_nps = self.to_input(datas)

        for param in reversed(self.backward_params):
            datas = self._transform_one_step(datas, types, param)

        datas = self.to_output(datas, is_nps)

        if len(datas) == 1:
            datas = datas[0]

        return datas

    def replay(self, *datas, types=[]):
        self._input_check(datas, types, need_image_size=False)
        datas, is_nps = self.to_input(datas)

        for param in self.forward_params:
            datas = self._transform_one_step(datas, types, param)

        datas = self.to_output(datas, is_nps)

        if len(datas) == 1:
            datas = datas[0]

        return datas

    def _to_dict(self, data, key):
        if isinstance(data, torch.Tensor):
            return {key: data}, True
        else:
            return data, False

    def _from_dict(self, data, key, tensor_out):
        if tensor_out:
            return data[key]
        else:
            return data

    def _transform_one_step(self, datas, types, param):
        param = param.copy()
        trans = param.pop('method')
        trans.param_check(param)
        output = []
        for data, type in zip(datas, types):
            if data is None or len(data) == 0:
                output.append(data)
                continue
            if type == 'image':
                if data.ndim == 4:    # in case of batch data
                    output.append(self._transform_image_batch(trans.image_transform, data, param))
                else:
                    output.append(trans.image_transform(data, param))
            elif type == 'mask':
                if data.ndim == 4:    # in case of batch data
                    output.append(self._transform_image_batch(trans.mask_transform, data, param))
                else:
                    output.append(trans.mask_transform(data, param))
            elif type == 'bbox':
                data, tensor_out = self._to_dict(data, 'bboxes')
                assert isinstance(data, dict) and 'bboxes' in data.keys()
                out = trans.bbox_transform(data.copy(), param)
                output.append(self._from_dict(out, 'bboxes', tensor_out))
            elif type == 'point':
                data, tensor_out = self._to_dict(data, 'points')
                assert isinstance(data, dict) and 'points' in data.keys()
                out = trans.point_transform(data.copy(), param)
                output.append(self._from_dict(out, 'points', tensor_out))
            elif type == 'event':
                data, tensor_out = self._to_dict(data, 'events')
                assert isinstance(data, dict) and 'events' in data.keys()
                out = trans.event_transform(data.copy(), param)
                output.append(self._from_dict(out, 'events', tensor_out))
            elif type == 'meta':
                assert isinstance(data, dict) and 'width' in data.keys() and 'height' in data.keys()
                output.append(trans.meta_transform(data.copy(), param))
        return tuple(output)

    def _transform_image_batch(self, function, batch_data, param):
        output = [ function(data, param) for data in batch_data ]
        return torch.stack(output)

    def _to_in(self, data):
        is_np = False
        if isinstance(data, dict):
            for k, v in data.items():
                is_np = isinstance(v, np.ndarray)
            data = { k: torch.from_numpy(v) if isinstance(v, np.ndarray) else v for k, v in data.items() }
        elif isinstance(data, np.ndarray):
            data = torch.from_numpy(data)
            is_np = True
        return data, is_np

    def _to_out(self, data, is_np):
        if isinstance(data, dict):
            data = { k: v.numpy() if is_np else v for k, v in data.items() }
        elif isinstance(data, torch.Tensor) and is_np:
            data = data.numpy()
        return data

    def to_input(self, datas):
        inputs = []
        is_nps = []
        for data in datas:
            data, is_np = self._to_in(data)
            inputs.append(data)
            is_nps.append(is_np)
        return inputs, is_nps

    def to_output(self, datas, is_nps):
        outputs = [ self._to_out(data, is_np) for data, is_np in zip(datas, is_nps) ]
        return outputs

    def _get_image_size(self, datas, types):
        for data, type in zip(datas, types):
            if type in ('image', 'mask') and data is not None and not isinstance(data, (list, tuple)):
                H, W = data.shape[-2:]
                return H, W
            elif type == 'meta':
                return data['height'], data['width']

class Compose(TransformBase):
    def __init__(self, list_transform):
        super().__init__()
        self.list_transform = list_transform

    def __call__(self, *datas, types=[], repeat=False):
        if len(self.list_transform) == 0:
            return datas

        self._input_check(datas, types, need_image_size=True)

        datas, is_nps = self.to_input(datas)

        if repeat and len(self.forward_params) > 0:
            return self.replay(*datas, types=types)

        self.forward_params = []
        self.backward_params = []
        for trans in self.list_transform:
            datas = self._transform(datas, types=types, trans=trans)

        datas = self.to_output(datas, is_nps)

        if len(datas) == 1:
            datas = datas[0]

        return datas

class TransformFunctions(TransformBase):
    def _transform(self, datas, types, trans):
        datas, is_nps = self.to_input(datas)
        datas = super()._transform(datas, types, trans)
        datas = self.to_output(datas, is_nps)
        if len(datas) == 1:
            datas = datas[0]
        return datas

    def numpy_to_tensor(self, *datas, types=[], input_band_order="CHW"):
        trans = NumpyToTensor(input_band_order)
        return self._transform(datas, types, trans)

    def resize(self, *datas, types=[], size=None, scale_factor=None, image_resampling='BILINEAR', mask_resampling='NEAREST', backend='Torch'):
        trans = Resize(size=size, scale_factor=scale_factor, image_resampling=image_resampling, mask_resampling=mask_resampling, backend=backend)
        return self._transform(datas, types, trans)

    def random_resize(self, *datas, types=[], scale_min=0.5, scale_range=15, image_resampling='BILINEAR', mask_resampling='NEAREST', backend='Torch'):
        trans = RandomResize(scale_min=scale_min, scale_range=scale_range, image_resampling=image_resampling, mask_resampling=mask_resampling, backend=backend)
        return self._transform(datas, types, trans)

    def resize_inside(self, *datas, types=[], size=None, image_resampling='BILINEAR', mask_resampling='NEAREST', backend='Torch'):
        trans = ResizeInside(size=size, image_resampling=image_resampling, mask_resampling=mask_resampling, backend=backend)
        return self._transform(datas, types, trans)

    def crop(self, *datas, types=[], crop_size=None, crop_position=None, const_image=0, const_mask=-1, bbox_filter_by_center=False, backend='Torch'):
        trans = Crop(crop_size=crop_size, crop_position=crop_position, const_image=const_image, const_mask=const_mask, bbox_filter_by_center=bbox_filter_by_center, backend=backend)
        return self._transform(datas, types, trans)

    def crop_center(self, *datas, types=[], crop_size=None, const_image=0, const_mask=-1, bbox_filter_by_center=False, backend='Torch'):
        trans = CropCenter(crop_size=crop_size, const_image=const_image, const_mask=const_mask, bbox_filter_by_center=bbox_filter_by_center, backend=backend)
        return self._transform(datas, types, trans)

    def random_crop(self, *datas, types=[], crop_size=None, const_image=0, const_mask=-1, bbox_filter_by_center=False, backend='Torch'):
        trans = RandomCrop(crop_size=crop_size, const_image=const_image, const_mask=const_mask, bbox_filter_by_center=bbox_filter_by_center, backend=backend)
        return self._transform(datas, types, trans)

    def iou_random_crop(self, *datas, types=[], min_crop_scale=0.3, min_ar=0.5, max_ar=2.0, iou_range=[0.1,0.3,0.5,0.7,0.9], iou_minmax='min',
                             const_image=0, const_mask=-1, bbox_filter_by_center=False, backend='Torch'):
        trans = IoURandomCrop(min_crop_scale=min_crop_scale, min_ar=min_ar, max_ar=max_ar, iou_range=iou_range, iou_minmax=iou_minmax,
                             const_image=const_image, const_mask=const_mask, bbox_filter_by_center=bbox_filter_by_center, backend=backend)
        return self._transform(datas, types, trans)

    def padding(self, *datas, types=[], size=None, padding=None, halign='center', valign='center', const_image=0, const_mask=-1, padding_mode='constant', backend='Torch'):
        trans = Padding(size=size, padding=padding, halign=halign, valign=valign, const_image=const_image, const_mask=const_mask, padding_mode=padding_mode, backend=backend)
        return self._transform(datas, types, trans)

    def flip(self, *datas, types=[], direction='H'):
        trans = Flip(direction=direction)
        return self._transform(datas, types, trans)

    def random_flip(self, *datas, types=[], direction='H', prob=0.5):
        trans = RandomFlip(direction=direction, prob=prob)
        return self._transform(datas, types, trans)

    def rotate90N(self, *datas, types=[], rot=0):
        trans = Rotate90N(rot=rot)
        return self._transform(datas, types, trans)

    def random_rotate90N(self, *datas, types=[]):
        trans = RandomRotate90N()
        return self._transform(datas, types, trans)

    def rotate(self, *datas, types=[], rot=0, image_resampling='BICUBIC', mask_resampling='NEAREST', const_image=0, const_mask=-1, backend='PIL'):
        trans = Rotate(rot=rot, image_resampling=image_resampling, mask_resampling=mask_resampling, const_image=const_image, const_mask=const_mask, backend=backend)
        return self._transform(datas, types, trans)

    def random_rotate(self, *datas, types=[], min_rot=0, max_rot=20, image_resampling='BICUBIC', mask_resampling='NEAREST', const_image=0, const_mask=-1, backend='PIL'):
        trans = RandomRotate(min_rot=min_rot, max_rot=max_rot, image_resampling=image_resampling, mask_resampling=mask_resampling, const_image=const_image, const_mask=const_mask, backend=backend)
        return self._transform(datas, types, trans)

class NumpyToTensor(object):
    def __init__(self, input_band_order="CHW"):
        self._input_band_order = input_band_order

    def get_param(self, image_size):
        forward_param  = {'method': NumpyToTensor, 'mode':  'forward', 'input_order': self._input_band_order}
        backward_param = {'method': NumpyToTensor, 'mode': 'backward', 'input_order': self._input_band_order}
        return forward_param, backward_param

    @classmethod
    def param_check(cls, param):
        required = ('image_size', 'mode', 'input_order')
        assert set(required) == set(param.keys())

    @classmethod
    def image_transform(cls, image, param):
        mode = param['mode']
        input_order = param['input_order']
        if mode == 'forward':
            image = torch.from_numpy(image)
            image = cls._input_to_chw(image, input_band_order)
            image = image.float()
        else:
            image = cls._chw_to_input(image, input_band_order)
            image = image.numpy()
        return image

    @classmethod
    def mask_transform(cls, mask, param):
        return cls._ann_transform(mask, param['mode'])
    @classmethod
    def bbox_transform(cls, bbox_dict, param):
        output = {}
        for key, data in bbox_dict.items():
            output[key] = cls._ann_transform(data, param['mode'])
        return output
    @classmethod
    def point_transform(cls, point_dict, param):
        output = {}
        for key, data in point_dict.items():
            output[key] = cls._ann_transform(data, param['mode'])
        return output
    @classmethod
    def event_transform(cls, event_dict, param):
        output = {}
        for key, data in event_dict.items():
            output[key] = cls._ann_transform(data, param['mode'])
        return output
    @classmethod
    def meta_transform(cls, meta, param):
        return meta

    @classmethod
    def _ann_transform(cls, ann, mode):
        if mode == 'forward':
            ann = torch.from_numpy(ann)
            ann = ann.long()
        else:
            ann = ann.numpy()
        return ann

    @classmethod
    def _input_to_chw(cls, tensor, input_order):
        if input_order == 'HWC':
            tensor = tensor.permute(2, 0, 1)
        return tensor

    @classmethod
    def _chw_to_input(cls, tensor, input_order):
        if input_order == 'HWC':
            tensor = tensor.permute(1, 2, 0)
        return tensor

class PhotometricTransform(object):
    @classmethod
    def image_transform(cls, image, param):
        raise NotImplementedError

    @classmethod
    def mask_transform(cls, mask, param):
        return mask

    @classmethod
    def bbox_transform(cls, bbox_dict, param):
        return bbox_dict

    @classmethod
    def point_transform(cls, point_dict, param):
        return point_dict

    @classmethod
    def event_transform(cls, event_dict, param):
        return event_dict

    @classmethod
    def meta_transform(cls, meta, param):
        return meta

class Normalize(PhotometricTransform):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)[:,None,None]
        self.std = torch.tensor(std)[:,None,None]

    def get_param(self, image_size):
        forward_param  = {'method': Normalize, 'mean': self.mean, 'std': self.std}
        backward_param = {'method': Denormalize, 'mean': self.mean, 'std': self.std}

    @classmethod
    def image_transform(cls, image, param):
        mean = param['mean']
        std = param['std']
        return (image - mean) / std

class Denormalize(PhotometricTransform):
    def __init__(self, mean, std):
        self.mean = torch.tensor(mean)[:,None,None]
        self.std = torch.tensor(std)[:,None,None]

    def get_param(self, image_size):
        forward_param  = {'method': Denormalize, 'mean': self.mean, 'std': self.std}
        backward_param = {'method': Normalize, 'mean': self.mean, 'std': self.std}

    @classmethod
    def image_transform(cls, image, param):
        mean = param['mean']
        std = param['std']
        return image * std + mean

class BandFlip(PhotometricTransform):
    def __init__(self, flip_indices=None):
        self.flip_indices = flip_indices

    def get_param(self, image_size):
        forward_param  = {'method': BandFlip, 'flip_indices': self.flip_indices}
        back_flip_indices = np.argsort(self.flip_indices).tolist()
        backward_param = {'method': BandFlip, 'flip_indices': back_flip_indices}

    @classmethod
    def param_check(cls, param):
        required = ('flip_indices',)
        assert set(required) == set(param.keys())

    @classmethod
    def image_transform(cls, image, param):
        flip_indices = param['flip_indices']
        return image[flip_indices,:,:]

class RGB2Gray(PhotometricTransform):
    def get_param(self, image_size):
        forward_param  = {'method': RGB2Gray}
        backward_param = {'method': Gray2RGB}

    @classmethod
    def param_check(cls, param):
        required = set()
        assert set(required) == set(param.keys())

    @classmethod
    def image_transform(cls, image, param):
        assert image.shape[0] == 3
        R, G, B = image[0], image[1], image[2]
        Gray = 0.2989 * R + 0.5871 * G + 0.1140 * B
        return Gray[None,:,:]

class Gray2RGB(PhotometricTransform):
    @classmethod
    def param_check(cls, param):
        required = set()
        assert set(required) == set(param.keys())

    @classmethod
    def image_transform(cls, image, param):
        assert image.shape[0] == 1
        return image.repeat(3,1,1)

class Resize(object):
    # Resize the input to the given size, 'size' is a 2-element tuple or list in the order of (h, w).
    def __init__(self, size=None, scale_factor=None, image_resampling='BILINEAR', mask_resampling='NEAREST', event_downsampling='UNIFORM', event_upsampling='KNN', event_resampling='BILINEAR', backend='Torch'):
        assert xor(size is None, scale_factor is None)
        if size is not None:
            assert (isinstance(size, (list, tuple)) and len(size) == 2)
        self.size = size
        if isinstance(scale_factor, (list, tuple)):
            self.scale_factor = scale_factor
        else:
            self.scale_factor = [scale_factor] * 2
        self.image_resampling = image_resampling
        self.mask_resampling = mask_resampling
        self.event_downsampling = event_downsampling
        self.event_upsampling = event_upsampling
        self.event_resampling = event_resampling
        self.backend = backend

    def get_param(self, image_size):
        forward_param  = {'method': Resize, 'image_resampling': self.image_resampling, 'mask_resampling': self.mask_resampling,
                          'event_downsampling': self.event_downsampling, 'event_upsampling': self.event_upsampling, 'event_resampling': self.event_resampling, 'backend': self.backend}
        backward_param = {'method': Resize, 'image_resampling': self.image_resampling, 'mask_resampling': self.mask_resampling,
                          'event_downsampling': self.event_downsampling, 'event_upsampling': self.event_upsampling, 'event_resampling': self.event_resampling, 'backend': self.backend}

        if self.size is not None:
            H, W = self.size
            h, w = image_size
        else:
            sy, sx = self.scale_factor
            h, w = image_size
            H = int(round(h * sy))
            W = int(round(w * sx))
        forward_param.update( {'method': Resize, 'size': (H, W), 'scale_factor': (float(H)/h, float(W)/w)})
        backward_param.update({'method': Resize, 'size': (h, w), 'scale_factor': (float(h)/H, float(w)/W)})
        return forward_param, backward_param

    @classmethod
    def param_check(cls, param):
        required = ('image_size', 'size', 'scale_factor', 'image_resampling', 'mask_resampling', 'event_downsampling', 'event_upsampling', 'event_resampling', 'backend')
        assert set(required) == set(param.keys())

    @classmethod
    def image_transform(cls, image, param):
        return _resize(image, param['size'], param['image_resampling'], param['backend'])

    @classmethod
    def mask_transform(cls, mask, param):
        return _resize(mask, param['size'], param['mask_resampling'], param['backend'])

    @classmethod
    def bbox_transform(cls, bbox_dict, param):
        bboxes = bbox_dict.pop('bboxes')
        scale_y, scale_x = param['scale_factor']
        bboxes = bboxes * torch.tensor([scale_x, scale_y, scale_x, scale_y], dtype=torch.float, device=bboxes.device)
        bbox_dict['bboxes'] = bboxes.long()
        return bbox_dict

    @classmethod
    def point_transform(cls, point_dict, param):
        points = point_dict.pop('points')
        scale_y, scale_x = param['scale_factor']
        points = points * torch.tensor([scale_x, scale_y], dtype=torch.float, device=points.device)
        point_dict['points'] = points.long()
        return point_dict

    @classmethod
    def event_transform(cls, event_dict, param):
        if len(event_dict['events']) == 0:
            return event_dict

        scale_y, scale_x = param['scale_factor']
        event_downsampling = param['event_downsampling']
        event_upsampling = param['event_upsampling']
        event_resampling = param['event_resampling']
        assert event_resampling in ('BILINEAR', 'NEAREST', 'NONE')
        events = event_dict.pop('events')
        height, width = param['image_size']

        # downsample events
        if scale_y <= 1 and scale_x <= 1:
            events = _downsample_events(events, scale_x, scale_y, width, height, downsampling=event_downsampling, resampling=event_resampling)

        # upsample events
        elif scale_y >= 1 and scale_x >= 1:
            events = _upsample_events(events, scale_x, scale_y, width, height, k=10, knn=event_dict.get('knn', None), upsampling=event_upsampling, resampling=event_resampling)

        event_dict['events'] = events
        return event_dict

    @classmethod
    def meta_transform(cls, meta, param):
        scale_y, scale_x = param['scale_factor']
        width = meta['width']
        height = meta['height']
        width = int(round(width * scale_x))
        height = int(round(height * scale_y))

        meta['width'] = width
        meta['height'] = height
        return meta

class RandomResize(Resize):
    def __init__(self, scale_min, scale_range, scale_mult=1.0, image_resampling='BILINEAR', mask_resampling='NEAREST', event_downsampling='NONE', event_upsampling='NONE', event_resampling='NONE', backend='Torch'):
        self.scale_min = scale_min
        self.scale_range = scale_range
        self.scale_mult = scale_mult
        self.image_resampling = image_resampling
        self.mask_resampling = mask_resampling
        self.event_downsampling = event_downsampling
        self.event_upsampling = event_upsampling
        self.event_resampling = event_resampling
        self.backend = backend

    def get_param(self, image_size):
        forward_param  = {'method': Resize, 'image_resampling': self.image_resampling, 'mask_resampling': self.mask_resampling,
                          'event_downsampling': self.event_downsampling, 'event_upsampling': self.event_upsampling, 'event_resampling': self.event_resampling, 'backend': self.backend}
        backward_param = {'method': Resize, 'image_resampling': self.image_resampling, 'mask_resampling': self.mask_resampling,
                          'event_downsampling': self.event_downsampling, 'event_upsampling': self.event_upsampling, 'event_resampling': self.event_resampling, 'backend': self.backend}

        scale = self.scale_min + random.randint(0, self.scale_range) / 10.0
        scale = self.scale_mult * scale
        h, w = image_size
        H = int(round(h * scale))
        W = int(round(w * scale))
        forward_param.update( {'method': Resize, 'size': (H, W), 'scale_factor': (float(H)/h, float(W)/w)})
        backward_param.update({'method': Resize, 'size': (h, w), 'scale_factor': (float(h)/H, float(w)/W)})
        return forward_param, backward_param

class ResizeInside(Resize):
    def __init__(self, size, image_resampling='BILINEAR', mask_resampling='NEAREST', event_downsampling='UNIFORM', event_upsampling='KNN', event_resampling='BILINEAR', backend='Torch'):
        assert (isinstance(size, (list, tuple)) and len(size) == 2)
        self.size = size
        self.image_resampling = image_resampling
        self.mask_resampling = mask_resampling
        self.event_downsampling = event_downsampling
        self.event_upsampling = event_upsampling
        self.event_resampling = event_resampling
        self.backend = backend

    def get_param(self, image_size):
        forward_param  = {'method': Resize, 'image_resampling': self.image_resampling, 'mask_resampling': self.mask_resampling,
                          'event_downsampling': self.event_downsampling, 'event_upsampling': self.event_upsampling, 'event_resampling': self.event_resampling, 'backend': self.backend}
        backward_param = {'method': Resize, 'image_resampling': self.image_resampling, 'mask_resampling': self.mask_resampling,
                          'event_downsampling': self.event_downsampling, 'event_upsampling': self.event_upsampling, 'event_resampling': self.event_resampling, 'backend': self.backend}

        h, w = image_size
        H, W = self.size
        scale = min(float(H)/h, float(W)/w)
        H = int(round(h * scale))
        W = int(round(w * scale))
        forward_param.update( {'method': Resize, 'size': (H, W), 'scale_factor': (float(H)/h, float(W)/w)})
        backward_param.update({'method': Resize, 'size': (h, w), 'scale_factor': (float(h)/H, float(w)/W)})
        return forward_param, backward_param


class Crop(object):
    def __init__(self, crop_size, crop_position, const_image=0, const_mask=-1, bbox_filter_by_center=False, clip_border=True, backend='Torch'):
        self.crop_size = crop_size
        self.crop_position = crop_position
        self.const_image = const_image
        self.const_mask = const_mask
        self.bbox_filter_by_center = bbox_filter_by_center
        self.clip_border = clip_border
        self.backend = backend
        assert clip_border or bbox_filter_by_center

    def get_param(self, image_size):
        crop_h, crop_w = self.crop_size
        sty, stx = self.crop_position
        h, w = image_size
        assert stx + crop_w <= w
        assert sty + crop_h <= h

        pad_left = stx
        pad_right = w - (stx + crop_w)
        pad_top = sty
        pad_bottom = h - (sty + crop_h)
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        forward_param  = {'method': Crop, 'crop_size': (crop_h, crop_w), 'crop_position': (sty, stx), 'bbox_filter_by_center': self.bbox_filter_by_center, 'clip_border': self.clip_border}
        backward_param = {'method': Padding, 'padding': padding, 'const_image': self.const_image, 'const_mask': self.const_mask, 'padding_mode': 'constant', 'backend': self.backend}
        return forward_param, backward_param

    @classmethod
    def param_check(cls, param):
        required = ('image_size', 'crop_size', 'crop_position', 'bbox_filter_by_center', 'clip_border')
        assert set(required) == set(param.keys())

    @classmethod
    def image_transform(cls, image, param):
        crop_h, crop_w = param['crop_size']
        sty, stx = param['crop_position']
        return image[:,sty:sty+crop_h, stx:stx+crop_w]

    @classmethod
    def mask_transform(cls, mask, param):
        ndim = mask.ndim
        if mask.ndim == 2:
            mask = mask[None,:,:]

        crop_h, crop_w = param['crop_size']
        sty, stx = param['crop_position']
        mask = mask[:, sty:sty+crop_h, stx:stx+crop_w]

        if ndim == 2:
            mask = mask[0]
        return mask

    @classmethod
    def bbox_transform(cls, bbox_dict, param):
        bboxes = bbox_dict.pop('bboxes')
        crop_h, crop_w = param['crop_size']
        sty, stx = param['crop_position']
        bbox_filter_by_center = param['bbox_filter_by_center']
        clip_border = param['clip_border']

        bboxes = bboxes - torch.tensor([stx,sty,stx,sty], dtype=bboxes.dtype, device=bboxes.device)

        if bbox_filter_by_center:
            center_x = (bboxes[:,0] + bboxes[:,2]) / 2
            center_y = (bboxes[:,1] + bboxes[:,3]) / 2

        if clip_border:
            bboxes[:, 0::2] = torch.clip(bboxes[:, 0::2], 0, crop_w)
            bboxes[:, 1::2] = torch.clip(bboxes[:, 1::2], 0, crop_h)

        if bbox_filter_by_center:
            valid = (center_x >= 0) & (center_x < crop_w) & (center_y >= 0) & (center_y < crop_h)
        else:
            valid = (bboxes[:, 2] > bboxes[:, 0]) & (bboxes[:, 3] > bboxes[:, 1])

        output = {}
        for key, data in bbox_dict.items():
            output[key] = data[valid]
        output['bboxes'] = bboxes[valid, :]

        return output

    @classmethod
    def point_transform(cls, point_dict, param):
        points = point_dict.pop('points')
        crop_h, crop_w = param['crop_size']
        sty, stx = param['crop_position']
        points = points - torch.tensor([stx,sty], dtype=points.dtype, device=points.device)
        valid = (points[:, 0] >= 0) & (points[:, 0] < crop_w) & (points[:, 1] >= 0) & (points[:, 1] < crop_h)

        output = {}
        for key, data in point_dict.items():
            output[key] = data[valid]
        output['points'] = points[valid, :]

        return output

    @classmethod
    def event_transform(cls, event_dict, param):
        if len(event_dict['events']) == 0:
            return event_dict

        events = event_dict.pop('events')
        crop_h, crop_w = param['crop_size']
        sty, stx = param['crop_position']

        events[:,1] = events[:,1] - stx
        events[:,2] = events[:,2] - sty

        valid = (events[:, 1] >= 0) & (events[:, 1] < crop_w) & (events[:, 2] >= 0) & (events[:, 2] < crop_h)

        output = {}
        output['events'] = events[valid]
        if 'knn' in event_dict and event_dict['knn'] is not None:
            output['knn'] = event_dict['knn'][valid]
        elif 'knn' in event_dict:
            output['knn'] = event_dict['knn']

        return output

    @classmethod
    def meta_transform(cls, meta, param):
        crop_h, crop_w = param['crop_size']
        meta['width'] = crop_w
        meta['height'] = crop_h
        return meta

class CropCenter(Crop):
    def __init__(self, crop_size, const_image=0, const_mask=-1, bbox_filter_by_center=False, clip_border=True, backend='Torch'):
        self.crop_size = crop_size
        self.const_image = const_image
        self.const_mask = const_mask
        self.bbox_filter_by_center = bbox_filter_by_center
        self.clip_border = clip_border
        self.backend = backend
        assert clip_border or bbox_filter_by_center

    def get_param(self, image_size):
        crop_h, crop_w = self.crop_size
        img_h, img_w = image_size
        crop_h = min(crop_h, img_h)
        crop_w = min(crop_w, img_w)
        sty = int((img_h - crop_h) * 0.5)
        stx = int((img_w - crop_w) * 0.5)

        pad_left = stx
        pad_right = img_w - (stx + crop_w)
        pad_top = sty
        pad_bottom = img_h - (sty + crop_h)
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        forward_param  = {'method': Crop, 'crop_size': (crop_h, crop_w), 'crop_position': (sty, stx), 'bbox_filter_by_center': self.bbox_filter_by_center, 'clip_border': self.clip_border}
        backward_param = {'method': Padding, 'padding': padding, 'const_image': self.const_image, 'const_mask': self.const_mask, 'padding_mode': 'constant', 'backend': self.backend}
        return forward_param, backward_param

class RandomCrop(Crop):
    def __init__(self, crop_size, const_image=0, const_mask=-1, bbox_filter_by_center=False, clip_border=True, backend='Torch'):
        self.crop_size = crop_size
        self.const_image = const_image
        self.const_mask = const_mask
        self.bbox_filter_by_center = bbox_filter_by_center
        self.clip_border = clip_border
        self.backend = backend
        assert clip_border or bbox_filter_by_center

    def get_param(self, image_size):
        crop_h, crop_w = self.crop_size
        img_h, img_w = image_size
        sty = random.randint(0, img_h - crop_h)
        stx = random.randint(0, img_w - crop_w)

        pad_left = stx
        pad_right = img_w - (stx + crop_w)
        pad_top = sty
        pad_bottom = img_h - (sty + crop_h)
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        forward_param  = {'method': Crop, 'crop_size': (crop_h, crop_w), 'crop_position': (sty, stx), 'bbox_filter_by_center': self.bbox_filter_by_center, 'clip_border': self.clip_border}
        backward_param = {'method': Padding, 'padding': padding, 'const_image': self.const_image, 'const_mask': self.const_mask, 'padding_mode': 'constant', 'backend': self.backend}
        return forward_param, backward_param

class IoURandomCrop(Crop):
    def __init__(self, size=None, min_crop_scale=0.3, min_ar=0.5, max_ar=2.0, iou_range=[0.1,0.3,0.5,0.7,0.9], iou_minmax='min',
                 const_image=0, const_mask=-1, bbox_filter_by_center=False, clip_border=True, backend='Torch'):
        assert iou_minmax in ('min', 'max')
        self.size = size
        self.min_crop_scale = min_crop_scale
        self.min_ar = min_ar
        self.max_ar = max_ar
        self.iou_range = [0] + iou_range + [1]
        self.iou_minmax = iou_minmax
        self.const_image = const_image
        self.const_mask = const_mask
        self.bbox_filter_by_center = bbox_filter_by_center
        self.clip_border = clip_border
        self.backend = backend
        assert clip_border or bbox_filter_by_center

    def get_param_using_data(self, image_size, datas, types):
        assert types.count('bbox') == 1
        bbox_dict = datas[types.index('bbox')]
        bboxes = bbox_dict['bboxes']
        img_h, img_w = image_size
        bbox_areas = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])

        while True:
            bboxes = bboxes.clone()

            # choose iou threshold
            iou_threshold = random.choice(self.iou_range)

            if iou_threshold == 1 and self.size is None:
                crop_h, crop_w = img_h, img_w
                stx, sty = 0, 0
                break

            # IoU threshold > 0 cannot be met
            if len(bboxes) == 0:
                iou_threshold = 0

            # choose crop size
            if self.size is None:
                crop_h = int(img_h * random.uniform(self.min_crop_scale, 1))
                crop_w = int(img_w * random.uniform(self.min_crop_scale, 1))

                # check aspect ratio in desired range
                ar = float(crop_h) / crop_w
                if ar < self.min_ar or ar > self.max_ar:
                    continue
            else:
                crop_h, crop_w = self.size

            # choose crop position
            sty = random.randint(0, img_h - crop_h)
            stx = random.randint(0, img_w - crop_w)

            if len(bboxes) == 0 and iou_threshold == 0:
                break

            crop_area = crop_h * crop_w

            # calculate intersection
            bboxes = bboxes - torch.tensor([stx,sty,stx,sty], dtype=bboxes.dtype, device=bboxes.device)
            bboxes[:, 0::2] = torch.clip(bboxes[:, 0::2], 0, crop_w)
            bboxes[:, 1::2] = torch.clip(bboxes[:, 1::2], 0, crop_h)
            intersection = (bboxes[:,2] - bboxes[:,0]) * (bboxes[:,3] - bboxes[:,1])

            iou = intersection / (crop_area + bbox_areas - intersection)

            if self.iou_minmax == 'min':
                target_iou = iou.min()
            elif self.iou_minmax == 'max':
                target_iou = iou.max()

            if target_iou >= iou_threshold:
                break

        pad_left = stx
        pad_right = img_w - (stx + crop_w)
        pad_top = sty
        pad_bottom = img_h - (sty + crop_h)
        padding = (pad_left, pad_top, pad_right, pad_bottom)

        forward_param  = {'method': Crop, 'crop_size': (crop_h, crop_w), 'crop_position': (sty, stx), 'bbox_filter_by_center': self.bbox_filter_by_center, 'clip_border': self.clip_border}
        backward_param = {'method': Padding, 'padding': padding, 'const_image': self.const_image, 'const_mask': self.const_mask, 'padding_mode': 'constant', 'backend': self.backend}
        return forward_param, backward_param


class Padding(object):
    def __init__(self, size=None, padding=None, halign='center', valign='center', const_image=0, const_mask=-1, padding_mode='constant', backend='Torch'):
        assert halign in ('center', 'left', 'right')
        assert valign in ('center', 'top', 'bottom')
        assert (size is not None and padding is None) or (size is None and padding is not None)
        self.size = size
        self.padding = padding
        self.halign = halign
        self.valign = valign
        self.const_image = const_image
        self.const_mask = const_mask
        self.padding_mode = padding_mode
        self.backend = backend

    def _pad_size(self, pad, align):
        if align == 'center':
            p1 = int(pad / 2)
            p2 = pad - p1
        elif align in ('left', 'top'):
            p1, p2 = 0, pad
        elif align in ('right', 'bottom'):
            p1, p2 = pad, 0
        return p1, p2

    def get_param(self, image_size):
        if self.size is not None:
            H, W = self.size
            h, w = image_size
            pad_y = max((H - h), 0)
            pad_x = max((W - w), 0)
            pad_left, pad_right = self._pad_size(pad_x, self.halign)
            pad_top, pad_bottom = self._pad_size(pad_y, self.valign)
            padding = (pad_left, pad_top, pad_right, pad_bottom)
        else:
            padding = self.padding
        forward_param = {'method': Padding, 'padding': padding, 'const_image': self.const_image, 'const_mask': self.const_mask, 'padding_mode': self.padding_mode, 'backend': self.backend}
        backward_param = {'method': Crop, 'crop_size': (h, w), 'crop_position': (pad_top, pad_left), 'bbox_filter_by_center': False, 'clip_border': True}
        return forward_param, backward_param

    @classmethod
    def param_check(cls, param):
        required = ('image_size', 'padding', 'const_image', 'const_mask', 'padding_mode', 'backend')
        assert set(required) == set(param.keys())

    @classmethod
    def image_transform(cls, image, param):
        return _pad(image, param['padding'], param['const_image'], param['padding_mode'], param['backend'])

    @classmethod
    def mask_transform(cls, mask, param):
        return _pad(mask, param['padding'], param['const_mask'], param['padding_mode'], param['backend'])

    @classmethod
    def bbox_transform(cls, bbox_dict, param):
        bboxes = bbox_dict.pop('bboxes')
        pad_left, pad_top, pad_right, pad_bottom = param['padding']
        bboxes = bboxes + torch.tensor([pad_left, pad_top, pad_left, pad_top], dtype=bboxes.dtype, device=bboxes.device)
        bbox_dict['bboxes'] = bboxes
        return bbox_dict

    @classmethod
    def point_transform(cls, point_dict, param):
        points = point_dict.pop('points')
        pad_left, pad_top, pad_right, pad_bottom = param['padding']
        points = points + torch.tensor([pad_left, pad_top], dtype=points.dtype, device=points.device)
        point_dict['points'] = points
        return point_dict

    @classmethod
    def event_transform(cls, event_dict, param):
        if len(event_dict['events']) == 0:
            return event_dict

        events = event_dict.pop('events')
        pad_left, pad_top, pad_right, pad_bottom = param['padding']
        events[:,1] += pad_left
        events[:,2] += pad_top
        event_dict['events'] = events
        return event_dict

    @classmethod
    def meta_transform(cls, meta, param):
        pad_left, pad_top, pad_right, pad_bottom = param['padding']
        width = meta['width']
        height = meta['height']
        width = width + pad_right + pad_left
        height = height + pad_top + pad_bottom

        meta['width'] = width
        meta['height'] = height
        return meta

class Flip(object):
    def __init__(self, direction='H'):
        assert direction in ('H', 'V')
        self.direction = direction

    def get_param(self, image_size):
        is_flip = True
        forward_param  = {'method': Flip, 'is_flip': is_flip, 'direction': self.direction}
        backward_param = {'method': Flip, 'is_flip': is_flip, 'direction': self.direction}
        return forward_param, backward_param

    @classmethod
    def param_check(cls, param):
        required = ('image_size', 'is_flip', 'direction')
        assert set(required) == set(param.keys())

    @classmethod
    def image_transform(cls, image, param):
        is_flip = param['is_flip']
        direction = param['direction']
        if is_flip and direction == 'H':
            image = TF.hflip(image)
        elif is_flip and direction == 'V':
            image = TF.vflip(image)
        return image

    @classmethod
    def mask_transform(cls, mask, param):
        is_flip = param['is_flip']
        direction = param['direction']
        if is_flip and direction == 'H':
            mask = TF.hflip(mask)
        elif is_flip and direction == 'V':
            mask = TF.vflip(mask)
        return mask

    @classmethod
    def bbox_transform(cls, bbox_dict, param):
        bboxes = bbox_dict.pop('bboxes')
        is_flip = param['is_flip']
        direction = param['direction']
        size = param['image_size']
        box = (bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3])

        if is_flip and direction == 'H':
            box, _ = _box_hflip(box, size)
        elif is_flip and direction == 'V':
            box, _ = _box_vflip(box, size)

        bbox_dict['bboxes'] = torch.stack(box, dim=-1)
        return bbox_dict

    @classmethod
    def point_transform(cls, point_dict, param):
        points = point_dict.pop('points')
        is_flip = param['is_flip']
        direction = param['direction']
        size = param['image_size']
        point = (points[:,0], points[:,1])

        if is_flip and direction == 'H':
            point, _ = _point_hflip(point, size)
        elif is_flip and direction == 'V':
            point, _ = _point_vflip(point, size)

        point_dict['points'] = torch.stack(point, dim=-1)
        return point_dict

    @classmethod
    def event_transform(cls, event_dict, param):
        if len(event_dict['events']) == 0:
            return event_dict

        events = event_dict.pop('events')
        is_flip = param['is_flip']
        direction = param['direction']
        size = param['image_size']
        point = (events[:,1], events[:,2])

        if is_flip and direction == 'H':
            point, _ = _point_hflip(point, size)
        elif is_flip and direction == 'V':
            point, _ = _point_vflip(point, size)

        events[:,1] = point[0]
        events[:,2] = point[1]

        event_dict['events'] = events
        return event_dict

    @classmethod
    def meta_transform(cls, meta, param):
        return meta

class RandomFlip(Flip):
    def __init__(self, direction='H', prob=0.5):
        assert direction in ('H', 'V')
        self.direction = direction
        self.prob = prob

    def get_param(self, image_size):
        is_flip = random.uniform(0,1) <= self.prob
        forward_param  = {'method': Flip, 'is_flip': is_flip, 'direction': self.direction}
        backward_param = {'method': Flip, 'is_flip': is_flip, 'direction': self.direction}
        return forward_param, backward_param

class Rotate90N(object):
    def __init__(self, rot):
        assert rot in (0, 90, 180, 270)
        self.rot = rot

    def get_param(self, image_size):
        rot = self.rot
        back_rot = (360 - rot) % 360
        forward_param  = {'method': Rotate90N, 'rot': rot}
        backward_param = {'method': Rotate90N, 'rot': back_rot}
        return forward_param, backward_param

    @classmethod
    def param_check(cls, param):
        required = ('image_size', 'rot')
        assert set(required) == set(param.keys())

    @classmethod
    def image_transform(cls, image, param):
        rot = param['rot']
        if rot == 90:
            image = image.transpose(1,2)
            image = TF.vflip(image)
        elif rot == 180:
            image = TF.hflip(image)
            image = TF.vflip(image)
        elif rot == 270:
            image = image.transpose(1,2)
            image = TF.hflip(image)
        return image

    @classmethod
    def mask_transform(cls, mask, param):
        ndim = mask.ndim
        if ndim == 2:
            mask = mask[None,:,:]

        rot = param['rot']
        if rot == 90:
            mask = mask.transpose(1,2)
            mask = TF.vflip(mask)
        elif rot == 180:
            mask = TF.hflip(mask)
            mask = TF.vflip(mask)
        elif rot == 270:
            mask = mask.transpose(1,2)
            mask = TF.hflip(mask)

        if ndim == 2:
            mask = mask[0]
        return mask

    @classmethod
    def bbox_transform(cls, bbox_dict, param):
        bboxes = bbox_dict.pop('bboxes')
        rot = param['rot']
        size = param['image_size']
        box = (bboxes[:,0], bboxes[:,1], bboxes[:,2], bboxes[:,3])
        if rot == 90:
            box, size = _box_transpose(box, size)
            box, size = _box_vflip(box, size)
        if rot == 180:
            box, size = _box_hflip(box, size)
            box, size = _box_vflip(box, size)
        if rot == 270:
            box, size = _box_transpose(box, size)
            box, size = _box_hflip(box, size)

        bbox_dict['bboxes'] = torch.stack(box, dim=-1)
        return bbox_dict

    @classmethod
    def point_transform(cls, point_dict, param):
        points = point_dict.pop('points')
        rot = param['rot']
        size = param['image_size']
        point = (points[:,0], points[:,1])
        if rot == 90:
            point, size = _point_transpose(point, size)
            point, size = _point_vflip(point, size)
        if rot == 180:
            point, size = _point_hflip(point, size)
            point, size = _point_vflip(point, size)
        if rot == 270:
            point, size = _point_transpose(point, size)
            point, size = _point_hflip(point, size)

        point_dict['points'] = torch.stack(point, dim=-1)
        return point_dict

    @classmethod
    def event_transform(cls, event_dict, param):
        if len(event_dict['events']) == 0:
            return event_dict

        events = event_dict.pop('events')
        rot = param['rot']
        size = param['image_size']
        point = (events[:,1], events[:,2])
        if rot == 90:
            point, size = _point_transpose(point, size)
            point, size = _point_vflip(point, size)
        if rot == 180:
            point, size = _point_hflip(point, size)
            point, size = _point_vflip(point, size)
        if rot == 270:
            point, size = _point_transpose(point, size)
            point, size = _point_hflip(point, size)

        events[:,1] = point[0]
        events[:,2] = point[1]
        event_dict['events'] = events
        return event_dict

    @classmethod
    def meta_transform(cls, meta, param):
        rot = param['rot']
        width = meta['width']
        height = meta['height']
        if rot in (90, 270):
            width, height = height, width

        meta['width'] = width
        meta['height'] = height
        return meta


class RandomRotate90N(Rotate90N):
    def __init__(self):
        self.choices = [0, 90, 180, 270]

    def get_param(self, image_size):
        rot = random.choice(self.choices)
        back_rot = (360 - rot) % 360
        forward_param  = {'method': Rotate90N, 'rot': rot}
        backward_param = {'method': Rotate90N, 'rot': back_rot}
        return forward_param, backward_param

class Rotate(object):
    def __init__(self, rot, image_resampling='BICUBIC', mask_resampling='NEAREST', const_image=0, const_mask=-1, clip_border=True, backend='PIL'):
        self.rot = rot
        self.image_resampling = image_resampling
        self.mask_resampling = mask_resampling
        self.const_image = const_image
        self.const_mask = const_mask
        self.clip_border = clip_border
        self.backend = backend

    def get_param(self, image_size):
        rot = self.rot
        forward_param = {'method': Rotate, 'rot': rot, 'image_resampling': self.image_resampling, 'mask_resampling': self.mask_resampling,
                         'const_image': self.const_image, 'const_mask': self.const_mask, 'clip_border': self.clip_border, 'backend': self.backend}
        backward_param = {'method': Rotate, 'rot': -rot, 'image_resampling': self.image_resampling, 'mask_resampling': self.mask_resampling,
                         'const_image': self.const_image, 'const_mask': self.const_mask, 'clip_border': self.clip_border, 'backend': self.backend}
        return forward_param, backward_param

    @classmethod
    def param_check(cls, param):
        required = ('image_size', 'rot', 'image_resampling', 'mask_resampling', 'const_image', 'const_mask', 'clip_border', 'backend')
        assert set(required) == set(param.keys())

    @classmethod
    def image_transform(cls, image, param):
        return _rotate(image, param['rot'], param['image_resampling'], param['const_image'], param['backend'])

    @classmethod
    def mask_transform(cls, mask, param):
        return _rotate(mask, param['rot'], param['mask_resampling'], param['const_mask'], param['backend'])

    @classmethod
    def bbox_transform(cls, bbox_dict, param):
        bboxes = bbox_dict.pop('bboxes')

        h, w = param['image_size']
        cx = float(w)/2 - 0.5
        cy = float(h)/2 - 0.5
        center = torch.tensor([cx,cy,cx,cy], dtype=torch.float, device=bboxes.device)
        bboxes = bboxes.float() - center
        cp1 = bboxes[:,:2]
        cp2 = bboxes[:,2:]
        cp3 = bboxes[:,0::3]
        cp4 = torch.stack([bboxes[:,2],bboxes[:,1]], dim=-1)

        # rotate points
        rot = param['rot'] / 180 * np.pi
        R = torch.tensor([[ np.cos(rot), -np.sin(rot)],
                          [ np.sin(rot),  np.cos(rot)]], dtype=torch.float, device=bboxes.device)
        center = center[:2]
        p1 = cp1 @ R + center
        p2 = cp2 @ R + center
        p3 = cp3 @ R + center
        p4 = cp4 @ R + center

        # get bounding rect
        P = torch.stack([p1, p2, p3, p4])
        p1 = P.amin(dim=0)
        p2 = P.amax(dim=0)

        if clip_border:
            p1[:,0] = torch.clip(p1[:,0], 0, w)
            p2[:,0] = torch.clip(p2[:,0], 0, w)
            p1[:,1] = torch.clip(p1[:,1], 0, h)
            p2[:,1] = torch.clip(p2[:,1], 0, h)
            valid = (p2[:,0] > p1[:,0]) & (p2[:,1] > p1[:,1])
        else:
            _p1 = p1.clone()
            _p2 = p2.clone()
            _p1[:,0] = torch.clip(_p1[:,0], 0, w)
            _p2[:,0] = torch.clip(_p2[:,0], 0, w)
            _p1[:,1] = torch.clip(_p1[:,1], 0, h)
            _p2[:,1] = torch.clip(_p2[:,1], 0, h)
            valid = (_p2[:,0] > _p1[:,0]) & (_p2[:,1] > _p1[:,1])
        bboxes = torch.cat([p1,p2], dim=1)[valid,:]

        output = {}
        for key, data in bbox_dict.items():
            output[key] = data[valid]
        output['bboxes'] = bboxes.long()
        return output
        
    @classmethod
    def point_transform(cls, point_dict, param):
        points = point_dict.pop('points')

        h, w = param['image_size']
        center = torch.tensor([float(w)/2 - 0.5, float(h)/2 - 0.5], dtype=torch.float, device=points.device)
        cp = points - center

        # rotate points
        rot = param['rot'] / 180 * np.pi
        R = torch.tensor([[ np.cos(rot), -np.sin(rot)],
                          [ np.sin(rot),  np.cos(rot)]], dtype=torch.float, device=points.device)
        points = cp @ R + center

        valid = (points[:,0] >= 0) & (points[:,0] < w) & (points[:,1] >= 0) & (points[:,1] < h)
        points = points[valid,:]

        output = {}
        for key, data in point_dict.items():
            output[key] = data[valid]
        output['points'] = points.long()
        return output

    @classmethod
    def event_transform(cls, event_dict, param):
        if len(event_dict['events']) == 0:
            return event_dict

        events = event_dict.pop('events')

        h, w = param['image_size']
        center = torch.tensor([float(w)/2 - 0.5, float(h)/2 - 0.5], dtype=torch.float, device=events.device)
        cp = events[:,1:3] - center

        # rotate points
        rot = param['rot'] / 180 * np.pi
        R = torch.tensor([[ np.cos(rot), -np.sin(rot)],
                          [ np.sin(rot),  np.cos(rot)]], dtype=torch.float, device=events.device)
        events[:,1:3] = cp @ R + center

        valid = (events[:,1] >= 0) & (events[:,1] < w) & (events[:,2] >= 0) & (events[:,2] < h)

        output = {}
        output['events'] = events[valid].long()
        if 'knn' in event_dict and event_dict['knn'] is not None:
            output['knn'] = event_dict['knn'][valid]
        elif 'knn' in event_dict:
            output['knn'] = event_dict['knn']

        return output

    @classmethod
    def meta_transform(cls, meta, param):
        return meta


class RandomRotate(Rotate):
    def __init__(self, min_rot, max_rot, image_resampling='BICUBIC', mask_resampling='NEAREST', const_image=0, const_mask=-1, clip_border=True, backend='PIL'):
        self.min_rot = min_rot
        self.max_rot = max_rot
        self.image_resampling = image_resampling
        self.mask_resampling = mask_resampling
        self.const_image = const_image
        self.const_mask = const_mask
        self.clip_border = clip_border
        self.backend = backend

    def get_param(self, image_size):
        rot = random.uniform(self.min_rot, self.max_rot)
        forward_param = {'method': Rotate, 'rot': rot, 'image_resampling': self.image_resampling, 'mask_resampling': self.mask_resampling,
                         'const_image': self.const_image, 'const_mask': self.const_mask, 'clip_border': self.clip_border, 'backend': self.backend}
        backward_param = {'method': Rotate, 'rot': -rot, 'image_resampling': self.image_resampling, 'mask_resampling': self.mask_resampling,
                         'const_image': self.const_image, 'const_mask': self.const_mask, 'clip_border': self.clip_border, 'backend': self.backend}
        return forward_param, backward_param

def _resize(img, size, resampling, backend):
    assert backend in ('PIL', 'Torch', 'OpenCV')
    h, w = img.shape[-2:]
    if h == size[0] and w == size[1]:
        return img

    ndim = img.ndim
    if ndim == 2:
        img = img[None,:,:]

    resampling = RESAMPLING[backend][resampling]
    if backend == 'PIL':
        img = _trans_in_numpy(img, _resize_pil, size=size, resampling=resampling)
    elif backend == 'Torch':
        img = _resize_torch(img, size, resampling)
    elif backend == 'OpenCV':
        img = _trans_in_numpy(img, _resize_cv2, size=size, resampling=resampling)

    if ndim == 2:
        img = img[0]

    return img

def _resize_pil(img, size, resampling):
    H, W = size
    bands = [ Image.fromarray(band) for band in _split_bands(img) ]
    bands = [ np.array(band.resize((W,H), resampling)) for band in bands ]
    return np.stack(bands)

def _resize_torch(img, size, resampling):
    #antialias = resampling == TF.InterpolationMode.BILINEAR
    antialias = False    # disable antialias because autodiff is not supported for the operation
    img = TF.resize(img, size, resampling, antialias=antialias)
    return img

def _resize_cv2(img, size, resampling):
    H, W = size
    h, w = img.shape[-2:]
    scale_x = float(W) / w
    scale_y = float(H) / h
    img = img.transpose(1,2,0)
    img = cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=resampling)
    if img.ndim == 2:
        img = img[:,:,None]
    return img.transpose(2,0,1)

def _pad(img, padding, fill_values, padding_mode, backend):
    assert backend in ('PIL', 'Torch', 'OpenCV')

    ndim = img.ndim
    if ndim == 2:
        img = img[None,:,:]

    if backend == 'PIL':
        img = _trans_in_numpy(img, _pad_pil, padding=padding, fill_values=fill_values, padding_mode=padding_mode)
    elif backend == 'Torch':
        img = _pad_torch(img, padding, fill_values, padding_mode)
    elif backend == 'OpenCV':
        img = _trans_in_numpy(img, _pad_cv2, padding=padding, fill_values=fill_values, padding_mode=padding_mode)

    if ndim == 2:
        img = img[0]

    return img

def _pad_pil(img, padding, fill_values, padding_mode):
    print('Warning: Padding is not implemented for PIL. Using OpenCV instead.')
    return _pad_cv2(img, padding, fill_values, padding_mode)

def _pad_torch(img, padding, fill_values, padding_mode):
    if isinstance(fill_values, (list, tuple)):
        assert len(fill_values) == len(img)
    else:
        fill_values = [fill_values] * len(img)
    bands = _split_bands(img)
    bands = [ TF.pad(band, padding, fill=fill, padding_mode=padding_mode) for band, fill in zip(bands, fill_values) ]
    img = torch.stack(bands)
    return img

def _pad_cv2(img, padding, fill_values, padding_mode):
    if isinstance(fill_values, (list, tuple)):
        assert len(fill_values) == len(img)
    else:
        fill_values = [fill_values] * len(img)

    if padding_mode == 'constant':
        mode = cv2.BORDER_CONSTANT
    elif padding_mode == 'symmetric':
        mode = cv2.BORDER_REFLECT
    elif padding_mode == 'reflect':
        mode = cv2.BORDER_REFLECT_101

    pad_l, pad_t, pad_r, pad_b = padding
    img = img.transpose(1,2,0)
    img = cv2.copyMakeBorder(img, pad_t, pad_b, pad_l, pad_r, mode, value=fill_values)
    if img.ndim == 2:
        img = img[:,:,None]
    return img.transpose(2,0,1)

def _pad_numpy(img, padding, fill_values, padding_mode):
    if isinstance(fill_values, (list, tuple)):
        assert len(fill_values) == len(img)
    else:
        fill_values = [fill_values] * len(img)

    pad_l, pad_t, pad_r, pad_b = padding
    if padding_mode == 'constant':
        b, h, w = img.shape
        out = np.ones([b, h+pad_u+pad_d, w+pad_l+pad_r], dtype=img.dtype) * np.array(fill_values)[:,None,None]
        out[:, pad_u:pad_u+h, pad_l:pad_l+w] = img
    elif mode == 'symmetric' or mode == 'reflect':
        out = np.pad(img, ((0,0), (pad_u,pad_d), (pad_l,pad_r)), mode=mode)
    return out


def _rotate(img, rot, resampling, fill_values, backend):
    assert backend in ('PIL', 'Torch', 'OpenCV')
    resampling = RESAMPLING[backend][resampling]

    ndim = img.ndim
    if ndim == 2:
        img = img[None,:,:]

    if backend == 'PIL':
        img = _trans_in_numpy(img, _rotate_pil, rot=rot, resampling=resampling, fill_values=fill_values)
    elif backend == 'Torch':
        img = _rotate_torch(img, rot, resampling, fill_values)
    elif backend == 'OpenCV':
        img = _trans_in_numpy(img, _rotate_cv2, rot=rot, resampling=resampling, fill_values=fill_values)

    if ndim == 2:
        img = img[0]

    return img

def _rotate_pil(img, rot, resampling, fill_values):
    bands = [ Image.fromarray(band) for band in _split_bands(img) ]
    if isinstance(fill_values, (list, tuple)):
        assert len(fill_values) == len(bands)
    else:
        fill_values = [fill_values] * len(bands)

    bands = [ np.array(band.rotate(rot, resampling, fillcolor=fill_value)) for band, fill_value in zip(bands, fill_values) ]
    return np.stack(bands)

def _rotate_torch(img, rot, resampling, fill_values):
    img = TF.rotate(img, rot, resampling, fill=fill_values)
    return img

def _rotate_cv2(img, rot, resampling, fill_values):
    if isinstance(fill_values, (list, tuple)):
        assert len(fill_values) == len(img)
    else:
        fill_values = [fill_values] * len(img)
    h, w = img.shape[-2:]
    img = img.transpose(1,2,0)
    matrix = cv2.getRotationMatrix2D((float(w) / 2 - 0.5, float(h) / 2 - 0.5), rot, 1)
    img = cv2.warpAffine(img, matrix, (w, h), flags=resampling, borderMode=cv2.BORDER_CONSTANT, borderValue=fill_values)
    if img.ndim == 2:
        img = img[:,:,None]
    return img.transpose(2,0,1)

def _box_transpose(box, size):
    H, W = size
    x1, y1, x2, y2 = box
    return (y1, x1, y2, x2), (W, H)

def _box_vflip(box, size):
    H, W = size
    x1, y1, x2, y2 = box
    y1, y2 = H-1 - y1, H-1 - y2
    return (x1, y2, x2, y1), (H, W)

def _box_hflip(box, size):
    H, W = size
    x1, y1, x2, y2 = box
    x1, x2 = W-1 - x1, W-1 - x2
    return (x2, y1, x1, y2), (H, W)

def _point_transpose(point, size):
    H, W = size
    x1, y1 = point
    return (y1, x1), (W, H)

def _point_vflip(point, size):
    H, W = size
    x1, y1 = point
    y1 = H-1 - y1
    return (x1, y1), (H, W)

def _point_hflip(point, size):
    H, W = size
    x1, y1 = point
    x1 = W-1 - x1
    return (x1, y1), (H, W)

def _downsample_events(events, scale_x, scale_y, width, height, downsampling='UNIFORM', resampling='BILINEAR'):

    # downsampling
    if downsampling == 'UNIFORM':
        indices = torch.randperm(len(events))
        num_sample = int(len(events) * scale_x * scale_y)
        indices = indices[:num_sample]
        indices = indices.sort()[0]
        events = events[indices]

    # resampling
    if resampling == 'BILINEAR':
        events = events.float()
        events[:,1] = events[:,1] * scale_x
        events[:,2] = events[:,2] * scale_y
        W = int(round(width * scale_x))
        H = int(round(height * scale_y))
        events = _bilinear_events(events, W, H, scale_x, scale_y)
    elif resampling == 'NEAREST':
        events[:,1] = (events[:,1] * scale_x).long()
        events[:,2] = (events[:,2] * scale_y).long()
    elif resampling == 'NONE':
        events = events.float()
        events[:,1] = events[:,1] * scale_x
        events[:,2] = events[:,2] * scale_y
    else:
        raise RuntimeError

    return events

def _upsample_events(events, scale_x, scale_y, width, height, k=4, knn=None, upsampling='KNN', resampling='BILINEAR'):
    sqrt2 = math.sqrt(2)
    OFFSET_TABLE = torch.tensor([[0,1],[1,1],[1,0],[1,-1],[0,-1],[-1,-1],[-1,0],[-1,1]])
    DIST_TABLE = torch.tensor([1, sqrt2, 1, sqrt2, 1, sqrt2, 1, sqrt2])

    if upsampling == 'KNN':
        if knn is not None:
            ev_index = knn[:,0]
            dir_code = knn[:,1]
            time_coef = knn[:,2]
        else:
            ev_index, dir_code, time_coef = _calc_event_edges(events, width, height, k=k)

    if resampling in ('BILINEAR', 'NONE'):
        events = events.float()
        events[:,1] = events[:,1] * scale_x
        events[:,2] = events[:,2] * scale_y
    elif resampling == 'NEAREST':
        events[:,1] = events[:,1] * scale_x
        events[:,2] = events[:,2] * scale_y
    else:
        raise RuntimeError

    if upsampling == 'KNN':
        offset = OFFSET_TABLE[dir_code]
        new_events = events[ev_index]

        #dir_code = knn[:,:,0]
        #time_coef = knn[:,:,1]
        #offset = OFFSET_TABLE[dir_code]

        #time_coef = time_coef.view(-1,1)
        #offset = offset.view(-1,2)
        #events = events[:,None,:].repeat(1,2,1).view(-1,4)

        #filt = time_coef <= 0 
        #time_coef = time_coef[filt]
        #offset = offset[filt]
        #new_events = events[filt]

        shift_y = offset[:,0] * scale_y * 0.5
        shift_x = offset[:,1] * scale_x * 0.5
        dist = DIST_TABLE[dir_code]
        new_y = new_events[:,2] + shift_y
        new_x = new_events[:,1] + shift_x
        new_t = new_events[:,0] + time_coef * (scale_x + scale_y - 2) * 0.5
        new_p = new_events[:,3].clone() * min(1, (scale_x + scale_y - 2) * 0.5)

        #assert new_y.min() >= 0 and new_x.min() >= 0

        new_events = torch.stack([new_t, new_x, new_y, new_p], dim=-1).to(events.dtype)

        # subsample
        indices = torch.randperm(len(new_events))
        num_sample = int(len(events) * (scale_x * scale_y - 1))
        indices = indices[:num_sample]
        new_events = new_events[indices]

        events = torch.cat([events, new_events], dim=0)

        # sort
        indices = torch.argsort(events[:,0])
        events = events[indices]

    if resampling == 'BILINEAR':
        W = int(round(width * scale_x))
        H = int(round(height * scale_y))
        events = _bilinear_events(events, W, H, scale_x, scale_y)

    return events

def _bilinear_events(events, width, height, scale_x, scale_y):

    t = events[:,0]
    x = events[:,1]
    y = events[:,2]
    #p = (events[:,3] - 0.5) * 2.0
    p = events[:,3]

    mode = 2

    if mode == 0:
        x1 = x.floor()
        x2 = x1 + 1
        y1 = y.floor()
        y2 = y1 + 1
        x0 = x1 - 1
        y0 = y1 - 1

        x012 = torch.stack([x0, x1, x2])
        y012 = torch.stack([y0, y1, y2])
        wx012 = 1. - (x012 - x).abs() / scale_x
        wy012 = 1. - (y012 - y).abs() / scale_y
        wx012 = wx012.clip(min=0)
        wy012 = wy012.clip(min=0)

        w012 = wx012[:,None,:] * wy012[None,:,:]
        p012 = w012 * p[None,None,:]
        x012 = x012[:,None,:].repeat(1,3,1)
        y012 = y012[None,:,:].repeat(3,1,1)
        t012 = t[None,None,:].repeat(3,3,1)

        fx0 = x0 >= 0
        fx1 = torch.ones_like(x1).bool()
        fx2 = x2 < width
        fy0 = y0 >= 0
        fy1 = torch.ones_like(y1).bool()
        fy2 = y2 < height
        fx012 = torch.stack([fx0, fx1, fx2])
        fy012 = torch.stack([fy0, fy1, fy2])
        f012 = fx012[:,None,:] & fy012[None,:,:]

        events = torch.stack([t012,x012,y012,p012], dim=-1).view(-1,4)
        filt = ((w012 > 0) & f012).view(-1)
        events = events[filt]

    elif mode == 1:
        x1 = x.floor()
        x2 = x1 + 1
        x3 = x1 + 2
        y1 = y.floor()
        y2 = y1 + 1
        y3 = y1 + 2
        x0 = x1 - 1
        y0 = y1 - 1

        x012 = torch.stack([x0, x1, x2, x3])
        y012 = torch.stack([y0, y1, y2, y3])
        wx012 = 1. - (x012 - x).abs() / scale_x
        wy012 = 1. - (y012 - y).abs() / scale_y
        wx012 = wx012.clip(min=0)
        wy012 = wy012.clip(min=0)

        w012 = wx012[:,None,:] * wy012[None,:,:]
        p012 = w012 * p[None,None,:]
        x012 = x012[:,None,:].repeat(1,4,1)
        y012 = y012[None,:,:].repeat(4,1,1)
        t012 = t[None,None,:].repeat(4,4,1)

        fx0 = x0 >= 0
        fx1 = torch.ones_like(x1).bool()
        fx2 = x2 < width
        fx3 = x3 < width
        fy0 = y0 >= 0
        fy1 = torch.ones_like(y1).bool()
        fy2 = y2 < height
        fy3 = y3 < height
        fx012 = torch.stack([fx0, fx1, fx2, fx3])
        fy012 = torch.stack([fy0, fy1, fy2, fy3])
        f012 = fx012[:,None,:] & fy012[None,:,:]

        events = torch.stack([t012,x012,y012,p012], dim=-1).view(-1,4)
        filt = ((w012 > 0) & f012).view(-1)
        events = events[filt]

    elif mode == 2:
        kx = math.ceil(scale_x * 2)
        ky = math.ceil(scale_y * 2)
        kx = max(2, kx)
        ky = max(2, ky)

        shift_x = torch.arange(kx) - int((kx-1) * 0.5)
        shift_y = torch.arange(ky) - int((ky-1) * 0.5)
        X = x.floor()[None,:] + shift_x[:,None]
        Y = y.floor()[None,:] + shift_y[:,None]
        Fx = (X >= 0) & (X < width)
        Fy = (Y >= 0) & (Y < height)

        Wx = 1. - (X - x).abs() / max(1, scale_x)
        Wy = 1. - (Y - y).abs() / max(1, scale_y)
        Wx = Wx.clip(min=0)
        Wy = Wy.clip(min=0)

        W = Wx[:,None,:] * Wy[None,:,:]
        P = W * p[None,None,:]
        X = X[:,None,:].repeat(1,kx,1)
        Y = Y[None,:,:].repeat(ky,1,1)
        T = t[None,None,:].repeat(ky,kx,1)
        F = Fx[:,None,:] & Fy[None,:,:]

        events = torch.stack([T,X,Y,P], dim=-1).transpose(0,2).contiguous().view(-1,4)
        filt = ((W > 0) & F).transpose(0,2).contiguous().view(-1)
        events = events[filt]

    elif mode == 3:
        x1 = x.floor()
        x2 = x1 + 1
        y1 = y.floor()
        y2 = y1 + 1

        wx1 = x2 - x
        wx2 = x - x1
        wy1 = y2 - y
        wy2 = y - y1

        w11 = wx1 * wy1
        w12 = wx1 * wy2
        w21 = wx2 * wy1
        w22 = wx2 * wy2

        p11 = p * w11
        p12 = p * w12
        p21 = p * w21
        p22 = p * w22

        x2 = x2.clip(max=width-1)
        y2 = y2.clip(max=height-1)

        e11 = torch.stack([t, x1, y1, p11], dim=1)
        e12 = torch.stack([t, x1, y2, p12], dim=1)
        e21 = torch.stack([t, x2, y1, p21], dim=1)
        e22 = torch.stack([t, x2, y2, p22], dim=1)

        events = torch.cat([e11, e12, e21, e22], dim=0)

    events

    return events

def _calc_event_edges(events, width, height, k):
    DIR_CODE = torch.tensor([
        [5, 6, 7],
        [4, 8, 0],
        [3, 2, 1],
    ])

    curr_time = events[0,0]
    t = (events[:,0].float() - curr_time) / float(width * 10000)
    x = events[:,1].float() / width
    y = events[:,2].float() / height
    p = events[:,3].float()
    A = torch.stack([t, x, y, p], dim=1).numpy().astype('float32')

    if len(A) > 4000:
        #nlist = int(4 * torch.sqrt(len(A)))
        nlist = 100
        quantizer = faiss.IndexFlatL2(A.shape[1])  # the other index
        index = faiss.IndexIVFFlat(quantizer, A.shape[1], nlist, faiss.METRIC_L2) # here we specify METRIC_L2, by default it performs inner-product search
        index.train(A)
        index.add(A)                  # add may be a bit slower as well
        dists, result = index.search(A, k=k)     # actual search
    else:
        index = faiss.IndexFlatL2(A.shape[1])   # build the index
        index.add(A)                  # add vectors to the index
        dists, result = index.search(A, k=k)     # actual search

    r = torch.from_numpy(result)[:,1:]
    diff = events[r][:,:,:] - events[:,None,:]
    valid_t = torch.logical_and(diff[:,:,0] <= 0, diff[:,:,0] > -50e3)
    valid_x = torch.abs(diff[:,:,1]) <= 1
    valid_y = torch.abs(diff[:,:,2]) <= 1
    exclude = torch.logical_and(diff[:,:,1] == 0, diff[:,:,2] == 0)
    valid = valid_t & valid_x & valid_y & ~exclude

    ev_index = torch.arange(len(A))[:,None].repeat(1, k-1)
    ev_index = ev_index[valid]
    txy = diff[valid][:,:3]
    time_coef = txy[:,0]
    dir_code = DIR_CODE[txy[:,2]+1, txy[:,1]+1]
    
    return ev_index, dir_code, time_coef

def _trans_in_numpy(img, func, **params):
    is_tensor = isinstance(img, torch.Tensor)
    if is_tensor:
        device = img.device
        img = img.cpu().numpy()

    dtype = img.dtype
    img = func(img.astype(np.float32), **params)
    img = img.astype(dtype)

    if is_tensor:
        img = torch.from_numpy(img).to(device)

    return img

def _split_bands(img):
    return [ band[0] for band in np.split(img, len(img)) ]

RESAMPLING = {
    'PIL': {
        'NEAREST' : Image.NEAREST,
        'BILINEAR': Image.BILINEAR,
        'BICUBIC' : Image.BICUBIC,
        'LANCZOS' : Image.LANCZOS,
    },
    'OpenCV': {
        'NEAREST' : cv2.INTER_NEAREST,
        'BILINEAR': cv2.INTER_LINEAR,
        'BICUBIC' : cv2.INTER_CUBIC,
        'LANCZOS' : cv2.INTER_LANCZOS4,
    },
    'Torch': {
        'NEAREST' : TF.InterpolationMode.NEAREST,
        'BILINEAR': TF.InterpolationMode.BILINEAR,
        'BICUBIC' : TF.InterpolationMode.BICUBIC,
    },
}




