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
import numpy as np
import os
import time

import torch
import torch.nn as nn
import torch.nn.functional as F

from ..base.blocks import BlockBase
from ..base.init import init_transformer
from ..base.backbone.builder import build_backbone
from ..base.neck.builder import build_neck
from ..base.head.builder import build_head
from hmnet.utils.common import Timer

from torch import Tensor
from typing import Tuple, List, Optional, Dict
Meta = Dict[str,Tuple[int,int,int]]

class HMDet(BlockBase):
    def __init__(self, backbone, neck, bbox_head, devices, test_aug=None) -> None:
        super().__init__()
        self.devices = devices
        self.test_aug = test_aug

        self.cfg_backbone = backbone
        self.cfg_neck = neck
        self.cfg_bbox_head = bbox_head

        self.backbone  = build_backbone(backbone)
        self.neck      = build_neck(neck)
        self.bbox_head = build_head(bbox_head)

        self.set_module_names()

    @property
    def no_decay_set(self):
        return {'bias'}

    def init_weights(self, pretrained=None):
        init_transformer(self.modules())
        self.backbone.init_weights()
        self.neck.init_weights()
        self.bbox_head.init_weights()

    def to_cuda(self):
        d0 = self.devices[0]
        if hasattr(self.backbone, 'to_cuda'):
            self.backbone.to_cuda(*self.devices[1:])
        else:
            self.backbone = self.backbone.to(d0)
        self.neck = self.neck.to(d0)
        self.bbox_head = self.bbox_head.to(d0)

    def compile(self, backend, fp16=False, input_shape=None):
        assert input_shape is not None

        sizes = self.cfg_backbone['latent_sizes']
        channels = self.cfg_bbox_head['in_channels']
        input_shapes = [ (1,C,H,W) for (H,W),C in zip(sizes, channels) ]

        self.bbox_head = self.bbox_head.compile(backend, fp16, input_shapes)

    def forward(self, list_events, list_image_metas, list_gt_bboxes, list_gt_labels, list_ignore_masks, init_states=True) -> Tensor:
        if init_states:
            self.idx_offset = 0

        # Gather gts for loss calculation
        gather_indices = self._identify_required_outputs_batch(list_gt_labels, self.idx_offset)
        out_image_metas  = self._gather(list_image_metas , gather_indices)
        out_gt_bboxes    = self._gather(list_gt_bboxes   , gather_indices)
        out_gt_labels    = self._gather(list_gt_labels   , gather_indices)
        out_ignore_masks = self._gather(list_ignore_masks, gather_indices)

        # Backbone
        features = self.backbone(list_events, list_image_metas, gather_indices, init_states=init_states, detach=True, fast_training=True)

        # Head
        loss, log_vars = self._forward_head(features, out_gt_bboxes, out_gt_labels, out_ignore_masks)

        # Make sure all parameters are involved in loss calculation
        loss = loss + sum([ 0. * params.sum() for params in self.parameters() ])

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(out_image_metas))

        return outputs

    def inference(self, list_events, list_image_metas, speed_test=False) -> Tensor:
        output = []
        output_image_metas = []
        d0 = self.devices[0]
        s0 = torch.cuda.Stream(device=d0)
        timer = Timer(enabled=speed_test)

        batch_size = len(list_events[0])
        height = list_image_metas[0][0]['height']
        width = list_image_metas[0][0]['width']

        self.backbone.prepair_for_inference(batch_size, image_size=(height, width))

        for idx, (events, image_metas) in enumerate(zip(list_events, list_image_metas)):
            events = to_device(events, d0)

            timer.lazy_start(43)

            events, image_metas = self._test_transform(events, image_metas)

            features = self.backbone.inference(events, image_metas)

            if None not in features:
                features = [ f.to(d0) for f in features ]

                s0.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s0):
                    x = self.neck(features)
                    list_bbox_dict = self._bbox_head_test(self.bbox_head, x, image_metas)
                    list_bbox_dict, image_metas = self._backward_transform(list_bbox_dict, image_metas)
                    output += list_bbox_dict
                    output_image_metas += image_metas

                    # debug
                    #list_bbox_dict, detections = self._bbox_head_test_debug(self.bbox_head, x, image_metas)
                    #if (idx+1) % 2000 == 0:
                    #    #torch.save(detections, f'./debug/out_{idx+1}.pth')
                    #    src = torch.load(f'./debug/out_{idx+1}.pth')
                    #    print(idx+1, diff(detections, src))
                    # /debug

                s0.synchronize()
                timer.tick(sync_method='none')  # already synced

        timer.display()

        self.backbone.termination()

        return output, output_image_metas

    def _test_transform(self, events, img_metas):
        if self.test_aug is None:
            return events, img_metas

        out_events, out_img_metas = [], []
        for evt, img_meta in zip(events, img_metas):
            evt, img_meta = self.test_aug(evt, img_meta, types=['event', 'meta'])
            out_events.append(evt)
            out_img_metas.append(img_meta)
        return out_events, out_img_metas

    def _backward_transform(self, list_bbox_dict, img_metas):
        if self.test_aug is None:
            return list_bbox_dict, img_metas

        out_bbox_dict = []
        out_img_metas = []
        for bbox_dict, img_meta in zip(list_bbox_dict, img_metas):
            bbox_dict, img_meta = self.test_aug.backward(bbox_dict, img_meta, types=['bbox', 'meta'])
            out_bbox_dict.append(bbox_dict)
            out_img_metas.append(img_meta)

        return out_bbox_dict, out_img_metas

    def _forward_head(self, features, out_gt_bboxes, out_gt_labels, out_ignore_masks):
        if len(features[0]) == 0:
            # dummy loss calculation for DDP
            out_gt_bboxes = [ torch.empty(0,4) ]
            out_gt_labels = [ torch.empty(0,1) ]
            out_ignore_masks = [ torch.empty(0,1) ]
            features = self.backbone.get_dummy_output()
            coef = 0
        else:
            coef = 1

        x = self.neck(features)
        outputs = self.bbox_head(x)
        loss, log_vars = self.bbox_head.loss(outputs, out_gt_bboxes, out_gt_labels, out_ignore_masks)

        loss = coef * loss
        log_vars = { key: coef * value for key, value in log_vars.items() }

        return loss, log_vars

    def _bbox_head_test(self, bbox_head, features, image_metas):
        detections = bbox_head.inference(features)
        detections = bbox_head.postprocess(detections)    # Tensor[B,N,7], [0:4] bbox(xyxy), [4,5] score, [6] label
        if detections[0] is None:
            outputs = [ {'bboxes': torch.empty(0,4), 'labels': torch.empty(0), 'scores': torch.empty(0)} for det in detections ]
        else:
            outputs = [ {'bboxes': det[:,0:4], 'labels': det[:,6], 'scores': det[:,4]*det[:,5]} for det in detections ]

        return outputs

    def _bbox_head_test_debug(self, bbox_head, features, image_metas):
        out = bbox_head.inference(features)
        detections = bbox_head.postprocess(out)    # Tensor[B,N,7], [0:4] bbox(xyxy), [4,5] score, [6] label
        if detections[0] is None:
            outputs = [ {'bboxes': torch.empty(0,4), 'labels': torch.empty(0), 'scores': torch.empty(0)} for det in detections ]
        else:
            outputs = [ {'bboxes': det[:,0:4], 'labels': det[:,6], 'scores': det[:,4]*det[:,5]} for det in detections ]

        return outputs, out

    def _gather(self, list_data, gather_indices):
        time_indices = gather_indices['time']
        batch_indices = gather_indices['batch']
        destination = torch.arange(len(time_indices))

        if len(time_indices) == 0:
            return []

        assert time_indices.max() < len(list_data)

        output = [None] * len(time_indices)
        for tidx, datas in enumerate(list_data):
            mask = time_indices == tidx
            b_inds = batch_indices[mask]
            d_inds = destination[mask]
            for didx, bidx in zip(d_inds, b_inds):
                output[didx] = datas[bidx]
        return output

    def _identify_required_outputs(self, list_gt, bidx, idx_offset):
        time_indices = []
        for tidx, gt_labels in enumerate(list_gt):
            labels = gt_labels[bidx]
            if labels is not None:
                time_indices.append(tidx)

        time_indices = torch.tensor(time_indices).long()
        mask = torch.logical_and(time_indices < len(list_gt), (time_indices + idx_offset) > self.backbone.warmup)
        time_indices = time_indices[mask]

        return time_indices

    def _identify_required_outputs_batch(self, list_gt, idx_offset):
        time_indices = []
        batch_indices = []
        for bidx in range(len(list_gt[0])):
            t_indices = self._identify_required_outputs(list_gt, bidx, idx_offset)
            b_indices = torch.ones_like(t_indices) * bidx
            time_indices.append(t_indices)
            batch_indices.append(b_indices)
        time_indices = torch.cat(time_indices)
        batch_indices = torch.cat(batch_indices)
        gather_indices = dict(time=time_indices, batch=batch_indices)
        return gather_indices

def diff(dst, src):
    # (B,N,C), C = [X,Y,W,H,OBJ,CLS]
    dst_box = dst[:,:,:4]
    dst_obj = dst[:,:,4]
    dst_cls = dst[:,:,5:]

    src_box = src[:,:,:4]
    src_obj = src[:,:,4]
    src_cls = src[:,:,5:]

    cls_idx = src_cls.argmax(dim=-1, keepdims=True)
    src_cls = src_cls.gather(dim=-1, index=cls_idx)
    dst_cls = dst_cls.gather(dim=-1, index=cls_idx)

    src_score = src_obj * src_cls
    dst_score = dst_obj * dst_cls

    weight = dst_score / dst_score.sum()

    score_diff = (weight * (src_score - dst_score).abs()).sum(-1).mean()
    box_diff = (weight * (src_box - dst_box).abs().mean(-1)).sum(-1).mean()

    return box_diff.item(), score_diff.item()

def to_device(data, device, non_blocking=True):
    if data is None:
        return data
    elif isinstance(data, (list, tuple)):
        return [ to_device(d, device, non_blocking) for d in data ]
    elif isinstance(data, torch.Tensor):
        if data.device != device:
            return data.to(device, non_blocking=non_blocking)
        else:
            return data
    else:
        return data






