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

import numpy as np
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

class HMSeg(BlockBase):
    def __init__(self, backbone, neck, seg_head, aux_head, devices, test_aug=None) -> None:
        super().__init__()
        self.devices = devices

        self.backbone = build_backbone(backbone)
        self.neck     = build_neck(neck)
        self.seg_head = build_head(seg_head)
        self.aux_head = build_head(aux_head)
        self.test_aug = test_aug
        self.set_module_names()

    def init_weights(self):
        init_transformer(self.modules())
        self.backbone.init_weights()
        self.neck.init_weights()
        self.seg_head.init_weights()

    def to_cuda(self):
        d0 = self.devices[0]
        if hasattr(self.backbone, 'to_cuda'):
            self.backbone.to_cuda(*self.devices[1:])
        else:
            self.backbone = self.backbone.to(d0)
        self.neck = self.neck.to(d0)
        self.seg_head = self.seg_head.to(d0)
        self.aux_head = self.aux_head.to(d0)

    def forward(self, list_events, list_images, list_image_metas, list_gt, init_states=True) -> Tensor:
        if init_states:
            self.idx_offset = 0

        # Gather gts for loss calculation
        gather_indices = self._identify_required_outputs_batch(list_gt, self.idx_offset)
        out_image_metas = self._gather(list_image_metas, gather_indices)
        out_gt = self._gather(list_gt, gather_indices)
        out_gt = torch.stack(out_gt)

        # Backbone
        outputs = self.backbone(list_events, list_image_metas, gather_indices, list_images=list_images, init_states=init_states, detach=True, fast_training=True)

        # Neck, Head
        loss, log_vars = self._forward_head(outputs, out_image_metas, out_gt, gather_indices['batch'])

        # Make sure all parameters are involved in loss calculation
        # e.g., parameters for event embedding will not be used when len(events)==0
        loss = loss + sum([ 0. * params.sum() for params in self.parameters() ])

        outputs = dict(loss=loss, log_vars=log_vars, num_samples=len(out_image_metas))

        self.idx_offset += len(list_events)

        return outputs

    def inference(self, list_events, list_images, list_image_metas, speed_test=False) -> Tensor:
        output = []
        output_image_metas = []
        d0 = self.devices[0]
        d3 = self.devices[-1]
        s0 = torch.cuda.Stream(device=d0)
        timer = Timer(enabled=speed_test)

        batch_size = len(list_events[0])
        height = list_image_metas[0][0]['height']
        width = list_image_metas[0][0]['width']

        self.backbone.prepair_for_inference(batch_size, image_size=(height, width))

        for idx, (events, images, image_metas) in enumerate(zip(list_events, list_images, list_image_metas)):
            events = to_device(events, d0)
            images = to_device(images, d3)

            timer.lazy_start(43)

            events, images, image_metas = self._test_transform(events, images, image_metas)

            features = self.backbone.inference(events, image_metas, images)

            valid = sum([ f is None for f in features ]) == 0

            if valid:
                features = [ f.to(d0) for f in features ]

                s0.wait_stream(torch.cuda.current_stream())
                with torch.cuda.stream(s0):
                    x = self.neck(features)
                    preds = self.seg_head.inference(x, image_metas)
                    preds, image_metas = self._backward_transform(preds, image_metas)

                s0.synchronize()
                timer.tick(sync_method='none')  # already synced

                if image_metas[0]['label_path'] is not None:    # correspond to gt timing
                    output += preds.cpu()
                    output_image_metas += image_metas

        timer.display()

        self.backbone.termination()

        return torch.stack(output), output_image_metas

    def _forward_head(self, features, image_metas, gt, batch_indices):
        # Neck
        x = self.neck(features)

        # Head
        pred = self.seg_head(x, image_metas)
        loss, log_vars = self.seg_head.loss(pred, gt, image_metas)

        pred_aux = self.aux_head(features[0], image_metas)
        loss_aux, log_vars_aux = self.aux_head.loss(pred_aux, gt, image_metas)

        loss = loss + loss_aux
        log_vars.update(log_vars_aux)

        return loss, log_vars

    def _gather(self, list_data, gather_indices):
        time_indices = gather_indices['time']
        batch_indices = gather_indices['batch']
        destination = torch.arange(len(time_indices))

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

    def _test_transform(self, events, images, img_metas):
        if self.test_aug is None:
            return events, images, img_metas

        out_events, out_images, out_img_metas = [], [], []
        for evt, img, img_meta in zip(events, images, img_metas):
            evt, img, img_meta = self.test_aug(evt, img, img_meta, types=['event', 'image', 'meta'])
            out_events.append(evt)
            out_images.append(img)
            out_img_metas.append(img_meta)

        if out_images[0] is not None:
            out_images = torch.stack(out_images)

        return out_events, out_images, out_img_metas

    def _backward_transform(self, preds, img_metas):
        if self.test_aug is None:
            return preds, img_metas

        out_preds, out_img_metas = [], []
        for pred, img_meta in zip(preds, img_metas):
            pred, img_meta = self.test_aug(pred, img_meta, types=['image', 'meta'])
            out_preds.append(pred)
            out_img_metas.append(img_meta)
        return torch.stack(out_preds), out_img_metas

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


