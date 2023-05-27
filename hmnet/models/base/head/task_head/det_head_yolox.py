#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# Copyright (c) Megvii Inc. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# 
# http://www.apache.org/licenses/LICENSE-2.0
# 
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.

# This file is modified from the original code at
# https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/yolo_head.py
# The list of modifications are as follows:
#   (1) In the function 'get_losses', the number of foreground bboxess
#       'num_fg' is calculated by the average of all distributed processes.
#   (2) A new class 'BBoxHead' is added to reimplement the head for each level as a module.
#   (3) The loss functions are modified so that it can handle ignore bbox.
#   (4) The functions 'postprocess', 'bboxes_iou' are moved from 
#       https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/utils/boxes.py
#   (5) The classes 'BaseConv', 'DWConv', and 'IOUloss' are moved from
#       https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/network_blocks.py
#       https://github.com/Megvii-BaseDetection/YOLOX/blob/main/yolox/models/losses.py


import math
from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributed as dist
import torchvision

from ...init import kaiming_uniform_silu
from ...utils import merge_conv_block, merge_conv2d
from ...blocks import BlockBase
from ...layers import ConvBlock

from torch import Tensor, Size
from typing import Tuple, List, Optional, Dict

class BBoxHead(BlockBase):
    def __init__(self, num_classes, n_anchors, in_channels, feat_channels, act="silu", depthwise=False, use_stem=False, stacked_convs=2):
        super().__init__()
        self.num_classes = num_classes
        self.stacked_convs = stacked_convs
        self.n_anchors = n_anchors

        Conv = DWConv if depthwise else BaseConv

        if use_stem:
            self.stem = BaseConv(
                in_channels=int(in_channels),
                out_channels=feat_channels,
                ksize=1,
                stride=1,
                act=act)
        else:
            self.stem = nn.Identity()

        self.cls_conv = nn.Sequential(*[ Conv(in_channels=feat_channels, out_channels=feat_channels, ksize=3, stride=1, act=act)
                                         for _ in range(stacked_convs) ])

        self.reg_conv = nn.Sequential(*[ Conv(in_channels=feat_channels, out_channels=feat_channels, ksize=3, stride=1, act=act)
                                         for _ in range(stacked_convs) ])

        self.cls_pred = nn.Conv2d(
            in_channels  = feat_channels,
            out_channels = n_anchors * num_classes,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
        )
        self.reg_pred = nn.Conv2d(
            in_channels  = feat_channels,
            out_channels = n_anchors * 4,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
        )
        self.obj_pred = nn.Conv2d(
            in_channels  = feat_channels,
            out_channels = n_anchors * 1,
            kernel_size  = 1,
            stride       = 1,
            padding      = 0,
        )

    @torch.jit.ignore
    def init_weights(self, prior_prob: float=1.0e-2) -> None:
        b = self.cls_pred.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.cls_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

        b = self.obj_pred.bias.view(self.n_anchors, -1)
        b.data.fill_(-math.log((1 - prior_prob) / prior_prob))
        self.obj_pred.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)

    def forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        cls_feat = self.cls_conv(x)
        reg_feat = self.reg_conv(x)
        cls_pred = self.cls_pred(cls_feat)
        reg_pred = self.reg_pred(reg_feat)
        obj_pred = self.obj_pred(reg_feat)
        return reg_pred, obj_pred, cls_pred

    def fast_forward(self, x: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
        x = self.stem(x)
        x = self.convs(x)

        if self.stacked_convs > 0:
            half_dim = int(x.shape[1] // 2)
            cls_feat = x[:,:half_dim]
            reg_feat = x[:,half_dim:]
        else:
            cls_feat = x
            reg_feat = x

        cls_pred = self.cls_pred(cls_feat)
        regobj_pred = self.regobj_pred(reg_feat)
        reg_pred = regobj_pred[:,:4]
        obj_pred = regobj_pred[:,4:]

        return reg_pred, obj_pred, cls_pred

    @torch.jit.ignore
    def _extract_conv_params(self, conv_module: nn.Module) -> None:
        pre_norm = False
        pre_act = False

        conv = conv_module.conv
        norm = conv_module.bn
        act = conv_module.act

        return dict(
            inc         = conv_module.conv.in_channels,
            outc        = conv_module.conv.out_channels,
            kernel_size = conv_module.conv.kernel_size,
            stride      = conv_module.conv.stride,
            padding     = conv_module.conv.padding,
            bias        = conv_module.conv.bias is not None,
            dilation    = conv_module.conv.dilation,
            groups      = conv_module.conv.groups,
            pre_norm    = pre_norm,
            pre_act     = pre_act,
            norm_layer  = type(norm),
            act_layer   = type(act),
        ), conv, norm, act

    @torch.jit.ignore
    def _to_convblock(self, conv_module: nn.Module) -> nn.Module:
        cfg_conv, conv, norm, act = self._extract_conv_params(conv_module)
        new_conv = ConvBlock(**cfg_conv)
        new_conv.conv = conv
        new_conv.norm = norm
        new_conv.act  = act
        return new_conv

    @torch.jit.ignore
    def to_convblock(self) -> nn.Module:
        for i in range(len(self.cls_conv)):
            self.cls_conv[i] = self._to_convblock(self.cls_conv[i])

        for i in range(len(self.reg_conv)):
            self.reg_conv[i] = self._to_convblock(self.reg_conv[i])

    @torch.jit.ignore
    def _to_fast_model(self) -> nn.Module:
        self.to_convblock()

        new_convs = []
        for i, (cls, reg) in enumerate(zip(self.cls_conv, self.reg_conv)):
            new_convs.append(merge_conv_block(cls, reg, group=i>0))

        self.convs = nn.Sequential(*new_convs)
        self.regobj_pred = merge_conv2d(self.reg_pred, self.obj_pred)

        del self.cls_conv
        del self.reg_conv
        del self.reg_pred
        del self.obj_pred

        self.forward = self.fast_forward

        if not self.training:
            self.eval()

        return self

class YOLOXHead(BlockBase):
    def __init__(
        self,
        num_classes,
        strides=[8, 16, 32],
        in_channels=[256, 512, 1024],
        feat_channels=256,
        act="silu",
        depthwise=False,
        use_stem = False,
        score_thr = 0.01,
        nms_iou_threshold = 0.65,
        stacked_convs = 2,
        ignore_bboxes_as_negative = True,
    ):
        """
        Args:
            act (str): activation type of conv. Defalut value: "silu".
            depthwise (bool): whether apply depthwise conv in conv branch. Defalut value: False.
        """
        super().__init__()

        self.n_anchors = 1
        self.num_classes = num_classes
        self.decode_in_inference = False # for deploy, set to False
        self.use_stem = use_stem
        self.score_thr = score_thr
        self.nms_iou_threshold = nms_iou_threshold
        self.stacked_convs = stacked_convs
        self.ignore_bboxes_as_negative = ignore_bboxes_as_negative

        self.levels = nn.ModuleList()

        for i in range(len(in_channels)):
            self.levels.append(
                BBoxHead(
                    num_classes,
                    self.n_anchors,
                    in_channels,
                    feat_channels,
                    act,
                    depthwise,
                    use_stem,
                    stacked_convs)
            )

        self.use_l1 = False
        self.l1_loss = nn.L1Loss(reduction="none")
        self.bcewithlog_loss = nn.BCEWithLogitsLoss(reduction="none")
        self.iou_loss = IOUloss(reduction="none")
        self.strides = strides
        self.grids = [torch.zeros(1)] * len(in_channels)

        for m in self.modules():
            if isinstance(m, nn.BatchNorm2d):
                m.eps = 1e-3
                m.momentum = 0.03

    @torch.jit.ignore
    def init_weights(self, prior_prob: float=1.0e-2) -> None:
        kaiming_uniform_silu(self.modules())
        for bbox_head in self.levels:
            bbox_head.init_weights(prior_prob)

    @torch.jit.ignore
    def _to_tensor_gt(self, gt_bboxes, gt_labels):
        if gt_labels is None and gt_bboxes is None:
            return None
        elif len(gt_labels) == 0:
            return torch.empty(0, 0, 5)
        else:
            B = len(gt_labels)
            N = max([ len(lbl) for lbl in gt_labels ])

            if N == 0:
                return torch.empty(B, N, 5)

            dtype = gt_labels[0].dtype
            device = gt_labels[0].device

            targets = torch.zeros([B, N, 5], dtype=dtype, device=device)

            for i, (bbox, lbl) in enumerate(zip(gt_bboxes, gt_labels)):
                # xyxy -> cxcywh
                bbox = bbox.clone()
                lbl = lbl.clone()
                bbox[:,2] = bbox[:,2] - bbox[:,0]        # w
                bbox[:,3] = bbox[:,3] - bbox[:,1]        # h
                bbox[:,0] = bbox[:,0] + bbox[:,2] * 0.5  # cx
                bbox[:,1] = bbox[:,1] + bbox[:,3] * 0.5  # cy
                num_gt = len(lbl)
                targets[i, :num_gt, 0 ] = lbl
                targets[i, :num_gt, 1:] = bbox

        return targets

    def forward(self, xin: List[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        return self._forward(xin)

    def _forward(self, xin: List[Tensor]) -> Tuple[List[Tensor], List[Tensor], List[Tensor]]:
        x1, x2, x3 = xin
        reg1, obj1, cls1 = self.levels[0](x1)
        reg2, obj2, cls2 = self.levels[1](x2)
        reg3, obj3, cls3 = self.levels[2](x3)

        return (reg1, reg2, reg3), (obj1, obj2, obj3), (cls1, cls2, cls3)

    def inference(self, xin: List[Tensor]) -> Tensor:
        reg_outputs, obj_outputs, cls_outputs = self._forward(xin)

        outputs = []
        for k in range(len(reg_outputs)):
            reg_output = reg_outputs[k]
            obj_output = obj_outputs[k]
            cls_output = cls_outputs[k]
            output = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            outputs.append(output)

        hw = [(x.shape[-2], x.shape[-1]) for x in outputs]
        outputs = torch.cat([x.flatten(start_dim=2) for x in outputs], dim=2).permute(0, 2, 1)    # (B, N_anchors_all, 4 + 1 + N_class)

        outputs = self.decode_outputs(outputs, hw, dtype=outputs[0][0].type())

        return outputs

    def inference_jit(self, x1: Tensor, x2: Tensor, x3: Tensor) -> Tensor:
        return self.inference([x1, x2, x3])

    @torch.jit.ignore
    def compile(self, backend, fp16=False, input_shapes=None):
        jit_compatible_head = YOLOXHeadJIT(self)
        return jit_compatible_head.compile(backend, fp16, input_shapes)

    @torch.jit.ignore
    def _filter_ignore(self, gt_bboxes, gt_labels, ignore_masks):
        out_bboxes, out_labels, out_masks = [], [], []
        for bboxes, labels, masks in zip(gt_bboxes, gt_labels, ignore_masks):
            if len(labels) > 0:
                valid = torch.logical_not(masks)
                bboxes = bboxes[valid]
                labels = labels[valid]
                masks = masks[valid]
            out_bboxes.append(bboxes)
            out_labels.append(labels)
            out_masks.append(masks)
        return out_bboxes, out_labels, out_masks

    @torch.jit.ignore
    def loss(self, outputs, gt_bboxes, gt_labels, ignore_masks, iou_only=False):
        reg_outputs, obj_outputs, cls_outputs = outputs

        if self.ignore_bboxes_as_negative:
            gt_bboxes, gt_labels, ignore_masks = self._filter_ignore(gt_bboxes, gt_labels, ignore_masks)

        labels = self._to_tensor_gt(gt_bboxes, gt_labels)

        outputs = []
        origin_preds = []
        x_shifts = []
        y_shifts = []
        expanded_strides = []

        for k, stride_this_level in enumerate(self.strides):
            reg_output = reg_outputs[k]
            obj_output = obj_outputs[k]
            cls_output = cls_outputs[k]

            output = torch.cat([reg_output, obj_output, cls_output], 1)
            output, grid = self.get_output_and_grid(output, k, stride_this_level, reg_output.type())
            x_shifts.append(grid[:, :, 0])
            y_shifts.append(grid[:, :, 1])
            expanded_strides.append(torch.zeros(1, grid.shape[1]).fill_(stride_this_level).type_as(reg_output))
            if self.use_l1:
                batch_size = reg_output.shape[0]
                hsize, wsize = reg_output.shape[-2:]
                reg_output = reg_output.view(batch_size, self.n_anchors, 4, hsize, wsize)
                reg_output = reg_output.permute(0, 1, 3, 4, 2).reshape(batch_size, -1, 4)
                origin_preds.append(reg_output.clone())
            outputs.append(output)

        if iou_only:
            loss, log_vars = self.get_bbox_loss(
                None,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=reg_output.dtype,
                ignore_masks=ignore_masks,
            )
        else:
            loss, log_vars = self.get_losses(
                None,
                x_shifts,
                y_shifts,
                expanded_strides,
                labels,
                torch.cat(outputs, 1),
                origin_preds,
                dtype=reg_output.dtype,
                ignore_masks=ignore_masks,
            )
        return loss, log_vars

    @torch.jit.ignore
    def get_output_and_grid(self, output, k, stride, dtype):
        grid = self.grids[k]

        batch_size = output.shape[0]
        n_ch = 5 + self.num_classes
        hsize, wsize = output.shape[-2:]
        if grid.shape[2:4] != output.shape[2:4]:
            yv, xv = torch.meshgrid(torch.arange(hsize, device=output.device), torch.arange(wsize, device=output.device), indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, 1, hsize, wsize, 2).type(dtype)
            self.grids[k] = grid

        output = output.view(batch_size, self.n_anchors, n_ch, hsize, wsize)
        output = output.permute(0, 1, 3, 4, 2).reshape(batch_size, self.n_anchors * hsize * wsize, n_ch)
        grid = grid.view(1, -1, 2)
        output[..., :2] = (output[..., :2] + grid) * stride
        output[..., 2:4] = torch.exp(output[..., 2:4]) * stride
        return output, grid

    def decode_outputs(self, outputs: Tensor, hw: List[Tuple[int,int]], dtype: torch.dtype) -> Tensor:
        grids = []
        strides = []
        for (hsize, wsize), stride in zip(hw, self.strides):
            yv, xv = torch.meshgrid(torch.arange(hsize, device=outputs.device), torch.arange(wsize, device=outputs.device), indexing='ij')
            grid = torch.stack((xv, yv), 2).view(1, -1, 2)
            grids.append(grid)
            #shape = grid.shape[:2]
            #strides.append(torch.full((*shape, 1), stride))
            strides.append(torch.full((1, hsize*wsize, 1), stride))

        grids = torch.cat(grids, dim=1).type(dtype)
        strides = torch.cat(strides, dim=1).type(dtype)

        outputs[..., :2] = (outputs[..., :2] + grids) * strides
        outputs[..., 2:4] = torch.exp(outputs[..., 2:4]) * strides
        return outputs

    @torch.jit.ignore
    def postprocess(self, detections):
        return postprocess(detections, self.num_classes, self.score_thr, self.nms_iou_threshold)

    @torch.jit.ignore
    def get_losses(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
        ignore_masks,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)
        if self.use_l1:
            origin_preds = torch.cat(origin_preds, 1)

        cls_targets = []
        reg_targets = []
        l1_targets = []
        obj_targets = []
        fg_masks = []
        valid_masks = []

        num_fg = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            if num_gt == 0:
                cls_target = outputs.new_zeros((0, self.num_classes))
                reg_target = outputs.new_zeros((0, 4))
                l1_target = outputs.new_zeros((0, 4))
                obj_target = outputs.new_zeros((total_num_anchors, 1))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                valid = outputs.new_ones(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                ignore_mask = ignore_masks[batch_idx]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        None,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    print(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        None,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img

                cls_target = F.one_hot(gt_matched_classes.to(torch.int64), self.num_classes) * pred_ious_this_matching.unsqueeze(-1)
                obj_target = fg_mask.unsqueeze(-1)
                reg_target = gt_bboxes_per_image[matched_gt_inds]
                if self.use_l1:
                    l1_target = self.get_l1_target(
                        outputs.new_zeros((num_fg_img, 4)),
                        gt_bboxes_per_image[matched_gt_inds],
                        expanded_strides[0][fg_mask],
                        x_shifts=x_shifts[0][fg_mask],
                        y_shifts=y_shifts[0][fg_mask],
                    )

                # handle ignore label
                ignore = ignore_mask[matched_gt_inds]
                valid_fg = torch.logical_not(ignore)
                valid = torch.ones_like(fg_mask).bool()
                indices = fg_mask.nonzero().view(-1)
                valid[indices[ignore]] = False

                cls_target = cls_target[valid_fg]
                obj_target = obj_target[valid]
                reg_target = reg_target[valid_fg]
                fg_mask = fg_mask[valid]
                if self.use_l1:
                    l1_target = l1_target[valid_fg]

            cls_targets.append(cls_target)
            reg_targets.append(reg_target)
            obj_targets.append(obj_target.to(dtype))
            fg_masks.append(fg_mask)
            valid_masks.append(valid)
            if self.use_l1:
                l1_targets.append(l1_target)

        cls_targets = torch.cat(cls_targets, 0)
        reg_targets = torch.cat(reg_targets, 0)
        obj_targets = torch.cat(obj_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        if self.use_l1:
            l1_targets = torch.cat(l1_targets, 0)

        num_fg = torch.tensor(num_fg, dtype=torch.float, device=cls_preds.device)
        num_fg = max(reduce_mean(num_fg), 1.0)

        # reshape preds
        bbox_preds = bbox_preds.view(-1,4)
        obj_preds = obj_preds.view(-1,1)
        cls_preds = cls_preds.view(-1,self.num_classes)
        if self.use_l1:
            origin_preds = origin_preds.view(-1, 4)

        # filter out preds assigned to ignore bboxes
        valid_masks = torch.cat(valid_masks, 0)
        bbox_preds = bbox_preds[valid_masks]
        obj_preds = obj_preds[valid_masks]
        cls_preds = cls_preds[valid_masks]
        if self.use_l1:
            origin_preds = origin_preds[valid_masks]

        loss_iou = self.iou_loss(bbox_preds[fg_masks], reg_targets).sum() / num_fg
        loss_obj = self.bcewithlog_loss(obj_preds, obj_targets).sum() / num_fg
        loss_cls = self.bcewithlog_loss(cls_preds[fg_masks], cls_targets).sum() / num_fg
        if self.use_l1:
            loss_l1 = self.l1_loss(origin_preds[fg_masks], l1_targets).sum() / num_fg
        else:
            loss_l1 = 0.0

        reg_weight = 5.0
        loss = reg_weight * loss_iou + loss_obj + loss_cls + loss_l1

        loss, log_vars = self._parse_losses(loss, reg_weight * loss_iou, loss_obj, loss_cls, loss_l1)

        return loss, log_vars

    @torch.jit.ignore
    def _parse_losses(self, loss, loss_iou, loss_obj, loss_cls, loss_l1):
        log_vars = OrderedDict()
        log_vars['loss_cls'] = loss_cls.item()
        log_vars['loss_bbox'] = loss_l1.item() if isinstance(loss_l1, torch.Tensor) else loss_l1
        log_vars['loss_obj'] = loss_obj.item()
        log_vars['loss_iou'] = loss_iou.item()
        log_vars['loss'] = loss.item()

        return loss, log_vars


    @torch.jit.ignore
    def get_l1_target(self, l1_target, gt, stride, x_shifts, y_shifts, eps=1e-8):
        l1_target[:, 0] = gt[:, 0] / stride - x_shifts
        l1_target[:, 1] = gt[:, 1] / stride - y_shifts
        l1_target[:, 2] = torch.log(gt[:, 2] / stride + eps)
        l1_target[:, 3] = torch.log(gt[:, 3] / stride + eps)
        return l1_target

    @torch.no_grad()
    @torch.jit.ignore
    def get_assignments(
        self,
        batch_idx,
        num_gt,
        total_num_anchors,
        gt_bboxes_per_image,
        gt_classes,
        bboxes_preds_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        cls_preds,
        bbox_preds,
        obj_preds,
        labels,
        imgs,
        mode="gpu",
    ):

        if mode == "cpu":
            print("------------CPU Mode for This Batch-------------")
            gt_bboxes_per_image = gt_bboxes_per_image.cpu().float()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu().float()
            gt_classes = gt_classes.cpu().float()
            expanded_strides = expanded_strides.cpu().float()
            x_shifts = x_shifts.cpu()
            y_shifts = y_shifts.cpu()

        fg_mask, is_in_boxes_and_center = self.get_in_boxes_info(
            gt_bboxes_per_image,
            expanded_strides,
            x_shifts,
            y_shifts,
            total_num_anchors,
            num_gt,
        )

        bboxes_preds_per_image = bboxes_preds_per_image[fg_mask]
        cls_preds_ = cls_preds[batch_idx][fg_mask]
        obj_preds_ = obj_preds[batch_idx][fg_mask]
        num_in_boxes_anchor = bboxes_preds_per_image.shape[0]

        if mode == "cpu":
            gt_bboxes_per_image = gt_bboxes_per_image.cpu()
            bboxes_preds_per_image = bboxes_preds_per_image.cpu()

        # (cx, cy, w, h) -> (x1, y1, x2, y2)
        gt_bboxes_xyxy = gt_bboxes_per_image.clone()
        gt_bboxes_xyxy[:,:2] = gt_bboxes_per_image[:,:2] - (gt_bboxes_per_image[:,2:] * 0.5).long()
        gt_bboxes_xyxy[:,2:] = gt_bboxes_xyxy[:,:2] + gt_bboxes_per_image[:,2:]
        bboxes_preds_xyxy = bboxes_preds_per_image.clone()
        bboxes_preds_xyxy[:,:2] = bboxes_preds_per_image[:,:2] - (bboxes_preds_per_image[:,2:] * 0.5)
        bboxes_preds_xyxy[:,2:] = bboxes_preds_xyxy[:,:2] + bboxes_preds_per_image[:,2:]

        pair_wise_ious = bboxes_iou(gt_bboxes_xyxy, bboxes_preds_xyxy, True)
        #pair_wise_ious = bboxes_iou(gt_bboxes_per_image, bboxes_preds_per_image, False)

        gt_cls_per_image = (
            F.one_hot(gt_classes.to(torch.int64), self.num_classes)
            .float()
            .unsqueeze(1)
            .repeat(1, num_in_boxes_anchor, 1)
        )
        pair_wise_ious_loss = -torch.log(pair_wise_ious + 1e-7)

        if mode == "cpu":
            cls_preds_, obj_preds_ = cls_preds_.cpu(), obj_preds_.cpu()

        with torch.cuda.amp.autocast(enabled=False):
            cls_preds_ = (
                cls_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
                * obj_preds_.float().unsqueeze(0).repeat(num_gt, 1, 1).sigmoid_()
            )
            pair_wise_cls_loss = F.binary_cross_entropy(
                cls_preds_.sqrt_(), gt_cls_per_image, reduction="none"
            ).sum(-1)
        del cls_preds_

        cost = (
            pair_wise_cls_loss
            + 3.0 * pair_wise_ious_loss
            + 100000.0 * (~is_in_boxes_and_center)
        )

        (
            num_fg,
            gt_matched_classes,
            pred_ious_this_matching,
            matched_gt_inds,
        ) = self.dynamic_k_matching(cost, pair_wise_ious, gt_classes, num_gt, fg_mask)
        del pair_wise_cls_loss, cost, pair_wise_ious, pair_wise_ious_loss

        if mode == "cpu":
            gt_matched_classes = gt_matched_classes.cuda()
            fg_mask = fg_mask.cuda()
            pred_ious_this_matching = pred_ious_this_matching.cuda()
            matched_gt_inds = matched_gt_inds.cuda()

        return (
            gt_matched_classes,
            fg_mask,
            pred_ious_this_matching,
            matched_gt_inds,
            num_fg,
        )

    @torch.jit.ignore
    def get_in_boxes_info(
        self,
        gt_bboxes_per_image,
        expanded_strides,
        x_shifts,
        y_shifts,
        total_num_anchors,
        num_gt,
    ):
        expanded_strides_per_image = expanded_strides[0]
        x_shifts_per_image = x_shifts[0] * expanded_strides_per_image
        y_shifts_per_image = y_shifts[0] * expanded_strides_per_image
        x_centers_per_image = (
            (x_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )  # [n_anchor] -> [n_gt, n_anchor]
        y_centers_per_image = (
            (y_shifts_per_image + 0.5 * expanded_strides_per_image)
            .unsqueeze(0)
            .repeat(num_gt, 1)
        )

        gt_bboxes_per_image_l = gt_bboxes_per_image[:, 0] - (0.5 * gt_bboxes_per_image[:, 2]).long()
        gt_bboxes_per_image_l = gt_bboxes_per_image_l[:,None].repeat(1, total_num_anchors)
        gt_bboxes_per_image_r = gt_bboxes_per_image_l + gt_bboxes_per_image[:,2].unsqueeze(1)

        gt_bboxes_per_image_t = gt_bboxes_per_image[:, 1] - (0.5 * gt_bboxes_per_image[:, 3]).long()
        gt_bboxes_per_image_t = gt_bboxes_per_image_t[:,None].repeat(1, total_num_anchors)
        gt_bboxes_per_image_b = gt_bboxes_per_image_t + gt_bboxes_per_image[:,3].unsqueeze(1)

        b_l = x_centers_per_image - gt_bboxes_per_image_l
        b_r = gt_bboxes_per_image_r - x_centers_per_image
        b_t = y_centers_per_image - gt_bboxes_per_image_t
        b_b = gt_bboxes_per_image_b - y_centers_per_image
        bbox_deltas = torch.stack([b_l, b_t, b_r, b_b], 2)

        is_in_boxes = bbox_deltas.min(dim=-1).values > 0.0
        is_in_boxes_all = is_in_boxes.sum(dim=0) > 0
        # in fixed center

        gt_center_x = (gt_bboxes_per_image_l + gt_bboxes_per_image_r) * 0.5    # (Ngt, Na)
        gt_center_y = (gt_bboxes_per_image_t + gt_bboxes_per_image_b) * 0.5    # (Ngt, Na)

        center_radius = 2.5
        radius = center_radius * expanded_strides_per_image[None,:]
        gt_bboxes_per_image_l = gt_center_x - radius
        gt_bboxes_per_image_r = gt_center_x + radius
        gt_bboxes_per_image_t = gt_center_y - radius
        gt_bboxes_per_image_b = gt_center_y + radius

        c_l = x_centers_per_image - gt_bboxes_per_image_l
        c_r = gt_bboxes_per_image_r - x_centers_per_image
        c_t = y_centers_per_image - gt_bboxes_per_image_t
        c_b = gt_bboxes_per_image_b - y_centers_per_image
        center_deltas = torch.stack([c_l, c_t, c_r, c_b], 2)
        is_in_centers = center_deltas.min(dim=-1).values > 0.0
        is_in_centers_all = is_in_centers.sum(dim=0) > 0

        # in boxes and in centers
        is_in_boxes_anchor = is_in_boxes_all | is_in_centers_all

        is_in_boxes_and_center = (
            is_in_boxes[:, is_in_boxes_anchor] & is_in_centers[:, is_in_boxes_anchor]
        )
        return is_in_boxes_anchor, is_in_boxes_and_center

    @torch.jit.ignore
    def dynamic_k_matching(self, cost, pair_wise_ious, gt_classes, num_gt, fg_mask):
        # Dynamic K
        # ---------------------------------------------------------------
        matching_matrix = torch.zeros_like(cost, dtype=torch.uint8)

        ious_in_boxes_matrix = pair_wise_ious
        n_candidate_k = min(10, ious_in_boxes_matrix.size(1))
        topk_ious, _ = torch.topk(ious_in_boxes_matrix, n_candidate_k, dim=1)
        dynamic_ks = torch.clamp(topk_ious.sum(1).int(), min=1)
        dynamic_ks = dynamic_ks.tolist()
        for gt_idx in range(num_gt):
            _, pos_idx = torch.topk(
                cost[gt_idx], k=dynamic_ks[gt_idx], largest=False
            )
            #matching_matrix[gt_idx][pos_idx] = torch.ones(len(pos_idx), device=matching_matrix.device, dtype=matching_matrix.dtype)
            matching_matrix[gt_idx][pos_idx] = 1

        del topk_ious, dynamic_ks, pos_idx

        anchor_matching_gt = matching_matrix.sum(0)
        if (anchor_matching_gt > 1).sum() > 0:
            _, cost_argmin = torch.min(cost[:, anchor_matching_gt > 1], dim=0)
            matching_matrix[:, anchor_matching_gt > 1] *= 0
            matching_matrix[cost_argmin, anchor_matching_gt > 1] = 1
        fg_mask_inboxes = matching_matrix.sum(0) > 0
        num_fg = fg_mask_inboxes.sum().item()

        fg_mask[fg_mask.clone()] = fg_mask_inboxes

        matched_gt_inds = matching_matrix[:, fg_mask_inboxes].argmax(0)
        gt_matched_classes = gt_classes[matched_gt_inds]

        pred_ious_this_matching = (matching_matrix * pair_wise_ious).sum(0)[
            fg_mask_inboxes
        ]
        return num_fg, gt_matched_classes, pred_ious_this_matching, matched_gt_inds

    @torch.jit.ignore
    def get_bbox_loss(
        self,
        imgs,
        x_shifts,
        y_shifts,
        expanded_strides,
        labels,
        outputs,
        origin_preds,
        dtype,
        ignore_masks,
    ):
        bbox_preds = outputs[:, :, :4]  # [batch, n_anchors_all, 4]
        obj_preds = outputs[:, :, 4].unsqueeze(-1)  # [batch, n_anchors_all, 1]
        cls_preds = outputs[:, :, 5:]  # [batch, n_anchors_all, n_cls]

        # calculate targets
        nlabel = (labels.sum(dim=2) > 0).sum(dim=1)  # number of objects

        total_num_anchors = outputs.shape[1]
        x_shifts = torch.cat(x_shifts, 1)  # [1, n_anchors_all]
        y_shifts = torch.cat(y_shifts, 1)  # [1, n_anchors_all]
        expanded_strides = torch.cat(expanded_strides, 1)

        reg_targets = []
        fg_masks = []
        valid_masks = []

        num_fg = 0.0

        for batch_idx in range(outputs.shape[0]):
            num_gt = int(nlabel[batch_idx])
            if num_gt == 0:
                reg_target = outputs.new_zeros((0, 4))
                fg_mask = outputs.new_zeros(total_num_anchors).bool()
                valid = outputs.new_ones(total_num_anchors).bool()
            else:
                gt_bboxes_per_image = labels[batch_idx, :num_gt, 1:5]
                gt_classes = labels[batch_idx, :num_gt, 0]
                ignore_mask = ignore_masks[batch_idx]
                bboxes_preds_per_image = bbox_preds[batch_idx]

                try:
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        None,
                    )
                except RuntimeError as e:
                    # TODO: the string might change, consider a better way
                    if "CUDA out of memory. " not in str(e):
                        raise  # RuntimeError might not caused by CUDA OOM

                    print(
                        "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                           CPU mode is applied in this batch. If you want to avoid this issue, \
                           try to reduce the batch size or image size."
                    )
                    torch.cuda.empty_cache()
                    (
                        gt_matched_classes,
                        fg_mask,
                        pred_ious_this_matching,
                        matched_gt_inds,
                        num_fg_img,
                    ) = self.get_assignments(  # noqa
                        batch_idx,
                        num_gt,
                        total_num_anchors,
                        gt_bboxes_per_image,
                        gt_classes,
                        bboxes_preds_per_image,
                        expanded_strides,
                        x_shifts,
                        y_shifts,
                        cls_preds,
                        bbox_preds,
                        obj_preds,
                        labels,
                        None,
                        "cpu",
                    )

                torch.cuda.empty_cache()
                num_fg += num_fg_img
                reg_target = gt_bboxes_per_image[matched_gt_inds]

                # handle ignore label
                ignore = ignore_mask[matched_gt_inds]
                valid_fg = torch.logical_not(ignore)
                valid = torch.ones_like(fg_mask).bool()
                indices = fg_mask.nonzero().view(-1)
                valid[indices[ignore]] = False

                reg_target = reg_target[valid_fg]
                fg_mask = fg_mask[valid]

            reg_targets.append(reg_target)
            fg_masks.append(fg_mask)
            valid_masks.append(valid)

        reg_targets = torch.cat(reg_targets, 0)
        fg_masks = torch.cat(fg_masks, 0)
        valid_masks = torch.cat(valid_masks, 0)

        num_fg = torch.tensor(num_fg, dtype=torch.float, device=cls_preds.device)
        num_fg = max(reduce_mean(num_fg), 1.0)

        bbox_preds = bbox_preds.view(-1, 4)[valid_masks]
        loss_iou = self.iou_loss(bbox_preds[fg_masks], reg_targets).sum() / num_fg

        reg_weight = 5.0
        loss = reg_weight * loss_iou

        log_vars = OrderedDict()
        log_vars['loss_iou'] = reg_weight * loss_iou.item()

        return loss, log_vars

class IOUloss(nn.Module):
    def __init__(self, reduction: str="none", loss_type: str="iou") -> None:
        super(IOUloss, self).__init__()
        self.reduction = reduction
        self.loss_type = loss_type

    @torch.jit.ignore
    def forward(self, pred, target):
        assert pred.shape[0] == target.shape[0]

        pred = pred.view(-1, 4)
        target = target.view(-1, 4)
        tl_p = pred[:, :2] - pred[:, 2:] / 2
        br_p = pred[:, :2] + pred[:, 2:] / 2
        tl_t = target[:, :2] - (target[:, 2:] * 0.5).long()
        br_t = tl_t + target[:, 2:]
        tl = torch.max(tl_p, tl_t)
        br = torch.min(br_p, br_t)

        area_p = torch.prod(pred[:, 2:], 1)
        area_g = torch.prod(target[:, 2:], 1)

        en = (tl < br).type(tl.type()).prod(dim=1)
        area_i = torch.prod(br - tl, 1) * en
        area_u = area_p + area_g - area_i
        iou = (area_i) / (area_u + 1e-16)

        if self.loss_type == "iou":
            loss = 1 - iou ** 2
        elif self.loss_type == "giou":
            c_tl = torch.min(
                (pred[:, :2] - pred[:, 2:] / 2), (target[:, :2] - target[:, 2:] / 2)
            )
            c_br = torch.max(
                (pred[:, :2] + pred[:, 2:] / 2), (target[:, :2] + target[:, 2:] / 2)
            )
            area_c = torch.prod(c_br - c_tl, 1)
            giou = iou - (area_c - area_u) / area_c.clamp(1e-16)
            loss = 1 - giou.clamp(min=-1.0, max=1.0)

        if self.reduction == "mean":
            loss = loss.mean()
        elif self.reduction == "sum":
            loss = loss.sum()

        return loss

@torch.jit.ignore
def reduce_mean(tensor):
    if not (dist.is_available() and dist.is_initialized()):
        return tensor
    tensor = tensor.clone()
    dist.all_reduce(tensor.div_(dist.get_world_size()), op=dist.ReduceOp.SUM)
    return tensor

@torch.jit.ignore
def postprocess(prediction, num_classes, conf_thre=0.7, nms_thre=0.45, class_agnostic=False):
    box_corner = prediction.new(prediction.shape)
    box_corner[:, :, 0] = prediction[:, :, 0] - prediction[:, :, 2] / 2
    box_corner[:, :, 1] = prediction[:, :, 1] - prediction[:, :, 3] / 2
    box_corner[:, :, 2] = prediction[:, :, 0] + prediction[:, :, 2] / 2
    box_corner[:, :, 3] = prediction[:, :, 1] + prediction[:, :, 3] / 2
    prediction[:, :, :4] = box_corner[:, :, :4]

    output = [None for _ in range(len(prediction))]
    for i, image_pred in enumerate(prediction):

        # If none are remaining => process next image
        if not image_pred.size(0):
            continue
        # Get score and class with highest confidence
        class_conf, class_pred = torch.max(image_pred[:, 5: 5 + num_classes], 1, keepdim=True)

        conf_mask = (image_pred[:, 4] * class_conf.squeeze() >= conf_thre).squeeze()
        # Detections ordered as (x1, y1, x2, y2, obj_conf, class_conf, class_pred)
        detections = torch.cat((image_pred[:, :5], class_conf, class_pred.float()), 1)
        detections = detections[conf_mask]
        if not detections.size(0):
            continue

        if class_agnostic:
            nms_out_index = torchvision.ops.nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                nms_thre,
            )
        else:
            nms_out_index = torchvision.ops.batched_nms(
                detections[:, :4],
                detections[:, 4] * detections[:, 5],
                detections[:, 6],
                nms_thre,
            )

        detections = detections[nms_out_index]
        if output[i] is None:
            output[i] = detections
        else:
            output[i] = torch.cat((output[i], detections))

    return output

@torch.jit.ignore
def bboxes_iou(bboxes_a, bboxes_b, xyxy=True):
    if bboxes_a.shape[1] != 4 or bboxes_b.shape[1] != 4:
        raise IndexError

    if xyxy:
        tl = torch.max(bboxes_a[:, None, :2], bboxes_b[:, :2])
        br = torch.min(bboxes_a[:, None, 2:], bboxes_b[:, 2:])
        area_a = torch.prod(bboxes_a[:, 2:] - bboxes_a[:, :2], 1)
        area_b = torch.prod(bboxes_b[:, 2:] - bboxes_b[:, :2], 1)
    else:
        tl_a = bboxes_a[:, :2] - (bboxes_a[:, 2:] * 0.5).long()
        tl_b = bboxes_b[:, :2] - (bboxes_b[:, 2:] * 0.5).long()
        br_a = tl_a + bboxes_a[:, 2:]
        br_b = tl_b + bboxes_b[:, 2:]
        tl = torch.max(tl_a[:,None], tl_b[None,:])
        br = torch.min(br_a[:,None], br_b[None,:])
        area_a = torch.prod(bboxes_a[:, 2:], 1)
        area_b = torch.prod(bboxes_b[:, 2:], 1)
    en = (tl < br).type(tl.type()).prod(dim=2)
    area_i = torch.prod(br - tl, 2) * en  # * ((tl < br).all())
    return area_i / (area_a[:, None] + area_b - area_i)

def get_activation(name="silu", inplace=True):
    if name == "silu":
        module = nn.SiLU(inplace=inplace)
    elif name == "relu":
        module = nn.ReLU(inplace=inplace)
    elif name == "lrelu":
        module = nn.LeakyReLU(0.1, inplace=inplace)
    else:
        raise AttributeError("Unsupported act type: {}".format(name))
    return module


class BaseConv(nn.Module):
    """A Conv2d -> Batchnorm -> silu/leaky relu block"""

    def __init__(
        self, in_channels, out_channels, ksize, stride, groups=1, bias=False, act="silu"
    ):
        super().__init__()
        # same padding
        pad = (ksize - 1) // 2
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ksize,
            stride=stride,
            padding=pad,
            groups=groups,
            bias=bias,
        )
        self.bn = nn.BatchNorm2d(out_channels)
        self.act = get_activation(act, inplace=True)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def fuseforward(self, x):
        return self.act(self.conv(x))


class DWConv(nn.Module):
    """Depthwise Conv + Conv"""

    def __init__(self, in_channels, out_channels, ksize, stride=1, act="silu"):
        super().__init__()
        self.dconv = BaseConv(
            in_channels,
            in_channels,
            ksize=ksize,
            stride=stride,
            groups=in_channels,
            act=act,
        )
        self.pconv = BaseConv(
            in_channels, out_channels, ksize=1, stride=1, groups=1, act=act
        )

    def forward(self, x):
        x = self.dconv(x)
        return self.pconv(x)


# JIT compatible head wrapper for TorchScript and TensorRT
class YOLOXHeadJIT(BlockBase):
    def __init__(self, head):
        super().__init__()
        self.head = head
        self.num_classes = head.num_classes
        self.score_thr = head.score_thr
        self.nms_iou_threshold = head.nms_iou_threshold

    def compile(self, backend, fp16, input_shapes):
        self.head.forward = self.head.inference_jit
        self.head = super(type(self.head), self.head).compile(backend, fp16, input_shapes)
        return self

    def postprocess(self, detections):
        return postprocess(detections, self.num_classes, self.score_thr, self.nms_iou_threshold)

    def inference(self, xin: List[Tensor]) -> Tensor:
        x1, x2, x3 = xin
        return self.head(x1, x2, x3)


