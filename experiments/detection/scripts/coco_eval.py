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

# Copyright (c) Prophesee S.A.
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
# https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/metrics/coco_eval.py
# The list of modifications are as follows:
# (1) add revised function "_match_times_rev" that matches GT boxes to detected boxes with nearest timestamp
# (2) "evaluate_detection" returns COCOeval instance

"""
Compute the COCO metric on bounding box files by matching timestamps

Copyright: (c) 2019-2020 Prophesee
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def evaluate_detection(gt_boxes_list, dt_boxes_list, classes=("car", "pedestrian"), height=240, width=304,
                       time_tol=50000):
    """
    Compute detection KPIs on list of boxes in the numpy format, using the COCO python API
    https://github.com/cocodataset/cocoapi
    KPIs are only computed on timestamps where there is actual at least one box
    (fully empty frames are not considered)

    :param gt_boxes_list: list of numpy array for GT boxes (one per file)
    :param dt_boxes_list: list of numpy array for detected boxes
    :param classes: iterable of classes names
    :param height: int for box size statistics
    :param width: int for box size statistics
    :param time_tol: int size of the temporal window in micro seconds to look for a detection around a gt box
    """
    flattened_gt = []
    flattened_dt = []
    for gt_boxes, dt_boxes in zip(gt_boxes_list, dt_boxes_list):

        assert np.all(gt_boxes['t'][1:] >= gt_boxes['t'][:-1])
        assert np.all(dt_boxes['t'][1:] >= dt_boxes['t'][:-1])

        all_ts = np.unique(gt_boxes['t'])
        n_steps = len(all_ts)

        gt_win, dt_win = _match_times_rev(all_ts, gt_boxes, dt_boxes, time_tol)
        flattened_gt = flattened_gt + gt_win
        flattened_dt = flattened_dt + dt_win

    return _coco_eval(flattened_gt, flattened_dt, height, width, labelmap=classes)


def _match_times_rev(all_ts, gt_boxes, dt_boxes, time_tol):
    windowed_gt = []
    windowed_dt = []

    dt_ts = np.unique(dt_boxes['t'])

    for ts in all_ts:
        windowed_gt.append(gt_boxes[gt_boxes['t'] == ts])

        # nearest neighbor search
        dist = np.abs(dt_ts - ts)
        nn_idx = np.argmin(dist)
        nn_ts = dt_ts[nn_idx]
        if dist[nn_idx] < time_tol:
            windowed_dt.append(dt_boxes[dt_boxes['t'] == nn_ts])
        else:
            windowed_dt.append(dt_boxes[:0])

    return windowed_gt, windowed_dt

def _match_times(all_ts, gt_boxes, dt_boxes, time_tol):
    """
    match ground truth boxes and ground truth detections at all timestamps using a specified tolerance
    return a list of boxes vectors
    """
    gt_size = len(gt_boxes)
    dt_size = len(dt_boxes)

    windowed_gt = []
    windowed_dt = []

    low_gt, high_gt = 0, 0
    low_dt, high_dt = 0, 0
    for ts in all_ts:

        while low_gt < gt_size and gt_boxes[low_gt]['t'] < ts:
            low_gt += 1
        # the high index is at least as big as the low one
        high_gt = max(low_gt, high_gt)
        while high_gt < gt_size and gt_boxes[high_gt]['t'] <= ts:
            high_gt += 1

        # detection are allowed to be inside a window around the right detection timestamp
        low = ts - time_tol
        high = ts + time_tol
        while low_dt < dt_size and dt_boxes[low_dt]['t'] < low:
            low_dt += 1
        # the high index is at least as big as the low one
        high_dt = max(low_dt, high_dt)
        while high_dt < dt_size and dt_boxes[high_dt]['t'] <= high:
            high_dt += 1

        windowed_gt.append(gt_boxes[low_gt:high_gt])
        windowed_dt.append(dt_boxes[low_dt:high_dt])

    return windowed_gt, windowed_dt


def _coco_eval(gts, detections, height, width, labelmap=("car", "pedestrian")):
    """simple helper function wrapping around COCO's Python API
    :params:  gts iterable of numpy boxes for the ground truth
    :params:  detections iterable of numpy boxes for the detections
    :params:  height int
    :params:  width int
    :params:  labelmap iterable of class labels
    """
    categories = [{"id": id + 1, "name": class_name, "supercategory": "none"}
                  for id, class_name in enumerate(labelmap)]

    dataset, results = _to_coco_format(gts, detections, categories, height=height, width=width)

    coco_gt = COCO()
    coco_gt.dataset = dataset
    coco_gt.createIndex()
    coco_pred = coco_gt.loadRes(results)

    coco_eval = COCOeval(coco_gt, coco_pred, 'bbox')
    coco_eval.params.imgIds = np.arange(1, len(gts) + 1, dtype=int)
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval


def _to_coco_format(gts, detections, categories, height=240, width=304):
    """
    utilitary function producing our data in a COCO usable format
    """
    annotations = []
    results = []
    images = []

    # to dictionary
    for image_id, (gt, pred) in enumerate(zip(gts, detections)):
        im_id = image_id + 1

        images.append(
            {"date_captured": "2019",
             "file_name": "n.a",
             "id": im_id,
             "license": 1,
             "url": "",
             "height": height,
             "width": width})

        for bbox in gt:
            x1, y1 = bbox['x'], bbox['y']
            w, h = bbox['w'], bbox['h']
            area = w * h

            annotation = {
                "area": float(area),
                "iscrowd": False,
                "image_id": im_id,
                "bbox": [x1, y1, w, h],
                "category_id": int(bbox['class_id']) + 1,
                "id": len(annotations) + 1
            }
            annotations.append(annotation)

        for bbox in pred:

            image_result = {
                'image_id': im_id,
                'category_id': int(bbox['class_id']) + 1,
                'score': float(bbox['class_confidence']),
                'bbox': [bbox['x'], bbox['y'], bbox['w'], bbox['h']],
            }
            results.append(image_result)

    dataset = {"info": {},
               "licenses": [],
               "type": 'instances',
               "images": images,
               "annotations": annotations,
               "categories": categories}
    return dataset, results