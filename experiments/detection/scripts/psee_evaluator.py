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
# https://github.com/prophesee-ai/prophesee-automotive-dataset-toolbox/blob/master/src/psee_evaluator.py
# The list of modifications are as follows:
# (1) "min_box_side" for box filtering is modified following the previous work:
#     Perot, Etienne, et al. "Learning to detect objects with a 1 megapixel event camera." Advances in Neural Information Processing Systems 33 (2020): 16639-16652.
# (2) Configs for GEN1 and GEN4 are added and passed to "evaluate_detection"

import glob
import numpy as np
import os
import argparse
import pickle as pkl

from coco_eval import evaluate_detection
from hmnet.utils.psee_toolbox.io.box_filtering import filter_boxes
from hmnet.utils.psee_toolbox.io.box_loading import reformat_boxes
from hmnet.utils.common import get_list, mkdir

EVAL_CONF_GEN1 = dict(
    classes = ('car', 'pedestrian'),
    width = 304,
    height = 240,
    time_tol = 25000, # +/- 25 msec (50 msec)
)

EVAL_CONF_GEN4 = dict(
    classes = ('pedestrian', 'two wheeler', 'car', 'truck', 'bus', 'traffic sign', 'traffic light'),
    width = 1280,
    height = 720,
    time_tol = 25000, # +/- 25 msec (50 msec)
)

def evaluate_folders(dt_folder, gt_lst, camera):
    dt_file_paths = get_list(dt_folder, ext='npy')
    gt_file_paths = get_list(gt_lst, ext='npy')
    assert len(dt_file_paths) == len(gt_file_paths)
    print("There are {} GT bboxes and {} PRED bboxes".format(len(gt_file_paths), len(dt_file_paths)))
    result_boxes_list = [np.load(p) for p in dt_file_paths]
    gt_boxes_list = [np.load(p) for p in gt_file_paths]

    result_boxes_list = [reformat_boxes(p) for p in result_boxes_list]
    gt_boxes_list = [reformat_boxes(p) for p in gt_boxes_list]

    min_box_diag = 60 if camera == 'GEN4' else 30
    min_box_side = 20 if camera == 'GEN4' else 10
    eval_conf = EVAL_CONF_GEN4 if camera == 'GEN4' else EVAL_CONF_GEN1

    filter_boxes_fn = lambda x:filter_boxes(x, int(5e5), min_box_diag, min_box_side)

    gt_boxes_list = map(filter_boxes_fn, gt_boxes_list)
    result_boxes_list = map(filter_boxes_fn, result_boxes_list)
    evaluate_detection(gt_boxes_list, result_boxes_list, **eval_conf)

def main():
    parser = argparse.ArgumentParser(prog='psee_evaluator.py')
    parser.add_argument('gt_lst', type=str, help='Text file contaiing list of GT .npy files')
    parser.add_argument('dt_folder', type=str, help='RESULT folder containing .npy files')
    parser.add_argument('--camera', type=str, default='GEN4', help='GEN1 (QVGA) or GEN4 (720p)')
    opt = parser.parse_args()
    evaluate_folders(opt.dt_folder, opt.gt_lst, opt.camera)

if __name__ == '__main__':
    main()
