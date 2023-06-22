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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('config', type=str, help='Config file')
    parser.add_argument('data_list', type=str, help='Path for directory containing file lists for testing')
    parser.add_argument('data_root', type=str, help='Path for dataset root directory')
    parser.add_argument('--mode', type=str, default='single_process', choices=('single_process', 'multi_process', 'cuda_stream'), help='')
    parser.add_argument('--speed_test', action='store_true', help='Measure inference time')
    parser.add_argument('--name', type=str, default=None, help='Name of the model. (default value is set by this script name)')
    parser.add_argument('--gpuid', type=str, default=0, help='GPU ID')
    parser.add_argument('--cpu', action='store_true', help='Run in CPU mode')
    parser.add_argument('--test_chunks', type=str, default='1/1', help='"{CHUNK_ID}/{NUM_CHUNKS}": Split test data into NUM_CHUNKS and run inference on a specified CHUNK_ID.')
    parser.add_argument('--pretrained', type=str, help='Path for the pretrained weight (checkpoint file in workspace will be loaded by default)')
    parser.add_argument('--random_init', action='store_true', help='Run without pretrained weights')
    parser.add_argument('--devices', type=int, nargs='*', help='')
    parser.add_argument('--fast', action='store_true', help='Convert to fast model')
    parser.add_argument('--fp16', action='store_true', help='Run in FP16 mode')
    parser.add_argument('--compile', type=str, choices=('jit', 'trt', 'onnx', 'otrt', 'inductor', 'aot_ts_nvfuser'), help='Compile and accelarate the model')
    args = parser.parse_args()

import os
import numpy as np
import sys
import copy
from importlib import machinery
from PIL import Image
from functools import partial
from numpy.lib import recfunctions as rfn

import torch
from torch.cuda.amp import autocast
from hmnet.dataset.custom_collate_fn import collate_keep_dict
from hmnet.utils.common import fix_seed, get_list, get_chunk, mkdir, makedirs, Timer

# cudnn benchmark mode
torch.backends.cudnn.benchmark = True

timer = Timer()

#os.environ['PYTORCH_CUDA_ALLOC_CONF'] = 'max_split_size_mb:128'
@torch.no_grad()
def main(config):
    # set device
    if config.cpu:
        config.device = torch.device('cpu')
    else:
        config.device = torch.device('cuda:%d' % config.gpuid)

    # set seed
    fix_seed(42)

    # get model
    model = config.get_model(config.devices, config.mode)

    if hasattr(model, 'to_cuda'):
        model.to_cuda()
    else:
        model = model.to(config.device)
    model.eval()

    # load pretrained weights
    if config.pretrained is None:
        fpath_checkpoint = config.dpath_work + '/' + config.checkpoint
    else:
        fpath_checkpoint = config.pretrained

    if not config.random_init:
        print("=> loading checkpoint '{}'".format(fpath_checkpoint))
        state_dict = get_state_dict(fpath_checkpoint, config.device)
        model.load_state_dict(state_dict)

    # convert to fast model
    if config.fast:
        model.to_fast_model()

    # get list
    list_fpath_evt = get_list(config.fpath_evt_lst, ext=None)
    list_fpath_lbl = get_list(config.fpath_lbl_lst, ext=None)

    # split targets into chunks
    list_fpath_evt = get_chunk(list_fpath_evt, chunk_str=config.test_chunks)
    list_fpath_lbl = get_chunk(list_fpath_lbl, chunk_str=config.test_chunks)

    for fpath_evt, fpath_lbl in zip(list_fpath_evt, list_fpath_lbl):
        # get dataset
        dataset = config.get_dataset(fpath_evt, fpath_lbl, config.fpath_meta, config.fpath_gt_duration, config.data_root, fast_mode=config.fast)
        loader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,  # MUST be 0 because dataset.event_transform is needed for backward transformation
                                             batch_size=config.batch_size,
                                             collate_fn=collate_keep_dict)

        results = []
        for i, data in enumerate(loader):
            events, image_metas = parse_event_data(data, config.device)

            if i == 0 and config.compile is not None:
                h = image_metas[0][0]['height']
                w = image_metas[0][0]['width']
                model.compile(config.compile, fp16=config.fp16, input_shape=[1,None,h,w])

            if getattr(config, 'to_device_in_model', False) == False:
                events = to_device(events, config.device)

            with autocast(enabled=config.fp16):
                list_bbox_dict, image_metas = model.inference(events, image_metas, speed_test=config.speed_test)    # outputs = list[tuple[bboxes, labels]], bboxes: N x (tl_x, tl_y, br_x, br_y, score), labels: N
            
            if loader.dataset.event_transform is not None:
                list_bbox_dict, image_metas = backward_transform(list_bbox_dict, image_metas, loader.dataset.event_transform)

            results += reformat_result(list_bbox_dict, image_metas)

        results = rfn.stack_arrays(results, usemask=False)
        print(f'\nwriting results')
        np.save(f"{config.dpath_out}/{fpath_evt.split('/')[-1]}", results)


def backward_transform(list_bbox_dict, img_metas, transform):
    out_bbox_dict = []
    out_img_metas = []
    for bbox_dict, img_meta in zip(list_bbox_dict, img_metas):
        bbox_dict, img_meta = transform.backward(bbox_dict, img_meta, types=['bbox', 'meta'])
        out_bbox_dict.append(bbox_dict)
        out_img_metas.append(img_meta)

    return out_bbox_dict, out_img_metas

def reformat_result(list_bbox_dict, image_metas):
    outputs = []
    BBOX_DTYPE = np.dtype({'names':['t','x','y','w','h','class_id','track_id','class_confidence'], 'formats':['<i8','<f4','<f4','<f4','<f4','<u4','<u4','<f4'], 'offsets':[0,8,12,16,20,24,28,32], 'itemsize':40})
    for bboxes_dict, image_meta in zip(list_bbox_dict, image_metas):
        if len(bboxes_dict['bboxes']) == 0:
            continue
        bboxes = bboxes_dict['bboxes'].cpu().numpy().astype(np.float32)
        labels = bboxes_dict['labels'].cpu().numpy().astype(np.uint32)
        scores = bboxes_dict['scores'].cpu().numpy().astype(np.float32)
        times = np.array(image_meta['curr_time_org'], dtype=np.int64).repeat(len(bboxes),0)

        output = np.zeros((len(bboxes),), dtype=BBOX_DTYPE) 
        output['t'] = times
        output['x'] = bboxes[:,0]
        output['y'] = bboxes[:,1]
        output['w'] = bboxes[:,2] - bboxes[:,0]
        output['h'] = bboxes[:,3] - bboxes[:,1]
        output['class_id'] = labels
        output['class_confidence'] = scores

        outputs.append(output)

    return outputs

def parse_event_data(data, device):
    datas, targets, metas= data

    if isinstance(datas[0], (tuple, list)):
        list_events, list_image_metas = [], []
        for datas, targets, metas in zip(datas, targets, metas):
            events, image_metas = parse_event_data([datas, targets, metas], device)
            list_events.append(events)
            list_image_metas.append(image_metas)
        return list_events, list_image_metas
    else:
        if isinstance(datas[0], dict):
            events = [ d['events'] for d in datas ]
        else:
            events = datas
        image_metas = [ meta['image_meta'] for meta in metas ]

        return events, image_metas

def to_device(data, device, non_blocking=True):
    if data is None:
        return data
    elif isinstance(data, (list, tuple)):
        return [ to_device(d, device, non_blocking) for d in data ]
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    else:
        return data

def get_state_dict(fpath, device):
    state_dict = torch.load(fpath, map_location=device)
    if 'state_dict' in state_dict:
        return state_dict['state_dict']
    else:
        return state_dict

def get_config(args):
    config_module = machinery.SourceFileLoader('config', args.config).load_module()
    config = config_module.TestSettings()

    config.fpath_evt_lst = f'{args.data_list}/events.txt'
    config.fpath_lbl_lst = f'{args.data_list}/labels.txt'
    config.fpath_meta    = f'{args.data_list}/meta.pkl'
    config.fpath_gt_duration = f'{args.data_list}/gt_interval.csv'
    config.data_root = args.data_root

    config.cpu = args.cpu
    config.gpuid = args.gpuid
    config.test_chunks = args.test_chunks
    config.pretrained = args.pretrained
    config.random_init = args.random_init
    config.fast = args.fast
    config.fp16 = args.fp16
    config.compile = args.compile
    config.devices = args.devices
    config.speed_test = args.speed_test
    config.mode = args.mode

    name = args.config.split('/')[-1].replace('.py', '')
    dirname = get_dirname(args.data_list)
    config.dpath_work = f'./workspace/{name}'
    config.dpath_out = f'./workspace/{name}/result/pred_{dirname}'

    return config

def get_dirname(path):
    if path.endswith('/'):
        path = path[:-1]
    return path.split('/')[-1].split('.')[0]

if __name__ == '__main__':
    __spec__ = None
    config = get_config(args)
    makedirs(config.dpath_out)
    main(config)

