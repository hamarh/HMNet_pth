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
    args = parser.parse_args()

# ========= for debug ==========
DEBUG = False
PREFIX = './debug/preds'
#PREFIX = './debug/preds_img'
# ==============================

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
    list_fpath_rgb = get_list(config.fpath_rgb_lst, ext=None)
    list_fpath_lbl = get_list(config.fpath_lbl_lst, ext=None)

    # split targets into chunks
    list_fpath_evt = get_chunk(list_fpath_evt, chunk_str=config.test_chunks)
    list_fpath_rgb = get_chunk(list_fpath_rgb, chunk_str=config.test_chunks)
    list_fpath_lbl = get_chunk(list_fpath_lbl, chunk_str=config.test_chunks)

    for fidx, (fpath_evt, fpath_rgb, fpath_lbl) in enumerate(zip(list_fpath_evt, list_fpath_rgb, list_fpath_lbl)):
        # get dataset
        dataset = config.get_dataset(fpath_evt, fpath_rgb, fpath_lbl, config.fpath_meta, config.fpath_video_duration, config.data_root, fast_mode=config.fast)
        loader = torch.utils.data.DataLoader(dataset,
                                             shuffle=False,
                                             pin_memory=True,
                                             num_workers=0,  # MUST be 0 because dataset.event_transform is needed for backward transformation
                                             batch_size=config.batch_size,
                                             collate_fn=collate_keep_dict)

        results = []
        for i, data in enumerate(loader):
            events, images, image_metas = parse_event_data(data)

            if getattr(config, 'to_device_in_model', False) == False:
                events = to_device(events, config.device)
                images = to_device(images, config.device)

            with autocast(enabled=config.fp16):
                preds, out_image_metas = model.inference(events, images, image_metas, speed_test=config.speed_test)
            
            # debug
            if DEBUG:
                preds_ref = torch.load(f'{PREFIX}_{fidx}_{i}.pth', map_location='cpu')
                abs_err = (preds[-1] - preds_ref).abs().mean().item()
                abs_rel = ((preds[-1] - preds_ref) / preds_ref).abs().mean().item()
                print('%.2e, %.2f%%' % (abs_err, abs_rel*100))
            # /debug

            if loader.dataset.event_transform is not None:
                preds, out_image_metas = backward_transform(preds, out_image_metas, loader.dataset.event_transform)

            preds, out_image_metas = get_pred_corresponding_gt_frame(preds, out_image_metas)

            results.append([preds, out_image_metas])

        print(f'\nwriting results')
        save_results(results, config.dpath_out)

def save_results(results, dpath_out):
    pred_times, pred_files = [], []
    for preds, metas in results:
        for pred, meta in zip(preds, metas):
            video_name = '.'.join(meta['ori_filename'].split('.')[:-1])
            label_name = meta['depth_path'].split('/')[-1]
            mkdir(f'{dpath_out}/{video_name}')
            fpath_out = f'{dpath_out}/{video_name}/{label_name}'
            curr_time = meta['curr_time_org']
            np.save(fpath_out, pred.cpu().numpy())
            pred_times.append(curr_time)
            pred_files.append(fpath_out)

    length = max([ len(fpath) for fpath in pred_files ])
    dtype_str = f'<U{length}'

    DTYPE = np.dtype({'names':['t','depth'], 'formats':['<i8',dtype_str], 'offsets':[0,8], 'itemsize':8 + length*4})
    pred_info = np.zeros((len(pred_times),), dtype=DTYPE)
    pred_info['t'] = pred_times
    pred_info['depth'] = pred_files
    np.save(f'{dpath_out}/{video_name}.npy', pred_info)


def get_pred_corresponding_gt_frame(preds, image_metas):
    out_preds, out_metas = [], []
    for pred, meta in zip(preds, image_metas):
        if meta['depth_path'] is not None:
            out_preds.append(pred)
            out_metas.append(meta)
    return out_preds, out_metas

def backward_transform(depth, img_metas, transform):
    out_depth = []
    out_img_metas = []
    for dpt, img_meta in zip(depth, img_metas):
        dpt, img_meta = transform.backward(dpt, img_meta, types=['image', 'meta'])
        out_depth.append(dpt)
        out_img_metas.append(img_meta)

    return torch.stack(out_depth), out_img_metas

def parse_event_data(data):
    def _nested_shape(lst, shape=[]):
        if isinstance(lst, (list, tuple)):
            shape += [len(lst)]
            return _nested_shape(lst[0], shape)
        else:
            return shape

    datas, targets, metas = data
    shape = _nested_shape(metas)

    if len(shape) == 2:
        list_events, list_images, list_image_metas = [], [], []
        for d, t, m in zip(datas, targets, metas):
            events, images, image_metas = parse_event_data([d, t, m])
            list_events.append(events)
            list_images.append(images)
            list_image_metas.append(image_metas)
        return list_events, list_images, list_image_metas
    else:
        events = [ d['events'] for d in datas ]
        images = [ d['images'] for d in datas ]
        image_metas = [ m['image_meta'] for m in metas ]

        if events[0].ndim == 3:
            events = torch.stack(events)

        return events, images, image_metas

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
    config = config_module.TestEventscape()

    config.fpath_evt_lst = f'{args.data_list}/events.txt'
    config.fpath_rgb_lst = f'{args.data_list}/images.txt'
    config.fpath_lbl_lst = f'{args.data_list}/labels.txt'
    config.fpath_meta    = f'{args.data_list}/meta.pkl'
    config.fpath_video_duration = f'{args.data_list}/video_duration.csv'
    config.data_root = args.data_root

    config.cpu = args.cpu
    config.gpuid = args.gpuid
    config.test_chunks = args.test_chunks
    config.pretrained = args.pretrained
    config.random_init = args.random_init
    config.fast = args.fast
    config.fp16 = args.fp16
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

