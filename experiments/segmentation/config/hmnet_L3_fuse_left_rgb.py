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

import sys
import os
import copy

import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrsch

from hmnet.utils.lr_lambda import CombinationV2
from hmnet.utils.common import CDict as dict
from hmnet.utils.transform import Compose, RandomCrop, RandomFlip, RandomResize, Padding, Resize, ResizeInside
from hmnet.dataset.dsec import EventPacketStream
from hmnet.dataset.custom_collate_fn import collate_keep_dict
from hmnet.models.segmentation import HMSeg

TRAIN_DURATION = 200e3
DELTA_T = 5e3
INPUT_SIZE = (440,640)

backbone = dict(
    type         = 'HMNet',
    latent_sizes = [(110, 160), (55, 80), (28, 40)],
    latent_dims  = [256, 256, 256],
    output_dims  = [256, 256, 256],
    num_heads    = [8, 8, 8],
    depth        = [1, 3, 9],

    cfg_embed = dict(
        input_size    = INPUT_SIZE,
        out_dim       = [32, 32, 32],
        duration      = DELTA_T,
        discrete_time = True,
        time_bins     = 100,
        dynamic       = [True,True,True],
        dynamic_dim   = [32, 32, 32],
    ),

    cfg_memory1 = dict(
        freq = 1,
        vector_latent = True,

        cfg_update = dict(
            layer = 'res',
            norm_layer = nn.GroupNorm,
            act_layer  = nn.SiLU,
        ),
    ),

    cfg_memory2 = dict(
        freq = 3,
        vector_latent = True,

        cfg_update = dict(
            layer = 'res',
            norm_layer = nn.GroupNorm,
            act_layer  = nn.SiLU,
        ),

        cfg_write = dict(
            downsample  = 'conv',
            input_proj  = True,
            window_size = (7,7),
            grouping    = 'intra-window',
            act_layer   = nn.GELU,
            norm_layer  = nn.LayerNorm,
            drop = 0.,
            pos_dynamic = False,
            pos_log_scale = False,
        ),

        cfg_message = dict(
            input_resize  = 'merge',
            latent_resize = 'none',
            input_proj    = True,
            latent_proj   = False,
            out_proj      = True,
            norm_layer    = nn.LayerNorm,
            drop          = 0.,
            window_size   = (7,7),
            grouping    = 'intra-window',
            pos_dynamic   = False,
            pos_log_scale = False
        ),
    ),

    cfg_memory3 = dict(
        freq = 9,
        vector_latent = True,

        cfg_update = dict(
            layer = 'res',
            norm_layer = nn.GroupNorm,
            act_layer  = nn.SiLU,
        ),

        cfg_write = dict(
            downsample  = 'conv',
            input_proj  = True,
            window_size = (7,7),
            grouping    = 'intra-window',
            act_layer   = nn.GELU,
            norm_layer  = nn.LayerNorm,
            drop = 0.,
            pos_dynamic = False,
            pos_log_scale = False,
        ),

        cfg_message = dict(
            input_resize  = 'merge',
            latent_resize = 'none',
            input_proj    = True,
            latent_proj   = False,
            out_proj      = True,
            norm_layer    = nn.LayerNorm,
            drop          = 0.,
            window_size   = (7,7),
            grouping    = 'intra-window',
            pos_dynamic   = False,
            pos_log_scale = False
        ),

        cfg_image_write = dict(
            window_size = (7,7),
            grouping    = 'intra-window',
            act_layer   = nn.GELU,
            norm_layer  = nn.LayerNorm,
            drop        = 0.,
            encoder_type   = 'ResNet',
            encoder_layers = [2,2,2],
            encoder_dims   = [64, 128, 256],
            encoder_drop_path = 0.,
        ),
    ),
)

neck = dict(
    type = 'Pyramid',
    input_proj     = False,
    in_channels    = [256, 256, 256],
    input_start_index = 0,
    dim            = 256,
    out_indices    = [0, 1, 2],
    cfg_pyramid_fuse = [
        dict(direction='bottom-up', pre_trans=False, fuse_method='add', post_trans=True, post_convs=False),
        dict(direction='top-down' , pre_trans=False, fuse_method='add', post_trans=True, post_convs=False),
    ],
)

aux_head = dict(
    type           = 'SegHead',
    inc            = 256,
    num_classes    = 11,
    num_extra_conv = 1,
    norm_layer     = nn.BatchNorm2d,
    act_layer      = nn.ReLU,
    drop           = 0.1,
    cfg_loss = dict(
        type = 'SegLoss',
        coef_ce = 0.4,
        prefix = 'Aux ',
        ignore_index = 255,
    ),
)

head = dict(
    type           = 'SegHead',
    inc            = 256,
    num_classes    = 11,
    num_extra_conv = 1,
    norm_layer     = nn.BatchNorm2d,
    act_layer      = nn.ReLU,
    drop           = 0.1,
    cfg_loss = dict(
        type = 'SegLoss',
        coef_ce = 1.0,
        prefix = '',
        ignore_index = 255,
    ),
)


class TrainSettings(object):
    # ======== train data settings ========
    def get_dataset(self):
        train_transform = Compose([
            RandomResize(scale_min=0.5, scale_range=15, event_downsampling='NONE', event_upsampling='NONE', event_resampling='NONE'),
            Padding(size=INPUT_SIZE, halign='center', valign='center', const_image=0, const_mask=255, padding_mode='constant'),
            RandomCrop(crop_size=INPUT_SIZE, const_image=0, clip_border=False, bbox_filter_by_center=True),
            RandomFlip(prob=0.5, direction='H'),
        ])

        train_dataset = EventPacketStream(
            fpath_evt_lst        = './data/dsec/list/train/events.txt',
            fpath_image_lst      = './data/dsec/list/train/images.txt',
            fpath_label_lst      = './data/dsec/list/train/labels.txt',
            base_path            = './data/dsec/',
            fpath_meta           = './data/dsec/list/train/meta.pkl',
            fpath_video_duration = './data/dsec/list/train/video_duration.csv',
            train_duration       = TRAIN_DURATION,
            delta_t              = DELTA_T,
            sampling             = 'label',
            skip_ts              = 0,
            use_nearest_label    = False,
            random_time_scaling  = False,
            min_time_scale       = 0.5,
            max_time_scale       = 2.0,
            start_index_aug_method = 'end',
            start_index_aug_ratio  = 0.25,
            event_transform          = train_transform,
            max_events_per_packet    = 6000000,
            downsample_packet_length = TRAIN_DURATION,
            skip_image_loading       = False,
            ignore_index             = 255,
        )

        return train_dataset

    loader_param = dict(
        batch_size  = 16,
        shuffle     = True,
        num_workers = 4,
        pin_memory  = True,
        drop_last   = True,
        collate_fn  = collate_keep_dict,
    )

    # ======== model settings ========
    def get_model(self):
        d0 = torch.device('cuda:0')
        model = HMSeg(backbone, neck, head, aux_head, devices=[d0,d0,d0,d0])
        model.init_weights()
        return model

    # ======== optimizer settings ========
    N_SAMPLES = 8082
    NUM_EPOCHS = 180
    bsize = loader_param['batch_size']
    iter_per_epoch = N_SAMPLES // bsize
    maxiter = iter_per_epoch * NUM_EPOCHS

    optimizer    = torch.optim.AdamW
    optim_params = dict(
        lr           = 2.0e-4,
        betas        = (0.9, 0.999),
        weight_decay = 0.01,
    )
    schedule = [
        {'method': 'cosine', 'range': (0,maxiter), 'start_lr': optim_params['lr'], 'end_lr': 1.0e-7}
    ]
    lr_scheduler = lrsch.LambdaLR
    lrsch_params = dict( lr_lambda = CombinationV2(schedule, optim_params['lr']) )

    # ======== DDP settings ========
    grad_as_bucket_view = True
    static_graph = False

    # ======== other settings ========
    resume      = ''
    print_freq  = 10
    fpath_script= sys.argv[0]

class TestSettings(object):
    def get_model(self, devices=None, mode='single_process'):
        assert mode in ('single_process', 'multi_process', 'cuda_stream')
        if devices is None:
            d = torch.device('cuda:0')
            devices = [d, d, d, d]

        model = HMSeg(backbone, neck, head, aux_head, devices=devices, test_aug=None)
        model.init_weights()

        if mode == 'multi_process':
            model.backbone.set_multi_process([False, True, True])
        elif mode == 'cuda_stream':
            model.backbone.set_cuda_stream([True, True, True])

        return model

    def get_dataset(self, fpath_evt, fpath_rgb, fpath_lbl, fpath_meta, fpath_video_duration, base_path, fast_mode=False, delta_t=None, debug=False):
        delta_t = delta_t or DELTA_T

        test_dataset = EventPacketStream(
            fpath_evt_lst      = [fpath_evt],
            fpath_image_lst    = [fpath_rgb],
            fpath_label_lst    = [fpath_lbl],
            base_path          = base_path,
            fpath_meta         = fpath_meta,
            fpath_video_duration = fpath_video_duration,
            train_duration     = 62e6 if not debug else 10e6,
            sampling_stride    = 62e6 if not debug else 10e6,
            delta_t            = delta_t,
            skip_ts            = 0,
            use_nearest_label  = False,
            sampling           = 'regular',
            start_index_aug_method = 'none',
            start_index_aug_ratio  = 0.,
            random_time_scaling = False,
            min_time_scale      = 0.5,
            max_time_scale      = 2.0,
            event_transform     = None,
            output_type         = 'long' if fast_mode else None,
            skip_image_loading  = False,
            ignore_index        = 255,
        )

        return test_dataset

    # ======== prediction settings ========
    checkpoint      = 'checkpoint.pth.tar'
    batch_size      = 1

    to_device_in_model = True


