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

import sys
import torch
import torch.nn as nn
import torch.optim.lr_scheduler as lrsch

from hmnet.utils.common import CDict as dict
from hmnet.utils.lr_lambda import CombinationV2
from hmnet.utils.transform import Compose, RandomCrop, RandomFlip, RandomResize, Padding, Resize, ResizeInside
from hmnet.models.detection import HMDet
from hmnet.dataset.gen1 import EventPacketStream
from hmnet.dataset.custom_collate_fn import collate_keep_dict

TRAIN_DURATION = 200e3
DELTA_T = 5e3
INPUT_SIZE = (240, 304)

backbone = dict(
    type         = 'HMNet',
    latent_sizes = [(60, 76), (30, 38), (15, 19)],
    latent_dims  = [128, 256, 256],
    output_dims  = [256, 256, 256],
    num_heads    = [4, 8,  8],
    depth        = [1, 3,  9],

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
        vector_latent = False,

        cfg_update = dict(
            layer = 'res',
            norm_layer = nn.GroupNorm,
            act_layer  = nn.SiLU,
        ),
    ),

    cfg_memory2 = dict(
        freq = 3,
        vector_latent = False,

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
        vector_latent = False,

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

head = dict(
    type = 'YOLOXHead',
    num_classes = 2,
    strides     = [4, 8, 16],
    in_channels = [256, 256, 256],
    stacked_convs = 2,
    feat_channels = 256,
    depthwise   = False,
    act         = "silu",
    score_thr   = 0.01,
    nms_iou_threshold = 0.65,
    ignore_bboxes_as_negative = True,
)


class TrainSettings(object):
    # ======== train data settings ========
    def get_dataset(self):
        train_transform = Compose([
            RandomResize(scale_min=0.5, scale_range=15, event_downsampling='NONE', event_upsampling='NONE', event_resampling='NONE'),
            Padding(size=INPUT_SIZE, halign='center', valign='center', const_image=0, const_mask=-1, padding_mode='constant'),
            RandomCrop(crop_size=INPUT_SIZE, const_image=0, clip_border=False, bbox_filter_by_center=True),
            RandomFlip(prob=0.5, direction='H'),
        ])

        train_dataset = EventPacketStream(
            fpath_evt_lst      = './data/gen1/list/train/events.txt',
            fpath_lbl_lst      = './data/gen1/list/train/labels.txt',
            base_path          = './data/gen1',
            fpath_meta         = './data/gen1/list/train/meta.pkl',
            fpath_gt_duration  = './data/gen1/list/train/gt_interval.csv',
            video_duration     = 60e6,
            train_duration     = TRAIN_DURATION,
            delta_t            = DELTA_T,
            skip_ts            = 0,
            use_nearest_label  = False,
            sampling           = 'label',
            min_box_diag       = 30,
            min_box_side       = 10,
            random_time_scaling = False,
            start_index_aug_method = 'end',
            start_index_aug_ratio = 0.25,
            event_transform    = train_transform,
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
        d1 = torch.device('cuda:0')
        d2 = torch.device('cuda:0')
        d3 = torch.device('cuda:0')
        model = HMDet(backbone, neck, head, devices=[d0,d1,d2,d3])
        model.init_weights()
        return model

    # ======== optimizer settings ========
    N_SAMPLES = 72371
    NUM_EPOCHS = 90
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
    def get_model(self, devices=None, speed_test='single_process'):
        assert speed_test in ('single_process', 'multi_process', 'cuda_stream')
        if devices is None:
            d = torch.device('cuda:0')
            devices = [d, d, d, d]

        test_transform = Compose([
            Padding(size=INPUT_SIZE, halign='center', valign='center', const_image=0, const_mask=-1, padding_mode='constant'),
        ])

        model = HMDet(backbone, neck, head, devices=devices, test_aug=test_transform)
        model.init_weights()

        if speed_test == 'multi_process':
            model.backbone.set_multi_process([False, True, True])
        elif speed_test == 'cuda_stream':
            model.backbone.set_cuda_stream([True, True, True])

        return model

    def get_dataset(self, fpath_evt, fpath_lbl, fpath_meta, fpath_gt_duration, base_path, fast_mode=False, delta_t=None):
        delta_t = delta_t or DELTA_T

        test_dataset = EventPacketStream(
            fpath_evt_lst      = [fpath_evt],
            fpath_lbl_lst      = [fpath_lbl],
            base_path          = base_path,
            fpath_meta         = fpath_meta,
            fpath_gt_duration  = fpath_gt_duration,
            video_duration     = 60e6,
            train_duration     = 60e6,
            sampling_stride    = 60e6,
            delta_t            = delta_t,
            skip_ts            = 0,
            use_nearest_label  = False,
            sampling           = 'regular',
            start_index_aug_method = 'none',
            min_box_diag       = 30,
            min_box_side       = 10,
            random_time_scaling = False,
            event_transform     = None,
            output_type         = 'long' if fast_mode else None,
        )

        return test_dataset

    # ======== prediction settings ========
    checkpoint      = 'checkpoint.pth.tar'
    batch_size      = 1



