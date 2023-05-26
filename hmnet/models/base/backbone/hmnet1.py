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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .latent_memory import LatentMemory, EventEmbedding
from ..blocks import BlockBase
from ..init import init_transformer

from torch import Tensor
from typing import Tuple, List, Optional, Dict
Meta = Dict[str,Tuple[int,int,int]]

from hmnet.utils.common import Timer
timer = Timer()

class HMNet1(BlockBase):
    def __init__(self, latent_sizes, latent_dims, output_dims, num_heads, depth, warmup=20,
                 cfg_embed=None, cfg_memory1=None, cfg_memory2=None, cfg_memory3=None) -> None:
        super().__init__()
        self.latent_dims = latent_dims
        self.output_dims = output_dims
        self.latent_sizes = latent_sizes
        self.warmup = warmup    # output is valid after the warmup time steps

        D0 = sum(cfg_embed['out_dim'])
        D1 = latent_dims
        O1 = output_dims
        L1 = latent_sizes
        H1 = num_heads
        N1 = depth

        self.memory1 = LatentMemory(latent_size=L1, input_dim=D0, latent_dim=D1, output_dim=O1, num_heads=H1, update_depth=N1, message_gen=False, event_write=True,  top_down=False,  **cfg_memory1, cfg_embed=cfg_embed)
        self.set_module_names()

    def init_weights(self, pretrained=None):
        init_transformer(self.modules())

    def to_cuda(self, device):
        self.devices = (device,)
        self.memory1 = self.memory1.to(device)

    def set_devices(self, device):
        self.devices = (device,)

    def set_multi_process(self, flag):
        self.memory1.use_multi_process = flag

    def set_cuda_stream(self, flag):
        self.memory1.use_cuda_stream = flag

    def prepair_for_inference(self, batch_size, image_size=None):
        device, = self.devices
        self.memory1.prepair_for_inference(batch_size, device=device)

    def forward(self, list_events, list_image_metas, gather_indices, list_images=None, init_states=True, detach=True, fast_training=False):
        if list_images is None:
            list_images = [None] * len(list_events)

        if init_states:
            batch_size = len(list_events[0])
            height = list_image_metas[0][0]['height']
            width = list_image_metas[0][0]['width']
            self.memory1.init_states(batch_size, image_size=(height, width))

        # set place holders for outputs
        num_output = len(gather_indices['time'])
        outputs1 = self.memory1.place_holder(num_output)

        if fast_training:
            # extract key value in advance for fast trainig
            list_events = self.memory1.embed.forward_fast_train(self.memory1, list_events, list_image_metas)

        for time_idx, (events, images, image_metas) in enumerate(zip(list_events, list_images, list_image_metas)):
            # forward one time step
            out1 = self._forward_one_step(events, image_metas, image_input=images, fast_training=fast_training)

            if out1 is None:
                continue

            # gather outputs of annotated timmings for loss calculation
            outputs1, = self._gather((outputs1,), (out1,), gather_indices, time_idx)

        # detach memory states (required for TBPTT)
        self.memory1.detach()

        return outputs1,

    def inference(self, events, image_metas, image_input=None) -> Tensor:
        out1 = self._forward_one_step(events, image_metas, image_input)
        return out1,

    def _forward_one_step(self, events, image_metas, image_input=None, fast_training=False) -> Tensor:
        # get event meta data
        curr_time = [ meta['curr_time_crop'] for meta in image_metas ]
        duration = [ meta['delta_t'] for meta in image_metas ]

        # get current state
        z1, message1 = self.memory1.sync_and_get_state()

        # forward one time step
        out1 = self.memory1(events, None, event_metas=(curr_time, duration), fast_training=fast_training)

        return out1

    def _gather(self, list_dst, list_src, gather_indices, time_idx):
        time_indices = gather_indices['time']
        batch_indices = gather_indices['batch']
        assert len(time_indices) == len(batch_indices)
        destination = torch.arange(len(time_indices))

        mask = time_indices == time_idx
        src_indices = batch_indices[mask]
        dst_indices = destination[mask]

        for dst, src in zip(list_dst, list_src):
            dst[dst_indices] = src[src_indices]

        return list_dst

    def termination(self):
        self.memory1.termination()



