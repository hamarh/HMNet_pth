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

import torch
import numpy as np

class PseudoEpochLoader:
    def __init__(self, iter_per_epoch, dataset, sampler, start_epoch=0, **loader_param):
        self._loader = torch.utils.data.DataLoader(dataset, sampler=sampler, **loader_param)
        self._iterator = iter(self._loader)
        self._iter_per_epoch = iter_per_epoch
        self._iter = -1
        self._epoch = 0
        self._pseudo_epoch = start_epoch

    def __iter__(self):
        return self

    def __len__(self):
        return self._iter_per_epoch

    def __next__(self):
        return self.next()

    @property
    def nowiter(self):
        return self._pseudo_epoch * self._iter_per_epoch + self._iter

    def next(self):
        self._iter += 1
        if self._iter >= self._iter_per_epoch:
            self._iter = -1
            self._pseudo_epoch += 1
            raise StopIteration

        try:
            data = next(self._iterator)
            return data
        except StopIteration:
            self._iterator = None
            self._epoch += 1
            if isinstance(self._loader.sampler, torch.utils.data.distributed.DistributedSampler):
                self._loader.sampler.set_epoch(self._epoch)
            self._iterator = iter(self._loader)
            data = next(self._iterator)
            return data

    @property
    def sampler(self):
        return self._loader.sampler

    @property
    def dataset(self):
        return self._loader.dataset

