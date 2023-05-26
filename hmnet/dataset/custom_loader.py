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

