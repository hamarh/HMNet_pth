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
from torch import Tensor
from ..blocks import BlockBase

class IdentityNeck(BlockBase):
    def __init__(self) -> None:
        super().__init__()
        pass

    @torch.jit.ignore
    def init_weights(self) -> None:
        pass

    def forward(self, inputs: Tensor) -> Tensor:
        return inputs

    def clip_grad(self, method: str='norm', max_norm: float=1.0, clip_value: float=0.5) -> None:
        pass

