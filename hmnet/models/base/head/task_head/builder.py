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

from .seg_head import SegHead
from .det_head_yolox import YOLOXHead
from .depth_reg_head import DepthRegHead

HEADS = {
    'SegHead': SegHead,
    'YOLOXHead': YOLOXHead,
    'DepthRegHead': DepthRegHead,
}

def build_task_head(param):
    param = param.copy()
    name = param.pop('type')
    cls = HEADS[name]
    return cls(**param)

