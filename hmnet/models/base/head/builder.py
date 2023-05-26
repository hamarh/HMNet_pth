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

from .uper_head import UPerHead
from .res_uhead import ResUHead
from .task_head.builder import build_task_head

HEADS = {
    'UPerHead': UPerHead,
    'ResUHead': ResUHead,
}

def build_head(param):
    param = param.copy()
    if param['type'] not in HEADS:
        return build_task_head(param)
    name = param.pop('type')
    cls = HEADS[name]
    return cls(**param)
