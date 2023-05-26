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

import numpy as np
import glob

from hmnet.utils.common import get_list

VAL_IDS = [
    8,   9,  23,  24,  35,  36,  37,  38,  39,  43, 
   44,  51,  52,  65,  85,  86,  95,  96,  98, 102, 
  103, 112, 113, 118, 119, 121, 127, 128, 131, 133, 
  135, 136, 139, 141, 143, 145, 146, 149, 152, 156, 
  159, 160, 161, 163, 164, 165, 169, 174, 177, 178, 
  179, 180, 189, 199, 200, 204, 206, 207, 208, 209, 
  210, 211, 212, 213, 214, 215, 216, 219, 220, 221, 
  222, 228, 230, 231, 232, 233, 234, 235, 236, 237, 
  242, 243, 244, 245, 249, 250, 270, 272, 273, 280, 
  281, 289, 290, 291, 292, 294, 295, 296, 297, 300, 
]

def get_filter(phase):
    get_seqid = lambda fpath: int(fpath.split('/')[3].split('_')[1])

    if phase == 'train':
        return lambda fpath: 'Town01' in fpath or 'Town02' in fpath or 'Town03' in fpath
    elif phase == 'val':
        return lambda fpath: 'Town05' in fpath and get_seqid(fpath) in VAL_IDS
    elif phase == 'test':
        return lambda fpath: 'Town05' in fpath and get_seqid(fpath) not in VAL_IDS

def save(phase):
    header = 'name,start,end'

    filter = get_filter(phase)

    list_fpath = glob.glob('./source/Town*/sequence_*/events/data/boundary_timestamps.txt')
    list_fpath = [ fpath for fpath in list_fpath if filter(fpath) ]

    out = ''
    for fpath in list_fpath:
        print(fpath)
        name = fpath.replace('./source/', '').replace('/events/data/boundary_timestamps.txt', '').replace('/', '_') + '_depth.npy'

        with open(fpath, 'r') as fp:
            lines = fp.read().split('\n')
        if lines[-1] == '':
            lines = lines[:-1]

        start = float(lines[0].split(' ')[1]) * 1.0e6
        end = float(lines[-1].split(' ')[2]) * 1.0e6
        out += f'\n{name},{start},{end}'

    with open(f'./list/{phase}/video_duration.csv', 'w') as fp:
        fp.write(header + out)


save('train')
save('val')
save('test')
