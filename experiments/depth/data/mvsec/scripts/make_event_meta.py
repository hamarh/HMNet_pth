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
import h5py

from hmnet.utils.common import mkdir

list_files = [
    './source/outdoor_day1_data.hdf5',
    './source/outdoor_day2_data.hdf5',
    './source/outdoor_night1_data.hdf5',
]


def main():
    for target_file in list_files:
        data = h5py.File(target_file)
        events = data['davis']['left']['events']

        t = (events[:,2] * 1e6).astype(int)
        t = t - t[0]
        diff = t[1:] - t[:-1]
        assert diff.min() >= 0

        segment_indices = t // 1000
        max_seg_idx = segment_indices.max()
        output = np.zeros([max_seg_idx+1,2])

        seg_idx, pos, count = np.unique(segment_indices, return_index=True, return_counts=True)
        output[seg_idx] = np.stack([pos, count], axis=1)
        output[:,0] = pad_index(output[:,0], output[:,1])
        fpath_out = target_file.replace('_data.hdf5', '') + '_meta.npy'
        np.save(fpath_out, output)
        print(fpath_out)
        print('N_seg: ', len(seg_idx))

def pad_index(pos, count):
    index = 0
    for i in reversed(range(len(pos))):
        if count[i] == 0:
            pos[i] = index
        else:
            index = pos[i]
    return pos

if __name__ == '__main__':
    main()
