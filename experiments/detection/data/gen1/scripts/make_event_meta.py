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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('target', type=str, choices=['train', 'val', 'test'], help='')
    parser.add_argument('--split' , type=str, default='1/1', help='')
    args = parser.parse_args()

import numpy as np
from hmnet.utils.common import get_list, get_chunk, mkdir

def main():
    dpath_out = args.target + '_meta'
    mkdir(dpath_out)

    list_fpath = get_list(f'./{args.target}_evt/', ext='npy')
    list_fpath = get_chunk(list_fpath, chunk_str=args.split)

    for fpath in list_fpath:
        data = np.load(fpath)
        output = np.zeros([60000,2])
        segment_indices = data['t'] // 1000
        seg_idx, pos, count = np.unique(segment_indices, return_index=True, return_counts=True)
        output[seg_idx] = np.stack([pos, count], axis=1)
        output[:,0] = pad_index(output[:,0], output[:,1])
        fpath_out = dpath_out + '/' + fpath.split('/')[-1]
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
