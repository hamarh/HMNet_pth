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

from hmnet.utils.psee_toolbox.io.dat_events_tools import load_td_data
from hmnet.utils.common import get_list, get_chunk, mkdir

def main():
    dpath_out = args.target + '_evt'
    mkdir(dpath_out)

    list_fpath = get_list(f'./source/detection_dataset_duration_60s_ratio_1.0/{args.target}', ext='dat')
    list_fpath = get_chunk(list_fpath, chunk_str=args.split)

    for fpath in list_fpath:
        data = load_td_data(fpath)
        fpath_out = dpath_out + '/' + fpath.split('/')[-1].replace('.dat', '.npy')

        t = data['t'].astype(int)
        diff = t[1:] - t[:-1]
        if diff.min() < 0:
            print(f"Time stamps are not sorted ({diff.min()}, {fpath.split('/')[-1]})")
            indices = np.argsort(t)
            data = data[indices]
        np.save(fpath_out, data)
        print(fpath_out)

if __name__ == '__main__':
    main()
