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
    parser.add_argument('target_dir', type=str, help='')
    parser.add_argument('--num_chunks', type=int, default=1, help='')
    args = parser.parse_args()

import h5py
import hdf5plugin
import numpy as np
import math

from hmnet.utils.transform import TransformFunctions
from hmnet.utils.common import get_list

HEIGHT = 480
WIDTH = 640
DISCARD = 40


def main(args):
    list_fpath = get_list(args.target_dir, ext='h5')

    for fpath in list_fpath:
        print(fpath)

        fp = h5py.File(fpath)
        data = fp['events']

        num_events = len(data['t'])
        N = math.ceil(num_events / args.num_chunks)

        fpath_out = fpath.replace('.h5', '.hdf5')
        h5file = h5py.File(fpath_out, 'w')
        out = h5file.create_dataset('events', (0,4), maxshape=(None,4), dtype=int)

        pointer = 0
        for i in range(args.num_chunks):
            print(i, end='', flush=True)
            st = i * N
            ed = (i+1) * N
            ed = min(num_events, ed)

            t = data['t'][st:ed].astype(int)
            x = data['x'][st:ed].astype(int)
            y = data['y'][st:ed].astype(int)
            p = data['p'][st:ed].astype(int)
            p = p * 2 - 1

            events = np.stack([t,x,y,p], axis=-1)

            trans = TransformFunctions()
            meta = dict(height=HEIGHT, width=WIDTH)
            events, meta = trans.crop(events, meta, types=['event', 'meta'], crop_size=(HEIGHT-DISCARD, WIDTH), crop_position=(0,0))

            ST = pointer
            ED = pointer + len(events)

            if out.shape[0] < ED:
                out.resize(ED, axis=0)

            out[ST:ED] = events

            pointer = ED

        print('')
        fp.close()
        h5file.close()

if __name__ == '__main__':
    main(args)



