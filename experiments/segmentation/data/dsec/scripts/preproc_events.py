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



