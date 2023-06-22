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
    parser.add_argument('--root', required=True, type=str, help='')
    parser.add_argument('--input_type', required=True, choices=['image', 'label'], type=str, help='')
    parser.add_argument('--num_chunks', type=int, default=1, help='')
    args = parser.parse_args()

import h5py
from PIL import Image
import numpy as np

from hmnet.utils.transform import TransformFunctions
from hmnet.utils.common import get_list, get_chunk

HEIGHT = 480
WIDTH = 640
DISCARD = 40

def main(args):
    list_fpath_info = get_list(args.target_dir, ext='npy')

    for fpath_info in list_fpath_info:
        print(fpath_info)
        info = np.load(fpath_info)
        ts = info['t']

        if args.input_type == 'image':
            list_fpath = info['image']
            n_channels = 3
        elif args.input_type == 'label':
            list_fpath = info['label']
            n_channels = 1
        else:
            raise RuntimeError
        list_fpath = [ args.root + '/' + strip(fpath) for fpath in list_fpath ]

        fpath_out = fpath_info.replace('_info.npy', '.hdf5')
        fp = h5py.File(fpath_out, 'w')
        data_ts = fp.create_dataset('data_ts', data=ts)
        data = fp.create_dataset('data', (0,n_channels,HEIGHT-DISCARD,WIDTH), maxshape=(None,n_channels,HEIGHT-DISCARD,WIDTH), dtype=np.uint8)

        pointer = 0
        for i in range(args.num_chunks):
            print(i, end='', flush=True)
            sublist = get_chunk(list_fpath, chunk_id=i, num_chunk=args.num_chunks)
            npys = to_npy(sublist, args.input_type)
            st = pointer
            ed = pointer + len(sublist)

            if data.shape[0] < ed:
                data.resize(ed, axis=0)

            data[st:ed] = npys

            pointer = ed

        print('')
        fp.close()

def to_npy(list_fpath, input_type):
    trans = TransformFunctions()
    npys = []
    for fpath in list_fpath:
        img = np.array(Image.open(fpath))
        #print(img.dtype, img.shape, fpath)
        if img.ndim == 3:
            img = img.transpose([2,0,1])
        else:
            img = img[None,:,:]
        img = img.astype(np.uint8)

        if input_type == 'image':
            img = trans.resize(img, types=['image'], size=(HEIGHT,WIDTH), image_resampling='BILINEAR')
            img = trans.crop(img, types=['image'], crop_size=(HEIGHT-DISCARD, WIDTH), crop_position=(0,0))
        #print(img.dtype, img.shape, fpath)

        npys.append(img)

    npys = np.array(npys)

    return npys

def strip(fpath):
    if fpath[0] in ('.', '/'):
        return strip(fpath[1:])
    else:
        return fpath


if __name__ == '__main__':
    main(args)
