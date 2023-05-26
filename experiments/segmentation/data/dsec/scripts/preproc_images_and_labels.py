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
