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

# This script convert field name 'ts' in GEN1 dataset into 't' as in 1 Megapixel dataset

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('dpath', type=str, help='')
    parser.add_argument('dpath_out', type=str, help='')
    args = parser.parse_args()

import numpy as np
import numpy.lib.recfunctions as rfn

from hmnet.utils.common import get_list, mkdir

def main(args):
    mkdir(args.dpath_out)

    list_fpath_lbl = get_list(args.dpath, ext='npy')

    for fpath in list_fpath_lbl:
        boxes = np.load(fpath)
        boxes = rfn.rename_fields(boxes, {'ts': 't'})

        fname = fpath.split('/')[-1]
        fpath_out = f'{args.dpath_out}/{fname}'
        np.save(fpath_out, boxes)

        print(f'{fname}, {boxes["t"].shape}')


if __name__ == '__main__':
    main(args)


