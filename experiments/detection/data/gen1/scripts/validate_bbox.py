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
    parser.add_argument('dpath', type=str, help='')
    parser.add_argument('dpath_out', type=str, help='')
    args = parser.parse_args()

import numpy as np
import numpy.lib.recfunctions as rfn

from hmnet.utils.common import get_list, mkdir

WIDTH = 304
HEIGHT = 240

def main(args):
    mkdir(args.dpath_out)

    list_fpath_lbl = get_list(args.dpath, ext='npy')

    total_boxes = 0
    total_invalid = 0
    total_detail = []

    for fpath in list_fpath_lbl:
        boxes = np.load(fpath)
        x1, y1, w, h = boxes['x'], boxes['y'], boxes['w'], boxes['h']
        x2 = x1 + w
        y2 = y1 + h

        # modify
        _x1 = x1.clip(0, WIDTH)
        _x2 = x2.clip(0, WIDTH)
        _y1 = y1.clip(0, HEIGHT)
        _y2 = y2.clip(0, HEIGHT)
        _w = _x2 - _x1
        _h = _y2 - _y1

        invalid = (_w <= 0) | (_h <= 0) | (_w == WIDTH) | (_h == HEIGHT)

        n1 = invalid.sum()
        total_invalid += n1
        total_boxes += len(boxes)

        boxes = rfn.append_fields(boxes, 'invalid', invalid, usemask=False)

        fname = fpath.split('/')[-1]
        fpath_out = f'{args.dpath_out}/{fname}'
        np.save(fpath_out, boxes)

    p = total_invalid / total_boxes * 100
    print('')
    print(args.dpath)
    print('='*50)
    print(f'Total: {total_invalid}/{total_boxes} ({p:.2f})')
    print('')

if __name__ == '__main__':
    main(args)

