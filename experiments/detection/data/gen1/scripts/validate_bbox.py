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

