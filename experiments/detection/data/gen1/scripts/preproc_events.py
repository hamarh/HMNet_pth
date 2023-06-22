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
