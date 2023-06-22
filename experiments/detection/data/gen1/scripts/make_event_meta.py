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
