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

import numpy as np
import h5py

from hmnet.utils.common import mkdir

list_files = [
    './source/outdoor_day1_data.hdf5',
    './source/outdoor_day2_data.hdf5',
    './source/outdoor_night1_data.hdf5',
]


def main():
    for target_file in list_files:
        data = h5py.File(target_file)
        events = data['davis']['left']['events']

        t = (events[:,2] * 1e6).astype(int)
        t = t - t[0]
        diff = t[1:] - t[:-1]
        assert diff.min() >= 0

        segment_indices = t // 1000
        max_seg_idx = segment_indices.max()
        output = np.zeros([max_seg_idx+1,2])

        seg_idx, pos, count = np.unique(segment_indices, return_index=True, return_counts=True)
        output[seg_idx] = np.stack([pos, count], axis=1)
        output[:,0] = pad_index(output[:,0], output[:,1])
        fpath_out = target_file.replace('_data.hdf5', '') + '_meta.npy'
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
