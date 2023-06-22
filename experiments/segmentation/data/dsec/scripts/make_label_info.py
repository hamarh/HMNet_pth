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
from multiprocessing import Pool
import glob
import h5py
import hdf5plugin

from hmnet.utils.common import get_list


TRAIN_DIRS = [
    'zurich_city_00_a',
    'zurich_city_01_a',
    'zurich_city_02_a',
    'zurich_city_04_a',
    'zurich_city_05_a',
    'zurich_city_06_a',
    'zurich_city_07_a',
    'zurich_city_08_a',
]

TEST_DIRS = [
    'zurich_city_13_a',
    'zurich_city_14_c',
    'zurich_city_15_a',
]


def main(phase):
    if phase == 'train':
        dirs = TRAIN_DIRS
    elif phase == 'test':
        dirs = TEST_DIRS

    for dir in dirs:
        run(dir, phase)

def run(dpath_in, phase):
    print(dpath_in, phase)
    list_fpath_image = get_list(f'{phase}_lbl/{dpath_in}_labels/', ext='png')
    length = max([ len(fpath) for fpath in list_fpath_image ])
    dtype_str = f'<U{length}'
    list_fpath_image = np.array(list_fpath_image, dtype=np.dtype(dtype_str))

    fpath_evt = f'{phase}_evt/{dpath_in}_events.h5'
    fpath_time = f'./source/{phase}/{dpath_in}/{dpath_in}_semantic_timestamps.txt'
    times = get_times(fpath_time, fpath_evt)

    assert len(list_fpath_image) == len(times)

    DTYPE = np.dtype({'names':['t','label'], 'formats':['<i8',dtype_str], 'offsets':[0,8], 'itemsize':8 + length*4})
    output = np.zeros((len(times),), dtype=DTYPE) 
    output['t'] = times
    output['label'] = list_fpath_image

    fpath_out = f'{phase}_lbl/{dpath_in}_label_info.npy'
    np.save(fpath_out, output)

    print(times[0], list_fpath_image[0])
    print(times[-1], list_fpath_image[-1])
    print('')


def get_times(fpath_time, fpath_evt):
    data = h5py.File(fpath_evt)
    t_offset = data['t_offset'][...]

    with open(fpath_time, 'r') as fp:
        lines = fp.read().split('\n')
    if lines[-1] == '':
        lines = lines[:-1]
    times = [ int(float(l) - t_offset) for l in lines ]

    data.close()

    return times


if __name__ == '__main__':
    main('train')
    main('test')




