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
    list_fpath_image = get_list(f'{phase}_img_right/{dpath_in}_images/', ext='png')
    length = max([ len(fpath) for fpath in list_fpath_image ])
    dtype_str = f'<U{length}'
    list_fpath_image = np.array(list_fpath_image, dtype=np.dtype(dtype_str))

    fpath_evt = f'{phase}_evt/{dpath_in}_events.h5'
    fpath_time = f'./source/{dpath_in}/images/timestamps.txt'
    times = get_times(fpath_time, fpath_evt)

    assert len(list_fpath_image) == len(times)

    DTYPE = np.dtype({'names':['t','image'], 'formats':['<i8',dtype_str], 'offsets':[0,8], 'itemsize':8 + length*4})
    output = np.zeros((len(times),), dtype=DTYPE) 
    output['t'] = times
    output['image'] = list_fpath_image

    fpath_out = f'{phase}_img_right/{dpath_in}_image_info.npy'
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




