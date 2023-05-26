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

from hmnet.utils.common import get_list, mkdir

VAL_IDS = [
    8,   9,  23,  24,  35,  36,  37,  38,  39,  43, 
   44,  51,  52,  65,  85,  86,  95,  96,  98, 102, 
  103, 112, 113, 118, 119, 121, 127, 128, 131, 133, 
  135, 136, 139, 141, 143, 145, 146, 149, 152, 156, 
  159, 160, 161, 163, 164, 165, 169, 174, 177, 178, 
  179, 180, 189, 199, 200, 204, 206, 207, 208, 209, 
  210, 211, 212, 213, 214, 215, 216, 219, 220, 221, 
  222, 228, 230, 231, 232, 233, 234, 235, 236, 237, 
  242, 243, 244, 245, 249, 250, 270, 272, 273, 280, 
  281, 289, 290, 291, 292, 294, 295, 296, 297, 300, 
]

def get_filter(phase):
    get_seqid = lambda fpath: int(fpath.split('/')[3].split('_')[1])

    if phase == 'train':
        return lambda fpath: 'Town01' in fpath or 'Town02' in fpath or 'Town03' in fpath
    elif phase == 'val':
        return lambda fpath: 'Town05' in fpath and get_seqid(fpath) in VAL_IDS
    elif phase == 'test':
        return lambda fpath: 'Town05' in fpath and get_seqid(fpath) not in VAL_IDS

def main(phase):
    dpath_out = f'./{phase}_img/'
    mkdir(dpath_out)

    filter = get_filter(phase)

    list_dpath_in = glob.glob('./source/Town*/sequence_*/rgb/data')
    list_dpath_in = [ fpath for fpath in list_dpath_in if filter(fpath) ]

    list_args = [ [dpath_in, dpath_out] for dpath_in in list_dpath_in ]

    #for arg in list_args:
    #    run(arg)
    p = Pool(10)
    p.map(run, list_args)
    p.close()

def run(arg):
    dpath_in, dpath_out = arg
    print(dpath_in)
    list_fpath_image = get_list(dpath_in, ext='png')
    length = max([ len(fpath) for fpath in list_fpath_image ])
    dtype_str = f'<U{length}'
    list_fpath_image = np.array(list_fpath_image, dtype=np.dtype(dtype_str))

    fpath_time = dpath_in + '/timestamps.txt'
    times = get_times(fpath_time)

    assert len(list_fpath_image) == len(times)

    DTYPE = np.dtype({'names':['t','image'], 'formats':['<i8',dtype_str], 'offsets':[0,8], 'itemsize':8 + length*4})
    output = np.zeros((len(times),), dtype=DTYPE) 
    output['t'] = times
    output['image'] = list_fpath_image

    elems = dpath_in.split('/')
    output_name = elems[2] + '_' + elems[3] + '_image.npy'
    fpath_out = dpath_out + '/' + output_name
    np.save(fpath_out, output)


def get_times(fpath_time):
    with open(fpath_time, 'r') as fp:
        lines = fp.read().split('\n')
    if lines[-1] == '':
        lines = lines[:-1]
    times = [ int(float(l.split(' ')[-1]) * 1.0e6) for l in lines ]
    return times


if __name__ == '__main__':
    main('train')
    main('val')
    main('test')
