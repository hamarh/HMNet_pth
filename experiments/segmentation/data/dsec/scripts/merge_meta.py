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
import pickle as pkl

from hmnet.utils.common import get_list

def merge(phase):
    list_fpath = get_list(f'./{phase}_meta/', ext='npy')

    dict_meta = {}
    for fpath in list_fpath:
        meta = np.load(fpath).astype(np.int64)
        key = fpath.split('/')[-1].replace('.npy', '')
        dict_meta[key] = (meta[:,0], meta[:,1])
        print(fpath)

    pkl.dump(dict_meta, open(f'./list/{phase}/meta.pkl', 'wb'))

#merge('val')
merge('test')
merge('train')
