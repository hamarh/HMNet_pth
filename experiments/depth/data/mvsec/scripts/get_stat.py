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

import h5py
import numpy as np

from hmnet.utils.common import get_list, mkdir

list_files = [
    './source/outdoor_day1_data.hdf5',
    './source/outdoor_day2_data.hdf5',
    './source/outdoor_night1_data.hdf5',
]

for fpath in list_files:
    print(fpath)
    data = h5py.File(fpath)
    images = data['davis']['left']['image_raw'][...]
    print(images.shape)
    data.close()

    mean = images.mean()
    std  = images.std()

    fpath_out = fpath.replace('_data.hdf5','') + '_image_mean_std.npy'
    np.save(fpath_out, np.array([mean, std]))
    print(mean, std)

