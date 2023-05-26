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
import h5py
import hdf5plugin

from hmnet.utils.common import get_list


def save(phase):
    header = 'name,start,end'

    list_fpath = get_list(f'./list/{phase}/events.txt', ext=None)
    out = ''

    for fpath in list_fpath:
        print(fpath)
        data = h5py.File(fpath)
        t = data['events'][:,0]
        name = fpath.split('/')[-1]
        start = t[0]
        end = t[-1]

        d = t[1:] - t[:-1]
        assert np.any(d < 0) == False

        out += f'\n{name},{start},{end}'

    with open(f'./list/{phase}/video_duration.csv', 'w') as fp:
        fp.write(header + out)
        print(out)

save('train')
save('test')

