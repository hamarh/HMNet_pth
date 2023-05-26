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

from hmnet.utils.common import get_list


def main(phase):
    output = ['name,interval']

    list_fpath = get_list(f'{phase}_lbl', ext='npy')
    for fpath in list_fpath:
        lbl = np.load(fpath)

        if len(lbl) < 1:
            interval = 1000000
        else:
            t = np.unique(lbl['t'])
            itvl = t[1:] - t[:-1]

            u_itvl = np.unique(itvl)
            u_itvl = u_itvl[u_itvl>0]
            
            if len(u_itvl) == 0:
                interval = 1000000
            else:
                interval = u_itvl.min()
                interval = min(1000000, interval)

        record = f"{fpath.split('/')[-1]},{interval}"
        output.append(record)

        print(interval, fpath.split('/')[-1])

    fpath_out = f"./list/{phase}/gt_interval.csv"
    with open(fpath_out, 'w') as fp:
        fp.write('\n'.join(output))

main('test')
main('val')
main('train')
