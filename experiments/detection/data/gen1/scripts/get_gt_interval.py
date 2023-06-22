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
