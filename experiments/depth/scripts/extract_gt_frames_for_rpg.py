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

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('target', type=str, help='Directory path for ground truth npy files')
parser.add_argument('outdir', type=str, help='')
args = parser.parse_args()

import math
import numpy as np
import glob
import os

MAX_DEPTH = 80
MIN_DEPTH = 1.97041

org_files = sorted(glob.glob(args.target + '/*.npy'))
org_files = org_files[3:]    # idx offset

if not os.path.isdir(args.outdir):
    os.mkdir(args.outdir)

def center_crop(lbl):
    return lbl[2:258,1:345]

for fpath in org_files:
    print(fpath)
    org_lbl = np.load(fpath)
    lbl = org_lbl.clip(MIN_DEPTH, MAX_DEPTH)
    lbl = center_crop(lbl)

    fpath_out = args.outdir + '/' + fpath.split('/')[-1]
    np.save(fpath_out, lbl)





