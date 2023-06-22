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

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--len', type=int, default=68, help='')
    parser.add_argument('--compact', action='store_true', help='')
    args = parser.parse_args()

import numpy as np
import sys
import os
import glob
import pandas as pd

TARGET = [
    'Model',
    'Pred',
    'mAP',
    'AP50',
    'AP75',
    'mAP(S)',
    'mAP(M)',
    'mAP(L)',
    'Spd_sG',
    'Spd_mG',
]

TARGET_COMPACT = [
    'Model',
    'Pred',
    'mAP',
    'mAP(S)',
    'mAP(M)',
    'mAP(L)',
]

def main(args):
    list_fpath = sorted(glob.glob('./workspace/*/result/pred_*/logs/result.txt'))

    target = TARGET_COMPACT if args.compact else TARGET
    table = []

    for fpath in list_fpath:
        results = get_results(fpath)
        model_name, pred_name = get_name(fpath, args.len)
        results.update({'Model': model_name, 'Pred': pred_name})

        row = []
        row += [ results[key] for key in target ]
        table.append(row)

    table = pd.DataFrame(table, columns=target)
    tb = table.to_markdown()
    L = len(tb.split('\n')[0])
    print('+' + '-'*(L-2) + '+')
    print(table.to_markdown())
    print('+' + '-'*(L-2) + '+')


def get_name(fpath, max_length):
    fpath = fpath.replace('workspace', ':')
    elems = fpath.split(':')[-1].split('/')
    model_name = elems[1]
    pred_name = elems[3].replace('pred_', '')

    model_name = model_name[:max_length]
    pred_name = pred_name[:max_length]

    return model_name, pred_name

def get_results(fpath_eval_result):
    is_mAP   = lambda line: 'AP' in line and 'IoU=0.50:0.95' in line and 'all' in line and 'small' not in line
    is_AP50  = lambda line: 'AP' in line and 'IoU=0.50' in line and 'IoU=0.50:0.95' not in line
    is_AP75  = lambda line: 'AP' in line and 'IoU=0.75' in line
    is_mAP_S = lambda line: 'AP' in line and 'IoU=0.50:0.95' in line and 'small' in line
    is_mAP_M = lambda line: 'AP' in line and 'IoU=0.50:0.95' in line and 'medium' in line
    is_mAP_L = lambda line: 'AP' in line and 'IoU=0.50:0.95' in line and 'large' in line
    is_AR1   = lambda line: 'AR' in line and 'all' in line and 'small' not in line and 'maxDets=  1' in line
    is_AR10  = lambda line: 'AR' in line and 'all' in line and 'small' not in line and 'maxDets= 10' in line
    is_AR100 = lambda line: 'AR' in line and 'all' in line and 'small' not in line and 'maxDets=100' in line
    is_AR_S  = lambda line: 'AR' in line and 'small' in line
    is_AR_M  = lambda line: 'AR' in line and 'medium' in line
    is_AR_L  = lambda line: 'AR' in line and 'large' in line
    is_THR   = lambda line: 'Threshold:' in line
    is_P     = lambda line: 'Precision:' in line
    is_R     = lambda line: 'Recall:' in line
    is_F1    = lambda line: 'F1:' in line
    is_latency_sgpu = lambda line: 'mean_sgpu:' in line
    is_latency_mgpu = lambda line: 'mean_mgpu:' in line

    get_value = lambda line: float(line.split('=')[-1].replace(' ', ''))

    with open(fpath_eval_result, 'r') as fp:
        lines = fp.read().split('\n')

    # get latency
    fpath_latency = '/'.join(fpath_eval_result.split('/')[:-1]) + '/latency.txt'
    if os.path.isfile(fpath_latency):
        with open(fpath_latency, 'r') as fp:
            lines += fp.read().split('\n')

    mAP, AP50, AP75, mAP_S, mAP_M, mAP_L, AR1, AR10, AR100, AR_S, AR_M, AR_L = -1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1,-1
    THR, P, R, F1 = -1,-1,-1,-1
    Latency_sGPU, Latency_mGPU = -1, -1
    for line in lines:
        if is_mAP(line):
            mAP = get_value(line)
        elif is_AP50(line):
            AP50 = get_value(line)
        elif is_AP75(line):
            AP75 = get_value(line)
        elif is_mAP_S(line):
            mAP_S = get_value(line)
        elif is_mAP_M(line):
            mAP_M = get_value(line)
        elif is_mAP_L(line):
            mAP_L = get_value(line)
        elif is_AR1(line):
            AR1 = get_value(line)
        elif is_AR10(line):
            AR10 = get_value(line)
        elif is_AR100(line):
            AR100 = get_value(line)
        elif is_AR_S(line):
            AR_S= get_value(line)
        elif is_AR_M(line):
            AR_M= get_value(line)
        elif is_AR_L(line):
            AR_L= get_value(line)
        elif is_THR(line):
            THR = float(line.split(' ')[-1])
        elif is_P(line):
            P = float(line.split(' ')[-1])
        elif is_R(line):
            R = float(line.split(' ')[-1])
        elif is_F1(line):
            F1 = float(line.split(' ')[-1])
        elif is_latency_sgpu(line):
            Latency_sGPU = float(line.split(' ')[1])
        elif is_latency_mgpu(line):
            Latency_mGPU = float(line.split(' ')[1])


    return {
        'mAP'    : mAP,
        'AP50'   : AP50,
        'AP75'   : AP75,
        'mAP(S)' : mAP_S,
        'mAP(M)' : mAP_M,
        'mAP(L)' : mAP_L,
        'AR1'    : AR1,
        'AR10'   : AR10,
        'AR100'  : AR100,
        'AR(S)'  : AR_S,
        'AR(M)'  : AR_M,
        'AR(L)'  : AR_L,
        'THR'    : THR,
        'P'      : P,
        'R'      : R,
        'F1'     : F1,
        'Spd_sG': Latency_sGPU,
        'Spd_mG': Latency_mGPU,
        }

if __name__ == '__main__':
    main(args)



