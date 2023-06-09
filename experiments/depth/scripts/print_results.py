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
    parser.add_argument('--len', type=int, default=60, help='')
    parser.add_argument('--compact', action='store_true', help='')
    args = parser.parse_args()

import numpy as np
import sys
import os
from collections import defaultdict
import pandas as pd
import glob

TARGET = [
    'Model',
    'Pred',
    'a1',
    'a2',
    'a3',
    #'Abs(10)',
    #'Abs(20)',
    #'Abs(30)',
    'AbsRel',
    #'SqRel',
    'RMSE',
    'RMSElog',
    #'SIlog',
    #'Log10',
    #'Spd_sG',
    #'Spd_mG',
]

TARGET_COMPACT = [
    'Model',
    'Pred',
    'Abs(30)',
    'AbsRel',
    'RMSE',
    'SIlog',
]

def main(args):
    list_fpath = sorted(glob.glob('./workspace/*/result/pred_day1/logs/result.txt'))
    print('[outdoor day1]')
    print_table(args, list_fpath)

    print('[outdoor night1]')
    list_fpath = sorted(glob.glob('./workspace/*/result/pred_night1/logs/result.txt'))
    print_table(args, list_fpath)

def print_table(args, list_fpath):
    target = TARGET_COMPACT if args.compact else TARGET
    table = []

    for fpath in list_fpath:
        model_name, pred_name = get_name(fpath, args.len)
        results = get_results(fpath)
        results.update(get_results(fpath, suffix='_cutoff30'))
        results.update(get_results(fpath, suffix='_cutoff20'))
        results.update(get_results(fpath, suffix='_cutoff10'))
        results.update(get_latency(fpath))
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

def get_latency(fpath_eval_result):
    is_latency_sgpu = lambda line: 'mean_sgpu:' in line
    is_latency_mgpu = lambda line: 'mean_mgpu:' in line

    fpath_latency = '/'.join(fpath_eval_result.split('/')[:-1]) + '/latency.txt'
    if not os.path.isfile(fpath_latency):
        return defaultdict(lambda: -1)

    with open(fpath_latency, 'r') as fp:
        lines = fp.read().split('\n')

    result = defaultdict(lambda: -1)
    get_value = lambda line, fmt: fmt % float(line.split(' ')[1])

    for line in lines:
        if is_latency_sgpu(line):
            result['Spd_sG'] = get_value(line, '%.2f')
        elif is_latency_mgpu(line):
            result['Spd_mG'] = get_value(line, '%.2f')

    return result


def get_results(fpath_eval_result, suffix=''):
    is_abs      = lambda line: f'Abs{suffix}:' in line
    is_abs_rel  = lambda line: f'AbsRel{suffix}:' in line
    is_sq_rel   = lambda line: f'SqRel{suffix}:' in line
    is_rmse     = lambda line: f'RMSE{suffix}:' in line
    is_rmse_log = lambda line: f'RMSElog{suffix}:' in line
    is_silog    = lambda line: f'SIlog{suffix}:' in line
    is_log10    = lambda line: f'Log10{suffix}:' in line
    is_a1       = lambda line: f'a1{suffix}:' in line
    is_a2       = lambda line: f'a2{suffix}:' in line
    is_a3       = lambda line: f'a3{suffix}:' in line

    with open(fpath_eval_result, 'r') as fp:
        lines = fp.read().split('\n')

    result = defaultdict(lambda: -1)

    get_value = lambda line, fmt: fmt % float(line.split(' ')[1])

    s = suffix.replace('_cutoff', '(') + ')' if suffix != '' else ''
    for line in lines:
        if is_abs(line):
            result[f'Abs{s}'] = get_value(line, '%.3f')
        elif is_abs_rel(line):
            result[f'AbsRel{s}'] = get_value(line, '%.3f')
        elif is_sq_rel(line):
            result[f'SqRel{s}'] = get_value(line, '%.3f')
        elif is_rmse(line):
            result[f'RMSE{s}'] = get_value(line, '%.3f')
        elif is_rmse_log(line):
            result[f'RMSElog{s}'] = get_value(line, '%.3f')
        elif is_silog(line):
            result[f'SIlog{s}'] = get_value(line, '%.3f')
        elif is_log10(line):
            result[f'Log10{s}'] = get_value(line, '%.3f')
        elif is_a1(line):
            result[f'a1{s}'] = get_value(line, '%.3f')
        elif is_a2(line):
            result[f'a2{s}'] = get_value(line, '%.3f')
        elif is_a3(line):
            result[f'a3{s}'] = get_value(line, '%.3f')

    return result


if __name__ == '__main__':
    main(args)



