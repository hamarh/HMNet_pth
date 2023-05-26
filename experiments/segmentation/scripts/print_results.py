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
    'mIoU',
    'Acc',
    'Spd_sG',
    'Spd_mG',
]

TARGET_COMPACT = [
    'Model',
    'Pred',
    'mIoU',
]

def main(args):
    table = []

    if args.compact:
        target = TARGET_COMPACT
    else:
        target = TARGET

    list_fpath = sorted(glob.glob('./workspace/*/result/pred_*/logs/result_eval_pixel_multiclass_margin0.txt'))

    for fpath in list_fpath:
        model_name, pred_name = get_name(fpath, args.len)
        results = get_results(fpath)
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
    is_latency_ss = lambda line: 'mean_ss:' in line or 'mean_sgpu:' in line
    is_latency_cs = lambda line: 'mean_cs:' in line
    is_latency_mp = lambda line: 'mean_mp:' in line
    is_latency_cs_mgpu = lambda line: 'mean_cs_mgpu:' in line
    is_latency_mp_mgpu = lambda line: 'mean_mp_mgpu:' in line or 'mean_mgpu:' in line

    is_latency_ss_fp16 = lambda line: 'mean_ss_fp16:' in line
    is_latency_cs_fp16 = lambda line: 'mean_cs_fp16:' in line
    is_latency_mp_fp16 = lambda line: 'mean_mp_fp16:' in line
    is_latency_cs_mgpu_fp16 = lambda line: 'mean_cs_mgpu_fp16:' in line
    is_latency_mp_mgpu_fp16 = lambda line: 'mean_mp_mgpu_fp16:' in line

    fpath_latency = '/'.join(fpath_eval_result.split('/')[:-1]) + '/latency.txt'
    if not os.path.isfile(fpath_latency):
        return defaultdict(lambda: -1)

    with open(fpath_latency, 'r') as fp:
        lines = fp.read().split('\n')

    result = defaultdict(lambda: -1)
    get_value = lambda line, fmt: fmt % float(line.split(' ')[1])

    for line in lines:
        if is_latency_ss(line):
            result['Spd_SS'] = get_value(line, '%.2f')
        elif is_latency_cs(line):
            result['Spd_CS'] = get_value(line, '%.2f')
        elif is_latency_mp(line):
            result['Spd_MP'] = get_value(line, '%.2f')
        elif is_latency_cs_mgpu(line):
            result['Spd_CSmG'] = get_value(line, '%.2f')
        elif is_latency_mp_mgpu(line):
            result['Spd_MPmG'] = get_value(line, '%.2f')

        elif is_latency_ss_fp16(line):
            result['Spd_SS_fp16'] = get_value(line, '%.2f')
        elif is_latency_cs_fp16(line):
            result['Spd_CS_fp16'] = get_value(line, '%.2f')
        elif is_latency_mp_fp16(line):
            result['Spd_MP_fp16'] = get_value(line, '%.2f')
        elif is_latency_cs_mgpu_fp16(line):
            result['Spd_CSmG_fp16'] = get_value(line, '%.2f')
        elif is_latency_mp_mgpu_fp16(line):
            result['Spd_MPmG_fp16'] = get_value(line, '%.2f')

    return result


def get_results(fpath_eval_result, suffix=''):
    is_miou = lambda line: f'mean IoU' in line
    is_acc  = lambda line: f'overall accuracy' in line

    with open(fpath_eval_result, 'r') as fp:
        lines = fp.read().split('\n')

    result = defaultdict(lambda: -1)

    get_value = lambda line, fmt: fmt % float(line.split(' ')[3].replace('%', ''))

    for line in lines:
        if is_miou(line):
            result[f'mIoU{suffix}'] = get_value(line, '%.2f')
        elif is_acc(line):
            result[f'Acc{suffix}'] = get_value(line, '%.2f')

    return result


if __name__ == '__main__':
    main(args)



