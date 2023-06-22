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
    parser.add_argument('dpath_pred', type=str, help='Directory path for prediction results or text file of the file list')
    parser.add_argument('dpath_gt', type=str, help='Directory path for ground truth images or text file of the file list')
    parser.add_argument('dpath_out', type=str, help='')
    parser.add_argument('--max_depth', type=float, required=True, help='')
    parser.add_argument('--min_depth', type=float, required=True, help='')
    parser.add_argument('--cutoff_depth', type=float, nargs='+', default=[], help='')
    parser.add_argument('--clip_pred', action='store_true', help='')
    parser.add_argument('--nlog_pred', action='store_true', help='')
    parser.add_argument('--gt_root', type=str, default='', help='')
    parser.add_argument('--input_type', type=str, default='dir', choices=['dir', 'info', 'mvsec'], help='')
    parser.add_argument('--skip_ts', type=float, help='msec')
    parser.add_argument('--skip_preds', type=int, default=0, help='')
    parser.add_argument('--skip_gts', type=int, default=0, help='')
    args = parser.parse_args()

import math
import numpy as np
import pandas as pd
import h5py

from hmnet.utils.common import get_list, mkdir

def eval_from_info(args):
    list_fpath_pred_info = sorted(get_list(args.dpath_pred, ext='npy'))
    list_fpath_gt_info   = sorted(get_list(args.dpath_gt  , ext='npy'))
    list_fpath_gt_info   = [ args.gt_root + '/' + strip(fpath) for fpath in list_fpath_gt_info ]

    list_fpath_pred, list_fpath_gt = [], []
    for fpath_pred_info, fpath_gt_info in zip(list_fpath_pred_info, list_fpath_gt_info):
        pred_info = np.load(fpath_pred_info)
        gt_info = np.load(fpath_gt_info)

        if args.skip_ts is not None:
            skip_ts = args.skip_ts * 1000
            base_time = gt_info['t'][0]

            mask = (pred_info['t'] - base_time) >= skip_ts
            predlist = pred_info['depth'][mask].tolist()

            mask = (gt_info['t'] - base_time) >= skip_ts
            gtlist = gt_info['depth'][mask].tolist()
        else:
            predlist = pred_info['depth'].tolist()
            gtlist = gt_info['depth'].tolist()

        assert len(predlist) == len(gtlist)

        list_fpath_pred += predlist
        list_fpath_gt += gtlist

    for fpath_pred, fpath_gt in zip(list_fpath_pred, list_fpath_gt):
        assert fpath_pred.split('/')[-1] == fpath_gt.split('/')[-1]

    list_fpath_gt = [ args.gt_root + '/' + strip(fpath) for fpath in list_fpath_gt ]

    eval_list(list_fpath_pred, list_fpath_gt, args.max_depth, args.min_depth, args.cutoff_depth, args.dpath_out, args.clip_pred, args.nlog_pred)

def eval_dir(args):
    list_fpath_pred = get_list(args.dpath_pred, ext='npy')
    list_fpath_gt   = get_list(args.dpath_gt  , ext='npy')
    list_fpath_gt = [ args.gt_root + strip(fpath) for fpath in list_fpath_gt ]
    eval_list(list_fpath_pred, list_fpath_gt, args.max_depth, args.min_depth, args.cutoff_depth, args.dpath_out, args.clip_pred, args.nlog_pred)

def Ndepth_to_depth(ndepth, max_depth, min_depth):
    alpha = math.log(max_depth/ min_depth)
    depth = max_depth * np.exp(alpha * (ndepth - 1))
    return depth

def eval_list(list_fpath_pred, list_fpath_gt, max_depth, min_depth, cutoff_depth, dpath_out, clip_pred, nlog_pred=False):
    assert len(list_fpath_pred) == len(list_fpath_gt)

    mkdir(dpath_out)

    eval_results = EvalResults()

    for i, (fpath_pred, fpath_gt) in enumerate(zip(list_fpath_pred, list_fpath_gt)):
        pred = np.load(fpath_pred).squeeze()
        if nlog_pred:
            pred = Ndepth_to_depth(pred, max_depth, min_depth)
        gt = np.load(fpath_gt).squeeze()
        result = evaluate_one_sample(pred, gt, max_depth, min_depth, cutoff_depth, clip_pred)
        eval_results.append(fpath_pred, result)
        print(f'{i} / {len(list_fpath_gt)}')

    save_results(eval_results, dpath_out)

def eval_mvsec(args):
    preds = np.load(args.dpath_pred)
    gtdata = h5py.File(args.dpath_gt)
    gts = gtdata['davis']['left']['depth_image_raw'][...]

    eval_npy(preds, gts, args.max_depth, args.min_depth, args.cutoff_depth, args.dpath_out, args.clip_pred, args.skip_preds, args.skip_gts)

def eval_npy(preds, gts, max_depth, min_depth, cutoff_depth, dpath_out, clip_pred, skip_preds, skip_gts):
    mkdir(dpath_out)

    preds = preds[skip_preds:]
    gts = gts[skip_gts:]

    assert len(preds) == len(gts)

    eval_results = EvalResults()

    for i, (pred, gt) in enumerate(zip(preds, gts)):
        if np.all(pred < 0) == True:
            print('Warning: valid prediction not exist. Using prediction at the previous frame.')
            pred = preds[i-1]
        result = evaluate_one_sample(pred.squeeze(), gt.squeeze(), max_depth, min_depth, cutoff_depth, clip_pred)
        eval_results.append(skip_gts + i, result)
        print(f'{i} / {len(gts)}')

    save_results(eval_results, dpath_out)


def save_results(eval_results, dpath_out):
    output = eval_results.mean()

    print('====================')
    print(output)

    report = ''
    for k,v in output.items():
        report += f'{k}: {v}\n'

    with open(f'{dpath_out}/result.txt', 'w') as fp:
        fp.write(report)

    eval_results.to_csv(f'{dpath_out}/all_result.csv')

def evaluate_one_sample(pred, gt, max_depth, min_depth, cutoff_depth, clip_pred):
    list_max_depth = [max_depth] + cutoff_depth

    result = _evaluate(pred, gt, max_depth, min_depth, clip_pred, suffix='')
    for cutoff in cutoff_depth:
        result.update(_evaluate(pred, gt, cutoff, min_depth, clip_pred, suffix=f'_cutoff{int(cutoff)}'))
    return result

def _evaluate(pred, gt, max_depth, min_depth, clip_pred, suffix=''):
    if clip_pred:
        pred = pred.clip(min_depth, max_depth)

    mask = (gt >= min_depth) & (gt <= max_depth)
    pred = pred[mask]
    gt = gt[mask]


    thresh = np.maximum((gt / pred), (pred / gt))
    a1 = (thresh < 1.25).mean()
    a2 = (thresh < 1.25 ** 2).mean()
    a3 = (thresh < 1.25 ** 3).mean()

    diff = gt - pred
    diff_log = np.log(gt) - np.log(pred)

    abs_err  = (np.abs(diff)).mean()
    abs_rel  = (np.abs(diff) / gt).mean()
    sq_rel   = ((diff ** 2) / gt).mean()
    rmse     = np.sqrt((diff ** 2).mean())
    rmse_log = np.sqrt((diff_log ** 2).mean())
    silog    = np.sqrt((diff_log ** 2).mean() - (diff_log.mean()) ** 2) * 100
    log_10   = (np.abs(np.log10(gt) - np.log10(pred))).mean()

    result = {
        f'Abs{suffix}'    : abs_err,
        f'AbsRel{suffix}' : abs_rel,
        f'SqRel{suffix}'  : sq_rel,
        f'RMSE{suffix}'   : rmse,
        f'RMSElog{suffix}': rmse_log,
        f'SIlog{suffix}'  : silog,
        f'Log10{suffix}'  : log_10,
        f'a1{suffix}'     : a1,
        f'a2{suffix}'     : a2,
        f'a3{suffix}'     : a3,
    }
    return result


class EvalResults:
    def __init__(self):
        self.files = []
        self.results = []

    def append(self, fpath, result):
        self.files.append(fpath)
        self.results.append(result)

    def mean(self):
        num = len(self.results)
        if num == 0:
            return self.results

        output = {}
        for key in self.results[0].keys():
            value = np.mean([ dic[key] for dic in self.results ])
            output[key] = value

        return output

    def numpy(self):
        results = [ list(dic.values()) for dic in self.results ]
        return np.array(results)

    def dataframe(self):
        columns = list(self.results[0].keys())
        data = self.numpy()
        return pd.DataFrame(data, index=self.files, columns=columns)

    def to_csv(self, fpath_out):
        df = self.dataframe()
        df.to_csv(fpath_out)


def strip(fpath):
    if fpath[:2] == './':
        fpath = fpath[2:]
    return fpath


if __name__ == '__main__':
    if args.input_type == 'dir':
        assert args.skip_ts is None
        eval_dir(args)
    elif args.input_type == 'info':
        eval_from_info(args)
    elif args.input_type == 'mvsec':
        eval_mvsec(args)


