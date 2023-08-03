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
parser.add_argument('config', type=str, help='Config file')
parser.add_argument('--seed', type=int, default=42, help='')
parser.add_argument('--finetune'  , action='store_true', help='Run finetuning on MVSEC')
parser.add_argument('--single'    , action='store_true', help='Train with single GPU')
parser.add_argument('--amp'       , action='store_true', help='Use automatic mixed precision')
parser.add_argument('--debug'     , action='store_true', help='Run in debug mode')
parser.add_argument('--clean'     , action='store_true', help='Clear workspace if already exist')
parser.add_argument('--overwrite' , action='store_true', help='Overwrite workspace')
parser.add_argument('-q','--quiet', action='store_true', help='Surpress standart output')
# Arguments for DDP
parser.add_argument('--distributed', action='store_true', help='Enable distributed data parallel')
parser.add_argument('--master', type=str, default='localhost', help='[DDP] IP address or name of a master node (default: "localhost")')
parser.add_argument('--node'  , type=str, default='1/1'      , help='[DDP] Specify node index and total number of nodes in the form of "{Node_Index}/{Total_Number_of_Nodes}" (e.g. 1/2, 2/2).\
                                                                     Master node must have node index = 1.\
                                                                     Specify "1/1" for single node DDP (default)')
args = parser.parse_args()

import os
import sys
import shutil
import numpy as np
import time
timer = time.perf_counter
import logging
from importlib import machinery
from collections import OrderedDict
import traceback
from copy import deepcopy

import torch
import torch.nn as nn
import torch.distributed as dist
from torch.cuda.amp import autocast, GradScaler
import torch.multiprocessing as mp

from hmnet.models.base.init import load_state_dict_matched
from hmnet.utils.common import makedirs, fix_seed, split_list, MovingAverageMeter
from hmnet.utils import common as utils
from hmnet.dataset.custom_loader import PseudoEpochLoader

torch.backends.cudnn.benchmark = True

def main_ddp(args, dist_settings, nprocs):
    mp.spawn(main, args=(args, dist_settings), nprocs=nprocs, join=True)

def main(local_rank, args, dist_settings=None):
    config = get_config(args)

    # set seed
    if config.seed is not None:
        fix_seed(config.seed)

    # init dist
    rank, world_size = init_ddp(local_rank, dist_settings, config) if config.distributed else (0, 1)

    # prepair workspace
    if rank == 0:
        prepair_workspace(config)

    # set log
    set_logger(rank, config)
    set_meter(rank, config)

    # set device
    config.device = torch.device("cuda:%d" % local_rank)
    torch.cuda.set_device(local_rank)

    # set model
    model = config.get_model()
    if config.distributed:
        model = nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = model.to(config.device)

    # reset seed for debug
    #if config.seed is not None:
    #    fix_seed(config.seed)

    # set optimizer
    optimizer, scheduler, scaler = set_optimizer(model, config)

    # load state dict
    load_params_if_specified(model, rank, config)

    # resume from checkpoint if specified
    resume_if_specified(model, optimizer, scaler, config)

    # free cache
    torch.cuda.empty_cache()

    # convert to DDP if needed (Note: must be after optimizer setting)
    if config.distributed:
        settings = dict(
            find_unused_parameters = False,
            broadcast_buffers      = False,
            gradient_as_bucket_view= getattr(config, 'grad_as_bucket_view', False),
            #static_graph           = getattr(config, 'static_graph', False),
        )
        model = nn.parallel.DistributedDataParallel(model, device_ids=[local_rank], **settings)

    # set dataset
    train_dataset = config.get_dataset()

    # get dataloader
    train_loader = get_dataloader(train_dataset, rank, world_size, config)

    print_log('Configure done: N_GPUS=%d' % world_size, rank, config)

    start_epoch = getattr(config, 'start_epoch', 0)
    num_epochs = (config.maxiter - 1) // config.iter_per_epoch + 1

    # iter epochs
    for epoch in range(start_epoch, num_epochs):

        train(epoch, train_loader, model, optimizer, scheduler, scaler, rank, config)

        # save
        if rank == 0:
            checkpoint = {
                'epoch'     : epoch + 1,
                'state_dict': get_state_dict(model),
                'optimizer' : optimizer.state_dict(),
                'scaler'    : scaler.state_dict(),
            }
            save_checkpoint(checkpoint, config.dpath_out, epoch, config)

    if config.distributed:
        dist.barrier()
        dist.destroy_process_group()

def train(epoch, loader, model, optimizer, scheduler, scaler, rank, config):
    print_log('start epoch', rank, config)
    
    # switch to train mode
    model.train()

    # initialize variables for report
    meter = config.meter
    meter.reset()
    meter.timer_start()

    for batch_idx, data in enumerate(loader):

        list_events, list_images, list_image_metas, list_labels = parse_event_data(data)
        meter.record_data_time()

        if hasattr(config, 'segment_duration'):
            segment_duration = adapt_segment_durations(list_events, list_image_metas, config.segment_duration, getattr(config, 'max_count_per_segment', 15000*162))

            seg_events, seg_images, seg_image_metas, seg_labels = \
                    split_into_segments(list_events, list_images, list_image_metas, list_labels, segment_duration=segment_duration, num_train_segments=config.num_train_segments)
        else:
            seg_events, seg_images, seg_image_metas, seg_labels = \
                [list_events], [list_images], [list_image_metas], [list_labels]

        for seg_idx, (events, images, image_metas, labels) in enumerate(zip(seg_events, seg_images, seg_image_metas, seg_labels)):

            events = to_device(events, config.device)
            images = to_device(images, config.device)
            labels = to_device(labels, config.device)

            # clear grad
            optimizer.zero_grad(set_to_none=True)

            # forward
            if config.amp:
                with autocast(enabled=True):
                    outputs = model(events, images, image_metas, labels, init_states=seg_idx==0)
                loss = outputs['loss']
                loss_reports = outputs['log_vars']
                meter.update(loss, loss_reports)
                scale_before_step = scaler.get_scale()
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                if scaler.get_scale() >= scale_before_step:
                    scheduler.step(loader.nowiter)
            else:
                outputs = model(events, images, image_metas, labels, init_states=seg_idx==0)
                loss = outputs['loss']
                loss_reports = outputs['log_vars']
                meter.update(loss, loss_reports)
                loss.backward()
                optimizer.step()
                scheduler.step(loader.nowiter)

        # measure elapsed time
        meter.record_batch_time()

        # report
        if batch_idx % config.print_freq == 0:
            meter.batch_report_train(epoch, loader.nowiter, config.maxiter, optimizer, config)

        meter.timer_start()

    # report epoch
    meter.epoch_report_train(epoch, config)

# ============ utils ==========================================================================================

def adapt_segment_durations(list_events, list_image_metas, segment_durations, threshold=15000*162):
    if not isinstance(segment_durations, (list, tuple)):
        return int(segment_durations)

    segment_durations = sorted(segment_durations, reverse=True)
    stride_t = list_image_metas[0][0]['stride_t']
    for segment_duration in segment_durations:
        segment_length = int(segment_duration // stride_t)
        total_length = len(list_events)
        assert total_length % segment_length == 0
        num_segments = total_length // segment_length

        max_count = max([ event_count(evseg) for evseg in split_list(list_events, num_segments) ])

        if max_count < threshold:
            return int(segment_duration)

    return int(segment_durations[-1])

def event_count(list_events):
    Nevt = np.array([ [len(e) for e in evt] for evt in list_events ])
    total_count = Nevt.sum(axis=0).max()
    return total_count

def split_into_segments(list_events, list_images, list_image_metas, list_labels, segment_duration, num_train_segments):
    stride_t = list_image_metas[0][0]['stride_t']
    segment_length = int(segment_duration // stride_t)
    total_length = len(list_events)
    assert total_length % segment_length == 0
    num_segments = total_length // segment_length

    list_events       = split_list(list_events,       num_segments)[-num_train_segments:]
    list_images       = split_list(list_images,       num_segments)[-num_train_segments:]
    list_image_metas  = split_list(list_image_metas,  num_segments)[-num_train_segments:]
    list_labels       = split_list(list_labels,       num_segments)[-num_train_segments:]

    return list_events, list_images, list_image_metas, list_labels

class Meter:
    def __init__(self, logfile_train, rank=0):
        self.rank = rank
        if rank == 0:
            self.reset()
            self.fp_train = open(logfile_train, 'w')

    def __del__(self):
        if self.rank == 0:
            self.fp_train.close()

    def reset(self):
        if self.rank == 0:
            # initialize variables for report
            self.timer_batch = MovingAverageMeter()
            self.timer_data  = MovingAverageMeter()
            self.timer_model = MovingAverageMeter()
            self.loss_meter  = MovingAverageMeter(0.999)
            self.loss_reports = OrderedDict()

    def timer_start(self):
        if self.rank == 0:
            self.st = timer()

    def record_batch_time(self):
        if self.rank == 0:
            self.timer_batch.update(timer() - self.st)

    def record_data_time(self):
        if self.rank == 0:
            self.timer_data.update(timer() - self.st)

    def record_model_time(self):
        if self.rank == 0:
            self.timer_model.update(timer() - self.st)

    def update(self, loss, loss_reports):
        loss = self._reduce(loss)
        loss_reports = { key: self._reduce(loss) for key, loss in loss_reports.items() }
        loss = self._to_number(loss)
        loss_reports = { key: self._to_number(value) for key, value in loss_reports.items() }
        if self.rank == 0:
            self.loss_meter.update(loss)
            for key, value in loss_reports.items():
                if key not in self.loss_reports:
                    self.loss_reports[key] = MovingAverageMeter()
                self.loss_reports[key].update(value)

    def batch_report_train(self, epoch, nowiter, maxiter, optimizer, config):
        if self.rank == 0:
            header = 'Epoch,Iter,Loss'
            csv = '%d,%d,%f' % (epoch+1, nowiter, self.loss_meter.mv_avg)
            logstr = '<< Batch Report >>'
            logstr += '\nTrain Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                epoch+1, nowiter, maxiter,
                100. * (nowiter) / maxiter, self.loss_meter.mv_avg )
            logstr, header, csv = self._print_time(logstr, header, csv)
            logstr, header, csv = self._print_loss(logstr, header, csv, meter='val')
            logstr, header, csv = self._print_optim(logstr, header, csv, optimizer)
            logstr, header, csv = self._print_gpu_usage(logstr, header, csv)
            logstr += '\n'
            header += '\n'
            csv += '\n'
            config.logger.info(logstr)
            if self.fp_train.tell() == 0:
                self.fp_train.write(header)
            self.fp_train.write(csv)

    def epoch_report_train(self, epoch, config):
        if self.rank == 0:
            logstr = '\nEpoch: {} Average loss train: {:.4f}\n'.format(epoch+1, self.loss_meter.avg)
            config.logger.info(logstr)

    def _print_loss(self, pstr, header, csv, meter='val'):
        for name, loss_meter in self.loss_reports.items():
            if meter == 'val':
                value = loss_meter.val
            elif meter == 'avg':
                value = loss_meter.avg
            elif meter == 'mv_avg':
                value = loss_meter.mv_avg
            pstr += '\n    {} = {:.4f}'.format(name, value)
            header += ',%s' % name
            csv += ',%f' % value
        return pstr, header, csv

    def _print_optim(self, pstr, header, csv, optimizer):
        pstr += '\n    lr = ('
        for i, group in enumerate(optimizer.param_groups):
            pstr += ' {:.2e},'.format(group['lr'])
            header += ',LR%d' % i
            csv += ',%f' % group['lr']
        pstr += ' )'
        return pstr, header, csv

    def _print_time(self, pstr, header, csv):
        pstr += '\n    Time for data load: {:.3f}sec (avg. {:.3f}sec)'.format(self.timer_data.val, self.timer_data.mv_avg)
        pstr += '\n    Time for batch    : {:.3f}sec (avg. {:.3f}sec)'.format(self.timer_batch.val, self.timer_batch.mv_avg)
        header += ',DataTime,BatchTime'
        csv += ',%f,%f' % (self.timer_data.mv_avg, self.timer_batch.mv_avg)
        return pstr, header, csv

    def _print_gpu_usage(self, pstr, header, csv):
        if torch.cuda.is_available():
            pstr += '\n    GPU usage: '
            for i in range(torch.cuda.device_count()):
                f, t = torch.cuda.mem_get_info(i)
                mem = (t - f) * 1e-9
                #mem = torch.cuda.memory_reserved(i) * 1e-9
                pstr += f'{mem:.1f}GB, '
                csv += f',{mem:.1f}'
                header += f',GPU{i}'
        return pstr, header, csv

    def _reduce(self, loss):
        return loss
        #if dist.is_initialized() and isinstance(loss, torch.Tensor):
        #    world_size = dist.get_world_size()
        #    dist.reduce(loss.clone(), dst=0)
        #    loss = loss / world_size
        #return loss

    def _to_number(self, loss):
        if isinstance(loss, torch.Tensor):
            loss = loss.item()
        return loss

def init_ddp(local_rank, dist_settings, config):
    rank_offset, world_size, master = dist_settings
    rank = local_rank + rank_offset

    if getattr(config, 'dist_comm_file', False) is True:
        dist.init_process_group(backend='nccl', init_method='file:///home/hama/tmp/%s.cmm' % config.name, rank=rank, world_size=world_size)
    else:
        os.environ['MASTER_ADDR'] = master
        os.environ['MASTER_PORT'] = '21289'
        dist.init_process_group(backend='nccl', rank=rank, world_size=world_size)
    dist.barrier()

    print('DDP setup: rank=%d, local_rank=%d, world_size=%d, master=%s, seed=%d' % (rank, local_rank, world_size, master, config.seed))
    sys.stdout.flush()

    return rank, world_size

def set_optimizer(model, config):

    base_lr = config.optim_params['lr']
    weight_decay = config.optim_params['weight_decay']
    param_groups = model.optim_settings(base_lr, weight_decay)

    optimizer = config.optimizer(param_groups, **config.optim_params)

    scheduler = config.lr_scheduler(optimizer, **config.lrsch_params)

    scaler = GradScaler()

    return optimizer, scheduler, scaler

def get_dataloader(dataset, rank, world_size, config, val=False):
    if dataset is None:
        return None

    loader_param = deepcopy(config.loader_param)

    if val == True:
        loader_param['drop_last'] = False

    if config.distributed:
        sampler = torch.utils.data.distributed.DistributedSampler(dataset, rank=rank, num_replicas=world_size, shuffle=loader_param.pop('shuffle'))
        loader_param['shuffle'] = False
        loader_param['pin_memory'] = False
        loader_param['num_workers'] = 4
        loader_param['drop_last'] = True
        loader_param['batch_size'] = int(loader_param['batch_size'] / world_size)
    else:
        sampler = None

    if val == True:
        config.loader_param['drop_last'] = False

    if val == False:
        loader = PseudoEpochLoader(dataset=dataset, sampler=sampler, **loader_param, iter_per_epoch=config.iter_per_epoch, start_epoch=config.start_epoch)
    else:
        loader = torch.utils.data.DataLoader(dataset, sampler=sampler, **loader_param)

    return loader

def to_device(data, device, non_blocking=True):
    if data is None:
        return data
    elif isinstance(data, (list, tuple)):
        return [ to_device(d, device, non_blocking) for d in data ]
    elif isinstance(data, torch.Tensor):
        return data.to(device, non_blocking=non_blocking)
    else:
        return data

def load_params_if_specified(model, rank, config):
    fpath_load = getattr(config, 'load', '')
    if fpath_load is not None and fpath_load != '':
        state_dict = torch.load(fpath_load, map_location=config.device)['state_dict']
        no_matching = load_state_dict_matched(model, state_dict)
        print_log("=> loading checkpoint '{}'".format(fpath_load), rank, config)
        for key in no_matching:
            print_log('No matching key. Skip loading %s' % key, rank, config)

def resume_if_specified(model, optimizer, scaler, config):
    if not isinstance(config.resume, str) or config.resume == '':
        return

    fpath_checkpoint = config.dpath_out + '/' + config.resume
    if os.path.isfile(fpath_checkpoint):
        print("=> loading checkpoint '{}'".format(fpath_checkpoint))
        checkpoint = torch.load(fpath_checkpoint, map_location=config.device)

        state_dict = checkpoint['state_dict']
        config.start_epoch = checkpoint['epoch']
        model.load_state_dict(state_dict)
        optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scaler' in checkpoint:
            scaler.load_state_dict(checkpoint['scaler'])

        print("=> loaded checkpoint '{}' (epoch {})".format(fpath_checkpoint, checkpoint['epoch']))
    else:
        print("=> no checkpoint found at '{}'".format(fpath_checkpoint))

def save_checkpoint(state, dpath_out, epoch, config):
    if config.finetune:
        fpath_chk = f'{dpath_out}/checkpoint_ft_{epoch+1}.pth.tar'
        fpath_latest = f'{dpath_out}/checkpoint_ft.pth.tar'
    else:
        fpath_chk = f'{dpath_out}/checkpoint_{epoch+1}.pth.tar'
        fpath_latest = f'{dpath_out}/checkpoint.pth.tar'
    torch.save(state, fpath_chk)
    shutil.copyfile(fpath_chk, fpath_latest)

def prepair_workspace(config):
    dpath_out    = config.dpath_out
    fpath_config = config.fpath_config

    if config.clean:
        os.system('rm -r %s' % dpath_out)
    if os.path.isdir(dpath_out):
        if config.overwrite:
            pass
        else:
            print('Experiment dir %s already exists.' % dpath_out)
            print('Specify "--clean" flag to clean dir')
            print('Specify "--overwrite" flag to overwrite')
            quit()
    else:
        makedirs(dpath_out)

    os.system('cp %s %s' % (fpath_config, dpath_out))

def set_logger(rank, config):
    if rank == 0:
        # setup logger
        level = getattr(config, 'log_level', 20)
        if config.finetune:
            logfile = config.dpath_out + '/pytorch_ft.log'
        else:
            logfile = config.dpath_out + '/pytorch.log'
        config.logger = utils.set_logger(config.quiet, logfile, level)

def set_meter(rank, config):
    # setup meter
    if config.finetune:
        log_train = config.dpath_out + '/train_ft.csv'
    else:
        log_train = config.dpath_out + '/train.csv'
    meter = Meter(log_train, rank)
    config.meter = meter

def get_config(args):
    config_module = machinery.SourceFileLoader('config', args.config).load_module()
    if args.finetune:
        config = config_module.FTSettings()
    else:
        config = config_module.TrainSettings()

    config.dpath_out = './workspace/'+args.name
    config.name = args.name
    config.fpath_config = config_module.__file__

    config.seed = args.seed
    config.single = args.single
    config.amp = args.amp
    config.clean = args.clean
    config.overwrite = args.overwrite
    config.quiet = args.quiet
    config.finetune = args.finetune
    config.start_epoch = 0
    config.distributed = args.distributed

    if args.debug == True:
        config.maxiter = 4000
        config.iter_per_epoch = 1000

    if args.single is True:
        config.loader_param['batch_size'] = int(config.loader_param['batch_size'] / 16)

    return config

def print_log(logstr, rank, config):
    if rank == 0:
        config.logger.info(logstr)

def get_state_dict(model):
    if isinstance(model, nn.DataParallel) or isinstance(model, nn.parallel.DistributedDataParallel):
        return model.module.state_dict()
    else:
        return model.state_dict()

def print_device(rank, name, tensor):
    if isinstance(tensor, torch.Tensor):
        print('rank%d %s: %r' % (rank, name, tensor.device))
        sys.stdout.flush()
    elif isinstance(tensor, tuple) or isinstance(tensor, list):
        for t in tensor:
            print_device(rank, name, t)
    
def get_ddp_settings(master, node):
    local_size = torch.cuda.device_count()

    if node == '1/1':
        # single node DDP
        master = 'localhost'
        rank_offset = 0
        world_size = local_size
    else:
        # multi-node DDP
        node_rank, node_size = map(int, node.split('/'))
        rank_offset = local_size * node_rank
        world_size = local_size * node_size

    return master, rank_offset, world_size, local_size
    
def parse_event_data(data):

    def _nested_shape(lst, shape=[]):
        if isinstance(lst, (list, tuple)):
            shape += [len(lst)]
            return _nested_shape(lst[0], shape)
        else:
            return shape

    datas, targets, metas = data
    shape = _nested_shape(metas)

    if len(shape) == 2:
        list_events, list_images, list_image_metas, list_labels = [], [], [], []
        for d, t, m in zip(datas, targets, metas):
            events, images, image_metas, labels = parse_event_data([d, t, m])
            list_events.append(events)
            list_images.append(images)
            list_image_metas.append(image_metas)
            list_labels.append(labels)
        return list_events, list_images, list_image_metas, list_labels
    else:
        events = [ d['events'] for d in datas ]
        images = [ d['images'] for d in datas ]
        labels = [ t['depth'] for t in targets]
        image_metas = [ m['image_meta'] for m in metas ]

        if events[0].ndim == 3:
            events = torch.stack(events)

        return events, images, image_metas, labels

if __name__ == '__main__':
    __spec__ = None
    args.name = args.config.split('/')[-1].replace('.py', '')

    if args.distributed:
        master, rank_offset, world_size, local_size = get_ddp_settings(args.master, args.node)
        dist_settings = [rank_offset, world_size, master]
        main_ddp(args, dist_settings, nprocs=local_size)
    else:
        main(0, args)





