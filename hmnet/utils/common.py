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

from collections import OrderedDict
from collections import MutableMapping
import logging
import os
import glob
import copy
import time
import numpy as np
from PIL import Image
from subprocess import getoutput
import torch
import random
import h5py
import hdf5plugin

def fix_seed(torch_seed, seed=None):
    seed = seed or torch_seed
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(torch_seed)
    torch.cuda.manual_seed_all(torch_seed)

def mkdir(dpath):
    if not os.path.isdir(dpath):
        try:
            os.mkdir(dpath)
        except FileExistsError:
            print('Already exists: %s' % dpath)
            pass
        except:
            print(traceback.format_exc())

def makedirs(dpath):
    if not os.path.isdir(dpath):
        try:
            os.makedirs(dpath)
        except FileExistsError:
            print('Already exists: %s' % dpath)
            pass
        except:
            print(traceback.format_exc())

# Get file list
def get_list(dpath, ext, root=None):
    if isinstance(dpath, (list, tuple)):
        fpathlist = dpath
    elif os.path.isdir(dpath):
        if isinstance(ext, str):
            ext = [ext]
        fpathlist = []
        for e in ext:
            fpathlist += glob.glob(f'{dpath}/*.{e}')
        fpathlist = sorted(fpathlist)
    elif os.path.isfile(dpath):
        fpathlist = [fpath for fpath in open(dpath, 'r').read().split('\n') if fpath != '']
    else:
        fpathlist = []

    if root is not None:
        strip = lambda x: x[1:] if x[0] == '.' else x
        fpathlist = [ root + '/' + strip(fpath) for fpath in fpathlist ]

    return fpathlist

def get_chunk(target_list, chunk_id=None, num_chunk=None, chunk_str=None):
    if chunk_str is not None:
        chunk_id, num_chunks = chunk_str.split('/')
        chunk_id = int(chunk_id) - 1
        num_chunk = int(num_chunks)

    if chunk_id is None or num_chunk is None:
        raise RuntimeError

    return split_list(target_list, num_chunk)[chunk_id]

def split_list(target_list, num):
    num_per_split = [len(target_list) // num]*num
    for i in range(len(target_list) % num):
        num_per_split[i] += 1

    output = []
    now = 0
    for N in num_per_split:
        output.append(target_list[now:now+N])
        now += N

    return output

def set_logger(quiet=False, logfile=None, level=20):
    # setup logger
    logger = logging.getLogger(__name__)
    logger.setLevel(level)
    formatter = logging.Formatter('[%(asctime)s - %(filename)s line %(lineno)d] %(levelname)s: %(message)s')

    if logfile is not None:
        fh = logging.FileHandler(logfile, 'w')
        logger.addHandler(fh)
        fh.setFormatter(formatter)

    if not quiet:
        sh = logging.StreamHandler()
        logger.addHandler(sh)
        sh.setFormatter(formatter)

    return logger

class CDict(MutableMapping, dict):
    def __init__(self, *args, **kargs):
        super(CDict, self).__setattr__('_dict', OrderedDict(*args, **kargs))

    def __setattr__(self, name, value):
        self._dict[name] = value

    def __getattr__(self, name):
        if name not in self._dict:
            raise AttributeError
        return self._dict[name]

    def __setitem__(self, key, value):
        self.__setattr__(key, value)

    def __getitem__(self, item):
        return self._dict[item]

    def __repr__(self):
        return '%s(%r)' % (self.__class__.__name__, self._dict.items())

    def __delitem__(self, key):
        del self._dict[key]
    
    def __iter__(self):
        return iter(self._dict)
    
    def __len__(self):
        return len(self._dict)
    
    def __contains__(self, item):
        return item in self._dict
        
    def __str__(self):
        return str(self._dict)

    def __call__(self, idx):
        key = list(self._dict.keys())[idx]
        return self.__getattr__(key)

    def __getstate__(self):
        return self.__dict__

    def __setstate__(self, state):
        self.__dict__.update(state)

    def __deepcopy__(self, memo):
        out = CDict()
        for key, value in self._dict.items():
            out[key] = copy.deepcopy(value, memo)
        return out

    def index(self, key):
        return list(self._dict.keys()).index(key)

    def copy(self):
        return self.clone()

    def clone(self):
        out = CDict()
        for key, value in self._dict.items():
            out[key] = copy.deepcopy(value)
        return out

    @classmethod
    def from_dict(cls, dic):
        cdic = cls()
        for key, value in dic.items():
            cdic[key] = value
        return cdic

class MovingAverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, fraction=0.99):
        self.reset()
        self.fraction = fraction

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

        self.mv_sum = 0
        self.mv_avg = 0
        self.mv_count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

        self.mv_sum = val + self.fraction*self.mv_sum
        self.mv_count = 1.0 + self.fraction*self.mv_count
        self.mv_avg = self.mv_sum / self.mv_count

    def __len__(self):
        return self.count

class Timer:
    SCALE = {
        'sec' : 1,
        'msec': 1e3,
        'usec': 1e6,
        'nsec': 1e9,
    }
    def __init__(self, name='Timer', unit='msec', devices=None, stream=None, sync_method='device', enabled=True):
        self.times = []
        self.st = None
        self.lazy_count = 0
        self.lazy_warmup = False
        self.name = name
        self.unit = unit
        self.devices = devices
        self.stream = stream
        self.sync_method = sync_method
        self.enabled = enabled

    def __enter__(self):
        self.start()

    def __exit__(self, ex_type, ex_value, trace):
        self.end()

    def _get_cfgs(self, name, unit):
        name = name or self.name
        unit = unit or self.unit
        return name, unit

    def reset(self):
        self.times = []
        self.st = None
        self.lazy_count = 0
        self.lazy_warmup = False

    def lazy_start(self, lazy_count, devices=None, stream=None, sync_method='device'):
        if not self.enabled:
            return
        if self.lazy_count == lazy_count:
            self.start(devices, stream, sync_method)
            self.lazy_warmup = False
        else:
            self.lazy_count += 1
            self.lazy_warmup = True

    def start(self, devices=None, stream=None, sync_method='device'):
        if not self.enabled:
            return
        devices = devices or self.devices
        self._cuda_sync(devices, stream, sync_method)
        self.st = time.perf_counter()

    def tick(self, name=None, unit=None, display=False, devices=None, stream=None, sync_method='device'):
        if not self.enabled or self.lazy_warmup:
            return
        name, unit = self._get_cfgs(name, unit)
        self._cuda_sync(devices, stream, sync_method)
        duration = time.perf_counter() - self.st
        self.times.append(duration)
        self.st = time.perf_counter()

        if display:
            duration *= self.SCALE[unit]
            print('[%s] %.2f' % (name, duration))

    def end(self, name=None, unit=None, devices=None, stream=None, sync_method='device'):
        if not self.enabled:
            return
        if self.st is None:
            print('Warning: "end" was called before the timer has been started')
            return
        name, unit = self._get_cfgs(name, unit)
        self._cuda_sync(devices, stream, sync_method)
        duration = time.perf_counter() - self.st
        scale = self.SCALE[unit]
        if len(self.times) > 0:
            self.times.append(duration)
            times = np.array(self.times) * scale
            print('[%s] mean: %.2f %s, (std: %.2f, min: %.2f, max: %.2f)' % (name, np.mean(times), unit, np.std(times), np.min(times), np.max(times)))
        else:
            duration *= scale
            print('[%s] %.2f %s' % (name, duration, unit))

        self.reset()

    def display(self, name=None, unit=None):
        if not self.enabled:
            return
        name, unit = self._get_cfgs(name, unit)
        scale = self.SCALE[unit]
        if len(self.times) > 1:
            times = np.array(self.times) * scale
            print('[%s] mean: %.2f %s, (std: %.2f, min: %.2f, max: %.2f)' % (name, np.mean(times), unit, np.std(times), np.min(times), np.max(times)))
        elif len(self.times) == 1:
            duration = self.times[0] * scale
            print('[%s] %.2f %s' % (name, duration, unit))
        else:
            print('Warning: timer has nothing to display')

    def _cuda_sync(self, devices=None, stream=None, sync_method='device'):
        if torch.cuda.is_available():
            assert sync_method in ('device', 'stream', 'none')
            devices = devices or self.devices
            stream = stream or self.stream
            sync_method = sync_method or self.sync_method
            if sync_method == 'device':
                if devices is None:
                    torch.cuda.synchronize()
                else:
                    for d in devices:
                        torch.cuda.synchronize(device=d)
            elif sync_method == 'stream':
                stream.synchronize()
            elif sync_method == 'none':
                pass

class ImageFiles:
    def __init__(self, data, data_type, num, timestamp=None, path=None, files=None):
        assert data_type in ('dir', 'npy', 'npy_files', 'hdf5', 'hdf5_files')
        self.data = data
        self.data_type = data_type
        self.timestamp = timestamp
        self.num = num
        self.path = path
        self.files = files
        self.pointer = 0

        if self.data_type in ('npy_files', 'info_files', 'hdf5_files'):
            self._start_indices = np.cumsum([0] + [ len(d) for d in self.data ])[:-1]

    @staticmethod
    def open_dir(dpath, extension):
        data = get_list(dpath, extension)
        data_type = 'dir'
        num = len(data)
        return ImageFiles(data, data_type, num, files=data)

    @staticmethod
    def open_npy(fpath_npy):
        data = np.load(fpath_npy, mmap_mode='r')
        data_type = 'npy'
        num = len(data)
        return ImageFiles(data, data_type, num, files=fpath_npy)

    @staticmethod
    def open_info(fpath_info, key, root=None, read_timestamp=False, timestamp_key='time'):
        strip = lambda x: x[1:] if x[0] == '.' else x
        npy = np.load(fpath_info)
        data = npy[key].tolist()
        data = [ root + '/' + strip(d) for d in data ]
        data_type = 'info'
        timestamp = npy[timestamp_key]
        num = len(data)
        return ImageFiles(data, data_type, num, key, files=data, timestamp=timestamp)

    @staticmethod
    def open_info_files(dpath, root=None):
        list_fpath_info = get_list(dpath, ext='npy', root=root)
        data = [ ImageFiles.open_info(fpath, path) for fpath in list_fpath_info ]
        data_type = 'info_files'
        num = sum([ len(d) for d in data ])
        return ImageFiles(data, data_type, num, files=list_fpath_info)

    @staticmethod
    def open_npy_files(dpath_npy):
        list_fpath = get_list(dpath_npy, ext='npy')
        data = [ ImageFiles.open_npy(fpath) for fpath in list_fpath ]
        data_type = 'npy_files'
        num = sum([ len(d) for d in data ])
        return ImageFiles(data, data_type, num, files=list_fpath)

    @staticmethod
    def open_hdf5(fpath_hdf5, path, read_timestamp=False, timestamp_path='data_ts'):
        data = h5py.File(fpath_hdf5)
        data_type = 'hdf5'

        dt = ImageFiles._ls_hdf5(data, path)
        num = len(dt)
        if read_timestamp:
            ts = ImageFiles._ls_hdf5(data, timestamp_path)[...]
        else:
            ts = None

        return ImageFiles(data, data_type, num, path=path, files=fpath_hdf5, timestamp=ts)

    @staticmethod
    def _ls_hdf5(data, path):
        path = [ p for p in path.split('/') if p != '' ]
        for p in path:
            data = data[p]
        return data

    @staticmethod
    def open_hdf5_files(dpath_hdf5, path):
        list_fpath = get_list(dpath_hdf5, ext=('hdf5', 'h5'))
        data = [ ImageFiles.open_hdf5(fpath, path) for fpath in list_fpath ]
        data_type = 'hdf5_files'
        num = sum([ len(d) for d in data ])
        return ImageFiles(data, data_type, num, files=list_fpath)

    def __len__(self):
        return self.num

    def __iter__(self):
        self.pointer = 0
        return self

    def __next__(self):
        return self.next()

    def next(self):
        if self.pointer >= self.__len__():
            raise StopIteration
        out = self.__getitem__(self.pointer)
        self.pointer += 1
        return out

    def __getitem__(self, index):
        if self.data_type == 'dir':
            return self._read_file(index)
        elif self.data_type == 'npy':
            return self._read_npy(index)
        elif self.data_type == 'npy_files':
            return self._read_npy_files(index)
        elif self.data_type == 'info':
            return self._read_file(index)
        elif self.data_type == 'info_files':
            return self._read_info_files(index)
        elif self.data_type == 'hdf5':
            return self._read_hdf5(index)
        elif self.data_type == 'hdf5_files':
            return self._read_hdf5_files(index)

    def timestamp(self, index):
        if self.data_type in ('dir', 'npy', 'npy_files'):
            return None
        elif self.data_type in ('info', 'hdf5'):
            return self.timestamp[index]
        elif self.data_type in ('info_files', 'hdf5_files'):
            ifile = np.flatnonzero(index >= self._start_indices).max()
            idx = index - self._start_indices[ifile]
            return self.data[ifile].timestamp[index]

    def load_delta_t(self, start_time, delta_t):
        if self.data_type in ('dir', 'npy', 'npy_files'):
            return None
        elif self.data_type in ('info', 'hdf5'):
            mask = (self.timestamp >= start_time) & (self.timestamp < start_time + delta_t)
            indices = np.flatnonzero(mask)
            return np.stack([ self.__getitem__(idx) for idx in indices ])
        elif self.data_type in ('info_files', 'hdf5_files'):
            data = [ d.load_delta_t(start_time, delta_t) for d in self.data ]
            data = [ d for d in data if len(d) > 0 ]
            return np.concatenate(data, axis=0)

    def filename(self, index):
        if self.data_type == 'dir':
            return self.files[index].split('/')[-1]
        elif self.data_type in ('npy', 'hdf5'):
            return self.files.split('/')[-1] + f'_{index}'
        elif self.data_type in ('npy_files', 'hdf5_files'):
            ifile = np.flatnonzero(index >= self._start_indices).max()
            idx = index - self._start_indices[ifile]
            return self.files[ifile].split('/')[-1] + f'_{idx}'

    def _read_file(self, index):
        fpath = self.data[index]
        if fpath.endswith('npy'):
            return np.load(fpath)
        else:
            img = np.array(Image.load(fpath))
            if img.ndim == 3:
                img = img.transpose([2,0,1])
            return img

    def _read_npy(self, index):
        return np.array(self.data[index])

    def _read_npy_files(self, index):
        ifile = np.flatnonzero(index >= self._start_indices).max()
        idx = index - self._start_indices[ifile]
        return np.array(self.data[ifile][idx])

    def _read_info_files(self, index):
        ifile = np.flatnonzero(index >= self._start_indices).max()
        idx = index - self._start_indices[ifile]
        return np.array(self.data[ifile][idx])

    def _read_hdf5(self, index):
        dt = self._ls_hdf5(self.data, self.path)
        return np.array(dt[index])

    def _read_hdf5_files(self, index):
        ifile = np.flatnonzero(index >= self._start_indices).max()
        idx = index - self._start_indices[ifile]
        return np.array(self.data[ifile][idx])


