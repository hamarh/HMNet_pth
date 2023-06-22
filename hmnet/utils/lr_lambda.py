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

import math

class Poly(object):
    def __init__(self, power, max_epochs, min_epochs=0, direction='desc', min_coef=0.0):
        self._power = power
        self._max_epochs = max_epochs
        self._min_epochs = min_epochs
        self._direction = direction
        self._min_coef = min_coef

    def __call__(self, epoch):
        # get progress
        progress   = float(epoch - self._min_epochs) / (self._max_epochs - self._min_epochs)
        
        # clip 0 to 1
        if progress > 1.: progress = 1.
        if progress < 0.: progress = 0.

        # calc coef
        if self._direction == 'desc':
            coef = 1. - progress ** self._power
        elif self._direction == 'asc':
            coef = progress ** self._power

        return max(coef, self._min_coef)

class Fixed(object):
    def __call__(self, epoch):
        coef = 1.
        return coef

#{'method': 'fixed', 'range': ( 0, 5), 'lr': 1.0e-2}
#{'method':  'poly', 'range': ( 5,10), 'start_lr': 1.0e-2, 'end_lr': 1.0e-5, 'power': 1.0}
#{'method': 'fixed', 'range': (10,15), 'lr': 1.0e-5}
#{'method': 'fixed', 'range': (15,None), 'lr': 0.0}
#{'method': 'warmup', 'range': (0,5), 'min_lr': 1.0e-6, 'shift': True}
#{'method': 'cosine', 'range': (5,30), 'start_lr': 0.01, 'end_lr': 1.0e-6}
class CombinationV2(object):
    def __init__(self, schedules, initial_lr):
        self._schedules = schedules
        self._initial_lr = initial_lr
        self._warmup_shift = 0

    def __call__(self, epoch):
        for sch in self._schedules:
            st, ed = sch['range']
            if epoch >= st and (ed is None or epoch < ed):
                method = getattr(self, sch['method'])
                return method(epoch, **sch)
        print('Coef for epoch %d is not defined in any scheduler:' % epoch)
        print(self._schedules)
        raise RuntimeError

    def poly(self, epoch, method, range, start_lr, end_lr, power):
        st = range[0] - self._warmup_shift
        ed = range[1]
        progress = float(epoch - st) / (ed - st)

        st_coef = start_lr / self._initial_lr
        ed_coef = end_lr / self._initial_lr
        coef = st_coef + (ed_coef - st_coef) * (progress ** power)
        return coef

    def fixed(self, epoch, method, range, lr):
        return lr / self._initial_lr

    def warmup(self, epoch, method, range, min_lr, shift):
        st = range[0]
        ed = range[1]
        progress = float(epoch - st) / (ed - st)

        st_coef = min_lr / self._initial_lr
        ed_coef = 1.0
        coef = st_coef + (ed_coef - st_coef) * (progress)

        if shift:
            self._warmup_shift = ed

        return coef

    def cosine(self, epoch, method, range, start_lr, end_lr):
        st = range[0] - self._warmup_shift
        ed = range[1]
        progress = float(epoch - st) / (st - ed)
        duration = st - ed

        lr = end_lr + 0.5 * (start_lr - end_lr) * (1 + math.cos(math.pi * progress))
        coef = lr / self._initial_lr

        return coef

class Mixed(object):
    def __init__(self, list_scheduler):
        self._list_scheduler = list_scheduler

    def __call__(self, epoch):
        coef = 1.0
        for sch in self._list_scheduler:
            coef *= sch(epoch)
        return coef














