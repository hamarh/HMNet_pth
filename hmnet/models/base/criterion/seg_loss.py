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

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lovasz_losses import lovasz_hinge

class SegLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=255, id2trainid=None, prefix='',
                     coef_ce=1.0, coef_rce=0., eps_rce=1.0e-4, coef_dice=0., coef_jaccard=0., coef_logdice=0., coef_logjaccard=0., coef_lovasz=0., coef_miou=0.):
        super().__init__()
        self.ce = CrossEntropyLoss(weight, reduction, ignore_index)
        self.dice = DiceLoss(smooth=1.)
        self.jaccard = JaccardLoss(smooth=1.)
        self.logdice = LogDiceLoss(smooth=1.)
        self.logjaccard = LogJaccardLoss(smooth=1.)
        self.lovasz = LovaszHingeLoss()
        self.miou = mIoULoss()

        self.coef_ce = coef_ce
        self.coef_rce = coef_rce
        self.eps_rce = eps_rce
        self.coef_dice = coef_dice
        self.coef_jaccard = coef_jaccard
        self.coef_logdice = coef_logdice
        self.coef_logjaccard = coef_logjaccard
        self.coef_lovasz = coef_lovasz
        self.coef_miou = coef_miou

        self.ignore_index = ignore_index
        self.id2trainid = id2trainid
        self.prefix = prefix
        self._report = []

    def forward(self, logit, target):
        loss = 0
        log_vars = {}
        prefix = self.prefix

        if self.id2trainid is not None:
            target = self.replace(target, self.id2trainid)

        l = self.ce(logit, target)
        loss = loss + self.coef_ce * l
        log_vars[prefix + 'CE  Loss'] = l.item()

        if self.coef_rce > 0:
            N, C, H, W = logit.shape
            pred = F.softmax(logit, dim=1)
            pred = pred.view(N, C, -1).transpose(1,2).contiguous().view(-1, C)
            label = torch.squeeze(target, 1).view(-1)

            if self.ignore_index is not None:
                mask = label != self.ignore_index
                pred = pred[mask]
                label = label[mask]

            label = F.one_hot(label)

            loss_rce = -(pred * (label + self.eps_rce).log()).sum(-1).mean()
            loss = loss + self.coef_rce * loss_rce
            log_vars[prefix + 'RCE  Loss'] = loss_rce.item()

        if self.coef_dice > 0:
            l = self.dice(logit, target)
            loss = loss + self.coef_dice * l
            log_vars[prefix + 'Dice Loss'] = l.item()

        if self.coef_jaccard > 0:
            l = self.jaccard(logit, target)
            loss = loss + self.coef_jaccard * l
            log_vars[prefix + 'Jacc Loss'] = l.item()

        if self.coef_logdice > 0:
            l = self.logdice(logit, target)
            loss = loss + self.coef_logdice * l
            log_vars[prefix + 'LogD Loss'] = l.item()

        if self.coef_logjaccard > 0:
            l = self.logjaccard(logit, target)
            loss = loss + self.coef_logjaccard * l
            log_vars[prefix + 'LogJ Loss'] = l.item()

        if self.coef_lovasz > 0:
            l = self.lovasz(logit, target)
            loss = loss + self.coef_lovasz * l
            log_vars[prefix + 'Lovasz Loss'] = l.item()

        if self.coef_miou > 0:
            l = self.miou(logit, target)
            loss = loss + self.coef_miou * l
            log_vars[prefix + 'mIoU Loss'] = l.item()

        return loss, log_vars

    def replace(self, target, id2trainid):
        output = target.clone()
        for src, dst in id2trainid.items():
            output[target==src] = dst
        return output

class CrossEntropyLoss(nn.Module):
    def __init__(self, weight=None, reduction='mean', ignore_index=-100):
        super(CrossEntropyLoss, self).__init__()
        self.ce_loss = nn.CrossEntropyLoss(weight, reduction=reduction, ignore_index=ignore_index)
        self.ignore_index = ignore_index

    # target size: (N,1,...)
    # input  size: (N,C,...)
    def forward(self, inputs, targets):
        if len(targets.shape) == 4:
            targets = torch.squeeze(targets, 1)
        return self.ce_loss(inputs, targets)


def _calc_dice_coef(targets, logits, smooth):
        N = targets.size(0)
        pred = F.softmax(logits, dim=1)
        iflat = pred[:,1,:,:].view(N, -1)
        tflat = targets.view(N, -1).float()
        intersection = (iflat * tflat).sum(dim=1)
        return (2. * intersection + smooth) / (iflat.sum(dim=1) + tflat.sum(dim=1) + smooth)

def _calc_jaccard_index(targets, logits, smooth):
        N = targets.size(0)
        pred = F.softmax(logits, dim=1)
        iflat = pred[:,1,:,:].view(N, -1)
        tflat = targets.view(N, -1).float()
        intersection = (iflat * tflat).sum(dim=1)
        return (intersection + smooth) / (iflat.sum(dim=1) + tflat.sum(dim=1) - intersection + smooth)

class DiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(DiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        dice_coef = _calc_dice_coef(targets, logits, self.smooth)
        return (1 - dice_coef).mean()


class JaccardLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(JaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        jaccard_index = _calc_jaccard_index(targets, logits, self.smooth)
        return (1 - jaccard_index).mean()


class LogDiceLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(LogDiceLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        dice_coef = _calc_dice_coef(targets, logits, self.smooth)
        return (dice_coef.mean() + 1.0e-7).log() * (-1)


class LogJaccardLoss(nn.Module):
    def __init__(self, smooth=1.):
        super(LogJaccardLoss, self).__init__()
        self.smooth = smooth

    def forward(self, logits, targets):
        jaccard_index = _calc_jaccard_index(targets, logits, self.smooth)
        return (jaccard_index.mean() + 1.0e-7).log() * (-1)

class LovaszHingeLoss(nn.Module):
    def __init__(self):
        super(LovaszHingeLoss, self).__init__()

    def forward(self, logits, targets):
        pos = logits[:,1,:,:]
        neg = logits[:,0,:,:]
        return lovasz_hinge(pos-neg, targets, per_image=True, ignore=None)

def to_one_hot_var(tensor, nClasses, requires_grad=False):

    n, h, w = tensor.size()
    one_hot = tensor.new(n, nClasses, h, w).fill_(0)
    one_hot = one_hot.scatter_(1, tensor.view(n, 1, h, w), 1)
    return one_hot

class mIoULoss(nn.Module):
    def __init__(self, weight=[0,1], size_average=True, n_classes=2):
        super(mIoULoss, self).__init__()
        self.classes = n_classes
        weight = torch.FloatTensor(weight)
        self.weights = weight * weight

    def forward(self, logits, targets):
        # inputs => N x Classes x H x W
        # target => N x H x W
        # target_oneHot => N x Classes x H x W

        weights = self.weights.to(inputs.device).view(1, self.classes)

        N = inputs.size()[0]
        target_oneHot = to_one_hot_var(targets, self.classes).float()

        # predicted probabilities for each pixel along channel
        inputs = F.softmax(inputs, dim=1)

        # Numerator Product
        inter = inputs * target_oneHot
        ## Sum over all pixels N x C x H x W => N x C
        inter = inter.view(N, self.classes, -1).sum(2)

        # Denominator
        union = inputs + target_oneHot - (inputs * target_oneHot)
        ## Sum over all pixels N x C x H x W => N x C
        union = union.view(N, self.classes, -1).sum(2)

        loss = (weights * inter) / (weights * union + 1e-8)

        ## Return average loss over classes and batch
        return -torch.mean(loss)

class OhemCrossEntropy(nn.Module): 
    def __init__(self, ignore_label=-1, thres=0.7, min_kept=100000, weight=None): 
        super(OhemCrossEntropy, self).__init__() 
        self.thresh = thres
        self.min_kept = max(1, min_kept)
        self.ignore_label = ignore_label 
        self.criterion = nn.CrossEntropyLoss(weight=weight, 
                                             ignore_index=ignore_label, 
                                             reduction='none') 
    
    def forward(self, score, target, **kwargs):
        target = torch.squeeze(target, 1)

        ph, pw = score.size(2), score.size(3)
        h, w = target.size(1), target.size(2)
        if ph != h or pw != w:
            score = F.upsample(input=score, size=(h, w), mode='bilinear')
        pred = F.softmax(score, dim=1)
        pixel_losses = self.criterion(score, target).contiguous().view(-1)
        mask = target.contiguous().view(-1) != self.ignore_label         
        
        tmp_target = target.clone() 
        tmp_target[tmp_target == self.ignore_label] = 0 
        pred = pred.gather(1, tmp_target.unsqueeze(1)) 
        pred, ind = pred.contiguous().view(-1,)[mask].contiguous().sort()
        min_value = pred[min(self.min_kept, pred.numel() - 1)] 
        threshold = max(min_value, self.thresh) 
        
        pixel_losses = pixel_losses[mask][ind]
        pixel_losses = pixel_losses[pred < threshold] 
        return pixel_losses.mean()

        

