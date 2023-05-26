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

import math
import torch
import torch.nn as nn
import torch.nn.functional as F

from ...init import kaiming_uniform_relu
from ...blocks import BlockBase
from ...layers import ConvBlock, UpConvBlock, ConvPixelShuffle

class DepthBaseHead(BlockBase):
    def __init__(self):
        super().__init__()

    def loss(self, pred, gt, image_metas):
        if self.clip_gt:
            valid_mask = gt > 0

            pred_org = pred.clone()
            gt_org = gt.clone()
            pred_org[gt_org==0] = self.min_depth  # padding region
            gt_org[gt_org==0] = self.min_depth    # padding region
            pred_org, gt_org = self._train_output(pred_org, gt_org)

            gt = gt.clip(min=self.min_depth, max=self.max_depth)
        else:
            pred_org = pred.clone()
            gt_org = gt.clone()
            pred_org[gt_org==0] = self.min_depth  # padding region
            gt_org[gt_org==0] = self.min_depth    # padding region
            pred_org, gt_org = self._train_output(pred_org, gt_org)

            valid_mask = (gt >= self.min_depth) & (gt <= self.max_depth)
            gt[torch.logical_not(valid_mask)] = self.min_depth    # temporary set min_depth to avoid inf

        pred, gt = self._train_output(pred, gt)

        B, _, H, W = pred.shape
        npix = valid_mask.view(B, -1).sum(-1)

        pred = (pred * valid_mask).view(B, -1)
        gt = (gt * valid_mask).view(B, -1)

        losses = []
        log_vars = {}
        for cfg in self.cfg_loss:
            loss_type = cfg['loss_type']
            coef = cfg['coef']

            if loss_type == 'SIGLoss':
                loss = coef * self._sigloss(pred, gt, npix)
            elif loss_type == 'S-SIGLoss':
                loss = coef * self._scaled_sigloss(pred, gt, npix, alpha=10, lam=0.85)
            elif loss_type == 'GMLoss':
                scales = cfg['grad_matching_scales']
                loss = coef * self._gradient_matching_loss(pred.view(B,1,H,W), gt.view(B,1,H,W), scales=scales)

            losses.append(loss)
            log_vars[loss_type] = loss.item()

        loss = torch.stack(losses).sum()

        return loss, log_vars


    def _sigloss(self, pred, gt, npix):
        d = pred - gt
        loss = (d.pow(2).sum(-1) / npix - d.sum(-1).pow(2) / npix.pow(2) * 0.5).mean()
        return loss

    def _scaled_sigloss(self, pred, gt, npix, alpha, lam):
        d = pred - gt
        loss = (alpha * (d.pow(2).sum(-1) / npix - d.sum(-1).pow(2) / npix.pow(2) * lam).sqrt()).mean()
        return loss

    def _gradient_matching_loss(self, pred, gt, scales):
        d = pred - gt
        losses = []
        npix = 0
        for scale in scales:
            k = int(1 / scale)
            scaled_d = F.avg_pool2d(d, k, k)
            grad_x, grad_y = self._sobel_grad(scaled_d)

            loss_sum = (grad_x.abs() + grad_y.abs()).sum()
            losses.append(loss_sum)
            npix += grad_x.numel()

        loss = torch.stack(losses).sum() / npix
        return loss

    def _sobel_grad(self, img):
        kx = img.new_tensor([[1, 0,-1], [2, 0,-2], [ 1, 0,-1]])[None,None,:,:]
        ky = img.new_tensor([[1, 2, 1], [0, 0, 0], [-1,-2,-1]])[None,None,:,:]
        sobel_x = F.conv2d(img, kx)
        sobel_y = F.conv2d(img, ky)
        return sobel_x, sobel_y

    def _depth_to_Ndepth(self, depth, max_depth, min_depth):
        alpha = math.log(max_depth / min_depth)
        ndepth = (depth / max_depth).log() / alpha + 1
        return ndepth

    def _Ndepth_to_depth(self, ndepth, max_depth, min_depth):
        alpha = math.log(max_depth / min_depth)
        depth = max_depth * (alpha * (ndepth - 1)).exp()
        return depth


class DepthRegHead(DepthBaseHead):
    def __init__(self, inc, outc, num_extra_conv=1, norm_layer=nn.GroupNorm, act_layer=nn.SiLU, drop=0.1, no_norm_at_extra_conv=False,
                       nlogdepth=False, max_depth=80, min_depth=1.978, cfg_loss=None, clip_gt=False, num_upsampling=0, upsampling_layer='upconv', reduce_dim_per_upsample=False):
        super().__init__()
        assert upsampling_layer in ('upconv', 'pixelshuffle')
        self.nlogdepth = nlogdepth
        self.max_depth = max_depth
        self.min_depth = min_depth
        self.clip_gt = clip_gt
        self.cfg_loss = cfg_loss
        self.upsampling_layer = upsampling_layer

        self.upsample = nn.ModuleList()
        dim = inc
        for _ in range(num_upsampling):
            if reduce_dim_per_upsample:
                dim = int(dim // 2)
            if upsampling_layer == 'upconv':
                up = UpConvBlock(inc, dim, kernel_size=3, padding=1, bias=norm_layer is None, norm_layer=norm_layer, act_layer=act_layer)
            elif upsampling_layer == 'pixelshuffle':
                up = ConvPixelShuffle(inc, dim, kernel_size=3, padding=1, scale_factor=2, norm_layer=norm_layer, act_layer=act_layer)
            self.upsample.append(up)
            inc = dim


        head = []

        bias_extra_conv = norm_layer is None or no_norm_at_extra_conv
        norm_extra = norm_layer if no_norm_at_extra_conv == False else None
        for _ in range(num_extra_conv):
            conv = ConvBlock(inc, inc, kernel_size=3, padding=1, bias=bias_extra_conv, norm_layer=norm_extra, act_layer=act_layer, drop=drop)
            head.append(conv)

        head.append( ConvBlock(inc, outc, kernel_size=1, bias=True, norm_layer=None, act_layer=None) )

        self.head = nn.Sequential(*head)

    @torch.jit.ignore
    def init_weights(self):
        kaiming_uniform_relu(self.modules())

    def forward(self, features, image_metas):
        W = image_metas[0]['width']
        H = image_metas[0]['height']

        x = features[0] if isinstance(features, (list, tuple)) else features

        for up in self.upsample:
            if self.upsampling_layer == 'upconv':
                x = up(x, scale_factor=2)
            elif self.upsampling_layer == 'pixelshuffle':
                x = up(x)

        pred = self.head(x)
        if pred.shape[-2:] != (H, W):
            pred = F.interpolate(pred, size=(H, W), mode='bilinear')

        return pred

    def inference(self, features, image_metas):
        pred = self.forward(features, image_metas)
        return self._test_output(pred)

    def _train_output(self, pred, gt):
        if self.nlogdepth:
            return pred.sigmoid(), self._depth_to_Ndepth(gt, self.max_depth, self.min_depth)
        else:
            return (pred.sigmoid() * self.max_depth).log(), gt.log()

    def _test_output(self, pred):
        if self.nlogdepth:
            return self._Ndepth_to_depth(pred.sigmoid(), self.max_depth, self.min_depth)
        else:
            return pred.sigmoid() * self.max_depth

