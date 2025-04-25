#
# Copyright (C) 2023, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import torch
import torch.nn.functional as F
from torch.autograd import Variable
from math import exp


def l1_loss(network_output, gt):
    return torch.abs((network_output - gt)).mean()


def kl_divergence(rho, rho_hat):
    rho_hat = torch.mean(torch.sigmoid(rho_hat), 0)
    rho = torch.tensor([rho] * len(rho_hat)).cuda()
    return torch.mean(
        rho * torch.log(rho / (rho_hat + 1e-5)) + (1 - rho) * torch.log((1 - rho) / (1 - rho_hat + 1e-5)))


def l2_loss(network_output, gt):
    return ((network_output - gt) ** 2).mean()


def gaussian(window_size, sigma):
    gauss = torch.Tensor([exp(-(x - window_size // 2) ** 2 / float(2 * sigma ** 2)) for x in range(window_size)])
    return gauss / gauss.sum()


def create_window(window_size, channel):
    _1D_window = gaussian(window_size, 1.5).unsqueeze(1)
    _2D_window = _1D_window.mm(_1D_window.t()).float().unsqueeze(0).unsqueeze(0)
    window = Variable(_2D_window.expand(channel, 1, window_size, window_size).contiguous())
    return window


def ssim(img1, img2, window_size=11, size_average=True):
    channel = img1.size(-3)
    window = create_window(window_size, channel)

    if img1.is_cuda:
        window = window.cuda(img1.get_device())
    window = window.type_as(img1)

    return _ssim(img1, img2, window, window_size, channel, size_average)


def _ssim(img1, img2, window, window_size, channel, size_average=True):
    mu1 = F.conv2d(img1, window, padding=window_size // 2, groups=channel)
    mu2 = F.conv2d(img2, window, padding=window_size // 2, groups=channel)

    mu1_sq = mu1.pow(2)
    mu2_sq = mu2.pow(2)
    mu1_mu2 = mu1 * mu2

    sigma1_sq = F.conv2d(img1 * img1, window, padding=window_size // 2, groups=channel) - mu1_sq
    sigma2_sq = F.conv2d(img2 * img2, window, padding=window_size // 2, groups=channel) - mu2_sq
    sigma12 = F.conv2d(img1 * img2, window, padding=window_size // 2, groups=channel) - mu1_mu2

    C1 = 0.01 ** 2
    C2 = 0.03 ** 2

    ssim_map = ((2 * mu1_mu2 + C1) * (2 * sigma12 + C2)) / ((mu1_sq + mu2_sq + C1) * (sigma1_sq + sigma2_sq + C2))

    if size_average:
        return ssim_map.mean()
    else:
        return ssim_map.mean(1).mean(1).mean(1)


# --------------BalancedL1Loss-----------------#
import numpy as np
import torch.nn as nn
import torch
import torch.nn.functional as F
# def balanced_l1_loss(pred,
#                      target,
#                      beta=1.0,
#                      alpha=0.5,
#                      gamma=1.5,
#                      reduction='mean'):
#     """Calculate balanced L1 loss.
#
#     Please see the `Libra R-CNN <https://arxiv.org/pdf/1904.02701.pdf>`_
#
#     Args:
#         pred (torch.Tensor): The prediction with shape (N, 4).
#         target (torch.Tensor): The learning target of the prediction with
#             shape (N, 4).
#         beta (float): The loss is a piecewise function of prediction and target
#             and ``beta`` serves as a threshold for the difference between the
#             prediction and target. Defaults to 1.0.
#         alpha (float): The denominator ``alpha`` in the balanced L1 loss.
#             Defaults to 0.5.
#         gamma (float): The ``gamma`` in the balanced L1 loss.
#             Defaults to 1.5.
#         reduction (str, optional): The method that reduces the loss to a
#             scalar. Options are "none", "mean" and "sum".
#
#     Returns:
#         torch.Tensor: The calculated loss
#     """
#     assert beta > 0
#     if target.numel() == 0:
#         return pred.sum() * 0
#
#     assert pred.size() == target.size()
#
#     diff = torch.abs(pred - target)
#     b = np.e ** (gamma / alpha) - 1
#     loss = torch.where(
#         diff < beta, alpha / b *
#         (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
#         gamma * diff + gamma / b - alpha * beta)
#
#     return loss

import torch

def balanced_l1_loss(pred,
                     target,
                     beta=1.0,
                     alpha=0.5,
                     gamma=1.5,
                     reduction='mean'):
    """Calculate balanced L1 loss."""

    assert beta > 0
    if target.numel() == 0:
        return torch.tensor(0.0, device=pred.device)  # 确保返回的张量在相同设备上

    assert pred.size() == target.size()

    diff = torch.abs(pred - target)

    # 将 gamma 和 alpha 转换为张量
    b = torch.exp(torch.tensor(gamma / alpha, device=pred.device)) - 1  # 使用 torch.exp

    # 计算 loss
    loss = torch.zeros_like(diff, device=pred.device)  # 初始化 loss 张量
    mask = diff < beta

    # 仅计算满足条件的部分
    loss[mask] = (alpha / b) * (b * diff[mask] + 1) * torch.log1p(b * diff[mask] / beta) - alpha * diff[mask]
    loss[~mask] = gamma * diff[~mask] + (gamma / b) - (alpha * beta)

    # 处理 reduction
    if reduction == 'mean':
        return loss.mean()
    elif reduction == 'sum':
        return loss.sum()
    else:
        return loss



class BalancedL1Loss(nn.Module):
    """Balanced L1 Loss.

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)

    Args:
        alpha (float): The denominator ``alpha`` in the balanced L1 loss.
            Defaults to 0.5.
        gamma (float): The ``gamma`` in the balanced L1 loss. Defaults to 1.5.
        beta (float, optional): The loss is a piecewise function of prediction
            and target. ``beta`` serves as a threshold for the difference
            between the prediction and target. Defaults to 1.0.
        reduction (str, optional): The method that reduces the loss to a
            scalar. Options are "none", "mean" and "sum".
        loss_weight (float, optional): The weight of the loss. Defaults to 1.0
    """

    def __init__(self,
                 alpha=0.5,
                 gamma=1.5,
                 beta=1.0,
                 reduction='mean',
                 loss_weight=1.0):
        super(BalancedL1Loss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.beta = beta
        self.reduction = reduction
        self.loss_weight = loss_weight

    def forward(self,
                pred,
                target,
                weight=None,
                avg_factor=None,
                reduction_override=None,
                **kwargs):
        """Forward function of loss.

        Args:
            pred (torch.Tensor): The prediction with shape (N, 4).
            target (torch.Tensor): The learning target of the prediction with
                shape (N, 4).
            weight (torch.Tensor, optional): Sample-wise loss weight with
                shape (N, ).
            avg_factor (int, optional): Average factor that is used to average
                the loss. Defaults to None.
            reduction_override (str, optional): The reduction method used to
                override the original reduction method of the loss.
                Options are "none", "mean" and "sum".

        Returns:
            torch.Tensor: The calculated loss
        """
        assert reduction_override in (None, 'none', 'mean', 'sum')
        reduction = (
            reduction_override if reduction_override else self.reduction)
        loss_bbox = self.loss_weight * balanced_l1_loss(
            pred,
            target,
            alpha=self.alpha,
            gamma=self.gamma,
            beta=self.beta,
            reduction=reduction,
            # avg_factor=avg_factor,
            **kwargs)
        return loss_bbox

    # --------------BalancedL1Loss  end-----------------#


from pytorch3d.ops.knn import knn_points
def translate_loss(gs_can, d_xyz, d_rotation, d_scaling, K=5):
    xyz_can = gs_can.get_xyz
    xyz_obs = xyz_can + d_xyz

    cov_can = gs_can.get_covariance()
    cov_obs = gs_can.get_covariance_obs(d_rotation, d_scaling)

    _, nn_ix, _ = knn_points(xyz_can.unsqueeze(0), xyz_can.unsqueeze(0), K=K, return_sorted=True)
    nn_ix = nn_ix.squeeze(0)

    dis_xyz_can = torch.cdist(xyz_can.unsqueeze(1), xyz_can[nn_ix])[:, 0, 1:]
    dis_xyz_obs = torch.cdist(xyz_obs.unsqueeze(1), xyz_obs[nn_ix])[:, 0, 1:]
    loss_pos = F.l1_loss(dis_xyz_can, dis_xyz_obs)

    dis_cov_can = torch.cdist(cov_can.unsqueeze(1), cov_can[nn_ix])[:, 0, 1:]
    dis_cov_obs = torch.cdist(cov_obs.unsqueeze(1), cov_obs[nn_ix])[:, 0, 1:]
    loss_cov = F.l1_loss(dis_cov_can, dis_cov_obs)

    return loss_pos, loss_cov
