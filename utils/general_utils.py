#
# Copyright (C) 2025
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
#

import torch
import sys
from datetime import datetime
import numpy as np
import random
from PIL import Image

def inverse_sigmoid(x):
    return torch.log(x / (1 - x))


def PILtoTorch(pil_image, resolution):
    if np.asarray(pil_image).shape[-1] == 4:
        # Process rgb and alpha respectively to avoid mask rgb with alpha
        rgb = Image.fromarray(np.asarray(pil_image)[..., :3])
        a = Image.fromarray(np.asarray(pil_image)[..., 3])
        rgb, a = np.asarray(rgb.resize(resolution)), np.asarray(a.resize(resolution))
        resized_image = torch.from_numpy(np.concatenate([rgb, a[..., None]], axis=-1)) / 255.0
    else:
        resized_image_PIL = pil_image.resize(resolution)
        resized_image = torch.from_numpy(np.array(resized_image_PIL)) / 255.0
    if len(resized_image.shape) == 3:
        return resized_image.permute(2, 0, 1)
    else:
        return resized_image.unsqueeze(dim=-1).permute(2, 0, 1)


def ArrayToTorch(array, resolution):
    # resized_image = np.resize(array, resolution)
    resized_image_torch = torch.from_numpy(array)

    if len(resized_image_torch.shape) == 3:
        return resized_image_torch.permute(2, 0, 1)
    else:
        return resized_image_torch.unsqueeze(dim=-1).permute(2, 0, 1)


def get_expon_lr_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = np.exp(np.log(lr_init) * (1 - t) + np.log(lr_final) * t)
        return delay_rate * log_lerp

    return helper


def get_linear_noise_func(
        lr_init, lr_final, lr_delay_steps=0, lr_delay_mult=1.0, max_steps=1000000
):
    """
    Copied from Plenoxels

    Continuous learning rate decay function. Adapted from JaxNeRF
    The returned rate is lr_init when step=0 and lr_final when step=max_steps, and
    is log-linearly interpolated elsewhere (equivalent to exponential decay).
    If lr_delay_steps>0 then the learning rate will be scaled by some smooth
    function of lr_delay_mult, such that the initial learning rate is
    lr_init*lr_delay_mult at the beginning of optimization but will be eased back
    to the normal learning rate when steps>lr_delay_steps.
    :param conf: config subtree 'lr' or similar
    :param max_steps: int, the number of steps during optimization.
    :return HoF which takes step as input
    """

    def helper(step):
        if step < 0 or (lr_init == 0.0 and lr_final == 0.0):
            # Disable this parameter
            return 0.0
        if lr_delay_steps > 0:
            # A kind of reverse cosine decay.
            delay_rate = lr_delay_mult + (1 - lr_delay_mult) * np.sin(
                0.5 * np.pi * np.clip(step / lr_delay_steps, 0, 1)
            )
        else:
            delay_rate = 1.0
        t = np.clip(step / max_steps, 0, 1)
        log_lerp = lr_init * (1 - t) + lr_final * t
        return delay_rate * log_lerp

    return helper

def get_staged_lr_func(lr_init, lr_final, warmup_steps=10000, max_steps=1000000):
    # 定义分阶段学习率策略函数
    decay_start = warmup_steps * 0.1
    lr_mid = lr_init/5
    def staged_lr_schedule(step):#, lr_init, lr_mid, lr_final, warmup_steps, decay_start, max_steps):
        if step < warmup_steps:
            # 阶段 1: 使用较小的初始学习率
            return lr_init/5
        elif step < decay_start:
            # 阶段 2: 增加学习率到中间值 lr_mid
            return lr_init
        else:
            # 阶段 3: 从 lr_mid 逐渐衰减到 lr_final
            t = np.clip((step - decay_start) / (max_steps - decay_start), 0, 1)
            return lr_mid * ((lr_final / lr_mid) ** t)  # 指数衰减
    return staged_lr_schedule


# # 定义分阶段学习率策略函数
def get_staged_trifunc_lr_func3(lr_init, lr_final, warmup_steps=25000, decay_start=14000, max_steps=1000000):
    lr_mid = lr_init  # 中间阶段的学习率
    # decay_start = warmup_steps  # 阶段 3 的起始步数
    decay_steps =  warmup_steps -decay_start
    def cos_sin_lr_schedule(step):
        if step < decay_start:
            return lr_init / 10
        elif step < warmup_steps:
            # 阶段 1: 使用 sin 函数从 lr_init/5 平滑上升到 lr_mid
            return lr_init / 10 + (lr_mid - lr_init / 10) * np.sin(0.5 * np.pi * (step-decay_start) / (warmup_steps-decay_start))
        # elif step < decay_start + decay_steps:
        #     # 阶段 2: 保持在 lr_mid
        #     return lr_mid
        else:
            # 阶段 3:从 lr_mid 平滑衰减到 lr_final
            return lr_final + (lr_mid - lr_final) * 0.5 * (1 + np.cos(np.pi * (step - decay_start - decay_steps) / (max_steps - decay_start - decay_steps)))
    return cos_sin_lr_schedule

# # 定义分阶段学习率策略函数
def get_staged_trifunc_lr_func4(lr_init, lr_final, warmup_steps=25000, decay_steps=0, max_steps=1000000):
    lr_mid = lr_init  # 中间阶段的学习率
    decay_start = warmup_steps  # 阶段 3 的起始步数
    # decay_steps =  warmup_steps -decay_start
    def cos_sin_lr_schedule(step):
        # if step < decay_start:
        #     return lr_init / 10
        if step < warmup_steps:
            # 阶段 1: 使用 sin 函数从 lr_init/5 平滑上升到 lr_mid
            return lr_init / 10 + (lr_mid - lr_init / 10) * np.sin(0.5 * np.pi * (step) / (warmup_steps))
        elif step < decay_start+decay_steps:
            # 阶段 2: 保持在 lr_mid
            return lr_mid
        else:
            # 阶段 3: 从 lr_mid 平滑衰减到 lr_final
            return lr_final + (lr_mid - lr_final) * 0.5 * (1 + np.cos(np.pi * (step - decay_start - decay_steps) / (max_steps - decay_start - decay_steps)))
    return cos_sin_lr_schedule

# # 定义学习率策略函数
def get_staged_trifunc_lr_func5(lr_init, lr_final, warmup_steps=25000, decay_steps=0, max_steps=1000000):
    lr_mid = lr_init  # 学习率
    decay_start = 5000  #起始步数
    # decay_steps =  warmup_steps -decay_start
    def cos_sin_lr_schedule(step):
        if step < decay_start:
            return lr_init / 10
        elif step < warmup_steps:
            # 阶段 1: 
            # return lr_init / 10 + (lr_mid - lr_init / 10) * np.sin(0.5 * np.pi * (step) / (warmup_steps))
           return lr_init / 10 + (lr_mid - lr_init / 10) * torch.sin(0.5 * torch.pi * step / warmup_steps)

        elif step < warmup_steps+decay_steps:
            # 阶段 2: 
            return lr_mid
        else:
            # 阶段 3: 
            return lr_final + (lr_mid - lr_final) * 0.5 * (1 + torch.cos(torch.pi * (step - warmup_steps) / (max_steps - warmup_steps)))
            # return lr_final + (lr_mid - lr_final) * 0.5 * (1 + np.cos(np.pi * (step - warmup_steps) / (max_steps - warmup_steps)))
    return cos_sin_lr_schedule

def get_staged_trifunc_lr_func6(lr_init, lr_final, warmup_steps=25000, decay_steps=0, max_steps=1000000):
    """
    先减小再增大，在减小
    """
    lr_delay_mult = 0.01
    lr_delay_steps = 0
    lr_mid = lr_init  # 中间阶段的学习率
    #decay_start = 5000  # 阶段 3 的起始步数
    # decay_steps =  warmup_steps -decay_start
    def cos_sin_lr_schedule(step):
        # if step < decay_start:
        #     return lr_init / 10
        if step < warmup_steps:
            # 阶段 1: 使用 cos 函数从 lr_init/10 平滑衰减到 lr_init /20
            # return lr_init /20+ (lr_init/10 - lr_init/20) * 0.5 * (1 + np.cos(np.pi * step / warmup_steps))
            delay_rate = 1.0
            t = np.clip(step / max_steps, 0, 1)
            log_lerp = np.exp(np.log(lr_init/10) * (1 - t) + np.log(lr_final) * t)
            return delay_rate * log_lerp

        elif step < warmup_steps+decay_steps:
            # 阶段 2: 使用 sin 函数从 lr_init/20 平滑上升到 lr_mid
            return lr_init / 20 + (lr_mid - lr_init / 20) * np.sin(0.5 * np.pi * (step-warmup_steps) / (decay_steps))
        else:
            return lr_final + (lr_mid - lr_final) * 0.5 * (1 + np.cos(np.pi * (step - warmup_steps-decay_steps) / (max_steps - warmup_steps-decay_steps)))
    return cos_sin_lr_schedule

def get_staged_trifunc_lr_func(lr_init, lr_final, warmup_steps=25000, decay_start=14000, max_steps=1000000):
    lr_mid = lr_init  # 中间阶段的学习率
    decay_steps =  warmup_steps -decay_start
    def cos_sin_lr_schedule(step):
        if step < decay_start:
            return lr_init/10

        elif step < warmup_steps:
            return lr_init / 10 + (lr_mid - lr_init / 10) * np.sin(0.5 * np.pi * (step-decay_start) / (warmup_steps-decay_start))
        else:
            return lr_final + (lr_mid - lr_final) * 0.5 * (1 + np.cos(np.pi * (step - decay_start - decay_steps) / (max_steps - decay_start - decay_steps)))
    return cos_sin_lr_schedule


def strip_lowerdiag(L):
    uncertainty = torch.zeros((L.shape[0], 6), dtype=torch.float, device="cuda")

    uncertainty[:, 0] = L[:, 0, 0]
    uncertainty[:, 1] = L[:, 0, 1]
    uncertainty[:, 2] = L[:, 0, 2]
    uncertainty[:, 3] = L[:, 1, 1]
    uncertainty[:, 4] = L[:, 1, 2]
    uncertainty[:, 5] = L[:, 2, 2]
    return uncertainty


def strip_symmetric(sym):
    return strip_lowerdiag(sym)


def build_rotation(r):
    norm = torch.sqrt(r[:, 0] * r[:, 0] + r[:, 1] * r[:, 1] + r[:, 2] * r[:, 2] + r[:, 3] * r[:, 3])

    q = r / norm[:, None]

    R = torch.zeros((q.size(0), 3, 3), device='cuda')

    r = q[:, 0]
    x = q[:, 1]
    y = q[:, 2]
    z = q[:, 3]

    R[:, 0, 0] = 1 - 2 * (y * y + z * z)
    R[:, 0, 1] = 2 * (x * y - r * z)
    R[:, 0, 2] = 2 * (x * z + r * y)
    R[:, 1, 0] = 2 * (x * y + r * z)
    R[:, 1, 1] = 1 - 2 * (x * x + z * z)
    R[:, 1, 2] = 2 * (y * z - r * x)
    R[:, 2, 0] = 2 * (x * z - r * y)
    R[:, 2, 1] = 2 * (y * z + r * x)
    R[:, 2, 2] = 1 - 2 * (x * x + y * y)
    return R


def build_scaling_rotation(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = s[:, 0]
    L[:, 1, 1] = s[:, 1]
    L[:, 2, 2] = s[:, 2]

    L = R @ L
    return L


def build_scaling_rotation_inverse(s, r):
    L = torch.zeros((s.shape[0], 3, 3), dtype=torch.float, device="cuda")
    R = build_rotation(r)

    L[:, 0, 0] = 1 / s[:, 0]
    L[:, 1, 1] = 1 / s[:, 1]
    L[:, 2, 2] = 1 / s[:, 2]

    L = R.permute(0, 2, 1) @ L
    return L


def safe_state(silent):
    old_f = sys.stdout

    class F:
        def __init__(self, silent):
            self.silent = silent

        def write(self, x):
            if not self.silent:
                if x.endswith("\n"):
                    old_f.write(x.replace("\n", " [{}]\n".format(str(datetime.now().strftime("%d/%m %H:%M:%S")))))
                else:
                    old_f.write(x)

        def flush(self):
            old_f.flush()

    sys.stdout = F(silent)

    random.seed(0)
    np.random.seed(0)
    torch.manual_seed(0)
    torch.cuda.set_device(torch.device("cuda:0"))
