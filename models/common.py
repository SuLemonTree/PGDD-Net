# YOLOv5 🚀 by Ultralytics, GPL-3.0 license
"""
Common modules
"""
import json
import math
import os
import platform
import warnings
from collections import OrderedDict, namedtuple
from copy import copy
from pathlib import Path
from torch.nn.init import trunc_normal_

import cv2
import numpy as np
import pandas as pd
import requests
import torch
import torch.nn as nn
import yaml
from PIL import Image
from mobile_sam import sam_model_registry
from torch import Tensor
from torch.cuda import amp
from functools import partial


from utils.datasets import exif_transpose, letterbox
from utils.general import (LOGGER, check_requirements, check_suffix, check_version, colorstr, increment_path,
                           make_divisible, non_max_suppression, scale_coords, xywh2xyxy, xyxy2xywh)
from utils.plots import Annotator, colors, save_one_box
from utils.torch_utils import copy_attr, time_sync

# from models.module import *
def autopad(k, p=None):  # kernel, padding
    # Pad to 'same'
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p
class Conv_SAM(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(256, 256, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(256)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class Conv(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
class Buffer_Pool(nn.Module):
    # Standard convolution
    def __init__(self,c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
    def forward(self, x):
        x =x
        # print(x.shape)
        return x
class Pretreatment(nn.Module):
    # Standard convolution
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = nn.Conv2d(3, c2, k, s, autopad(k, p), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act is True else (act if isinstance(act, nn.Module) else nn.Identity())

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))
#分组SPPCSPC 分组后参数量和计算量与原本差距不大，不知道效果怎么样
class SPPCSPC_group(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC_group, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, g=4)
        self.cv2 = Conv(c1, c_, 1, 1, g=4)
        self.cv3 = Conv(c_, c_, 3, 1, g=4)
        self.cv4 = Conv(c_, c_, 1, 1, g=4)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1, g=4)
        self.cv6 = Conv(c_, c_, 3, 1, g=4)
        self.cv7 = Conv(2 * c_, c2, 1, 1, g=4)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class DWConv(Conv):
    # Depth-wise convolution class
    def __init__(self, c1, c2, k=1, s=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__(c1, c2, k, s, g=math.gcd(c1, c2), act=act)


class TransformerLayer(nn.Module):
    # Transformer layer https://arxiv.org/abs/2010.11929 (LayerNorm layers removed for better performance)
    def __init__(self, c, num_heads):
        super().__init__()
        self.q = nn.Linear(c, c, bias=False)
        self.k = nn.Linear(c, c, bias=False)
        self.v = nn.Linear(c, c, bias=False)
        self.ma = nn.MultiheadAttention(embed_dim=c, num_heads=num_heads)
        self.fc1 = nn.Linear(c, c, bias=False)
        self.fc2 = nn.Linear(c, c, bias=False)

    def forward(self, x):
        x = self.ma(self.q(x), self.k(x), self.v(x))[0] + x
        x = self.fc2(self.fc1(x)) + x
        return x

class TransformerBlock(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        if self.conv is not None:
            x = self.conv(x)
        b, _, w, h = x.shape
        p = x.flatten(2).permute(2, 0, 1)
        return self.tr(p + self.linear(p)).permute(1, 2, 0).reshape(b, self.c2, w, h)

# class TransformerBlock(nn.Module):
#     # Vision Transformer https://arxiv.org/abs/2010.11929
#     def __init__(self, c1, c2, num_heads, num_layers):
#         super().__init__()
#         self.conv = None
#         if c1 != c2:
#             self.conv = Conv(c1, c2)
#         self.linear = nn.Linear(c2, c2)  # learnable position embedding
#         self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
#         self.c2 = c2
#
#         ##对张量进行抽样，达到长宽减半的效果
#
#     def chouyang(x0,x):
#
#         batch, kin, width, hight = x.size()
#         t = torch.randn(batch, kin, width // 2, hight // 2)
#         p = torch.randn(batch, kin, width // 2, hight // 2)
#         s = torch.randn(batch, kin, width // 2, hight // 2)
#         u = torch.randn(batch, kin, width // 2, hight // 2)
#         t,p,s,u = t.cuda(),p.cuda(),s.cuda(),u.cuda()
#         for b in range(int(batch)):
#             for k in range(int(kin)):
#                 for w in range(int(width)):
#                     for h in range(int(hight)):
#                         if (w % 2) == 0:
#                             if (h % 2) == 0:
#                                 # x[b,k,w,h] = 0
#                                 t[b, k, int(w // 2), int(h // 2)] = x[b, k, w, h]
#                                 p[b, k, int(w // 2), int(h // 2)] = x[b, k, w - 1, h - 1]
#                                 s[b, k, int(w // 2), int(h // 2)] = x[b, k, w, h - 1]
#                                 u[b, k, int(w // 2), int(h // 2)] = x[b, k, w - 1, h]
#
#                         else:
#                             if (h % 2) != 0:
#                                 # x[b,k,w,h] = 0
#                                 t[b, k, int(w // 2), int(h // 2)] = x[b, k, w, h]
#                                 p[b, k, int(w // 2), int(h // 2)] = x[b, k, w - 1, h - 1]
#                                 s[b, k, int(w // 2), int(h // 2)] = x[b, k, w, h - 1]
#                                 u[b, k, int(w // 2), int(h // 2)] = x[b, k, w - 1, h]
#         #print(p)
#         return p, u, t, s
#
#     ###将四个小方阵组合为原来的方阵
#     def zuhe(x,x1, x2, x3, x4):
#         batch, kin, width, hight = x1.size()
#         x10 = torch.randn(batch, kin, width * 2, hight * 2)
#         x10 = x10.cuda()
#         for b in range(int(batch)):
#             for k in range(int(kin)):
#                 for w in range(int(width)):
#                     for h in range(int(hight)):
#                         t = w * 2
#                         p = h * 2
#                         x10[b, k, t, p] = x1[b, k, int(w), int(h)]
#                         x10[b, k, t, p + 1] = x2[b, k, int(w), int(h)]
#                         x10[b, k, t + 1, p] = x4[b, k, int(w), int(h)]
#                         x10[b, k, t + 1, p + 1] = x3[b, k, int(w), int(h)]
#
#         return x10
#
#     def forward(self, x):
#
#         x1,x2,x3,x4 = self.chouyang(x)
#
#         x = self.zuhe(x1,x2,x3,x4)

#         if self.conv is not None:
#             x = self.conv(x)
#         b, _, w, h = x.shape
#         p1 = x.flatten(2).permute(2, 0, 1)
#         # return x
#         return self.tr(p1 + self.linear(p1)).permute(1, 2, 0).reshape(b, self.c2, w, h).cuda()
class TransformerBlock2(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2

    def forward(self, x):
        identity = x
        x1 = x[..., ::2, ::2]
        if self.conv is not None:
            x1 = self.conv(x1)
        b1, _, w1, h1 = x1.shape
        p1 = x1.flatten(2).permute(2, 0, 1)
        x1 = self.tr(p1 + self.linear(p1))
        x1 = x1.permute(1, 2, 0)
        x1 = x1.reshape(b1, self.c2, w1, h1)
        x[..., ::2, ::2] = x1
        # print(x)
        x2 = x[..., 1::2, ::2]
        if self.conv is not None:
            x2 = self.conv(x2)
        b1, _, w1, h1 = x2.shape
        p1 = x2.flatten(2).permute(2, 0, 1)
        x2 = self.tr(p1 + self.linear(p1)).permute(1, 2, 0).reshape(b1, self.c2, w1, h1)
        x[..., 1::2, ::2] = x2

        x3 = x[..., ::2, 1::2]
        if self.conv is not None:
            x3 = self.conv(x3)
        b1, _, w1, h1 = x3.shape
        p1 = x3.flatten(2).permute(2, 0, 1)
        x3 = self.tr(p1 + self.linear(p1)).permute(1, 2, 0).reshape(b1, self.c2, w1, h1)
        x[..., ::2, 1::2] = x3
        x4 = x[..., 1::2, 1::2]
        if self.conv is not None:
            x4 = self.conv(x4)
        b1, _, w1, h1 = x4.shape
        #dim从第几个维度往后展开展成一个
        #4 1 1
        p1 = x4.flatten(2).permute(2, 0, 1)
        x4 = self.tr(p1 + self.linear(p1)).permute(1, 2, 0).reshape(b1, self.c2, w1, h1)
        x[..., 1::2, 1::2] = x4
        x = x + identity
        return x

class Channel_Att(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att, self).__init__()
        self.channels = channels

        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)

    def forward(self, x):
        residual = x

        x = self.bn2(x)
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()

        x = torch.sigmoid(x) * residual  #

        return x


class NAMAttention(nn.Module):
    def __init__(self, channels, out_channels=None, no_spatial=True):
        super(NAMAttention, self).__init__()
        self.Channel_Att = Channel_Att(channels)

    def forward(self, x):
        x_out1 = self.Channel_Att(x)

        return x_out1
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CA(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CA, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        #C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out
class TransformerBlock3(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2
        self.nam = NAMAttention(c2,c2)
        self.ca = CA(c2,c2)
        self.convjiang = Conv(2 * c2, c2)
    def forward(self, x):
        identity = x
        # y = self.nam(x)
        b, c, h, w = x.size()
        # t1 = torch.rand(b, c, h, w)
        # t1[..., :1:1, ::1] = x[..., :1:1, ::1]  # 第一行
        # # print(t1)
        #
        # t2 = torch.rand(b, c, h, w)
        # t2[..., ::1, :1:1] = x[..., ::1, :1:1]  # 第一列
        # # print(t2)
        #
        # t3 = torch.rand(b, c, h, w)
        # t3[..., 1::1, 1::1] = x[..., 1::1, 1::1]  # 剩下的
        # # print(t3)
        #
        # # t4 = torch.rand(b, c, h, w)
        # #t4[..., 1:h - 1:1, 1:w - 1:1] = x[..., 1:h - 1:1, 1:w - 1:1]  # 中间的
        # # print(t4)
        #
        # t11 = torch.rand(b, c, h, w)
        # t11[..., h - 1::1, ::1] = x[..., h - 1::1, ::1]  # 最后一行
        # # print(t11)
        #
        # t22 = torch.rand(b, c, h, w)
        # t22[..., ::1, w - 1::1] = x[..., ::1, w - 1::1]  # 最右边一列
        # # print(t22)
        #
        # t33 = torch.rand(b, c, h, w)
        # t33[..., :h - 1:1, :w - 1:1] = x[..., :h - 1:1, :w - 1:1]  # 还原剩下的
        # # print(t33)
        # x[..., h - 1::1, ::1] = t1[..., :1:1, ::1]
        # # print(x)
        #
        # x[..., ::1, w - 1::1] = t2[..., ::1, :1:1]
        # # print(x)
        #
        # x[..., :h - 1:1, :w - 1:1] = t3[..., 1::1, 1::1]
        # print(x)
        # b, c, h, w = x.size()
        # p1 = torch.rand(b, c, h, w)
        x = torch.roll(x, int(h-1), 2)  # 变1
        # print(x)
        x = torch.roll(x, int(w-1), 3)  # 变2
        p1 = x[..., :int(h / 2):1, :int(w / 2):1]  # 第一块
        # print(p1.shape)
        if self.conv is not None:
            p1 = self.conv(p1)
        # b1, _, w1, h1 = p1.shape
        pp1 = p1.flatten(2).permute(2, 0, 1)
        # print(pp1.shape)
        x2 = self.tr(pp1 + self.linear(pp1)).permute(1, 2, 0).reshape(b, self.c2, int(h / 2), int(w / 2))
        # pp1 = self.tr(pp1 + self.linear(pp1))
        # print(pp1.shape)
        # pp1 = pp1.permute(1, 2, 0)
        # print(pp1.shape)
        # pp1 = pp1.reshape(b, self.c2, h, w) #1 1 2 2
        # print(pp1.shape)
        # x1= self.Channel_Att1(x)
        x[..., :int(h / 2):1, :int(w / 2):1] = x2
        # print(x)
        # p2 = torch.rand(b, c, h, w)
        p2 = x[..., :int(h / 2):1, int(w / 2):w:1]  # 第一块
        # print(p2)
        if self.conv is not None:
            p2 = self.conv(p2)
        # b1, _, w1, h1 = p1.shape
        pp2 = p2.flatten(2).permute(2, 0, 1)
        pp2 = self.tr(pp2 + self.linear(pp2))
        pp2 = pp2.permute(1, 2, 0)
        pp2 = pp2.reshape(b, self.c2,int(h / 2), int(w / 2))  # 1 1 2 2
        # x1= self.Channel_Att1(x)
        x[..., :int(h / 2):1, int(w / 2):w:1] = pp2
        # print(x)
        # p3 = torch.rand(b, c, h, w)
        p3 = x[..., int(h / 2):h:1, :int(w / 2):1]  # 第一块
        # print(p3)
        if self.conv is not None:
            p3 = self.conv(p3)
        # b1, _, w1, h1 = p1.shape
        pp3 = p3.flatten(2).permute(2, 0, 1)
        pp3 = self.tr(pp3 + self.linear(pp3))
        pp3 = pp3.permute(1, 2, 0)
        pp3 = pp3.reshape(b, self.c2, int(h / 2), int(w / 2))  # 1 1 2 2
        # x1= self.Channel_Att1(x)
        x[..., int(h / 2):h:1, :int(w / 2):1] = pp3
        # print(x)
        # p4 = torch.rand(b, c, h, w)
        p4 = x[..., int(h / 2):h:1, int(w / 2):w:1]  # 第一块
        # print(p4)
        if self.conv is not None:
            p4 = self.conv(p4)
        # b1, _, w1, h1 = p1.shape
        pp4 = p4.flatten(2).permute(2, 0, 1)
        pp4 = self.tr(pp4 + self.linear(pp4))
        pp4 = pp4.permute(1, 2, 0)
        pp4 = pp4.reshape(b, self.c2, int(h / 2),int(w / 2))  # 1 1 2 2
        # x1= self.Channel_Att1(x)
        x[..., int(h / 2):h:1, int(w / 2):w:1] = pp4
        # print(x)
        x = torch.roll(x,1,2) #还原1
        # print(x)
        x = torch.roll(x,1,3) #还原2
        # y = self.ca(x)
        # x = torch.cat((x, y), dim=1)
        # x = self.convjiang(x)
        x = x + identity
        return x
class TransformerBlock5(nn.Module):
    # Vision Transformer https://arxiv.org/abs/2010.11929
    def __init__(self, c1, c2, num_heads, num_layers):
        super().__init__()
        self.conv = None
        if c1 != c2:
            self.conv = Conv(c1, c2)
        self.linear = nn.Linear(c2, c2)  # learnable position embedding
        self.tr = nn.Sequential(*(TransformerLayer(c2, num_heads) for _ in range(num_layers)))
        self.c2 = c2
        self.nam = NAMAttention(c2,c2)
        self.ca = CA(c2,c2)
        self.convjiang = Conv(2 * c2, c2)

    def forward(self, x):
        identity = x
        y = self.ca(x)
        b, c, h, w = x.size()
        # t1 = torch.rand(b, c, h, w)
        # t1[..., :1:1, ::1] = x[..., :1:1, ::1]  # 第一行
        # # # print(t1)
        # #
        # t2 = torch.rand(b, c, h, w)
        # t2[..., ::1, :1:1] = x[..., ::1, :1:1]  # 第一列
        # # # print(t2)
        # #
        # t3 = torch.rand(b, c, h, w)
        # t3[..., 1::1, 1::1] = x[..., 1::1, 1::1]  # 剩下的
        # print(t3)
        # t4 = torch.rand(b, c, h, w)
        #t4[..., 1:h - 1:1, 1:w - 1:1] = x[..., 1:h - 1:1, 1:w - 1:1]  # 中间的
        # print(t4)
        #
        # t11 = torch.rand(b, c, h, w)
        # t11[..., h - 1::1, ::1] = x[..., h - 1::1, ::1]  # 最后一行
        # # print(t11)
        #
        # t22 = torch.rand(b, c, h, w)
        # t22[..., ::1, w - 1::1] = x[..., ::1, w - 1::1]  # 最右边一列
        # # print(t22)
        #
        # t33 = torch.rand(b, c, h, w)
        # t33[..., :h - 1:1, :w - 1:1] = x[..., :h - 1:1, :w - 1:1]  # 还原剩下的
        # # print(t33)
        # x[..., h - 1::1, ::1] = t1[..., :1:1, ::1]
        # # # print(x)
        # #
        # x[..., ::1, w - 1::1] = t2[..., ::1, :1:1]
        # # # print(x)
        # #
        # x[..., :h - 1:1, :w - 1:1] = t3[..., 1::1, 1::1]
        # print(x)
        # b, c, h, w = x.size()
        # p1 = torch.rand(b, c, h, w)
        x = torch.roll(x, int(h-1), 2)  # 变1
        # print(x)
        x = torch.roll(x, int(w-1), 3)  # 变2
        t = x.permute(0, 1, 3, 2)
        # print(x)
        t = torch.roll(t, int(h-1), 2)  # 变1
        # print(y)
        t = torch.roll(t,int(w-1), 3)  # 变2
        q1 = t[..., :int(h / 2):1, :int(w / 2):1]  # 第一块
        q2 = t[..., :int(h / 2):1, int(w / 2):w:1]  # 第一块
        q3 = t[..., int(h / 2):h:1, :int(w / 2):1]  # 第一块
        q4 = t[..., int(h / 2):h:1, int(w / 2):w:1]  # 第一块
        # print(x)
        p1 = x[..., :int(h / 2):1, :int(w / 2):1]  # 第一块
        p1 = p1 + q1
        # print(p1.shape)
        if self.conv is not None:
            p1 = self.conv(p1)
        # b1, _, w1, h1 = p1.shape
        pp1 = p1.flatten(2)
        pp1= pp1.permute(2, 0, 1)
        # print(pp1.shape)
        x2 = self.tr(pp1 + self.linear(pp1)).permute(1, 2, 0).reshape(b, self.c2, int(h / 2), int(w / 2))
        # pp1 = self.tr(pp1 + self.linear(pp1))
        # print(pp1.shape)
        # pp1 = pp1.permute(1, 2, 0)
        # print(pp1.shape)
        # pp1 = pp1.reshape(b, self.c2, h, w) #1 1 2 2
        # print(pp1.shape)
        # x1= self.Channel_Att1(x)
        x[..., :int(h / 2):1, :int(w / 2):1] = x2
        # print(x)

        # p2 = torch.rand(b, c, h, w)

        p2 = x[..., :int(h / 2):1, int(w / 2):w:1]  # 第一块
        p2 = p2 + q2
        # print(p2)
        if self.conv is not None:
            p2 = self.conv(p2)
        # b1, _, w1, h1 = p1.shape
        pp2 = p2.flatten(2).permute(2, 0, 1)
        pp2 = self.tr(pp2 + self.linear(pp2))
        pp2 = pp2.permute(1, 2, 0)
        pp2 = pp2.reshape(b, self.c2,int(h / 2), int(w / 2))  # 1 1 2 2
        # x1= self.Channel_Att1(x)
        x[..., :int(h / 2):1, int(w / 2):w:1] = pp2
        # print(x)

        # p3 = torch.rand(b, c, h, w)

        p3 = x[..., int(h / 2):h:1, :int(w / 2):1]  # 第一块
        p3 = p3 + q3
        # print(p3)
        if self.conv is not None:
            p3 = self.conv(p3)
        # b1, _, w1, h1 = p1.shape
        pp3 = p3.flatten(2).permute(2, 0, 1)
        pp3 = self.tr(pp3 + self.linear(pp3))
        pp3 = pp3.permute(1, 2, 0)
        pp3 = pp3.reshape(b, self.c2, int(h / 2), int(w / 2))  # 1 1 2 2
        # x1= self.Channel_Att1(x)
        x[..., int(h / 2):h:1, :int(w / 2):1] = pp3
        # print(x)

        # p4 = torch.rand(b, c, h, w)
        p4 = x[..., int(h / 2):h:1, int(w / 2):w:1]  # 第一块
        p4 = p4 + q4
        # print(p4)
        if self.conv is not None:
            p4 = self.conv(p4)
        # b1, _, w1, h1 = p1.shape
        pp4 = p4.flatten(2).permute(2, 0, 1)
        pp4 = self.tr(pp4 + self.linear(pp4))
        pp4 = pp4.permute(1, 2, 0)
        pp4 = pp4.reshape(b, self.c2, int(h / 2),int(w / 2))  # 1 1 2 2
        # x1= self.Channel_Att1(x)
        x[..., int(h / 2):h:1, int(w / 2):w:1] = pp4
        x = torch.roll(x,1,2) #还原1
        # print(x)
        x = torch.roll(x,1,3) #还原2
        # x = self.ca(x)
        # print(x)
        # x = torch.cat((x, y), dim=1)
        # x= self.convjiang(x)
        x = x + identity
        # x = x*y
        return x

class DepthWiseConv(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DepthWiseConv, self).__init__()
        # 逐通道卷积
        self.depth_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=in_channel,
            kernel_size=3,
            stride=1,
            padding=1,
            groups=in_channel
        )
        # groups是一个数，当groups=in_channel时,表示做逐通道卷积
        # 逐点卷积
        self.point_conv = nn.Conv2d(
            in_channels=in_channel,
            out_channels=out_channel,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
    def forward(self, input):
        out = self.depth_conv(input)
        out = self.point_conv(out)
        return out
class Bottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # self.cv3 = DepthWiseConv(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class Bottleneck_mamba(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        # self.cv3 = DepthWiseConv(c_, c2)
        self.add = shortcut and c1 == c2

    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))
class Bottleneckszz(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DepthWiseConv(c_, c2)
        self.cv3 = Conv(2*c_,c2)
        self.cv4 = Conv(int(c2/2), int(c2/2), 1, 1)
        # self.cv5 = nn.Conv2d(int(c2/2), int(c2/2), 3, 1,padding=2,dilation=2)
        # self.cv6 = nn.Conv2d(int(c2/2), int(c2/2), 3, 1,padding=3,dilation=3)
        self.bn = nn.BatchNorm2d(int(c2/2))
        self.act = nn.SiLU()
        self.add = shortcut and c1 == c2
        self.c2 = c2
    def forward(self, x):
        # return x + self.cv3(self.cv2(self.cv1(x))) if self.add else self.cv3(self.cv2(self.cv1(x)))
        ident= x
        x1 = x[:,0:int(self.c2/2),...]
        x11 = self.cv4(x1)
        # x11 = self.cv5(x1)
        # x11 = self.bn(x11)
        # x11 = self.act(x11)
        x21 = self.cv1(x)
        x22 = self.cv2(x21)
        x3 = x21[:,int(self.c2/2):self.c2,...]
        x31 = self.cv4(x3)
        # x31 = self.cv6(x3)
        # x31 = self.bn(x31)
        # x31 = self.act(x31)
        x1131 = torch.cat((x11,x31),dim=1)
        x4 = torch.cat((x1131,x22),dim=1)
        x5 = self.cv3(x4)
        x6 = self.cv1(x)
        return x5+ident+x6

class Bottleneckszzjinjie(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5,s=1,w = 1.0):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        cw = int(c2*w)
        self.cv11 = Conv(c_,c1, (1, 3), (1, s))
        self.cv12 = Conv(c1, cw, (3, 1), (s, 1), g=g)
        self.cv21 = Conv(c_,c1, (1, 5), (1, s))
        self.cv22 = Conv(c1, cw, (5, 1), (s, 1), g=g)
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = DepthWiseConv(c_, c2)
        self.bn = nn.BatchNorm2d(c2)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        # self.cv3 = Conv(c_,c2)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        # x1 = x[:,0:int(self.c2/2),...]
        # print(x1.shape)
        # x2 = x[:,int(self.c2/2):self.c2,...]
        # print(x2.shape)
        # x1 = self.cv11(x1)
        # x1 = self.cv12(x1)
        # print(x1.shape)
        # x2 = self.cv21(x2)
        # x2 = self.cv22(x2)
        # print(x2.shape)
        return x + self.act(self.bn(self.cv2(self.cv1(x)))) if self.add else self.act(self.bn(self.cv2(self.cv1(x))))

class BottleneckCSP(nn.Module):
    # CSP Bottleneck https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = nn.Conv2d(c1, c_, 1, 1, bias=False)
        self.cv3 = nn.Conv2d(c_, c_, 1, 1, bias=False)
        self.cv4 = Conv(2 * c_, c2, 1, 1)
        self.bn = nn.BatchNorm2d(2 * c_)  # applied to cat(cv2, cv3)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y1 = self.cv3(self.m(self.cv1(x)))
        y2 = self.cv2(x)
        return self.cv4(self.act(self.bn(torch.cat((y1, y2), dim=1))))

class CrossConv(nn.Module):
    # Cross Convolution Downsample
    def __init__(self, c1, c2, k=3, s=1, g=1, e=1.0, shortcut=False):
        # ch_in, ch_out, kernel, stride, groups, expansion, shortcut
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, (1, k), (1, s))
        self.cv2 = Conv(c_, c2, (k, 1), (s, 1), g=g)
        self.add = shortcut and c1 == c2
    def forward(self, x):
        return x + self.cv2(self.cv1(x)) if self.add else self.cv2(self.cv1(x))

class C3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        print(x.shape)
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
class C3MB33(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(128 * e)  # hidden channels
        self.cv1 = Conv(128, c_, 1, 1)
        self.cv2 = Conv(128, c_, 1, 1)
        self.cv3 = Conv(2 * c_, 256, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        print(x.shape)
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
class MRF_Mamba44(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(512 * e)  # hidden channels
        self.cv1 = Conv(512, c_, 1, 1)
        self.cv2 = Conv(512, c_, 1, 1)
        self.cv3 = Conv(2 * c_, 512, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):

        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
class C3cuda(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1).to("cuda:0")
        self.cv2 = Conv(c1, c_, 1, 1).to("cuda:0")
        self.cv3 = Conv(2 * c_, c2, 1).to("cuda:0")  # act=FReLU(c2)
        # self.cv1 = Conv(c1, c1, 1, 1).to("cuda:0")
        # self.cv2 = Conv(c1, c1, 1, 1).to("cuda:0")
        # self.cv3 = Conv(2 * c1, c2, 1).to("cuda:0")  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class D1(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = DepthWiseConv(c1, c_)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneckszz(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class S1(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
class S2(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,s=1,w = 1.0):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        cw = int(c2 * w)
        self.c2 = c2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.cv11 = Conv(c_,c_, (1, 3), (1, s))
        self.cv12 = Conv(c_, c_, (3, 1), (s, 1), g=g)
        self.cv21 = Conv(c_,c_, (1, 5), (1, s))
        self.cv22 = Conv(c_, c_, (5, 1), (s, 1), g=g)
        self.conv1 = Conv(c1,c_,1,1)
        self.conv2 = Conv(c1,c_,3,1)
        self.conv3 = nn.Conv2d(c1,c_,3,1,2,2)
        self.conv4 = nn.Conv2d(c1,c_,3,1,4,4)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU()
        self.m = nn.Sequential(*(Bottleneckszz(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        # print(x.shape)
        ident = x
        # x0 = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        x00 = self.conv1(x)
        # print(x00.shape)
        x01 = self.conv2(x)
        # print(x01.shape)
        x02 = self.conv3(x)
        x022 = self.act(x02)
        # print(x022.shape)
        x03 = self.conv4(x)
        x033 = self.act(x03)
        # print(x033.shape)
        x000 = x00+x01
        x001 =x022+x033
        x0 = torch.cat((x000,x001), dim=1)
        # print(x0.shape)
        x1 = x[:,0:int(self.c2/2),...]
        # print(int(self.c2/2))
        # print(x1.shape)
        x2 = x[:,int(self.c2/2):self.c2,...]
        # print(x2.shape)
        x1 = self.cv11(x1)
        x1 = self.cv12(x1)
        # print(x1.shape)
        x2 = self.cv21(x2)
        x2 = self.cv22(x2)
        # print(x2.shape)
        x3 = torch.cat((x1,x2), dim=1)
        # print(x3.shape)
        x = x3+ident
        return x
class S4(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,s=1,w = 1.0):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        cw = int(c1 * e)
        self.c2 = c2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv4 = Conv(c1,cw,1,1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.cv11 = Conv(c_,c_, (1, 3), (1, s))
        self.cv12 = Conv(c_, c_, (3, 1), (s, 1), g=g)
        self.cv21 = Conv(c_,c_, (1, 5), (1, s))
        self.cv22 = Conv(c_, c_, (5, 1), (s, 1), g=g)
        self.m = nn.Sequential(*(Bottleneckszz(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        # print(x.shape)
        ident = self.cv4(x)
        # print(ident.shape)
        # ident
        x0 = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        # print(x0.shape)
        x1 = x[:,0:int(self.c2/2),...]
        # print(int(self.c2/2))
        # print(x1.shape)
        x2 = x[:,int(self.c2/2):self.c2,...]
        # print(x2.shape)
        x1 = self.cv11(x1)
        x1 = self.cv12(x1)
        # print(x1.shape)
        x2 = self.cv21(x2)
        x2 = self.cv22(x2)
        # print(x2.shape)
        x3 = torch.cat((x1,x2), dim=1)
        # print(x3.shape)
        x = x3+ident
        return x
class S3(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5,s=1,w = 0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        cw = int(c_ * w)
        self.c2 = c2
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv4 = Conv(c1,2*c1,1,1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.cv11 = Conv(cw,c_, (1, 3), (1, s))
        self.cv12 = Conv(c_, c_, (3, 1), (s, 1), g=g)
        self.cv21 = Conv(cw,c_, (1, 5), (1, s))
        self.cv22 = Conv(c_, c_, (5, 1), (s, 1), g=g)
        self.m = nn.Sequential(*(Bottleneckszz(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        ident = self.cv4(x)
        # print(x.shape)
        x0 = self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
        # print(x0.shape)
        x1 = x[:,0:int(self.c2/4),...]
        # print(x1.shape)
        x2 = x[:,int(self.c2/4):int(self.c2/2),...]
        # print(x2.shape)
        x1 = self.cv11(x1)
        x1 = self.cv12(x1)
        # print(x1.shape)
        x2 = self.cv21(x2)
        # print(x2.shape)
        x2 = self.cv22(x2)
        x3 = torch.cat((x1,x2), dim=1)
        # print(x3.shape)
        x = ident+x3
        return x
class C4(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c1 = 384
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
class C5(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c1 = 1024
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
class C3C2(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.conv = nn.Conv2d(c1, c_, 1, 1, autopad(1, None), groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c_)
        self.act = nn.SiLU()
        self.cv1 = Conv(2 * c_, c2, 1, act=nn.Mish())
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))

    def forward(self, x):
        y = self.conv(x)
        return self.cv1(torch.cat((self.m(self.act(self.bn(y))), y), dim=1))

class C3x(C3):
    # C3 module with cross-convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = nn.Sequential(*(CrossConv(c_, c_, 3, 1, g, 1.0, shortcut) for _ in range(n)))

class C3TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = TransformerBlock(c_, c_, 4, n)

    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
import math
from functools import partial
from typing import Optional, Callable

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from einops import rearrange, repeat
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
try:
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
except:
    pass
class PatchEmbed2D(nn.Module):
    r""" Image to Patch Embedding
    Args:
        patch_size (int): Patch token size. Default: 4.
        in_chans (int): Number of input image channels. Default: 3.
        embed_dim (int): Number of linear projection output channels. Default: 96.
        norm_layer (nn.Module, optional): Normalization layer. Default: None
    """

    def __init__(self, patch_size=4, in_chans=3, embed_dim=96, norm_layer=None, **kwargs):
        super().__init__()
        if isinstance(patch_size, int):
            patch_size = (patch_size, patch_size)
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)
        if norm_layer is not None:
            self.norm = norm_layer(embed_dim)
        else:
            self.norm = None

    def forward(self, x):
        x = self.proj(x).permute(0, 2, 3, 1)  #3 3 256 256 _>3 64 64 96
        if self.norm is not None:
            x = self.norm(x)
        return x

class PatchMerging2D(nn.Module):
    r""" Patch Merging Layer.
    Args:
        input_resolution (tuple[int]): Resolution of input feature.
        dim (int): Number of input channels.
        norm_layer (nn.Module, optional): Normalization layer.  Default: nn.LayerNorm
    """

    def __init__(self, dim, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim
        self.reduction = nn.Linear(4 * dim, 2 * dim, bias=False)
        self.norm = norm_layer(4 * dim)

    def forward(self, x):
        B, H, W, C = x.shape

        SHAPE_FIX = [-1, -1]
        if (W % 2 != 0) or (H % 2 != 0):
            print(f"Warning, x.shape {x.shape} is not match even ===========", flush=True)
            SHAPE_FIX[0] = H // 2
            SHAPE_FIX[1] = W // 2

        x0 = x[:, 0::2, 0::2, :]  # B H/2 W/2 C
        x1 = x[:, 1::2, 0::2, :]  # B H/2 W/2 C
        x2 = x[:, 0::2, 1::2, :]  # B H/2 W/2 C
        x3 = x[:, 1::2, 1::2, :]  # B H/2 W/2 C

        if SHAPE_FIX[0] > 0:
            x0 = x0[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x1 = x1[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x2 = x2[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]
            x3 = x3[:, :SHAPE_FIX[0], :SHAPE_FIX[1], :]

        x = torch.cat([x0, x1, x2, x3], -1)  # B H/2 W/2 4*C
        x = x.view(B, H // 2, W // 2, 4 * C)  # B H/2*W/2 4*C

        x = self.norm(x)
        x = self.reduction(x)

        return x


class PatchExpand2D(nn.Module):
    def __init__(self, dim, dim_scale=2, norm_layer=nn.LayerNorm):
        super().__init__()
        self.dim = dim * 2
        self.dim_scale = dim_scale
        self.expand = nn.Linear(self.dim, dim_scale * self.dim, bias=False)
        self.norm = norm_layer(self.dim // dim_scale)

    def forward(self, x):
        B, H, W, C = x.shape
        x = self.expand(x)

        x = rearrange(x, 'b h w (p1 p2 c)-> b (h p1) (w p2) c', p1=self.dim_scale, p2=self.dim_scale,
                      c=C // self.dim_scale)
        x = self.norm(x)

        return x
# class SS2D(nn.Module):
#     def __init__(
#             self,
#             d_model,  # 96
#             d_state=16,
#             d_conv=3,
#             expand=2,
#             dt_rank="auto",
#             dt_min=0.001,
#             dt_max=0.1,
#             dt_init="random",
#             dt_scale=1.0,
#             dt_init_floor=1e-4,
#             dropout=0.,
#             conv_bias=True,
#             bias=False,
#     ):
#         super().__init__()
#         self.d_model = d_model  # 96
#         self.d_state = d_state  # 16
#         self.d_conv = d_conv  # 3
#         self.expand = expand  # 2
#         self.d_inner = int(self.expand * self.d_model)  # 192
#         self.dt_rank = math.ceil(self.d_model / 16)  # 6
#
#         #                           96                 384
#         self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
#         self.conv2d = nn.Conv2d(
#             in_channels=self.d_inner,  # 192
#             out_channels=self.d_inner,  # 192
#             kernel_size=d_conv,  # 3
#             padding=(d_conv - 1) // 2,  # 1
#             bias=conv_bias,
#             groups=self.d_inner,  # 192
#         )
#         self.act = nn.SiLU()
#
#         self.x_proj = (
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
#             nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
#         )
#         # 4*38*192的数据 初始化x的数据
#         self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
#         del self.x_proj
#
#         # 初始化dt的数据吧
#         self.dt_projs = (
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
#             self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
#         )
#         self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
#         self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
#         del self.dt_projs
#         # 初始化A和D
#         self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
#         self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)
#
#         # ss2d
#         self.forward_core = self.forward_corev0
#
#         self.out_norm = nn.LayerNorm(self.d_inner)
#         self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
#         self.dropout = nn.Dropout(dropout) if dropout > 0. else None
#
#     @staticmethod
#     def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
#                 **factory_kwargs):
#         dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)
#
#         # Initialize special dt projection to preserve variance at initialization
#         dt_init_std = dt_rank ** -0.5 * dt_scale
#         if dt_init == "constant":
#             nn.init.constant_(dt_proj.weight, dt_init_std)
#         elif dt_init == "random":
#             nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
#         else:
#             raise NotImplementedError
#
#         # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
#         dt = torch.exp(
#             torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
#             + math.log(dt_min)
#         ).clamp(min=dt_init_floor)
#         # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
#         inv_dt = dt + torch.log(-torch.expm1(-dt))
#         with torch.no_grad():
#             dt_proj.bias.copy_(inv_dt)
#         # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
#         dt_proj.bias._no_reinit = True
#
#         return dt_proj
#
#     @staticmethod
#     def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
#         # S4D real initialization
#         A = repeat(
#             torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
#             "n -> d n",
#             d=d_inner,
#         ).contiguous()
#         A_log = torch.log(A)  # Keep A_log in fp32
#         if copies > 1:
#             A_log = repeat(A_log, "d n -> r d n", r=copies)
#             if merge:
#                 A_log = A_log.flatten(0, 1)
#         A_log = nn.Parameter(A_log)
#         A_log._no_weight_decay = True
#         return A_log
#
#     @staticmethod
#     def D_init(d_inner, copies=1, device=None, merge=True):
#         # D "skip" parameter
#         D = torch.ones(d_inner, device=device)
#         if copies > 1:
#             D = repeat(D, "n1 -> r n1", r=copies)
#             if merge:
#                 D = D.flatten(0, 1)
#         D = nn.Parameter(D)  # Keep in fp32
#         D._no_weight_decay = True
#         return D
#
#     def forward_corev0(self, x: torch.Tensor):
#         self.selective_scan = selective_scan_fn
#
#         B, C, H, W = x.shape # 3 192 64 64
#         L = H * W  #64cheng64 4096
#         K = 4
#
#         x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
#                              dim=1).view(B, 2, -1, L)  #3 2 492 4096
#         xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)  3 4 192 4096
#
#         x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # 3 4 38 4096  x_proj_weight:4096 4*38*192的数据 初始化x的数据
#         dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #3 4 6 4096 3 4 16 4096 3 4 16 4096
#         dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # 3 4 192 4096   #proj_weight: 4*192*6的数据 初始化x的数据
#
#         xs = xs.float().view(B, -1, L)  # (b, k * d, l)  #3 768 4096
#         dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) # 3  768 4096
#         Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
#         Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
#         Ds = self.Ds.float().view(-1)  # (k * d) 768   # (K=4, D, N)
#         As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  #768 16 (k * d, d_state16)
#         dt_projs_bias = self.dt_projs_bias.float().view(-1)  #768 (k * d)
#
#         out_y = self.selective_scan(
#             xs, dts,
#             As, Bs, Cs, Ds, z=None,
#             delta_bias=dt_projs_bias,
#             delta_softplus=True,
#             return_last_state=False,
#         ).view(B, K, -1, L)  #3 4 192 4096
#         assert out_y.dtype == torch.float
#
#         inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) #3 2 192 4096
#         wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096
#         invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096
#
#         return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
#
#     def forward(self, x: torch.Tensor):
#         B, H, W, C = x.shape
#
#         xz = self.in_proj(x)  #3 64 64 96 _>3 64 64 384
#         x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  # x走的是ss2d的路径 3 64 64 192 3 64 64 192
#
#         x = x.permute(0, 3, 1, 2).contiguous() #3 192 64 64
#         x = self.act(self.conv2d(x))  # (b, d, h, w) #3 192 64 64
#         y1, y2, y3, y4 = self.forward_core(x) # 3 192 4096    3 192 4096 3 192 4096  3 192 4096
#         assert y1.dtype == torch.float32
#         y = y1 + y2 + y3 + y4
#         y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) #3 64 64 192
#         y = self.out_norm(y)
#         y = y * F.silu(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
#         out = self.out_proj(y) #3 64 64 96
#         if self.dropout is not None:
#             out = self.dropout(out)
#         return out  #3 64 64 96
class SS2D_At64(nn.Module):
    def __init__(
            self,
            d_model,  # 96
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            in_planes=64, out_planes=64,inp=64,oup=64,c1=64, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1
    ):

        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // 16)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.actat = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / 4), c1),
        )
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision, dilation=vision, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        # self.relu = nn.LeakyReLU()
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv  # 3
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,  # 1
            bias=conv_bias,
            groups=self.d_inner,  # 192
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape # 3 192 64 64
        L = H * W  #64cheng64 4096
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)  #3 2 492 4096
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)  3 4 192 4096

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # 3 4 38 4096  x_proj_weight:4096 4*38*192的数据 初始化x的数据
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #3 4 6 4096 3 4 16 4096 3 4 16 4096
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # 3 4 192 4096   #proj_weight: 4*192*6的数据 初始化x的数据

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)  #3 768 4096
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) # 3  768 4096
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d) 768   # (K=4, D, N)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  #768 16 (k * d, d_state16)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  #768 (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  #3 4 192 4096
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) #3 2 192 4096
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def forward_RFB(self, x):
        # print(x.shape)
        x0 = self.branch0(x)  # 1 16 64 64
        x1 = self.branch1(x)  # 1 16 64 64
        x2 = self.branch2(x)  # 1 16 64 64

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        # print(out.shape)
        short = self.shortcut(x)
        out = out * self.scale + short
        # out = self.relu(out)

        return out
    def forward_ATCA(self, x):
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        identity = x

        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        #C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.actat(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        out = out.permute(0, 2, 3, 1)
        return out
    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape   #1 64 64 32
        xz = self.in_proj(x)  #1 64 64 128
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  # x走的是ss2d的路径 3 64 64 192 3 64 64 192

        x = x.permute(0, 3, 1, 2).contiguous() #3 192 64 64
        # x_RFB= self.conv2d(x) #yuanlai
        # x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64
        x_RFB = self.forward_RFB(x) #xiugai1
        x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64

        y1, y2, y3, y4 = self.forward_core(x) # 3 192 4096    3 192 4096 3 192 4096  3 192 4096
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) #3 64 64 192
        y = self.out_norm(y)

        # y = y * F.silu(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        # y = y*self.forward_ATCA(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        b, h, w, c = z.shape
        z_permute = z.view(b, -1, c) #1 4096 64
        z_att_permute = self.channel_attention(z_permute)
        z_att_permute =z_att_permute.view(b, h, w, c)
        z_channel_att = z_att_permute
        zz = z * z_channel_att
        y = y* zz #3 64 64 192 # 这里的z忘记了一个Linear吧
        out = self.out_proj(y) #3 64 64 96
        if self.dropout is not None:
            out = self.dropout(out)
        return out  #3 64 64 96
class SS2D_At128(nn.Module):
    def __init__(
            self,
            d_model,  # 96
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            in_planes=128, out_planes=128,inp=128,oup=128,c1=128, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1
    ):

        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // 16)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.actat = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / 4), c1),
        )
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision, dilation=vision, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv  # 3
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,  # 1
            bias=conv_bias,
            groups=self.d_inner,  # 192
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape # 3 192 64 64
        L = H * W  #64cheng64 4096
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)  #3 2 492 4096
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)  3 4 192 4096

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # 3 4 38 4096  x_proj_weight:4096 4*38*192的数据 初始化x的数据
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #3 4 6 4096 3 4 16 4096 3 4 16 4096
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # 3 4 192 4096   #proj_weight: 4*192*6的数据 初始化x的数据

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)  #3 768 4096
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) # 3  768 4096
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d) 768   # (K=4, D, N)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  #768 16 (k * d, d_state16)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  #768 (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  #3 4 192 4096
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) #3 2 192 4096
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def forward_RFB(self, x):
        # print(x.shape)
        x0 = self.branch0(x)  # 1 16 64 64
        x1 = self.branch1(x)  # 1 16 64 64
        x2 = self.branch2(x)  # 1 16 64 64

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        # print(out.shape)
        short = self.shortcut(x)
        out = out * self.scale + short
        # out = self.relu(out)

        return out
    def forward_ATCA(self, x):
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        identity = x

        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        #C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.actat(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        out = out.permute(0, 2, 3, 1)
        return out
    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)  #3 64 64 96 _>3 64 64 384
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  # x走的是ss2d的路径 3 64 64 192 3 64 64 192

        x = x.permute(0, 3, 1, 2).contiguous() #3 192 64 64
        # x_RFB= self.conv2d(x) #yuanlai
        # x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64
        x_RFB = self.forward_RFB(x) #xiugai1
        x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64

        y1, y2, y3, y4 = self.forward_core(x) # 3 192 4096    3 192 4096 3 192 4096  3 192 4096
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) #3 64 64 192
        y = self.out_norm(y)

        # y = y * F.silu(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        # y = y*self.forward_ATCA(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        b, h, w, c = z.shape
        # zcheng = z.permute(0, 3, 1, 2)
        z_permute = z.view(b, -1, c) #1 4096 64
        z_att_permute = self.channel_attention(z_permute)
        z_att_permute =z_att_permute.view(b, h, w, c)
        z_channel_att = z_att_permute
        zz = z * z_channel_att
        y = y* zz #3 64 64 192 # 这里的z忘记了一个Linear吧
        out = self.out_proj(y) #3 64 64 96
        if self.dropout is not None:
            out = self.dropout(out)
        return out  #3 64 64 96
class SS2D_At256(nn.Module):
    def __init__(
            self,
            d_model,  # 96
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            in_planes=256, out_planes=256,inp=256,oup=256,c1=256, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1
    ):

        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // 16)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.actat = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / 4), c1),
        )
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision, dilation=vision, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv  # 3
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,  # 1
            bias=conv_bias,
            groups=self.d_inner,  # 192
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape # 3 192 64 64
        L = H * W  #64cheng64 4096
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)  #3 2 492 4096
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)  3 4 192 4096

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # 3 4 38 4096  x_proj_weight:4096 4*38*192的数据 初始化x的数据
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #3 4 6 4096 3 4 16 4096 3 4 16 4096
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # 3 4 192 4096   #proj_weight: 4*192*6的数据 初始化x的数据

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)  #3 768 4096
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) # 3  768 4096
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d) 768   # (K=4, D, N)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  #768 16 (k * d, d_state16)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  #768 (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  #3 4 192 4096
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) #3 2 192 4096
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def forward_RFB(self, x):
        # print(x.shape)
        x0 = self.branch0(x)  # 1 16 64 64
        x1 = self.branch1(x)  # 1 16 64 64
        x2 = self.branch2(x)  # 1 16 64 64

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        # print(out.shape)
        short = self.shortcut(x)
        out = out * self.scale + short
        # out = self.relu(out)

        return out
    def forward_ATCA(self, x):
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        identity = x

        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        #C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.actat(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        out = out.permute(0, 2, 3, 1)
        return out
    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)  #3 64 64 96 _>3 64 64 384
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  # x走的是ss2d的路径 3 64 64 192 3 64 64 192

        x = x.permute(0, 3, 1, 2).contiguous() #3 192 64 64
        # x_RFB= self.conv2d(x) #yuanlai
        # x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64
        x_RFB = self.forward_RFB(x) #xiugai1
        x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64

        y1, y2, y3, y4 = self.forward_core(x) # 3 192 4096    3 192 4096 3 192 4096  3 192 4096
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) #3 64 64 192
        y = self.out_norm(y)

        # y = y * F.silu(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        # y = y*self.forward_ATCA(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        b, h, w, c = z.shape
        z_permute = z.view(b, -1, c) #1 4096 64
        z_att_permute = self.channel_attention(z_permute)
        z_att_permute =z_att_permute.view(b, h, w, c)
        z_channel_att = z_att_permute
        zz = z * z_channel_att
        y = y* zz #3 64 64 192 # 这里的z忘记了一个Linear吧
        out = self.out_proj(y) #3 64 64 96
        if self.dropout is not None:
            out = self.dropout(out)
        return out  #3 64 64 96
class SS2D_At512(nn.Module):
    def __init__(
            self,
            d_model,  # 96
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            in_planes=512, out_planes=512,inp=512,oup=512,c1=512, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1
    ):

        super().__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // 16)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.actat = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce
        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / 4)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / 4), c1),
        )
        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision, dilation=vision, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv  # 3
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,  # 1
            bias=conv_bias,
            groups=self.d_inner,  # 192
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape # 3 192 64 64
        L = H * W  #64cheng64 4096
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)  #3 2 492 4096
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)  3 4 192 4096

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # 3 4 38 4096  x_proj_weight:4096 4*38*192的数据 初始化x的数据
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #3 4 6 4096 3 4 16 4096 3 4 16 4096
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # 3 4 192 4096   #proj_weight: 4*192*6的数据 初始化x的数据

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)  #3 768 4096
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) # 3  768 4096
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d) 768   # (K=4, D, N)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  #768 16 (k * d, d_state16)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  #768 (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  #3 4 192 4096
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) #3 2 192 4096
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def forward_RFB(self, x):
        # print(x.shape)
        x0 = self.branch0(x)  # 1 16 64 64
        x1 = self.branch1(x)  # 1 16 64 64
        x2 = self.branch2(x)  # 1 16 64 64

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        # print(out.shape)
        short = self.shortcut(x)
        out = out * self.scale + short
        # out = self.relu(out)

        return out
    def forward_ATCA(self, x):
        # print(x.shape)
        x = x.permute(0, 3, 1, 2)
        identity = x

        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        #C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.actat(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out =identity * a_w * a_h
        out = out.permute(0, 2, 3, 1)
        return out
    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)  #3 64 64 96 _>3 64 64 384
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  # x走的是ss2d的路径 3 64 64 192 3 64 64 192

        x = x.permute(0, 3, 1, 2).contiguous() #3 192 64 64
        # x_RFB= self.conv2d(x) #yuanlai
        # x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64
        x_RFB = self.forward_RFB(x) #xiugai1
        x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64

        y1, y2, y3, y4 = self.forward_core(x) # 3 192 4096    3 192 4096 3 192 4096  3 192 4096
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) #3 64 64 192
        y = self.out_norm(y)

        # y = y * F.silu(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        # y = y*self.forward_ATCA(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        b, h, w, c = z.shape
        z_permute = z.view(b, -1, c) #1 4096 64
        z_att_permute = self.channel_attention(z_permute)
        z_att_permute =z_att_permute.view(b, h, w, c)
        z_channel_att = z_att_permute
        zz = z * z_channel_att
        y = y* zz #3 64 64 192 # 这里的z忘记了一个Linear吧
        out = self.out_proj(y) #3 64 64 96
        if self.dropout is not None:
            out = self.dropout(out)
        return out  #3 64 64 96
class SS2D64(nn.Module):
    def __init__(
            self,
            d_model,  # 96
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            in_planes=64, out_planes=64, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1
    ):

        super().__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision, dilation=vision, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv  # 3
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,  # 1
            bias=conv_bias,
            groups=self.d_inner,  # 192
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape # 3 192 64 64
        L = H * W  #64cheng64 4096
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)  #3 2 492 4096
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)  3 4 192 4096

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # 3 4 38 4096  x_proj_weight:4096 4*38*192的数据 初始化x的数据
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #3 4 6 4096 3 4 16 4096 3 4 16 4096
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # 3 4 192 4096   #proj_weight: 4*192*6的数据 初始化x的数据

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)  #3 768 4096
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) # 3  768 4096
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d) 768   # (K=4, D, N)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  #768 16 (k * d, d_state16)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  #768 (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  #3 4 192 4096
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) #3 2 192 4096
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def forward_RFB(self, x):
        # print(x.shape)
        x0 = self.branch0(x)  # 1 16 64 64
        x1 = self.branch1(x)  # 1 16 64 64
        x2 = self.branch2(x)  # 1 16 64 64

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        # print(out.shape)
        short = self.shortcut(x)
        out = out * self.scale + short
        # out = self.relu(out)

        return out

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)  #3 64 64 96 _>3 64 64 384
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  # x走的是ss2d的路径 3 64 64 192 3 64 64 192

        x = x.permute(0, 3, 1, 2).contiguous() #3 192 64 64
        # x_RFB= self.conv2d(x) #yuanlai
        x_RFB = self.forward_RFB(x) #xiugai1
        x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64
        y1, y2, y3, y4 = self.forward_core(x) # 3 192 4096    3 192 4096 3 192 4096  3 192 4096
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) #3 64 64 192
        y = self.out_norm(y)

        y = y * F.silu(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        out = self.out_proj(y) #3 64 64 96
        if self.dropout is not None:
            out = self.dropout(out)
        return out  #3 64 64 96
class SS2D128(nn.Module):
    def __init__(
            self,
            d_model,  # 96
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            in_planes=128, out_planes=128, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1
    ):

        super().__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision, dilation=vision, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv  # 3
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,  # 1
            bias=conv_bias,
            groups=self.d_inner,  # 192
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape # 3 192 64 64
        L = H * W  #64cheng64 4096
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)  #3 2 492 4096
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)  3 4 192 4096

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # 3 4 38 4096  x_proj_weight:4096 4*38*192的数据 初始化x的数据
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #3 4 6 4096 3 4 16 4096 3 4 16 4096
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # 3 4 192 4096   #proj_weight: 4*192*6的数据 初始化x的数据

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)  #3 768 4096
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) # 3  768 4096
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d) 768   # (K=4, D, N)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  #768 16 (k * d, d_state16)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  #768 (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  #3 4 192 4096
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) #3 2 192 4096
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def forward_RFB(self, x):
        # print(x.shape)
        x0 = self.branch0(x)  # 1 16 64 64
        x1 = self.branch1(x)  # 1 16 64 64
        x2 = self.branch2(x)  # 1 16 64 64

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        # print(out.shape)
        short = self.shortcut(x)
        out = out * self.scale + short
        # out = self.relu(out)

        return out

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)  #3 64 64 96 _>3 64 64 384
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  # x走的是ss2d的路径 3 64 64 192 3 64 64 192

        x = x.permute(0, 3, 1, 2).contiguous() #3 192 64 64
        # x_RFB= self.conv2d(x) #yuanlai
        x_RFB = self.forward_RFB(x) #xiugai1
        x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64
        y1, y2, y3, y4 = self.forward_core(x) # 3 192 4096    3 192 4096 3 192 4096  3 192 4096
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) #3 64 64 192
        y = self.out_norm(y)

        y = y * F.silu(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        out = self.out_proj(y) #3 64 64 96
        if self.dropout is not None:
            out = self.dropout(out)
        return out  #3 64 64 96
class SS2D256(nn.Module):
    def __init__(
            self,
            d_model,  # 96
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            in_planes=256, out_planes=256, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1
    ):

        super().__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision, dilation=vision, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv  # 3
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,  # 1
            bias=conv_bias,
            groups=self.d_inner,  # 192
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape # 3 192 64 64
        L = H * W  #64cheng64 4096
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)  #3 2 492 4096
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)  3 4 192 4096

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # 3 4 38 4096  x_proj_weight:4096 4*38*192的数据 初始化x的数据
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #3 4 6 4096 3 4 16 4096 3 4 16 4096
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # 3 4 192 4096   #proj_weight: 4*192*6的数据 初始化x的数据

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)  #3 768 4096
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) # 3  768 4096
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d) 768   # (K=4, D, N)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  #768 16 (k * d, d_state16)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  #768 (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  #3 4 192 4096
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) #3 2 192 4096
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def forward_RFB(self, x):
        # print(x.shape)
        x0 = self.branch0(x)  # 1 16 64 64
        x1 = self.branch1(x)  # 1 16 64 64
        x2 = self.branch2(x)  # 1 16 64 64

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        # print(out.shape)
        short = self.shortcut(x)
        out = out * self.scale + short
        # out = self.relu(out)

        return out

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)  #3 64 64 96 _>3 64 64 384
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  # x走的是ss2d的路径 3 64 64 192 3 64 64 192

        x = x.permute(0, 3, 1, 2).contiguous() #3 192 64 64
        # x_RFB= self.conv2d(x) #yuanlai
        x_RFB = self.forward_RFB(x) #xiugai1
        x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64
        y1, y2, y3, y4 = self.forward_core(x) # 3 192 4096    3 192 4096 3 192 4096  3 192 4096
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) #3 64 64 192
        y = self.out_norm(y)

        y = y * F.silu(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        out = self.out_proj(y) #3 64 64 96
        if self.dropout is not None:
            out = self.dropout(out)
        return out  #3 64 64 96
class SS2D512(nn.Module):
    def __init__(
            self,
            d_model,  # 96
            d_state=16,
            d_conv=3,
            expand=2,
            dt_rank="auto",
            dt_min=0.001,
            dt_max=0.1,
            dt_init="random",
            dt_scale=1.0,
            dt_init_floor=1e-4,
            dropout=0.,
            conv_bias=True,
            bias=False,
            in_planes=512, out_planes=512, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1
    ):

        super().__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision, dilation=vision, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)
        self.d_model = d_model  # 96
        self.d_state = d_state  # 16
        self.d_conv = d_conv  # 3
        self.expand = expand  # 2
        self.d_inner = int(self.expand * self.d_model)  # 192
        self.dt_rank = math.ceil(self.d_model / 16)  # 6

        #                           96                 384
        self.in_proj = nn.Linear(self.d_model, self.d_inner * 2, bias=bias)
        self.conv2d = nn.Conv2d(
            in_channels=self.d_inner,  # 192
            out_channels=self.d_inner,  # 192
            kernel_size=d_conv,  # 3
            padding=(d_conv - 1) // 2,  # 1
            bias=conv_bias,
            groups=self.d_inner,  # 192
        )
        self.act = nn.SiLU()

        self.x_proj = (
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
            nn.Linear(self.d_inner, (self.dt_rank + self.d_state * 2), bias=False),
        )
        # 4*38*192的数据 初始化x的数据
        self.x_proj_weight = nn.Parameter(torch.stack([t.weight for t in self.x_proj], dim=0))  # (K=4, N, inner)
        del self.x_proj

        # 初始化dt的数据吧
        self.dt_projs = (
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
            self.dt_init(self.dt_rank, self.d_inner, dt_scale, dt_init, dt_min, dt_max, dt_init_floor),
        )
        self.dt_projs_weight = nn.Parameter(torch.stack([t.weight for t in self.dt_projs], dim=0))  # (K=4, inner, rank)
        self.dt_projs_bias = nn.Parameter(torch.stack([t.bias for t in self.dt_projs], dim=0))  # (K=4, inner)
        del self.dt_projs
        # 初始化A和D
        self.A_logs = self.A_log_init(self.d_state, self.d_inner, copies=4, merge=True)  # (K=4, D, N)
        self.Ds = self.D_init(self.d_inner, copies=4, merge=True)  # (K=4, D, N)

        # ss2d
        self.forward_core = self.forward_corev0

        self.out_norm = nn.LayerNorm(self.d_inner)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=bias)
        self.dropout = nn.Dropout(dropout) if dropout > 0. else None

    @staticmethod
    def dt_init(dt_rank, d_inner, dt_scale=1.0, dt_init="random", dt_min=0.001, dt_max=0.1, dt_init_floor=1e-4,
                **factory_kwargs):
        dt_proj = nn.Linear(dt_rank, d_inner, bias=True, **factory_kwargs)

        # Initialize special dt projection to preserve variance at initialization
        dt_init_std = dt_rank ** -0.5 * dt_scale
        if dt_init == "constant":
            nn.init.constant_(dt_proj.weight, dt_init_std)
        elif dt_init == "random":
            nn.init.uniform_(dt_proj.weight, -dt_init_std, dt_init_std)
        else:
            raise NotImplementedError

        # Initialize dt bias so that F.softplus(dt_bias) is between dt_min and dt_max
        dt = torch.exp(
            torch.rand(d_inner, **factory_kwargs) * (math.log(dt_max) - math.log(dt_min))
            + math.log(dt_min)
        ).clamp(min=dt_init_floor)
        # Inverse of softplus: https://github.com/pytorch/pytorch/issues/72759
        inv_dt = dt + torch.log(-torch.expm1(-dt))
        with torch.no_grad():
            dt_proj.bias.copy_(inv_dt)
        # Our initialization would set all Linear.bias to zero, need to mark this one as _no_reinit
        dt_proj.bias._no_reinit = True

        return dt_proj

    @staticmethod
    def A_log_init(d_state, d_inner, copies=1, device=None, merge=True):
        # S4D real initialization
        A = repeat(
            torch.arange(1, d_state + 1, dtype=torch.float32, device=device),
            "n -> d n",
            d=d_inner,
        ).contiguous()
        A_log = torch.log(A)  # Keep A_log in fp32
        if copies > 1:
            A_log = repeat(A_log, "d n -> r d n", r=copies)
            if merge:
                A_log = A_log.flatten(0, 1)
        A_log = nn.Parameter(A_log)
        A_log._no_weight_decay = True
        return A_log

    @staticmethod
    def D_init(d_inner, copies=1, device=None, merge=True):
        # D "skip" parameter
        D = torch.ones(d_inner, device=device)
        if copies > 1:
            D = repeat(D, "n1 -> r n1", r=copies)
            if merge:
                D = D.flatten(0, 1)
        D = nn.Parameter(D)  # Keep in fp32
        D._no_weight_decay = True
        return D

    def forward_corev0(self, x: torch.Tensor):
        self.selective_scan = selective_scan_fn

        B, C, H, W = x.shape # 3 192 64 64
        L = H * W  #64cheng64 4096
        K = 4

        x_hwwh = torch.stack([x.view(B, -1, L), torch.transpose(x, dim0=2, dim1=3).contiguous().view(B, -1, L)],
                             dim=1).view(B, 2, -1, L)  #3 2 492 4096
        xs = torch.cat([x_hwwh, torch.flip(x_hwwh, dims=[-1])], dim=1)  # (b, k, d, l)  3 4 192 4096

        x_dbl = torch.einsum("b k d l, k c d -> b k c l", xs.view(B, K, -1, L), self.x_proj_weight) # 3 4 38 4096  x_proj_weight:4096 4*38*192的数据 初始化x的数据
        dts, Bs, Cs = torch.split(x_dbl, [self.dt_rank, self.d_state, self.d_state], dim=2) #3 4 6 4096 3 4 16 4096 3 4 16 4096
        dts = torch.einsum("b k r l, k d r -> b k d l", dts.view(B, K, -1, L), self.dt_projs_weight) # 3 4 192 4096   #proj_weight: 4*192*6的数据 初始化x的数据

        xs = xs.float().view(B, -1, L)  # (b, k * d, l)  #3 768 4096
        dts = dts.contiguous().float().view(B, -1, L)  # (b, k * d, l) # 3  768 4096
        Bs = Bs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Cs = Cs.float().view(B, K, -1, L)  # (b, k, d_state, l)
        Ds = self.Ds.float().view(-1)  # (k * d) 768   # (K=4, D, N)
        As = -torch.exp(self.A_logs.float()).view(-1, self.d_state)  #768 16 (k * d, d_state16)
        dt_projs_bias = self.dt_projs_bias.float().view(-1)  #768 (k * d)

        out_y = self.selective_scan(
            xs, dts,
            As, Bs, Cs, Ds, z=None,
            delta_bias=dt_projs_bias,
            delta_softplus=True,
            return_last_state=False,
        ).view(B, K, -1, L)  #3 4 192 4096
        assert out_y.dtype == torch.float

        inv_y = torch.flip(out_y[:, 2:4], dims=[-1]).view(B, 2, -1, L) #3 2 192 4096
        wh_y = torch.transpose(out_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096
        invwh_y = torch.transpose(inv_y[:, 1].view(B, -1, W, H), dim0=2, dim1=3).contiguous().view(B, -1, L) #3 192 4096

        return out_y[:, 0], inv_y[:, 0], wh_y, invwh_y
    def forward_RFB(self, x):
        # print(x.shape)
        x0 = self.branch0(x)  # 1 16 64 64
        x1 = self.branch1(x)  # 1 16 64 64
        x2 = self.branch2(x)  # 1 16 64 64

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        # print(out.shape)
        short = self.shortcut(x)
        out = out * self.scale + short
        # out = self.relu(out)

        return out

    def forward(self, x: torch.Tensor):
        B, H, W, C = x.shape

        xz = self.in_proj(x)  #3 64 64 96 _>3 64 64 384
        x, z = xz.chunk(2, dim=-1)  # (b, h, w, d)  # x走的是ss2d的路径 3 64 64 192 3 64 64 192

        x = x.permute(0, 3, 1, 2).contiguous() #3 192 64 64
        # x_RFB= self.conv2d(x) #yuanlai
        x_RFB = self.forward_RFB(x) #xiugai1
        x = self.act(x_RFB)  # (b, d, h, w) #3 192 64 64
        y1, y2, y3, y4 = self.forward_core(x) # 3 192 4096    3 192 4096 3 192 4096  3 192 4096
        assert y1.dtype == torch.float32
        y = y1 + y2 + y3 + y4
        y = torch.transpose(y, dim0=1, dim1=2).contiguous().view(B, H, W, -1) #3 64 64 192
        y = self.out_norm(y)

        y = y * F.silu(z) #3 64 64 192 # 这里的z忘记了一个Linear吧
        out = self.out_proj(y) #3 64 64 96
        if self.dropout is not None:
            out = self.dropout(out)
        return out  #3 64 64 96
class VSSBlock64(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,  # 96
            drop_path: float = 0,  # 0.2
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
            attn_drop_rate: float = 0,  # 0
            d_state: int = 16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)  # 96             0.2                   16
        # self.self_attention = SS2D64(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,)
        self.self_attention = SS2D_At64(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # print(input.shape, "传入模块的大小")
        x = self.drop_path(self.self_attention(self.ln_1(input)))
        x = input + x

        return x
class VSSBlock128(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,  # 96
            drop_path: float = 0,  # 0.2
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
            attn_drop_rate: float = 0,  # 0
            d_state: int = 16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)  # 96             0.2                   16
        # self.self_attention = SS2D128(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,)
        self.self_attention = SS2D_At128(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # print(input.shape, "传入模块的大小")
        x = self.drop_path(self.self_attention(self.ln_1(input)))
        x = input + x

        return x
class VSSBlock256(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,  # 96
            drop_path: float = 0,  # 0.2
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
            attn_drop_rate: float = 0,  # 0
            d_state: int = 16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)  # 96             0.2                   16
        # self.self_attention = SS2D256(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,)
        self.self_attention = SS2D_At256(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # print(input.shape, "传入模块的大小")
        x = self.drop_path(self.self_attention(self.ln_1(input)))
        x = input + x

        return x
class VSSBlock512(nn.Module):
    def __init__(
            self,
            hidden_dim: int = 0,  # 96
            drop_path: float = 0,  # 0.2
            norm_layer: Callable[..., torch.nn.Module] = partial(nn.LayerNorm, eps=1e-6),  # nn.LN
            attn_drop_rate: float = 0,  # 0
            d_state: int = 16,
    ):
        super().__init__()
        self.ln_1 = norm_layer(hidden_dim)  # 96             0.2                   16
        # self.self_attention = SS2D512(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,)
        self.self_attention = SS2D_At512(d_model=hidden_dim, dropout=attn_drop_rate, d_state=d_state,)
        self.drop_path = DropPath(drop_path)

    def forward(self, input: torch.Tensor):
        # print(input.shape, "传入模块的大小")
        x = self.drop_path(self.self_attention(self.ln_1(input)))
        x = input + x

        return x
class C3cuda512(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(512, c_, 1, 1).to("cuda:0")
        self.cv2 = Conv(512, c_, 1, 1).to("cuda:0")
        self.cv3 = Conv(2 * c_, c2, 1).to("cuda:0")  # act=FReLU(c2)
        # self.cv1 = Conv(c1, c1, 1, 1).to("cuda:0")
        # self.cv2 = Conv(c1, c1, 1, 1).to("cuda:0")
        # self.cv3 = Conv(2 * c1, c2, 1).to("cuda:0")  # act=FReLU(c2)
        self.m = nn.Sequential(*(Bottleneck(c_, c_, shortcut, g, e=1.0) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))
class MRF_Mamba1(C3cuda):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # self.cv1 = Conv(c1, c_, 1, 1).to("cuda:0")
        # self.cv2 = Conv(c1, c_, 1, 1).to("cuda:0")
        # self.cv3 = Conv(2 * c_, c2, 1).to("cuda:0")  # act=FReLU(c2)
        self.m = VSSBlock64(hidden_dim=32, drop_path=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6), attn_drop_rate=0, d_state=16).to("cuda:0")

    def forward(self, x):
        if x.device.type == 'cpu':
            b = 1
        else:
            b = 0
        x = x.to("cuda:0")
        xmb = self.cv1(x)
        xmb = xmb.permute(0, 2, 3, 1).contiguous()
        xmb = self.m(xmb)
        xmb = xmb.permute(0, 3, 1, 2).contiguous()
        x=self.cv3(torch.cat((xmb, self.cv2(x)), dim=1))
        if b == 1:
            x = x.to("cpu")

        # x = x.to("cpu")
        return x
class MRF_Mamba2(C3cuda):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = VSSBlock128(hidden_dim=64, drop_path=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6), attn_drop_rate=0, d_state=16).to("cuda:0")

    def forward(self, x):
        if x.device.type == 'cpu':
            b = 1
        else:
            b = 0
        x = x.to("cuda:0")
        xmb = self.cv1(x)
        xmb = xmb.permute(0, 2, 3, 1).contiguous()
        xmb = self.m(xmb)
        xmb = xmb.permute(0, 3, 1, 2).contiguous()
        x=self.cv3(torch.cat((xmb, self.cv2(x)), dim=1))
        if b == 1:
            x = x.to("cpu")

        # x = x.to("cpu")
        return x
class MRF_Mamba3(C3cuda):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # self.cv1 = Conv(c1, c_, 1, 1).to("cuda:0")
        # self.cv2 = Conv(c1, c_, 1, 1).to("cuda:0")
        # self.cv3 = Conv(2 * c_, c2, 1).to("cuda:0")  # act=FReLU(c2)
        self.m = VSSBlock256(hidden_dim=128, drop_path=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6), attn_drop_rate=0, d_state=16).to("cuda:0")

    def forward(self, x):
        if x.device.type == 'cpu':
            b = 1
        else:
            b = 0
        x = x.to("cuda:0")
        xmb = self.cv1(x)
        xmb = xmb.permute(0, 2, 3, 1).contiguous()
        xmb = self.m(xmb)
        xmb = xmb.permute(0, 3, 1, 2).contiguous()
        x=self.cv3(torch.cat((xmb, self.cv2(x)), dim=1))
        if b == 1:
            x = x.to("cpu")

        # x = x.to("cpu")
        return x
class MRF_Mamba4(C3cuda512):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        # self.cv1 = Conv(c1, c_, 1, 1).to("cuda:0")
        # self.cv2 = Conv(c1, c_, 1, 1).to("cuda:0")
        # self.cv3 = Conv(2 * c_, c2, 1).to("cuda:0")  # act=FReLU(c2)
        self.m = VSSBlock512(hidden_dim=256, drop_path=0.2, norm_layer=partial(nn.LayerNorm, eps=1e-6), attn_drop_rate=0, d_state=16).to("cuda:0")

    def forward(self, x):
        if x.device.type == 'cpu':
            b = 1
        else:
            b = 0
        x = x.to("cuda:0")
        xmb = self.cv1(x)
        xmb = xmb.permute(0, 2, 3, 1).contiguous()
        xmb = self.m(xmb)
        xmb = xmb.permute(0, 3, 1, 2).contiguous()
        x=self.cv3(torch.cat((xmb, self.cv2(x)), dim=1))
        if b == 1:
            x = x.to("cpu")

        # x = x.to("cpu")
        return x
class S1TR(C3):
    # C3 module with TransformerBlock()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.cv4 = GhostConv(c_,c_)
        self.m = TransformerBlock5(c_, c_, 4, n)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 =  self.cv2(x)
        x21 = self.cv4(x2)
        x3 = self.m(x1)
        x4 = x1+x3
        x5 = torch.cat((x4,x21),dim=1)
        x6 = self.cv3(x5)
        return x6
class C3SPP(C3):
    # C3 module with SPP()
    def __init__(self, c1, c2, k=(5, 9, 13), n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)
        self.m = SPP(c_, c_, k)


class C3Ghost(C3):
    # C3 module with GhostBottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(GhostBottleneck(c_, c_) for _ in range(n)))


class SPP(nn.Module):
    # Spatial Pyramid Pooling (SPP) layer https://arxiv.org/abs/1406.4729
    def __init__(self, c1, c2, k=(5, 9, 13)):
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * (len(k) + 1), c2, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            return self.cv2(torch.cat([x] + [m(x) for m in self.m], 1))


class SPPF(nn.Module):
    # Spatial Pyramid Pooling - Fast (SPPF) layer for YOLOv5 by Glenn Jocher
    def __init__(self, c1, c2, k=5):  # equivalent to SPP(k=(5, 9, 13))
        super().__init__()
        c_ = c1 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_ * 4, c2, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)

    def forward(self, x):
        x = self.cv1(x)
        with warnings.catch_warnings():
            warnings.simplefilter('ignore')  # suppress torch 1.9.0 max_pool2d() warning
            y1 = self.m(x)
            y2 = self.m(y1)
            return self.cv2(torch.cat([x, y1, y2, self.m(y2)], 1))


class Focus(nn.Module):
    # Focus wh information into c-space
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.conv = Conv(c1 * 4, c2, k, s, p, g, act)
        # self.contract = Contract(gain=2)

    def forward(self, x):  # x(b,c,w,h) -> y(b,4c,w/2,h/2)
        return self.conv(torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1))
        # return self.conv(self.contract(x))


class GhostConv(nn.Module):
    # Ghost Convolution https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):  # ch_in, ch_out, kernel, stride, groups
        super().__init__()
        c_ = c2 // 2  # hidden channels
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)

    def forward(self, x):
        y = self.cv1(x)
        return torch.cat([y, self.cv2(y)], 1)
class GSConv(nn.Module):
    """
        GSConv is used to merge the channel information of DSConv and BaseConv
        You can refer to https://github.com/AlanLi1997/slim-neck-by-gsconv for more details
    """
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__()
        c_ = c2 // 2
        self.cv1 = Conv(c1, c_, k, s, None, g, act)
        self.cv2 = Conv(c_, c_, 5, 1, None, c_, act)
    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)

        b, n, h, w = x2.data.size()
        b_n = b * n // 2
        y = x2.reshape(b_n, 2, h * w)
        y = y.permute(1, 0, 2)
        y = y.reshape(2, -1, n // 2, h, w)

        return torch.cat((y[0], y[1]), 1)

class GSConvns(GSConv):
    # GSConv with a normative-shuffle https://github.com/AlanLi1997/slim-neck-by-gsconv # AIEAGNY
    def __init__(self, c1, c2, k=1, s=1, g=1, act=True):
        super().__init__(c1, c2, k=1, s=1, g=1, act=True)
        c_ = c2 // 2
        self.shuf = nn.Conv2d(c_ * 2, c2, 1, 1, 0, bias=False)

    def forward(self, x):
        x1 = self.cv1(x)
        x2 = torch.cat((x1, self.cv2(x1)), 1)
        # normative-shuffle, TRT supported
        return nn.ReLU(self.shuf(x2))

class GSBottleneck(nn.Module):
    def __init__(self, c1, c2, k=3, s=1):
        super().__init__()
        c_ = c2 // 2
        # for lighting
        self.conv_lighting = nn.Sequential(
            GSConv(c1, c_, 1, 1),
            GSConv(c_, c2, 1, 1, act=False))
        # for receptive field
        self.conv = nn.Sequential(
            GSConv(c1, c_, 3, 1),
            GSConv(c_, c2, 3, 1, act=False))
        self.shortcut = Conv(c1, c2, 3, 1, act=False)

    def forward(self, x):
        return self.conv_lighting(x) + self.shortcut(x)

class GhostBottleneck(nn.Module):
    # Ghost Bottleneck https://github.com/huawei-noah/ghostnet
    def __init__(self, c1, c2, k=3, s=1):  # ch_in, ch_out, kernel, stride
        super().__init__()
        c_ = c2 // 2
        self.conv = nn.Sequential(GhostConv(c1, c_, 1, 1),  # pw
                                  DWConv(c_, c_, k, s, act=False) if s == 2 else nn.Identity(),  # dw
                                  GhostConv(c_, c2, 1, 1, act=False))  # pw-linear
        self.shortcut = nn.Sequential(DWConv(c1, c1, k, s, act=False),
                                      Conv(c1, c2, 1, 1, act=False)) if s == 2 else nn.Identity()

    def forward(self, x):
        return self.conv(x) + self.shortcut(x)


class Contract(nn.Module):
    # Contract width-height into channels, i.e. x(1,64,80,80) to x(1,256,40,40)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert (h / s == 0) and (W / s == 0), 'Indivisible gain'
        s = self.gain
        x = x.view(b, c, h // s, s, w // s, s)  # x(1,64,40,2,40,2)
        x = x.permute(0, 3, 5, 1, 2, 4).contiguous()  # x(1,2,2,64,40,40)
        return x.view(b, c * s * s, h // s, w // s)  # x(1,256,40,40)


class Expand(nn.Module):
    # Expand channels into width-height, i.e. x(1,64,80,80) to x(1,16,160,160)
    def __init__(self, gain=2):
        super().__init__()
        self.gain = gain

    def forward(self, x):
        b, c, h, w = x.size()  # assert C / s ** 2 == 0, 'Indivisible gain'
        s = self.gain
        x = x.view(b, s, s, c // s ** 2, h, w)  # x(1,2,2,16,80,80)
        x = x.permute(0, 3, 4, 1, 5, 2).contiguous()  # x(1,16,80,2,80,2)
        return x.view(b, c // s ** 2, h * s, w * s)  # x(1,16,160,160)


# CBAM
class ChannelAttention(nn.Module):
    def __init__(self, in_planes, ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.f1 = nn.Conv2d(in_planes, in_planes // ratio, 1, bias=False)
        self.relu = nn.ReLU()
        self.f2 = nn.Conv2d(in_planes // ratio, in_planes, 1, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_out = self.f2(self.relu(self.f1(self.avg_pool(x))))
        max_out = self.f2(self.relu(self.f1(self.max_pool(x))))
        out = self.sigmoid(avg_out + max_out)
        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)

#CA
class h_sigmoid(nn.Module):
    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)
    def forward(self, x):
        return self.relu(x + 3) / 6
class h_swish(nn.Module):
    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)
    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(CoordAtt, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x, x2):
        identity = x2
        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        y = torch.cat([x_h, x_w], dim=2)
        #C*1*(h+w)
        y = self.conv1(y)
        y = self.bn1(y)
        y = self.act(y)
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)
        a_h = self.conv_h(x_h).sigmoid()
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h
        return out


class CBAM(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(c1, ratio)
        self.spatial_attention = SpatialAttention(kernel_size)

    def forward(self, x, x1):
        out = self.channel_attention(x) * x1
        # c*h*w
        # c*h*w * 1*h*w
        out = self.spatial_attention(x) * out
        return out

class ConV1(nn.Module):
      def __init__(self, c1, c2):
         super().__init__()
         self.d = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
      def forward(self,x):
         out = self.d(x)
         return out
class ConV2(nn.Module):
      def __init__(self, c1, c2):
         super().__init__()
         self.d = nn.Conv2d(c1, c2, 1, 1, 0, bias=False)
      def forward(self,x):
         out = self.d(x)
         return out
class Concat30(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = ConV1(512,256)
        self.conv1 = ConV2(256,512)
        self.conv = nn.Conv2d(512, 256, kernel_size=1, stride=1, padding=0)
        self.cbam = CoordAtt(256,256)
        
    def forward(self, x):
        size = len(x)
        if size < 3:
            return torch.cat(x, self.d)
        else:
            x1 = x[0]
            b1,c1,w1,h1 = x1.size()
            x2 = x[1]
            x3 = x[2]
            b3, c3, w3, h3 = x3.size()
            x3 = self.conv(x3)
            x3 = self.up(x3)
            #x_h = F.softmax(self.conv(x3), dim=0)
            #y1 = x_h * w1
            #x_w = F.softmax(self.conv(y1), dim=1)
            #x2 = x_w * x2
            x2 = self.cbam(x3,x2)
            x[1] = x2
            x3 = self.conv1(x2)
            x[2] = x3
            
            return torch.cat(x, self.d)
class EBiFPN_Add2(nn.Module):
    def __init__(self, dimension=1):
        super(EBiFPN_Add2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # 设置可学习参数 nn.Parameter的作用是：将一个不可训练的类型Tensor转换成可以训练的类型
        # 并且会向宿主模型注册该参数 成为其一部分 即model.parameters()会包含这个parameter
        # 从而在参数优化的时候可以自动一起优化
        self.conv = Conv(128,64,1)
    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        x= torch.cat(x, self.d)
        x =self.conv(x)
        return x
class Concat_sam_embed_add(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.cv1 = nn.Conv2d(256, 64,1)
        # self.cv1 = nn.Conv2d(768,512,1)
        self.up = nn.Upsample()
    def forward(self, x):
        # device = torch.device('cuda:0')
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        # x= torch.cat(x, self.d)
        x1 = weight[0] * x[0]
        x2 = weight[1] * x[1]
        # print(x1.shape)
        # print(x2.shape)
        x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='nearest')
        # x1 = x[0]
        # x2 = x[1]
        # print("1",x1.shape)  # 输出：cuda:0
        # print("2",x2.shape)  # 输出：cuda:0 sam
        # x = torch.cat((x1, x2), self.d)
        # print(x.shape)
        # print(x.device)  # 输出：cuda:0
        # x = torch.cat(x, self.d)
        x2 = self.cv1(x2)
        x = x1+x2
        # x = x.to(device)
        # print(x.shape)
        return x
class Concat_sam_embed_add2(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.cv1 = nn.Conv2d(256, 128,1)
        # self.cv1 = nn.Conv2d(768,512,1)
        self.up = nn.Upsample()
    def forward(self, x):
        # device = torch.device('cuda:0')
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        # x= torch.cat(x, self.d)
        x1 = weight[0] * x[0]
        x2 = weight[1] * x[1]
        # print(x1.shape)
        # print(x2.shape)
        x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='nearest')
        # x1 = x[0]
        # x2 = x[1]
        # print("1",x1.shape)  # 输出：cuda:0
        # print("2",x2.shape)  # 输出：cuda:0 sam
        # x = torch.cat((x1, x2), self.d)
        # print(x.shape)
        # print(x.device)  # 输出：cuda:0
        # x = torch.cat(x, self.d)
        x2 = self.cv1(x2)
        x = x1+x2
        # x = x.to(device)
        # print(x.shape)
        return x
class Concat_sam_embed_add3(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.cv1 = nn.Conv2d(256, 256,1)
        # self.cv1 = nn.Conv2d(768,512,1)
        self.up = nn.Upsample()
    def forward(self, x):
        # device = torch.device('cuda:0')
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        # x= torch.cat(x, self.d)
        x1 = weight[0] * x[0]
        x2 = weight[1] * x[1]
        # print(x1.shape)
        # print(x2.shape)
        x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='nearest')
        # x1 = x[0]
        # x2 = x[1]
        # print("1",x1.shape)  # 输出：cuda:0
        # print("2",x2.shape)  # 输出：cuda:0 sam
        # x = torch.cat((x1, x2), self.d)
        # print(x.shape)
        # print(x.device)  # 输出：cuda:0
        # x = torch.cat(x, self.d)
        x2 = self.cv1(x2)
        x = x1+x2
        # x = x.to(device)
        # print(x.shape)
        return x
class Concat_sam_embed_add4(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.cv1 = nn.Conv2d(256, 512,1)
        # self.cv1 = nn.Conv2d(768,512,1)
        self.up = nn.Upsample()
    def forward(self, x):
        # device = torch.device('cuda:0')
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        # x= torch.cat(x, self.d)
        x1 = weight[0] * x[0]
        x2 = weight[1] * x[1]
        # print(x1.shape)
        # print(x2.shape)
        x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='nearest')
        # x1 = x[0]
        # x2 = x[1]
        # print("1",x1.shape)  # 输出：cuda:0
        # print("2",x2.shape)  # 输出：cuda:0 sam
        # x = torch.cat((x1, x2), self.d)
        # print(x.shape)
        # print(x.device)  # 输出：cuda:0
        # x = torch.cat(x, self.d)
        x2 = self.cv1(x2)
        x = x1+x2
        # x = x.to(device)
        # print(x.shape)
        return x
class PWE(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.cv1 = nn.Conv2d(320, 64,1)  #yolov5s
        # self.cv1 = nn.Conv2d(384, 128,1)  #yolov5l
        # self.cv1 = nn.Conv2d(352, 128,1)  #yolov5m
        # self.cv1 = nn.Conv2d(768,512,1)
        self.up = nn.Upsample()
    def forward(self, x):
        # device = torch.device('cuda:0')
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        # x= torch.cat(x, self.d)
        x1 = weight[0] * x[0]
        x2 = weight[1] * x[1]
        # print(x1.shape)
        # print(x2.shape)
        x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='nearest')
        # x1 = x[0]
        # x2 = x[1]
        # print("1",x1.shape)  # 输出：cuda:0
        # print("2",x2.shape)  # 输出：cuda:0 sam
        x = torch.cat((x1, x2), self.d)
        # print(x.shape)
        # print(x.device)  # 输出：cuda:0
        # x = torch.cat(x, self.d)
        x = self.cv1(x)
        # x = x.to("cuda:0")
        # print(x.shape)
        return x
class PWE2(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.cv1 = nn.Conv2d(384, 128,1) #v5s
        # self.cv1 = nn.Conv2d(512, 256,1) #v5m
        # self.cv1 = nn.Conv2d(768,512,1)#v5l
        self.up = nn.Upsample()
    def forward(self, x):
        # device = torch.device('cuda:0')
        x1 = x[0]
        x2 = x[1]
        # print(x1.shape)
        # print(x2.shape)
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        # x= torch.cat(x, self.d)
        x1 = weight[0] * x[0]
        x2 = weight[1] * x[1]
        x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear')
        # x1 = x[0]
        # x2 = x[1]
        # print("1",x1.shape)  # 输出：cuda:0
        # print("2",x2.shape)  # 输出：cuda:0 sam
        x = torch.cat((x1, x2), self.d)
        # print(x.shape)
        # print(x.device)  # 输出：cuda:0
        # x = torch.cat(x, self.d)
        x = self.cv1(x)
        # x = x.to(device)
        # print(x.shape)
        return x
class PWE3(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.cv1 = nn.Conv2d(512, 256,1)  #v5s
        # self.cv1 = nn.Conv2d(768,512,1)  #v5l
        self.up = nn.Upsample()
    def forward(self, x):
        # device = torch.device('cuda:0')
        x1 = x[0]
        x2 = x[1]
        # print(x1.shape)
        # print(x2.shape)
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        # x= torch.cat(x, self.d)
        x1 = weight[0] * x[0]
        x2 = weight[1] * x[1]
        x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear')
        # x1 = x[0]
        # x2 = x[1]
        # print("1",x1.shape)  # 输出：cuda:0
        # print("2",x2.shape)  # 输出：cuda:0 sam
        x = torch.cat((x1, x2), self.d)
        # print(x.shape)
        # print(x.device)  # 输出：cuda:0
        # x = torch.cat(x, self.d)
        x = self.cv1(x)
        # x = x.to(device)
        # print(x.shape)
        return x
class PWE4(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # self.cv1 = nn.Conv2d(320, 64,1)
        self.cv1 = nn.Conv2d(768,512,1) #v5s
        # self.cv1 = nn.Conv2d(1280,1024,1) #v5l
        self.up = nn.Upsample()
    def forward(self, x):
        # device = torch.device('cuda:0')
        x1 = x[0]
        x2 = x[1]
        # print(x1.shape)
        # print(x2.shape)
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  # 将权重进行归一化
        # Fast normalized fusion
        # x= torch.cat(x, self.d)
        x1 = weight[0] * x[0]
        x2 = weight[1] * x[1]
        x2 = torch.nn.functional.interpolate(x2, size=x1.shape[2:], mode='bilinear')
        # x1 = x[0]
        # x2 = x[1]
        # print("1",x1.shape)  # 输出：cuda:0
        # print("2",x2.shape)  # 输出：cuda:0 sam
        x = torch.cat((x1, x2), self.d)
        # print(x.shape)
        # print(x.device)  # 输出：cuda:0
        # x = torch.cat(x, self.d)
        x = self.cv1(x)
        # x = x.to(device)
        # print(x.shape)
        return x
class Concat40(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        self.up = nn.UpsamplingNearest2d(scale_factor=2)
        self.conv = ConV1(256,128)
        self.conv1 = ConV2(128,256)
        self.conv = nn.Conv2d(256, 128, kernel_size=1, stride=1, padding=0)
        self.cbam = CoordAtt(128,128)
    def forward(self, x):
        size = len(x)
        if size < 3:
            return torch.cat(x, self.d)
        else:
            x1 = x[0]
            b1,c1,w1,h1 = x1.size()
            x2 = x[1]
            x3 = x[2]
            
            b3, c3, w3, h3 = x3.size()
            
           
            x3 = self.conv(x3)
            x3 = self.up(x3)
            #x_h = F.softmax(self.conv(x3), dim=0)
            #y1 = x_h * w1
            #x_w = F.softmax(self.conv(y1), dim=1)
            #x2 = x_w * x2
            x2 = self.cbam(x3,x2)
            x[1] = x2
            x3 = self.conv1(x2)
            x[2] = x2
            
            return torch.cat(x, self.d)

class Concat(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
    def forward(self, x):
        # print(x)
        return torch.cat(x, self.d)

class TwinNet01(nn.Module):
    def __init__(self, dim=1):
        super(TwinNet01, self).__init__()
        self.d = dim
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # self.sigmoid = torch.sigmoid()
        self.con1 = nn.Conv2d(128,64,1)
    def con(self,x):
        self.con = nn.Conv2d(x.shape[1], 1, 1)
    def forward(self, x):
        # print(x.shape)
        x1,x2 = x
        # print(x1.shape)
        # print(x2.shape)
        w = self.w
        weight1 = torch.sum(w, dim=0)
        weight = w / (weight1 + self.epsilon)
        # print(weight[0])
        # print(weight[1])
        x = [weight[0] * x1, weight[1] * x2]
        x6 = torch.cat(x, self.d)
        # print(x6.shape)
        x6 = self.con1(x6)
        # print(x6.shape)
        return x6
class TwinNet02(nn.Module):
    def __init__(self, dim=1):
        super(TwinNet02, self).__init__()
        self.d = dim
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # self.sigmoid = torch.sigmoid()
        self.con1 = nn.Conv2d(256,128,1)
    def con(self,x):
        self.con = nn.Conv2d(x.shape[1], 1, 1)
    def forward(self, x):
        # print(x.shape)
        x1,x2 = x
        # print(x1.shape)
        # print(x2.shape)
        w = self.w
        weight1 = torch.sum(w, dim=0)
        weight = w / (weight1 + self.epsilon)
        # print(weight[0])
        # print(weight[1])
        x = [weight[0] * x1, weight[1] * x2]
        x6 = torch.cat(x, self.d)
        # print(x6.shape)
        x6 = self.con1(x6)
        # print(x6.shape)
        return x6
class TwinNet03(nn.Module):
    def __init__(self, dim=1):
        super(TwinNet03, self).__init__()
        self.d = dim
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # self.sigmoid = torch.sigmoid()
        self.con1 = nn.Conv2d(512,256,1)
    def con(self,x):
        self.con = nn.Conv2d(x.shape[1], 1, 1)
    def forward(self, x):
        # print(x.shape)
        x1,x2 = x
        # print(x1.shape)
        # print(x2.shape)
        w = self.w
        weight1 = torch.sum(w, dim=0)
        weight = w / (weight1 + self.epsilon)
        # print(weight[0])
        # print(weight[1])
        x = [weight[0] * x1, weight[1] * x2]
        x6 = torch.cat(x, self.d)
        # print(x6.shape)
        x6 = self.con1(x6)
        # print(x6.shape)
        return x6
class TwinNet04(nn.Module):
    def __init__(self, dim=1):
        super(TwinNet04, self).__init__()
        self.d = dim
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # self.sigmoid = torch.sigmoid()
        self.con1 = nn.Conv2d(1024,512,1)
    def con(self,x):
        self.con = nn.Conv2d(x.shape[1], 1, 1)
    def forward(self, x):
        # print(x.shape)
        x1,x2 = x
        # print(x1.shape)
        # print(x2.shape)
        w = self.w
        weight1 = torch.sum(w, dim=0)
        weight = w / (weight1 + self.epsilon)
        # print(weight[0])
        # print(weight[1])
        x = [weight[0] * x1, weight[1] * x2]
        x6 = torch.cat(x, self.d)
        # print(x6.shape)
        x6 = self.con1(x6)
        # print(x6.shape)
        return x6
class TwinNet011(nn.Module):
    def __init__(self, dim=1):
        super(TwinNet011, self).__init__()
        self.d = dim
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        # self.sigmoid = torch.sigmoid()
    def con(self,x):
        self.con = nn.Conv2d(x.shape[1], 1, 1)
    def forward(self, x):
        # print(x.shape)
        x1,x2 = x
        print(x1.shape)
        print(x2.shape)
        x3 = torch.abs(x1 - x2)
        x33 = x1+x2
        x333 = x3//x33
        b,c,h,w = x333.size()
        print(x333.shape)
        # x4 = x3.view([b,-1])
        # print(x4.shape)
        x4 = torch.sigmoid(x3)
        print(x4.shape)
        x5 = [x1*x4,x2*x4]
        print(x5.shape)
        x6 = torch.cat(x5, self.d)
        print(x6.shape)
        return x6

class Concatszz01(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.conv = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)
        self.conv01 = Conv(1024, 512, 1, 1)
        self.conv02 = C3Ghost(512,512)
    def forward(self, x):
            x1 = x[0]
            # print(x1.shape)
            identy1 = x1
            x2 = x[1]
            # print(x2.shape)
            identy2 = x2
            # identy = identy1+identy2
            # x3 = torch.cat((x1,x2),1)
            x3 = x1 + x2
            # print(x3.shape)
            x4 = self.conv02(x3)
            # x4 = x1+x2
            # return x4+identy
            return x4
class Concatszz011(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.conv = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)
        self.conv01 = Conv(1024, 512, 1, 1)
        self.conv02 = C3Ghost(512,512)
    def forward(self, x):
            x1 = x[0]
            # print(x1.shape)
            b1, c1, h1, w1 = x1.size()
            x2 = x[1]
            # print(x2.shape)
            b2, c2, h2, w2 = x2.size()
            identy2 = x2
            # identy = identy1+identy2
            # x3 = torch.cat((x1,x2),1)
            # x11 = x1.view(b1,-1)
            # # print(x11.shape)
            # x22 = x2.view(b2,-1)
            # print(x22.shape)
            # siml1 = torch.cosine_similarity(x1, x2, dim=1)
            siml1 = torch.cosine_similarity(x1, x2, dim=2)
            siml2 = torch.cosine_similarity(x1,x2,dim=3)
            # print(siml.shape)
            siml1 = siml1.unsqueeze(2)
            siml2 = siml2.unsqueeze(3)
            # siml = siml.unsqueeze(2)
            # siml = siml.unsqueeze(3)
            # print(siml.shape)
            siml = siml1+siml2
            x3 = x1 + x2
            # print(x3.shape)
            x3 = x3 *siml
            x4 = self.conv02(x3)
            # x4 = x1+x2
            # return x4+identy
            return x4
class Concatszz021(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.conv = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)
        self.conv01 = Conv(1024, 512, 1, 1)
        self.conv02 = C3(256,256)
    def forward(self, x):
            x1 = x[0]
            # print(x1.shape)
            b1, c1, h1, w1 = x1.size()
            x2 = x[1]
            # print(x2.shape)
            b2, c2, h2, w2 = x2.size()
            identy2 = x2
            # identy = identy1+identy2
            # x3 = torch.cat((x1,x2),1)
            # x11 = x1.view(b1,-1)
            # # print(x11.shape)
            # x22 = x2.view(b2,-1)
            # print(x22.shape)
            # siml1 = torch.cosine_similarity(x1, x2, dim=1)
            siml1 = torch.cosine_similarity(x1, x2, dim=2)
            siml2 = torch.cosine_similarity(x1,x2,dim=3)
            # print(siml.shape)
            siml1 = siml1.unsqueeze(2)
            siml2 = siml2.unsqueeze(3)
            # siml = siml.unsqueeze(2)
            # siml = siml.unsqueeze(3)
            # print(siml.shape)
            siml = siml1+siml2
            x3 = x1 + x2
            # print(x3.shape)
            x3 = x3 *siml
            x4 = self.conv02(x3)
            # x4 = x1+x2
            # return x4+identy
            return x4
class Concatszz02(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.conv = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)
        self.conv01 = Conv(1024, 512, 1, 1)
        self.conv02 = C3Ghost(256,256)
    def forward(self, x):
            x1 = x[0]
            # print(x1.shape)
            identy1 = x1
            x2 = x[1]
            # print(x2.shape)
            identy2 = x2

            # identy = identy1+identy2
            # x3 = torch.cat((x1,x2),1)
            x3 = x1 + x2
            # print(x3.shape)
            x4 = self.conv02(x3)
            # print(x4.shape)
            # x4 = x1+x2
            # return x4+identy
            return x4
class Concatszz012(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.conv = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)
        self.conv01 = Conv(1024, 512, 1, 1)
        self.conv02 = C3Ghost(256,256)
    def forward(self, x):
            x1 = x[0]
            # print(x1.shape)
            b1, c1, h1, w1 = x1.size()
            x2 = x[1]
            # print(x2.shape)
            b2, c2, h2, w2 = x2.size()
            identy2 = x2
            # identy = identy1+identy2
            # x3 = torch.cat((x1,x2),1)
            # x11 = x1.view(b1,-1)
            # # print(x11.shape)
            # x22 = x2.view(b2,-1)
            # print(x22.shape)
            # siml1 = torch.cosine_similarity(x1, x2, dim=1)
            siml1 = torch.cosine_similarity(x1, x2, dim=2)
            siml2 = torch.cosine_similarity(x1,x2,dim=3)
            # print(siml.shape)
            siml1 = siml1.unsqueeze(2)
            siml2 = siml2.unsqueeze(3)
            # siml = siml.unsqueeze(2)
            # siml = siml.unsqueeze(3)
            # print(siml.shape)
            siml = siml1+siml2
            x3 = x1 + x2
            # print(x3.shape)
            x3 = x3 *siml
            x4 = self.conv02(x3)
            # x4 = x1+x2
            # return x4+identy
            return x4
class Concatszz022(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.conv = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)
        self.conv01 = Conv(1024, 512, 1, 1)
        self.conv02 = C3(128,128)
    def forward(self, x):
            x1 = x[0]
            # print(x1.shape)
            b1, c1, h1, w1 = x1.size()
            x2 = x[1]
            # print(x2.shape)
            b2, c2, h2, w2 = x2.size()
            identy2 = x2
            # identy = identy1+identy2
            # x3 = torch.cat((x1,x2),1)
            # x11 = x1.view(b1,-1)
            # # print(x11.shape)
            # x22 = x2.view(b2,-1)
            # print(x22.shape)
            # siml1 = torch.cosine_similarity(x1, x2, dim=1)
            siml1 = torch.cosine_similarity(x1, x2, dim=2)
            siml2 = torch.cosine_similarity(x1,x2,dim=3)
            # print(siml.shape)
            siml1 = siml1.unsqueeze(2)
            siml2 = siml2.unsqueeze(3)
            # siml = siml.unsqueeze(2)
            # siml = siml.unsqueeze(3)
            # print(siml.shape)
            siml = siml1+siml2
            x3 = x1 + x2
            # print(x3.shape)
            x3 = x3 *siml
            x4 = self.conv02(x3)
            # x4 = x1+x2
            # return x4+identy
            return x4
class Concatszz03(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.conv = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)
        self.conv01 = Conv(1024, 512, 1, 1)
        self.conv02 = C3Ghost(128,128)
    def forward(self, x):
            x1 = x[0]
            # print(x1.shape)
            identy1 = x1
            x2 = x[1]
            # print(x2.shape)
            identy2 = x2
            # identy = identy1+identy2
            # x3 = torch.cat((x1,x2),1)
            x3 = x1 + x2
            # print(x3.shape)
            x4 = self.conv02(x3)
            # print(x4.shape)
            # x4 = x1+x2
            # return x4+identy
            return x4
class Concatszz023(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.conv = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)
        self.conv01 = Conv(1024, 512, 1, 1)
        self.conv02 = C3(128,128)
        self.d = dimension
        # self.j1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
    def forward(self, x):
            x1 = x[0]
            # print(x1.shape)
            b1, c1, h1, w1 = x1.size()
            x2 = x[1]
            # print(x2.shape)
            b2, c2, h2, w2 = x2.size()
            identy2 = x2
            siml1 = torch.cosine_similarity(x1, x2, dim=2)
            siml2 = torch.cosine_similarity(x1,x2,dim=3)
            # print(siml.shape)
            siml1 = siml1.unsqueeze(2)
            siml2 = siml2.unsqueeze(3)
            w = self.w
            weight = w / (torch.sum(w, dim=0) + self.epsilon)
            siml = weight[0] * siml1 + weight[1] * siml2
            x3 = x1 + x2
            # print(x3.shape)
            x3 = x3 *siml
            x4 = self.conv02(x3)
            # x4 = x1+x2
            # return x4+identy
            return x4
class Concatszz04(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.conv = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)
        self.conv01 = Conv(1024, 512, 1, 1)
        self.conv02 = C3(64,64)
    def forward(self, x):
            x1 = x[0]
            # print(x1.shape)
            identy1 = x1
            x2 = x[1]
            # print(x2.shape)
            identy2 = x2

            # identy = identy1+identy2
            # x3 = torch.cat((x1,x2),1)
            x3 = x1 + x2
            # print(x3.shape)
            x4 = self.conv02(x3)
            # print(x4.shape)
            # x4 = x1+x2
            # return x4+identy
            return x4
class Concatszz024(nn.Module):
    # Concatenate a list of tensors along dimension
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension
        # self.conv = nn.Conv2d(c2, c1, kernel_size=1, stride=1, padding=0)
        self.conv01 = Conv(1024, 512, 1, 1)
        self.conv02 = C3(256,256)
        self.d = dimension
        # self.j1 = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
    def forward(self, x):
            x1 = x[0]
            # print(x1.shape)
            b1, c1, h1, w1 = x1.size()
            x2 = x[1]
            # print(x2.shape)
            b2, c2, h2, w2 = x2.size()
            identy2 = x2
            # identy = identy1+identy2
            # x3 = torch.cat((x1,x2),1)
            # x11 = x1.view(b1,-1)
            # # print(x11.shape)
            # x22 = x2.view(b2,-1)
            # print(x22.shape)
            # siml1 = torch.cosine_similarity(x1, x2, dim=1)
            siml1 = torch.cosine_similarity(x1, x2, dim=2)
            siml2 = torch.cosine_similarity(x1,x2,dim=3)
            # print(siml.shape)
            siml1 = siml1.unsqueeze(2)
            siml2 = siml2.unsqueeze(3)
            # siml = siml.unsqueeze(2)
            # siml = siml.unsqueeze(3)
            # print(siml.shape)
            w = self.w
            weight = w / (torch.sum(w, dim=0) + self.epsilon)  # Ȩ ؽ  й һ
            # Fast normalized fusion
            # x = [weight[0] * x[0], weight[1] * x[1]]
            siml = weight[0] * siml1 + weight[1] * siml2
            x3 = x1 + x2
            # print(x3.shape)
            x3 = x3 *siml
            x4 = self.conv02(x3)
            # print(x4.shape)
            # x4 = x1+x2
            # return x4+identy
            return x4
# class DetectMultiBackend(nn.Module):
#     # YOLOv5 MultiBackend class for python inference on various backends
#     def __init__(self, weights='yolov5s.pt', device=None, dnn=False, data=None):
#         # Usage:
#         #   PyTorch:              weights = *.pt
#         #   TorchScript:                    *.torchscript
#         #   ONNX Runtime:                   *.onnx
#         #   ONNX OpenCV DNN:                *.onnx with --dnn
#         #   OpenVINO:                       *.xml
#         #   CoreML:                         *.mlmodel
#         #   TensorRT:                       *.engine
#         #   TensorFlow SavedModel:          *_saved_model
#         #   TensorFlow GraphDef:            *.pb
#         #   TensorFlow Lite:                *.tflite
#         #   TensorFlow Edge TPU:            *_edgetpu.tflite
#
#
#         super().__init__()
#         w = str(weights[0] if isinstance(weights, list) else weights)
#         pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs = self.model_type(w)  # get backend
#         stride, names = 64, [f'class{i}' for i in range(1000)]  # assign defaults
#         w = attempt_download(w)  # download if not local
#         if data:  # data.yaml path (optional)
#             with open(data, errors='ignore') as f:
#                 names = yaml.safe_load(f)['names']  # class names
#
#         if pt:  # PyTorch
#             model = attempt_load(weights if isinstance(weights, list) else w, map_location=device)
#             stride = max(int(model.stride.max()), 32)  # model stride
#             names = model.module.names if hasattr(model, 'module') else model.names  # get class names
#             self.model = model  # explicitly assign for to(), cpu(), cuda(), half()
#         elif jit:  # TorchScript
#             LOGGER.info(f'Loading {w} for TorchScript inference...')
#             extra_files = {'config.txt': ''}  # model metadata
#             model = torch.jit.load(w, _extra_files=extra_files)
#             if extra_files['config.txt']:
#                 d = json.loads(extra_files['config.txt'])  # extra_files dict
#                 stride, names = int(d['stride']), d['names']
#         elif dnn:  # ONNX OpenCV DNN
#             LOGGER.info(f'Loading {w} for ONNX OpenCV DNN inference...')
#             check_requirements(('opencv-python>=4.5.4',))
#             net = cv2.dnn.readNetFromONNX(w)
#         elif onnx:  # ONNX Runtime
#             LOGGER.info(f'Loading {w} for ONNX Runtime inference...')
#             cuda = torch.cuda.is_available()
#             check_requirements(('onnx', 'onnxruntime-gpu' if cuda else 'onnxruntime'))
#             import onnxruntime
#             providers = ['CUDAExecutionProvider', 'CPUExecutionProvider'] if cuda else ['CPUExecutionProvider']
#             session = onnxruntime.InferenceSession(w, providers=providers)
#         elif xml:  # OpenVINO
#             LOGGER.info(f'Loading {w} for OpenVINO inference...')
#             check_requirements(('openvino-dev',))  # requires openvino-dev: https://pypi.org/project/openvino-dev/
#             import openvino.inference_engine as ie
#             core = ie.IECore()
#             if not Path(w).is_file():  # if not *.xml
#                 w = next(Path(w).glob('*.xml'))  # get *.xml file from *_openvino_model dir
#             network = core.read_network(model=w, weights=Path(w).with_suffix('.bin'))  # *.xml, *.bin paths
#             executable_network = core.load_network(network, device_name='CPU', num_requests=1)
#         elif engine:  # TensorRT
#             LOGGER.info(f'Loading {w} for TensorRT inference...')
#             import tensorrt as trt  # https://developer.nvidia.com/nvidia-tensorrt-download
#             check_version(trt.__version__, '7.0.0', hard=True)  # require tensorrt>=7.0.0
#             Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
#             logger = trt.Logger(trt.Logger.INFO)
#             with open(w, 'rb') as f, trt.Runtime(logger) as runtime:
#                 model = runtime.deserialize_cuda_engine(f.read())
#             bindings = OrderedDict()
#             for index in range(model.num_bindings):
#                 name = model.get_binding_name(index)
#                 dtype = trt.nptype(model.get_binding_dtype(index))
#                 shape = tuple(model.get_binding_shape(index))
#                 data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
#                 bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
#             binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
#             context = model.create_execution_context()
#             batch_size = bindings['images'].shape[0]
#         elif coreml:  # CoreML
#             LOGGER.info(f'Loading {w} for CoreML inference...')
#             import coremltools as ct
#             model = ct.models.MLModel(w)
#         else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
#             if saved_model:  # SavedModel
#                 LOGGER.info(f'Loading {w} for TensorFlow SavedModel inference...')
#                 import tensorflow as tf
#                 keras = False  # assume TF1 saved_model
#                 model = tf.keras.models.load_model(w) if keras else tf.saved_model.load(w)
#             elif pb:  # GraphDef https://www.tensorflow.org/guide/migrate#a_graphpb_or_graphpbtxt
#                 LOGGER.info(f'Loading {w} for TensorFlow GraphDef inference...')
#                 import tensorflow as tf
#
#                 def wrap_frozen_graph(gd, inputs, outputs):
#                     x = tf.compat.v1.wrap_function(lambda: tf.compat.v1.import_graph_def(gd, name=""), [])  # wrapped
#                     ge = x.graph.as_graph_element
#                     return x.prune(tf.nest.map_structure(ge, inputs), tf.nest.map_structure(ge, outputs))
#
#                 gd = tf.Graph().as_graph_def()  # graph_def
#                 gd.ParseFromString(open(w, 'rb').read())
#                 frozen_func = wrap_frozen_graph(gd, inputs="x:0", outputs="Identity:0")
#             elif tflite or edgetpu:  # https://www.tensorflow.org/lite/guide/python#install_tensorflow_lite_for_python
#                 try:  # https://coral.ai/docs/edgetpu/tflite-python/#update-existing-tf-lite-code-for-the-edge-tpu
#                     from tflite_runtime.interpreter import Interpreter, load_delegate
#                 except ImportError:
#                     import tensorflow as tf
#                     Interpreter, load_delegate = tf.lite.Interpreter, tf.lite.experimental.load_delegate,
#                 if edgetpu:  # Edge TPU https://coral.ai/software/#edgetpu-runtime
#                     LOGGER.info(f'Loading {w} for TensorFlow Lite Edge TPU inference...')
#                     delegate = {'Linux': 'libedgetpu.so.1',
#                                 'Darwin': 'libedgetpu.1.dylib',
#                                 'Windows': 'edgetpu.dll'}[platform.system()]
#                     interpreter = Interpreter(model_path=w, experimental_delegates=[load_delegate(delegate)])
#                 else:  # Lite
#                     LOGGER.info(f'Loading {w} for TensorFlow Lite inference...')
#                     interpreter = Interpreter(model_path=w)  # load TFLite model
#                 interpreter.allocate_tensors()  # allocate
#                 input_details = interpreter.get_input_details()  # inputs
#                 output_details = interpreter.get_output_details()  # outputs
#             elif tfjs: # https://github.com/iscyy/yoloair
#                 raise Exception('ERROR: YOLOv5 TF.js inference is not supported')
#         self.__dict__.update(locals())  # assign all variables to self
#
#     def forward(self, im, augment=False, visualize=False, val=False):
#         # YOLOv5 MultiBackend inference
#         b, ch, h, w = im.shape  # batch, channel, height, width
#         if self.pt or self.jit:  # PyTorch
#             y = self.model(im) if self.jit else self.model(im, augment=augment, visualize=visualize)
#             return y if val else y[0]
#         elif self.dnn:  # ONNX OpenCV DNN
#             im = im.cpu().numpy()  # torch to numpy
#             self.net.setInput(im)
#             y = self.net.forward()
#         elif self.onnx:  # ONNX Runtime
#             im = im.cpu().numpy()  # torch to numpy
#             y = self.session.run([self.session.get_outputs()[0].name], {self.session.get_inputs()[0].name: im})[0]
#         elif self.xml:  # OpenVINO
#             im = im.cpu().numpy()  # FP32
#             desc = self.ie.TensorDesc(precision='FP32', dims=im.shape, layout='NCHW')  # Tensor Description
#             request = self.executable_network.requests[0]  # inference request
#             request.set_blob(blob_name='images', blob=self.ie.Blob(desc, im))  # name=next(iter(request.input_blobs))
#             request.infer()
#             y = request.output_blobs['output'].buffer  # name=next(iter(request.output_blobs))
#         elif self.engine:  # TensorRT
#             assert im.shape == self.bindings['images'].shape, (im.shape, self.bindings['images'].shape)
#             self.binding_addrs['images'] = int(im.data_ptr())
#             self.context.execute_v2(list(self.binding_addrs.values()))
#             y = self.bindings['output'].data
#         elif self.coreml:  # CoreML
#             im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
#             im = Image.fromarray((im[0] * 255).astype('uint8'))
#             # im = im.resize((192, 320), Image.ANTIALIAS)
#             y = self.model.predict({'image': im})  # coordinates are xywh normalized
#             if 'confidence' in y:
#                 box = xywh2xyxy(y['coordinates'] * [[w, h, w, h]])  # xyxy pixels
#                 conf, cls = y['confidence'].max(1), y['confidence'].argmax(1).astype(np.float)
#                 y = np.concatenate((box, conf.reshape(-1, 1), cls.reshape(-1, 1)), 1)
#             else:
#                 k = 'var_' + str(sorted(int(k.replace('var_', '')) for k in y)[-1])  # output key
#                 y = y[k]  # output
#         else:  # TensorFlow (SavedModel, GraphDef, Lite, Edge TPU)
#             im = im.permute(0, 2, 3, 1).cpu().numpy()  # torch BCHW to numpy BHWC shape(1,320,192,3)
#             if self.saved_model:  # SavedModel
#                 y = (self.model(im, training=False) if self.keras else self.model(im)[0]).numpy()
#             elif self.pb:  # GraphDef# https://github.com/iscyy/yoloair
#                 y = self.frozen_func(x=self.tf.constant(im)).numpy()
#             else:  # Lite or Edge TPU
#                 input, output = self.input_details[0], self.output_details[0]
#                 int8 = input['dtype'] == np.uint8  # is TFLite quantized uint8 model
#                 if int8:
#                     scale, zero_point = input['quantization']
#                     im = (im / scale + zero_point).astype(np.uint8)  # de-scale
#                 self.interpreter.set_tensor(input['index'], im)
#                 self.interpreter.invoke()
#                 y = self.interpreter.get_tensor(output['index'])
#                 if int8:
#                     scale, zero_point = output['quantization']
#                     y = (y.astype(np.float32) - zero_point) * scale  # re-scale
#             y[..., :4] *= [w, h, w, h]  # xywh normalized to pixels
#
#         y = torch.tensor(y) if isinstance(y, np.ndarray) else y
#         return (y, []) if val else y
#
#     def warmup(self, imgsz=(1, 3, 640, 640), half=False):
#         # Warmup model by running inference once
#         if self.pt or self.jit or self.onnx or self.engine:  # warmup types
#             if isinstance(self.device, torch.device) and self.device.type != 'cpu':  # only warmup GPU models
#                 im = torch.zeros(*imgsz).to(self.device).type(torch.half if half else torch.float)  # input image
#                 self.forward(im)  # warmup
#
#     @staticmethod
#     def model_type(p='path/to/model.pt'):
#         # Return model type from model path, i.e. path='path/to/model.onnx' -> type=onnx
#         from tools.export import export_formats
#         suffixes = list(export_formats().Suffix) + ['.xml']  # export suffixes
#         check_suffix(p, suffixes)  # checks
#         p = Path(p).name  # eliminate trailing separators
#         pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs, xml2 = (s in p for s in suffixes)
#         xml |= xml2  # *_openvino_model or *.xml
#         tflite &= not edgetpu  # *.tflite
#         return pt, jit, onnx, xml, engine, coreml, saved_model, pb, tflite, edgetpu, tfjs


class AutoShape(nn.Module):
    # YOLOv5 input-robust model wrapper for passing cv2/np/PIL/torch inputs. Includes preprocessing, inference and NMS
    conf = 0.25  # NMS confidence threshold
    iou = 0.45  # NMS IoU threshold
    agnostic = False  # NMS class-agnostic
    multi_label = False  # NMS multiple labels per box
    classes = None  # (optional list) filter by class, i.e. = [0, 15, 16] for COCO persons, cats and dogs
    max_det = 1000  # maximum number of detections per image
    amp = False  # Automatic Mixed Precision (AMP) inference

    def __init__(self, model):
        super().__init__()
        LOGGER.info('Adding AutoShape... ')
        copy_attr(self, model, include=('yaml', 'nc', 'hyp', 'names', 'stride', 'abc'), exclude=())  # copy attributes
        # self.dmb = isinstance(model, DetectMultiBackend)  # DetectMultiBackend() instance
        self.pt = not self.dmb or model.pt  # PyTorch model
        self.model = model.eval()

    def _apply(self, fn):
        # Apply to(), cpu(), cuda(), half() to model tensors that are not parameters or registered buffers
        self = super()._apply(fn)# https://github.com/iscyy/yoloair
        if self.pt:
            m = self.model.model.model[-1] if self.dmb else self.model.model[-1]  # Detect()
            m.stride = fn(m.stride)
            m.grid = list(map(fn, m.grid))
            if isinstance(m.anchor_grid, list):
                m.anchor_grid = list(map(fn, m.anchor_grid))
        return self

    @torch.no_grad()
    def forward(self, imgs, size=640, augment=False, profile=False):
        # Inference from various sources. For height=640, width=1280, RGB images example inputs are:
        #   file:       imgs = 'data/images/zidane.jpg'  # str or PosixPath
        #   URI:             = 'https://ultralytics.com/images/zidane.jpg'
        #   OpenCV:          = cv2.imread('image.jpg')[:,:,::-1]  # HWC BGR to RGB x(640,1280,3)
        #   PIL:             = Image.open('image.jpg') or ImageGrab.grab()  # HWC x(640,1280,3)
        #   numpy:           = np.zeros((640,1280,3))  # HWC
        #   torch:           = torch.zeros(16,3,320,640)  # BCHW (scaled to size=640, 0-1 values)
        #   multiple:        = [Image.open('image1.jpg'), Image.open('image2.jpg'), ...]  # list of images

        t = [time_sync()]
        p = next(self.model.parameters()) if self.pt else torch.zeros(1)  # for device and type
        autocast = self.amp and (p.device.type != 'cpu')  # Automatic Mixed Precision (AMP) inference
        if isinstance(imgs, torch.Tensor):  # torch
            with amp.autocast(enabled=autocast):
                return self.model(imgs.to(p.device).type_as(p), augment, profile)  # inference

        # Pre-process
        n, imgs = (len(imgs), imgs) if isinstance(imgs, list) else (1, [imgs])  # number of images, list of images
        shape0, shape1, files = [], [], []  # image and inference shapes, filenames
        for i, im in enumerate(imgs):
            f = f'image{i}'  # filename
            if isinstance(im, (str, Path)):  # filename or uri
                im, f = Image.open(requests.get(im, stream=True).raw if str(im).startswith('http') else im), im
                im = np.asarray(exif_transpose(im))
            elif isinstance(im, Image.Image):  # PIL Image
                im, f = np.asarray(exif_transpose(im)), getattr(im, 'filename', f) or f
            files.append(Path(f).with_suffix('.jpg').name)
            if im.shape[0] < 5:  # image in CHW
                im = im.transpose((1, 2, 0))  # reverse dataloader .transpose(2, 0, 1)
            im = im[..., :3] if im.ndim == 3 else np.tile(im[..., None], 3)  # enforce 3ch input
            s = im.shape[:2]  # HWC
            shape0.append(s)  # image shape
            g = (size / max(s))  # gain
            shape1.append([y * g for y in s])
            imgs[i] = im if im.data.contiguous else np.ascontiguousarray(im)  # update
        shape1 = [make_divisible(x, self.stride) for x in np.stack(shape1, 0).max(0)]  # inference shape
        x = [letterbox(im, new_shape=shape1 if self.pt else size, auto=False)[0] for im in imgs]  # pad
        x = np.stack(x, 0) if n > 1 else x[0][None]  # stack
        x = np.ascontiguousarray(x.transpose((0, 3, 1, 2)))  # BHWC to BCHW
        x = torch.from_numpy(x).to(p.device).type_as(p) / 255  # uint8 to fp16/32
        t.append(time_sync())

        with amp.autocast(enabled=autocast):
            # Inference
            y = self.model(x, augment, profile)  # forward
            t.append(time_sync())

            # Post-process
            y = non_max_suppression(y if self.dmb else y[0], self.conf, iou_thres=self.iou, classes=self.classes,
                                    agnostic=self.agnostic, multi_label=self.multi_label, max_det=self.max_det)  # NMS
            for i in range(n):
                scale_coords(shape1, y[i][:, :4], shape0[i])

            t.append(time_sync())
            return Detections(imgs, y, files, t, self.names, x.shape)

# update
# class MobileOne(nn.Module):
#     # MobileOne
#     def __init__(self, in_channels, out_channels, n, k,
#                  stride=1, dilation=1, padding_mode='zeros', deploy=False, use_se=False):
#         super().__init__()
#         self.m = nn.Sequential(*[MobileOneBlock(in_channels, out_channels, k, stride, deploy) for _ in range(n)])
#
#     def forward(self, x):
#         x = self.m(x)
#         return x

class Detections:
    # YOLOv5 detections class for inference results
    def __init__(self, imgs, pred, files, times=(0, 0, 0, 0), names=None, shape=None):
        super().__init__()
        d = pred[0].device  # device
        gn = [torch.tensor([*(im.shape[i] for i in [1, 0, 1, 0]), 1, 1], device=d) for im in imgs]  # normalizations
        self.imgs = imgs  # list of images as numpy arrays
        self.pred = pred  # list of tensors pred[0] = (xyxy, conf, cls)
        self.names = names  # class names
        self.files = files  # image filenames
        self.times = times  # profiling times
        self.xyxy = pred  # xyxy pixels
        self.xywh = [xyxy2xywh(x) for x in pred]  # xywh pixels
        self.xyxyn = [x / g for x, g in zip(self.xyxy, gn)]  # xyxy normalized
        self.xywhn = [x / g for x, g in zip(self.xywh, gn)]  # xywh normalized
        self.n = len(self.pred)  # number of images (batch size)
        self.t = tuple((times[i + 1] - times[i]) * 1000 / self.n for i in range(3))  # timestamps (ms)
        self.s = shape  # inference BCHW shape

    def display(self, pprint=False, show=False, save=False, crop=False, render=False, save_dir=Path('')):
        crops = []
        for i, (im, pred) in enumerate(zip(self.imgs, self.pred)):
            s = f'image {i + 1}/{len(self.pred)}: {im.shape[0]}x{im.shape[1]} '  # string
            if pred.shape[0]:# https://github.com/iscyy/yoloair
                for c in pred[:, -1].unique():
                    n = (pred[:, -1] == c).sum()  # detections per class
                    s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string
                if show or save or render or crop:
                    annotator = Annotator(im, example=str(self.names))
                    for *box, conf, cls in reversed(pred):  # xyxy, confidence, class
                        label = f'{self.names[int(cls)]} {conf:.2f}'
                        if crop:
                            file = save_dir / 'crops' / self.names[int(cls)] / self.files[i] if save else None
                            crops.append({'box': box, 'conf': conf, 'cls': cls, 'label': label,
                                          'im': save_one_box(box, im, file=file, save=save)})
                        else:  # all others
                            annotator.box_label(box, label, color=colors(cls))
                    im = annotator.im
            else:
                s += '(no detections)'

            im = Image.fromarray(im.astype(np.uint8)) if isinstance(im, np.ndarray) else im  # from np
            if pprint:
                LOGGER.info(s.rstrip(', '))
            if show:
                im.show(self.files[i])  # show
            if save:
                f = self.files[i]
                im.save(save_dir / f)  # save
                if i == self.n - 1:
                    LOGGER.info(f"Saved {self.n} image{'s' * (self.n > 1)} to {colorstr('bold', save_dir)}")
            if render:
                self.imgs[i] = np.asarray(im)
        if crop:
            if save:
                LOGGER.info(f'Saved results to {save_dir}\n')
            return crops

    def print(self):
        self.display(pprint=True)  # print results
        LOGGER.info(f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS per image at shape {tuple(self.s)}' %
                    self.t)

    def show(self):
        self.display(show=True)  # show results

    def save(self, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True)  # increment save_dir
        self.display(save=True, save_dir=save_dir)  # save results

    def crop(self, save=True, save_dir='runs/detect/exp'):
        save_dir = increment_path(save_dir, exist_ok=save_dir != 'runs/detect/exp', mkdir=True) if save else None
        return self.display(crop=True, save=save, save_dir=save_dir)  # crop results

    def render(self):
        self.display(render=True)  # render results
        return self.imgs

    def pandas(self):
        # return detections as pandas DataFrames, i.e. print(results.pandas().xyxy[0])
        new = copy(self)  # return copy
        ca = 'xmin', 'ymin', 'xmax', 'ymax', 'confidence', 'class', 'name'  # xyxy columns
        cb = 'xcenter', 'ycenter', 'width', 'height', 'confidence', 'class', 'name'  # xywh columns
        for k, c in zip(['xyxy', 'xyxyn', 'xywh', 'xywhn'], [ca, ca, cb, cb]):
            a = [[x[:5] + [int(x[5]), self.names[int(x[5])]] for x in x.tolist()] for x in getattr(self, k)]  # update
            setattr(new, k, [pd.DataFrame(x, columns=c) for x in a])
        return new

    def tolist(self):
        # return a list of Detections objects, i.e. 'for result in results.tolist():'
        r = range(self.n)  # iterable
        x = [Detections([self.imgs[i]], [self.pred[i]], [self.files[i]], self.times, self.names, self.s) for i in r]
        # for d in x:
        #    for k in ['imgs', 'pred', 'xyxy', 'xyxyn', 'xywh', 'xywhn']:
        #        setattr(d, k, getattr(d, k)[0])  # pop out of list
        return x

    def __len__(self):
        return self.n


class Classify(nn.Module):
    # Classification head, i.e. x(b,c1,20,20) to x(b,c2)
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1):  # ch_in, ch_out, kernel, stride, padding, groups
        super().__init__()
        self.aap = nn.AdaptiveAvgPool2d(1)  # to x(b,c1,1,1)
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p), groups=g)  # to x(b,c2,1,1)
        self.flat = nn.Flatten()

    def forward(self, x):
        z = torch.cat([self.aap(y) for y in (x if isinstance(x, list) else [x])], 1)  # cat if list
        return self.flat(self.conv(z))  # flatten to x(b,c2)

# acmix
import torch.nn.functional as F
import time

def position(H, W, is_cuda=True):
    if is_cuda:
        loc_w = torch.linspace(-1.0, 1.0, W).cuda().unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).cuda().unsqueeze(1).repeat(1, W)
    else:
        loc_w = torch.linspace(-1.0, 1.0, W).unsqueeze(0).repeat(H, 1)
        loc_h = torch.linspace(-1.0, 1.0, H).unsqueeze(1).repeat(1, W)
    loc = torch.cat([loc_w.unsqueeze(0), loc_h.unsqueeze(0)], 0).unsqueeze(0)
    return loc


def stride(x, stride):
    b, c, h, w = x.shape
    return x[:, :, ::stride, ::stride]

def init_rate_half(tensor):
    if tensor is not None:
        tensor.data.fill_(0.5)

def init_rate_0(tensor):
    if tensor is not None:
        tensor.data.fill_(0.)


class ACmix(nn.Module):
    def __init__(self, in_planes, kernel_att=7, head=4, kernel_conv=3, stride=1, dilation=1):
        super(ACmix, self).__init__()
        self.in_planes = in_planes
        self.out_planes = in_planes
        self.head = head
        self.kernel_att = kernel_att
        self.kernel_conv = kernel_conv
        self.stride = stride
        self.dilation = dilation
        self.rate1 = torch.nn.Parameter(torch.Tensor(1))
        self.rate2 = torch.nn.Parameter(torch.Tensor(1))
        self.head_dim = self.out_planes // self.head

        self.conv1 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv2 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv3 = nn.Conv2d(in_planes, self.out_planes, kernel_size=1)
        self.conv_p = nn.Conv2d(2, self.head_dim, kernel_size=1)

        self.padding_att = (self.dilation * (self.kernel_att - 1) + 1) // 2
        self.pad_att = torch.nn.ReflectionPad2d(self.padding_att)
        self.unfold = nn.Unfold(kernel_size=self.kernel_att, padding=0, stride=self.stride)
        self.softmax = torch.nn.Softmax(dim=1)

        self.fc = nn.Conv2d(3*self.head, self.kernel_conv * self.kernel_conv, kernel_size=1, bias=False)
        self.dep_conv = nn.Conv2d(self.kernel_conv * self.kernel_conv * self.head_dim, self.out_planes, kernel_size=self.kernel_conv, bias=True, groups=self.head_dim, padding=1, stride=stride)

        self.reset_parameters()
    
    def reset_parameters(self):
        init_rate_half(self.rate1)
        init_rate_half(self.rate2)
        kernel = torch.zeros(self.kernel_conv * self.kernel_conv, self.kernel_conv, self.kernel_conv)
        for i in range(self.kernel_conv * self.kernel_conv):
            kernel[i, i//self.kernel_conv, i%self.kernel_conv] = 1.
        kernel = kernel.squeeze(0).repeat(self.out_planes, 1, 1, 1)
        self.dep_conv.weight = nn.Parameter(data=kernel, requires_grad=True)
        self.dep_conv.bias = init_rate_0(self.dep_conv.bias)

    def forward(self, x):
        q, k, v = self.conv1(x), self.conv2(x), self.conv3(x)
        scaling = float(self.head_dim) ** -0.5
        b, c, h, w = q.shape
        h_out, w_out = h//self.stride, w//self.stride


        # ### att
        # ## positional encoding https://github.com/iscyy/yoloair
        pe = self.conv_p(position(h, w, x.is_cuda))

        q_att = q.view(b*self.head, self.head_dim, h, w) * scaling
        k_att = k.view(b*self.head, self.head_dim, h, w)
        v_att = v.view(b*self.head, self.head_dim, h, w)

        if self.stride > 1:
            q_att = stride(q_att, self.stride)
            q_pe = stride(pe, self.stride)
        else:
            q_pe = pe

        unfold_k = self.unfold(self.pad_att(k_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # b*head, head_dim, k_att^2, h_out, w_out
        unfold_rpe = self.unfold(self.pad_att(pe)).view(1, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out) # 1, head_dim, k_att^2, h_out, w_out
        
        att = (q_att.unsqueeze(2)*(unfold_k + q_pe.unsqueeze(2) - unfold_rpe)).sum(1) # (b*head, head_dim, 1, h_out, w_out) * (b*head, head_dim, k_att^2, h_out, w_out) -> (b*head, k_att^2, h_out, w_out)
        att = self.softmax(att)

        out_att = self.unfold(self.pad_att(v_att)).view(b*self.head, self.head_dim, self.kernel_att*self.kernel_att, h_out, w_out)
        out_att = (att.unsqueeze(1) * out_att).sum(2).view(b, self.out_planes, h_out, w_out)

        ## conv
        f_all = self.fc(torch.cat([q.view(b, self.head, self.head_dim, h*w), k.view(b, self.head, self.head_dim, h*w), v.view(b, self.head, self.head_dim, h*w)], 1))
        f_conv = f_all.permute(0, 2, 1, 3).reshape(x.shape[0], -1, x.shape[-2], x.shape[-1])
        
        out_conv = self.dep_conv(f_conv)

        return self.rate1 * out_att + self.rate2 * out_conv

class ScaledDotProductAttention(nn.Module):
    '''
    Scaled dot-product attention
    '''

    def __init__(self, d_model, d_k, d_v, h,dropout=.1):
        '''
        :param d_model: Output dimensionality of the model
        :param d_k: Dimensionality of queries and keys
        :param d_v: Dimensionality of values
        :param h: Number of heads
        '''
        super(ScaledDotProductAttention, self).__init__()
        self.fc_q = nn.Linear(d_model, h * d_k)
        self.fc_k = nn.Linear(d_model, h * d_k)
        self.fc_v = nn.Linear(d_model, h * d_v)
        self.fc_o = nn.Linear(h * d_v, d_model)
        self.dropout=nn.Dropout(dropout)

        self.d_model = d_model
        self.d_k = d_k
        self.d_v = d_v
        self.h = h

        self.init_weights()


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, std=0.001)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, queries, keys, values, attention_mask=None, attention_weights=None):
        '''
        Computes
        :param queries: Queries (b_s, nq, d_model)
        :param keys: Keys (b_s, nk, d_model)
        :param values: Values (b_s, nk, d_model)
        :param attention_mask: Mask over attention values (b_s, h, nq, nk). True indicates masking.
        :param attention_weights: Multiplicative weights for attention values (b_s, h, nq, nk).
        :return:
        # https://github.com/iscyy/yoloair
        '''
        b_s, nq = queries.shape[:2]
        nk = keys.shape[1]

        q = self.fc_q(queries).view(b_s, nq, self.h, self.d_k).permute(0, 2, 1, 3)  # (b_s, h, nq, d_k)
        k = self.fc_k(keys).view(b_s, nk, self.h, self.d_k).permute(0, 2, 3, 1)  # (b_s, h, d_k, nk)
        v = self.fc_v(values).view(b_s, nk, self.h, self.d_v).permute(0, 2, 1, 3)  # (b_s, h, nk, d_v)

        att = torch.matmul(q, k) / np.sqrt(self.d_k)  # (b_s, h, nq, nk)
        if attention_weights is not None:
            att = att * attention_weights
        if attention_mask is not None:
            att = att.masked_fill(attention_mask, -np.inf)
        att = torch.softmax(att, -1)
        att=self.dropout(att)

        out = torch.matmul(att, v).permute(0, 2, 1, 3).contiguous().view(b_s, nq, self.h * self.d_v)  # (b_s, nq, h*d_v)
        out = self.fc_o(out)  # (b_s, nq, d_model)
        return out


# without BN version
class ASPP(nn.Module):
    def __init__(self, in_channel=512, out_channel=256):
        super(ASPP, self).__init__()
        self.mean = nn.AdaptiveAvgPool2d((1, 1))  # (1,1)means ouput_dim
        self.conv = nn.Conv2d(in_channel,out_channel, 1, 1)
        self.atrous_block1 = nn.Conv2d(in_channel, out_channel, 1, 1)
        self.atrous_block6 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=6, dilation=6)
        self.atrous_block12 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=12, dilation=12)
        self.atrous_block18 = nn.Conv2d(in_channel, out_channel, 3, 1, padding=18, dilation=18)
        self.conv_1x1_output = nn.Conv2d(out_channel * 5, out_channel, 1, 1)

    def forward(self, x):
        size = x.shape[2:]

        image_features = self.mean(x)
        image_features = self.conv(image_features)
        image_features = F.upsample(image_features, size=size, mode='bilinear')

        atrous_block1 = self.atrous_block1(x)
        atrous_block6 = self.atrous_block6(x)
        atrous_block12 = self.atrous_block12(x)
        atrous_block18 = self.atrous_block18(x)

        net = self.conv_1x1_output(torch.cat([image_features, atrous_block1, atrous_block6,
                                              atrous_block12, atrous_block18], dim=1))
        return net


class BasicConv(nn.Module):

    def __init__(self, in_planes, out_planes, kernel_size, stride=1, padding=0, dilation=1, groups=1, relu=True, bn=True):
        super(BasicConv, self).__init__()
        self.out_channels = out_planes
        if bn:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=False)
            self.bn = nn.BatchNorm2d(out_planes, eps=1e-5, momentum=0.01, affine=True)
            self.relu = nn.ReLU(inplace=False) if relu else None
        else:
            self.conv = nn.Conv2d(in_planes, out_planes, kernel_size=kernel_size, stride=stride, padding=padding, dilation=dilation, groups=groups, bias=True)
            self.bn = None
            self.relu = nn.ReLU(inplace=False) if relu else None

    def forward(self, x):
        x = self.conv(x)
        if self.bn is not None:
            x = self.bn(x)
        if self.relu is not None:
            x = self.relu(x)
        return x


class BasicRFB(nn.Module):

    def __init__(self, in_planes, out_planes, stride=1, scale=0.1, map_reduce=8, vision=1, groups=1):
        super(BasicRFB, self).__init__()
        self.scale = scale
        self.out_channels = out_planes
        inter_planes = in_planes // map_reduce

        self.branch0 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 1, dilation=vision + 1, relu=False, groups=groups)
        )
        self.branch1 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, 2 * inter_planes, kernel_size=(3, 3), stride=stride, padding=(1, 1), groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 2, dilation=vision + 2, relu=False, groups=groups)
        )
        self.branch2 = nn.Sequential(
            BasicConv(in_planes, inter_planes, kernel_size=1, stride=1, groups=groups, relu=False),
            BasicConv(inter_planes, (inter_planes // 2) * 3, kernel_size=3, stride=1, padding=1, groups=groups),
            BasicConv((inter_planes // 2) * 3, 2 * inter_planes, kernel_size=3, stride=stride, padding=1, groups=groups),
            BasicConv(2 * inter_planes, 2 * inter_planes, kernel_size=3, stride=1, padding=vision + 4, dilation=vision + 4, relu=False, groups=groups)
        )

        self.ConvLinear = BasicConv(6 * inter_planes, out_planes, kernel_size=1, stride=1, relu=False)
        self.shortcut = BasicConv(in_planes, out_planes, kernel_size=1, stride=stride, relu=False)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x):
        x0 = self.branch0(x)
        x1 = self.branch1(x)
        x2 = self.branch2(x)

        out = torch.cat((x0, x1, x2), 1)
        out = self.ConvLinear(out)
        short = self.shortcut(x)
        out = out * self.scale + short
        out = self.relu(out)

        return out


class SPPCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class SPPFCSPC(nn.Module):
    # CSP https://github.com/WongKinYiu/CrossStagePartialNetworks
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=5):
        super(SPPFCSPC, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(c_, c_, 3, 1)
        self.cv4 = Conv(c_, c_, 1, 1)
        self.m = nn.MaxPool2d(kernel_size=k, stride=1, padding=k // 2)
        self.cv5 = Conv(4 * c_, c_, 1, 1)
        self.cv6 = Conv(c_, c_, 3, 1)
        self.cv7 = Conv(2 * c_, c2, 1, 1)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        x2 = self.m(x1)
        x3 = self.m(x2)
        y1 = self.cv6(self.cv5(torch.cat((x1,x2,x3, self.m(x3)),1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class SPPCSPC_group(nn.Module):
    def __init__(self, c1, c2, n=1, shortcut=False, g=1, e=0.5, k=(5, 9, 13)):
        super(SPPCSPC_group, self).__init__()
        c_ = int(2 * c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1, g=4)
        self.cv2 = Conv(c1, c_, 1, 1, g=4)
        self.cv3 = Conv(c_, c_, 3, 1, g=4)
        self.cv4 = Conv(c_, c_, 1, 1, g=4)
        self.m = nn.ModuleList([nn.MaxPool2d(kernel_size=x, stride=1, padding=x // 2) for x in k])
        self.cv5 = Conv(4 * c_, c_, 1, 1, g=4)
        self.cv6 = Conv(c_, c_, 3, 1, g=4)
        self.cv7 = Conv(2 * c_, c2, 1, 1, g=4)

    def forward(self, x):
        x1 = self.cv4(self.cv3(self.cv1(x)))
        y1 = self.cv6(self.cv5(torch.cat([x1] + [m(x1) for m in self.m], 1)))
        y2 = self.cv2(x)
        return self.cv7(torch.cat((y1, y2), dim=1))

class C3HB(nn.Module):
    # CSP HorBlock with 3 convolutions by iscyy/yoloair
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c1, c_, 1, 1)
        self.cv3 = Conv(2 * c_, c2, 1)
        self.m = nn.Sequential(*(HorBlock(c_) for _ in range(n)))
    def forward(self, x):
        return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class space_to_depth(nn.Module):
    # Changing the dimension of the Tensor
    def __init__(self, dimension=1):
        super().__init__()
        self.d = dimension

    def forward(self, x):
         return torch.cat([x[..., ::2, ::2], x[..., 1::2, ::2], x[..., ::2, 1::2], x[..., 1::2, 1::2]], 1)

class HorLayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).# https://ar5iv.labs.arxiv.org/html/2207.14284
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError # by iscyy/air
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DDFA(nn.Module):
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = HorLayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = HorLayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x

class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )
        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        # B, C, H, W = x.shape gnconv [512]by iscyy/air
        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)
        dw_abc = self.dwconv(abc) * self.scale
        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]
        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]
        x = self.proj_out(x)

        return x

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)


class HorBlock(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = HorLayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim)
        self.norm2 = HorLayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape # [512]
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x

class HorNet(nn.Module):

    def __init__(self, index, in_chans, depths, dim_base, drop_path_rate=0.,layer_scale_init_value=1e-6, 
        gnconv=[
            partial(gnconv, order=2, s=1.0/3.0),
            partial(gnconv, order=3, s=1.0/3.0),
            partial(gnconv, order=4, s=1.0/3.0),
            partial(gnconv, order=5, s=1.0/3.0), # DDFA
        ],
    ):
        super().__init__()
        dims = [dim_base, dim_base * 2, dim_base * 4, dim_base * 8]
        self.index = index
        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers by iscyy/air
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            HorLayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    HorLayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks by iscyy/air
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 

        if not isinstance(gnconv, list):
            gnconv = [gnconv, gnconv, gnconv, gnconv]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[HorBlock(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i] # by iscyy/air

        self.apply(self._init_weights)

    def _init_weights(self, m):
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                    nn.init.trunc_normal_(m.weight, std=.02)
                    nn.init.constant_(m.bias, 0)
    def forward(self, x):
        x = self.downsample_layers[self.index](x)
        x = self.stages[self.index](x)
        return x

# class CNeB(nn.Module):
#     # CSP ConvNextBlock with 3 convolutions by iscyy/yoloair
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):  # ch_in, ch_out, number, shortcut, groups, expansion
#         super().__init__()
#         c_ = int(c2 * e)  # hidden channels
#         self.cv1 = Conv(c1, c_, 1, 1)
#         self.cv2 = Conv(c1, c_, 1, 1)
#         self.cv3 = Conv(2 * c_, c2, 1)
#         self.m = nn.Sequential(*(ConvNextBlock(c_) for _ in range(n)))
#     def forward(self, x):
#         return self.cv3(torch.cat((self.m(self.cv1(x)), self.cv2(x)), dim=1))

class SP(nn.Module):
    def __init__(self, k=3, s=1):
        super(SP, self).__init__()
        self.m = nn.MaxPool2d(kernel_size=k, stride=s, padding=k // 2)

    def forward(self, x):
        return self.m(x)

class MPC(nn.Module):
    def __init__(self, c1, k=2):
        super().__init__()
        c2 = c1 // 2
        self.mp = nn.MaxPool2d((k, k), k)
        self.cv1 = Conv(c1, c2, k=1)
        self.cv2 = nn.Sequential(
            Conv(c1, c2, k=1),
            Conv(c2, c2, k=3, p=1, s=2)
        )
    def forward(self, x):
        x1 = self.cv1(self.mp(x))
        x2 = self.cv2(x)
        out = torch.cat([x1, x2], dim=1)
        return out

class CAMethod(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        # self.ca=CoordAtt(c1,c2,ratio)
        self.convh = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0)
        self.convw = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0)

    def forward(self, x1):
        a, b, c, d = x1.size()

        x_h = F.softmax(self.convh(x1), dim=0)
        y1 = x_h * c
        x_w = F.softmax(self.convw(y1), dim=1)
        y = x_w
        out = x1 * y
        return out


class CABottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5, ratio=32):  # ch_in, ch_out, shortcut, groups, expansion
        super().__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        # self.ca=CoordAtt(c1,c2,ratio)
        self.convh = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0)
        self.convw = nn.Conv2d(c1, c1, kernel_size=1, stride=1, padding=0)
        self.conv2c = nn.Conv2d(2*c1, c1, kernel_size=1, stride=1, padding=0)
        self.relu = nn.ReLU6(inplace=True)
        self.conv1 = nn.Conv2d(c1, c_, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(c_)
        self.conv2 = nn.Conv2d(c_, c2, kernel_size=1, stride=1, padding=0)
        self.camethod = CAMethod(c1,c2)
    def forward(self, x):
        x1 = self.cv2(self.cv1(x))
        #x_h = F.softmax(self.convh(x1), dim=0)
        a,b,c,d = x.size()
        y1 = self.camethod(x1)
        y2 = self.camethod(x1)
        #y3 = self.camethod(x1)
        #y4 = self.camethod(x1)
        z =  list()
        z.append(y1)
        z.append(y2)
        #z.append(y3)
        #z.append(y4)
        y_out =  torch.cat(z, 1)
        y = self.conv2c(y_out)
        # x_h = F.softmax(self.convh(x1), dim=0)
        # y1 = x * x_h * c
        # x_w = F.softmax(self.convw(y1), dim=1)
        # y = x_w
        out = x1 * y
        #out = self.conv1(out2)
        return x + out if self.add else out

class Channel_Att1(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att1, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
    def forward(self, x):
        residual = x
        x = self.bn2(x)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_bn, x)
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x)  #
        return x

class Channel_Att2(nn.Module):
    def __init__(self, channels, t=16):
        super(Channel_Att2, self).__init__()
        self.channels = channels
        self.bn2 = nn.BatchNorm2d(self.channels, affine=True)
        self.ln2 = nn.LayerNorm(self.channels)
    def forward(self, x):
        residual = x
        x = x.permute(0, 2, 3, 1)
        # x = self.bn2(x)
        x = self.ln2(x)
        x = x.permute(0, 3, 1, 2)
        # residual = self.bn2(residual)
        # 式2的计算，即Mc的计算
        weight_bn = self.bn2.weight.data.abs() / torch.sum(self.bn2.weight.data.abs())
        weight_ln = self.ln2.weight.data.abs() / torch.sum(self.ln2.weight.data.abs())
        # weight = weight_ln + weight_bn
        x = x.permute(0, 2, 3, 1).contiguous()
        x = torch.mul(weight_ln, x)
        # residual = torch.mul(weight_bn, residual)
        # x =   x + residual
        x = x.permute(0, 3, 1, 2).contiguous()
        x = torch.sigmoid(x)  #
        return x
class SpatialAttention1(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention1, self).__init__()
        assert kernel_size in (3, 7), 'kernel size must be 3 or 7'
        padding = 3 if kernel_size == 7 else 1
        # (特征图的大小-算子的size+2*padding)/步长+1
        self.conv = nn.Conv2d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        # 1*h*w
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        x = torch.cat([avg_out, max_out], dim=1)
        # 2*h*w
        x = self.conv(x)
        # 1*h*w
        return self.sigmoid(x)
class ECA(nn.Module):
    """Constructs a ECA module.
    Args:
        channel: Number of channels of the input feature map
        k_size: Adaptive selection of kernel size
    """

    def __init__(self, c1,c2, k_size=3):
        super(ECA, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv = nn.Conv1d(1, 1, kernel_size=k_size, padding=(k_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # feature descriptor on the global spatial information
        y = self.avg_pool(x)
        # print(y.shape,y.squeeze(-1).shape,y.squeeze(-1).transpose(-1, -2).shape)
        # Two different branches of ECA module
        # 50*C*1*1
        #50*C*1
        #50*1*C
        y = self.conv(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        # Multi-scale information fusion
        y = self.sigmoid(y)
        return y.expand_as(x)
class Channels(nn.Module):
    def __init__(self, inp, oup, reduction=32):
        super(Channels, self).__init__()
        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))
        self.pool_w = nn.AdaptiveAvgPool2d((1, None))
        mip = max(8, inp // reduction)
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)
        self.act = h_swish()
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
    def forward(self, x):
        identity = x
        n, c, h, w = x.size()
        #c*1*W
        x_h = self.pool_h(x)
        # print(x_h.shape)
        #c*H*1
        #C*1*h
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        # print(x_w.shape)
        # y = torch.cat([x_h, x_w], dim=2)
        # print(y.shape)
        #C*1*(h+w)
        x_h = self.conv1(x_h)
        x_w = self.conv1(x_w)
        # print(x_h.shape)
        # print(x_w.shape)
        x_h = self.bn1(x_h)
        x_h = self.act(x_h)
        x_w = self.bn1(x_w)
        x_w = self.act(x_w)
        a_h = self.conv_h(x_h).sigmoid()
        # print(a_h.shape)
        a_w = self.conv_w(x_w).sigmoid()
        # print(a_w.shape)
        # print(identity.shape)
        out = identity * a_w * a_h
        return out + identity
class Attention(nn.Module):
    # CSP Bottleneck with 3 convolutions
    def __init__(self, c1, c2, ratio=16, kernel_size=7):  # ch_in, ch_out, number, shortcut, groups, expansion
        super(Attention, self).__init__()
        self.conv3 = nn.Conv2d(int(c1/2), int(c2/2), 3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(int(c1/2), int(c2/2), 5,stride=1,padding=2)
        self.c2 = c2
        self.channel_attention = Channel_Att1(int(c1 / 2))
        # self.channel_attention = ECA(int(c1/2),int(c2/2))
        # self.channel_attention = ChannelAttention(int(c1 / 2), ratio)
        self.spatial_attention = SpatialAttention1(kernel_size)
    def forward(self, x):
        ident = x
        x3 = x[:,0:int(self.c2/2),...]
        x5 = x[:,int(self.c2/2):self.c2,...]
        x3 = self.conv3(x3)
        x5 = self.conv5(x5)
        out1 = self.channel_attention(x3) * x3
        # print(out1.shape)
        # # c*h*w
        # # c*h*w * 1*h*w
        out2 = self.spatial_attention(x5) * x5
        # print(out2.shape)
        out = torch.cat((out1,out2),dim=1)
        return out

class AttentionBottleneck(nn.Module):
    # Standard bottleneck
    def __init__(self, c1, c2, shortcut=True, g=1, e=0.5,ratio=16,kernel_size=3):  # ch_in, ch_out, shortcut, groups, expansion
        super(AttentionBottleneck,self).__init__()
        c_ = int(c2 * e)  # hidden channels
        self.cv1 = Conv(c1, c_, 1, 1)
        self.cv2 = Conv(c_, c2, 3, 1, g=g)
        self.add = shortcut and c1 == c2
        self.conv3 = nn.Conv2d(int(c1/2), int(c2/2), 3,stride=1,padding=1)
        self.conv5 = nn.Conv2d(int(c1/2), int(c2/2), 5,stride=1,padding=2)
        self.c2 = c2
        self.channel_attention = Channels(int(c1 / 2), int(c2 / 2))
        # self.channel_attention = Channel_Att1(int(c1 / 2))
        # self.channel_attention = ECA(int(c1/2),int(c2/2))
        # self.channel_attention = ChannelAttention(int(c1 / 2), ratio)
        # self.spatial_attention = SpatialAttention1(kernel_size)
        self.spatial_attention = Channel_Att2(int(c1 / 2))
    def forward(self, x):
        x = self.cv2(self.cv1(x))
        ident = x
        x3 = x[:,0:int(self.c2/2),...]
        x5 = x[:,int(self.c2/2):self.c2,...]
        x3 = self.conv3(x3)
        x5 = self.conv5(x5)
        out1 = self.spatial_attention(x3) * x3
        out1 = out1+x3
        # print(out1.shape)
        # # c*h*w
        # # c*h*w * 1*h*w
        out2 = self.channel_attention(x5) * x5
        out2 = out2 + x5
        # print(out2.shape)
        out = torch.cat((out1,out2),dim=1)
        # return out
        return out +ident
class C3CA(C3):
    # C3 module with CABottleneck()
    def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
        super().__init__(c1, c2, n, shortcut, g, e)
        c_ = int(c2 * e)  # hidden channels
        self.m = nn.Sequential(*(AttentionBottleneck(c_, c_, shortcut) for _ in range(n)))

# class C3CA(C3):
#     # C3 module with CABottleneck()
#     def __init__(self, c1, c2, n=1, shortcut=True, g=1, e=0.5):
#         super().__init__(c1, c2, n, shortcut, g, e)
#         c_ = int(c2 * e)  # hidden channels
#         self.m = nn.Sequential(*(CABottleneck(c_, c_, shortcut) for _ in range(n)))

class BiFPN_Concat2(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat2, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        print(x[0].shape)
        print(x[1].shape)
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  #   Ȩ ؽ  й һ  
         #Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        x4 =torch.cat(x, self.d)
        print(x4.shape)
        return x4



class BiFPN_Concat3(nn.Module):
    def __init__(self, dimension=1):
        super(BiFPN_Concat3, self).__init__()
        self.d = dimension
        self.w = nn.Parameter(torch.ones(2, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001

    def forward(self, x):
        print(x)
        x[2] = x[1]
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)  #   Ȩ ؽ  й һ  
        # Fast normalized fusion
        x = [weight[0] * x[0], weight[1] * x[1]]
        return torch.cat(x, self.d)
class BiFPN_Add3(nn.Module):
    def __init__(self, c1, c2):
        super(BiFPN_Add3, self).__init__()
        self.w = nn.Parameter(torch.ones(3, dtype=torch.float32), requires_grad=True)
        self.epsilon = 0.0001
        self.conv = nn.Conv2d(c1, c2, kernel_size=1, stride=1, padding=0)
        self.silu = nn.SiLU()

    def forward(self, x):
        w = self.w
        weight = w / (torch.sum(w, dim=0) + self.epsilon)
        # Fast normalized fusion
        return self.conv(self.silu(weight[0] * x[0] + weight[1] * x[1] + weight[2] * x[2]))
class EPEncoder(nn.Module):
    def __init__(self, c1=256, c2=256, n=1, shortcut=True, g=1, e=0.5):
        """
        nonlin_first: if True you get conv -> nonlin -> norm. Else it's conv -> norm -> nonlin
        """
        super().__init__()
        # if isinstance(n_conv_per_stage, int):
        #     n_conv_per_stage = [n_conv_per_stage] * n_stages
        # if isinstance(n_conv_per_stage_decoder, int):
        #     n_conv_per_stage_decoder = [n_conv_per_stage_decoder] * (n_stages - 1)
        # assert len(n_conv_per_stage) == n_stages, "n_conv_per_stage must have as many entries as we have " \
        #                                           f"resolution stages. here: {n_stages}. " \
        #                                           f"n_conv_per_stage: {n_conv_per_stage}"
        # assert len(n_conv_per_stage_decoder) == (n_stages - 1), "n_conv_per_stage_decoder must have one less entries " \
        #                                                         f"as we have resolution stages. here: {n_stages} " \
        #                                                         f"stages, so it should have {n_stages - 1} entries. " \
        #                                                         f"n_conv_per_stage_decoder: {n_conv_per_stage_decoder}"
        # self.encoder = PlainConvEncoder(input_channels, n_stages, features_per_stage, conv_op, kernel_sizes, strides,
        #                                 n_conv_per_stage, conv_bias, norm_op, norm_op_kwargs, dropout_op,
        #                                 dropout_op_kwargs, nonlin, nonlin_kwargs, return_skips=True,
        #                                 nonlin_first=nonclin_first)
        # self.decoder = SAMDecoder(self.encoder, num_classes, n_conv_per_stage_decoder, deep_supervision,
        #                           nonlin_first=nonlin_first)

        save_path = "/home/szz/projects/yoloair04/"

        model_weight_path = os.path.join(save_path, "mobile_sam.pt")

        # if not os.path.exists(model_weight_path):
        #     download_model(url='https://github.com/ChaoningZhang/MobileSAM/raw/master/weights/mobile_sam.pt',
        #                    destination=model_weight_path)

        model_type = "vit_t"
        device = "cuda:0" if torch.cuda.is_available() else "cpu"

        mobile_sam = sam_model_registry[model_type](checkpoint=model_weight_path)
        mobile_sam.to("cuda:0")

        self.sam_image_encoder = mobile_sam.image_encoder
        for param in self.sam_image_encoder.parameters():
            param.requires_grad = False
        self.cv1 = Conv_SAM(c1,c2)  #256 64 64
    def forward(self, x):
        # print(x.shape)
        if x.device.type == 'cpu':
            b = 1
        else:
            b = 0
        sam_input = x.detach()
        sam_input = sam_input.to("cuda:0")
        if sam_input.shape[1] == 1:
            sam_input = sam_input.repeat(1, 3, 1, 1)
        sam_input = F.interpolate(sam_input, size=(1024, 1024), mode='bilinear', align_corners=True)
        sam_embed = self.sam_image_encoder(sam_input)
        if b == 1:
            sam_embed = sam_embed.to("cpu")
        # print(sam_embed.device)  # 输出：cuda:0
        # print(sam_embed.shape)
        sam_embed = self.cv1(sam_embed)
        return sam_embed  #256 64 64

class LinearLayer(nn.Module):
    def __init__(self, dim_in, dim_out, k, model):
        super(LinearLayer, self).__init__()
        if 'ViT' in model:
            self.fc = nn.ModuleList([nn.Linear(dim_in, dim_out) for i in range(k)])
        else:
            self.fc = nn.ModuleList([nn.Linear(dim_in * 2 ** (i + 2), dim_out) for i in range(k)])

    def forward(self, tokens):
        for i in range(len(tokens)):
            if len(tokens[i].shape) == 3:
                tokens[i] = self.fc[i](tokens[i][:, 1:, :])
            else:
                B, C, H, W = tokens[i].shape
                tokens[i] = self.fc[i](tokens[i].view(B, C, -1).permute(0, 2, 1).contiguous())
        return tokens


import numpy as np
import torch
from torch import nn
from torch.nn import init


def channel_shuffle(x, groups=2):
    B, C, H, W = x.size()
    out = x.view(B, groups, C // groups, H, W).permute(0, 2, 1, 3, 4).contiguous()
    out = out.view(B, C, H, W)
    return out


class GAMAttention(nn.Module):

    def __init__(self, c1, c2, group=True, rate=4):
        super(GAMAttention, self).__init__()

        self.channel_attention = nn.Sequential(
            nn.Linear(c1, int(c1 / rate)),
            nn.ReLU(inplace=True),
            nn.Linear(int(c1 / rate), c1),
        )
        self.spatial_attention = nn.Sequential(
            (
                nn.Conv2d(c1, c1 // rate, kernel_size=7, padding=3, groups=rate)
                if group
                else nn.Conv2d(c1, int(c1 / rate), kernel_size=7, padding=3)
            ),
            nn.BatchNorm2d(int(c1 / rate)),
            nn.ReLU(inplace=True),
            (
                nn.Conv2d(c1 // rate, c2, kernel_size=7, padding=3, groups=rate)
                if group
                else nn.Conv2d(int(c1 / rate), c2, kernel_size=7, padding=3)
            ),
            nn.BatchNorm2d(c2),
        )

    def forward(self, x):
        b, c, h, w = x.shape
        x_permute = x.permute(0, 2, 3, 1).view(b, -1, c)
        x_att_permute = self.channel_attention(x_permute).view(b, h, w, c)
        x_channel_att = x_att_permute.permute(0, 3, 1, 2)
        x = x * x_channel_att

        x_spatial_att = self.spatial_attention(x).sigmoid()
        x_spatial_att = channel_shuffle(x_spatial_att, 4)
        out = x * x_spatial_att
        return out


class RS_Mapping1(nn.Module):
    def __init__(self, dim,h=14, w=8, s=1.0):
        super().__init__()
        self.order = 3
        self.dims = [128 // 2 ** i for i in range(self.order)]#v5s
        # self.dims = [256 // 2 ** i for i in range(self.order)]#v5l
        self.dims.reverse()
        self.proj_in = nn.Conv2d(128, 2 * 128, 1)  #v5s
        # self.proj_in = nn.Conv2d(256, 2 * 256, 1)  #v5l

        # if gflayer is None:
        self.dwconv = nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=(7 - 1) // 2, bias=True, groups=sum(self.dims))
        # else:
        #     self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        # print(x.size())  #1 128 80 80
        fused_x = self.proj_in(x)# 1 128 64 64
        pwa, abc = torch.split(fused_x, (self.dims[0],sum(self.dims)), dim=1) #16 总-16=112
        pwb = abc[:,0:self.dims[1],:,:]
        pwc = abc[:,self.dims[1]:-self.dims[0],:,:]
        dw_abc = self.dwconv(abc) * self.scale #1 112 64 64
        dw_list = torch.split(dw_abc, self.dims, dim=1) # 1 16 64 64 | 1 32 64 64 | 1 64 64 64
        x1 = pwa * dw_list[0]  #1 16 64 64
        x2 = pwb * dw_list[1]  #1 16 64 64
        x3 = pwc * dw_list[2]  #1 16 64 64
        return x1,x2,x3

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel - 1) // 2, bias=bias, groups=dim)

class RS_Mapping2(nn.Module):
    def __init__(self, dim,h=14, w=8, s=1.0):
        super().__init__()
        self.order = 3
        self.dims = [256 // 2 ** i for i in range(self.order)] #v5s
        # self.dims = [512 // 2 ** i for i in range(self.order)] #v5l
        self.dims.reverse()
        self.proj_in = nn.Conv2d(256, 2 * 256, 1)#v5s
        # self.proj_in = nn.Conv2d(512, 2 * 512, 1) #v5l

        # if gflayer is None:
        #     self.dwconv = get_dwconv(sum(self.dims), 7, True)
        self.dwconv = nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=(7 - 1) // 2, bias=True,
                                groups=sum(self.dims))
        # else:
        #     self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        fused_x = self.proj_in(x)# 1 128 64 64
        pwa, abc = torch.split(fused_x, (self.dims[0],sum(self.dims)), dim=1) #16 总-16=112
        pwb = abc[:,0:self.dims[1],:,:]
        pwc = abc[:,self.dims[1]:-self.dims[0],:,:]
        dw_abc = self.dwconv(abc) * self.scale #1 112 64 64
        dw_list = torch.split(dw_abc, self.dims, dim=1) # 1 16 64 64 | 1 32 64 64 | 1 64 64 64
        x1 = pwa * dw_list[0]  #1 16 64 64
        x2 = pwb * dw_list[1]  #1 16 64 64
        x3 = pwc * dw_list[2]  #1 16 64 64
        return x1,x2,x3

class RS_Mapping3(nn.Module):
    def __init__(self, dim,  h=14, w=8, s=1.0):
        super().__init__()
        self.order = 3
        self.dims = [512 // 2 ** i for i in range(self.order)] #v5s
        # self.dims = [1024 // 2 ** i for i in range(self.order)] #v5l
        self.dims.reverse()
        self.proj_in = nn.Conv2d(512, 2 * 512, 1) #v5s
        # self.proj_in = nn.Conv2d(1024, 2 * 1024, 1) #v5l

        # if gflayer is None:
        #     self.dwconv = get_dwconv(sum(self.dims), 7, True)
        self.dwconv = nn.Conv2d(sum(self.dims), sum(self.dims), kernel_size=7, padding=(7 - 1) // 2, bias=True,
                                groups=sum(self.dims))
        # else:
        #     self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        fused_x = self.proj_in(x)# 1 128 64 64
        pwa, abc = torch.split(fused_x, (self.dims[0],sum(self.dims)), dim=1) #16 总-16=112
        pwb = abc[:,0:self.dims[1],:,:]
        pwc = abc[:,self.dims[1]:-self.dims[0],:,:]
        dw_abc = self.dwconv(abc) * self.scale #1 112 64 64
        dw_list = torch.split(dw_abc, self.dims, dim=1) # 1 16 64 64 | 1 32 64 64 | 1 64 64 64
        x1 = pwa * dw_list[0]  #1 16 64 64
        x2 = pwb * dw_list[1]  #1 16 64 64
        x3 = pwc * dw_list[2]  #1 16 64 64
        return x1,x2,x3

class HorLayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first.
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs
    with shape (batch_size, channels, height, width).# https://ar5iv.labs.arxiv.org/html/2207.14284
    """

    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError  # by iscyy/air
        self.normalized_shape = (normalized_shape,)

    def forward(self, x):
        # 最后一位表示特征维度
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

class DDFA2(nn.Module):
    def __init__(self, dim, h=14, w=8,order=3):
        super().__init__()
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        dim1 = self.dims[0]
        dim2 = self.dims[1]
        dim3 = self.dims[2]
        self.dw = nn.Conv2d(dim3 , dim3 , kernel_size=3, padding=1, bias=False, groups=dim3 // 2)
        self.complex_weight_x2 = nn.Parameter(torch.randn(dim3 // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_x2, std=.02)
        self.pre_norm1 = HorLayerNorm(dim3, eps=1e-6, data_format='channels_first')
        self.pre_norm2 = HorLayerNorm(dim2, eps=1e-6, data_format='channels_first')
        self.pre_norm3 = HorLayerNorm(dim1, eps=1e-6, data_format='channels_first')
        self.post_norm2 = HorLayerNorm(int(dim3+dim2), eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x_1 = x[0]   #  256
        x1 = x[1][0]  # 32
        x2 = x[2][1]   #64

        # x1 = x[0][2]
        # x2 = x[1]
        # x1 = x[0][3]
        # x2 = x[1][2]
        # x2 = x[3]
        x1 = self.pre_norm1(x1)
        x2 = self.pre_norm2(x2)
        x_1 = self.dw(x_1) #1 64 64 64
        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')  # 1 32 64 33

        weight = self.complex_weight  # 32 14 8 2
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2

        weight = torch.view_as_complex(weight.contiguous())  # 32 64 33

        x2 = x2 * weight  # 1 32 64 33
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64
        x = torch.cat([x1, x2], dim=1) #1 64 64 64
        x = self.post_norm2(x) #1 64 64 64
        return x

class DDFA3(nn.Module):
    def __init__(self, dim, h=14, w=8,order=3):
        super().__init__()
        self.dims = [128 // 2 ** i for i in range(order)]
        self.dims.reverse()
        # dim1 = self.dims[0]
        # dim2 = self.dims[1]
        # dim3 = self.dims[2]
        self.dw = nn.Conv2d(256 , 256 , kernel_size=3, padding=1, bias=False, groups=256 // 2)
        self.complex_weight_x2 = nn.Parameter(torch.randn(32, h, w, 2, dtype=torch.float32) * 0.02) #v5s
        self.complex_weight_x3 = nn.Parameter(torch.randn(128, h, w, 2, dtype=torch.float32) * 0.02)#v5s
        # self.complex_weight_x2 = nn.Parameter(torch.randn(64, h, w, 2, dtype=torch.float32) * 0.02) #v5l
        # self.complex_weight_x3 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02)#v5l
        trunc_normal_(self.complex_weight_x2, std=.02)
        trunc_normal_(self.complex_weight_x3, std=.02)
        self.pre_norm1 = HorLayerNorm(256, eps=1e-6, data_format='channels_first') #v5s
        # self.pre_norm1 = HorLayerNorm(512, eps=1e-6, data_format='channels_first') #v5l
        self.pre_norm2 = HorLayerNorm(32, eps=1e-6, data_format='channels_first')  #v5s
        # self.pre_norm2 = HorLayerNorm(64, eps=1e-6, data_format='channels_first')  #v5l
        self.pre_norm3 = HorLayerNorm(128, eps=1e-6, data_format='channels_first')#v5s
        # self.pre_norm3 = HorLayerNorm(256, eps=1e-6, data_format='channels_first')#v5l
        self.post_norm3 = HorLayerNorm(416, eps=1e-6, data_format='channels_first')#v5s
        # self.post_norm3 = HorLayerNorm(832, eps=1e-6, data_format='channels_first')#v5l
        self.conv1 = nn.Conv2d(416, 256, kernel_size=1, padding=0, bias=False)#v5s
        self.convx2 = ComplexConv2d(32, 32, kernel_size=1, padding=0, bias=False)
        self.convx3 = ComplexConv2d(128, 128, kernel_size=1, padding=0, bias=False)
        # self.conv1 = nn.Conv2d(832, 512, kernel_size=1, padding=0, bias=False)#v5l
        self.adavpool = nn.AdaptiveAvgPool2d((16, 16))
        # self.ATT = GAMAttention(256,256)
    def forward(self, x):
        x_1 = x[0]   #  256
        x2 = x[1][0]  # 32
        x3 = x[2][1]   #128
        x_1 = self.pre_norm1(x_1)
        x2 = self.pre_norm2(x2)
        x3 = self.pre_norm3(x3)
        # x_1 = self.dw(x_1) #1 32 64 64

        # x2
        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')  # 1 32 64 33
        weight2 = self.complex_weight_x2  # 32 14 8 2
        if not weight2.shape[1:3] == x2.shape[2:4]:
            weight2 = F.interpolate(weight2.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2

        weight2 = torch.view_as_complex(weight2.contiguous())  # 32 64 33

        x2 = x2 * weight2  # 1 32 64 33
        x2 = self.convx2(x2)
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64

        #x3

        x3 = x3.to(torch.float32)
        B, C, a, b = x3.shape
        x3 = torch.fft.rfft2(x3, dim=(2, 3), norm='ortho')  # 1 32 64 33
        weight3 = self.complex_weight_x3  # 32 14 8 2
        if not weight3.shape[1:3] == x3.shape[2:4]:
            weight3 = F.interpolate(weight3.permute(3, 0, 1, 2), size=x3.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2

        weight3 = torch.view_as_complex(weight3.contiguous())  # 32 64 33

        x3 = x3 * weight3  # 1 32 64 33
        x3 = self.convx3(x3)
        x3 = torch.fft.irfft2(x3, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64

        # x2 = F.interpolate(x2, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        # x3 = F.interpolate(x3, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        b,c,h,w = x_1.size()
        ampool = nn.AdaptiveAvgPool2d((h, w))
        # ampool = nn.AdaptiveMaxPool2d((h, w))
        x2 = ampool(x2)
        x3 = ampool(x3)
        x = torch.cat([x_1, x2,x3], dim=1) #1 64 64 64

        x = self.post_norm3(x) #1 64 64 64
        x = self.conv1(x)
        # x = self.ATT(x)
        return x
def apply_complex(fr, fi, input, dtype = torch.complex64):
    return (fr(input.real)-fi(input.imag)).type(dtype) \
            + 1j*(fr(input.imag)+fi(input.real)).type(dtype)


class ComplexConv2d(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=0,
                 dilation=1, groups=1, bias=True):
        super(ComplexConv2d, self).__init__()
        self.conv_r = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)
        self.conv_i = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, dilation, groups, bias)

    def forward(self, input):
        return apply_complex(self.conv_r, self.conv_i, input)
class GlobalLocalFilter3_2(nn.Module):
    def __init__(self, dim, h=14, w=8,order=3):
        super().__init__()
        self.dims = [128 // 2 ** i for i in range(order)]
        self.dims.reverse()
        dim1 = self.dims[0]
        dim2 = self.dims[1]
        dim3 = self.dims[2]
        self.dw = nn.Conv2d(128 , 128 , kernel_size=3, padding=1, bias=False, groups=128 // 2)
        self.complex_weight_x2 = nn.Parameter(torch.randn(64, h, w, 2, dtype=torch.float32) * 0.02)
        self.complex_weight_x3 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight_x2, std=.02)
        trunc_normal_(self.complex_weight_x3, std=.02)
        self.pre_norm1 = HorLayerNorm(128, eps=1e-6, data_format='channels_first')
        self.pre_norm2 = HorLayerNorm(64, eps=1e-6, data_format='channels_first')
        self.pre_norm3 = HorLayerNorm(256, eps=1e-6, data_format='channels_first')
        self.post_norm3 = HorLayerNorm(448, eps=1e-6, data_format='channels_first')
        self.conv1 = nn.Conv2d(448, 128, kernel_size=1, padding=0, bias=False)
        self.convx2 = ComplexConv2d(64, 64, kernel_size=1, padding=0, bias=False)
        self.convx3 = ComplexConv2d(256, 256, kernel_size=1, padding=0, bias=False)

    def forward(self, x):
        x_1 = x[0]   #  128
        x2 = x[1][1]  # 64
        x3 = x[2][2]   #256
        x_1 = self.pre_norm1(x_1)
        x2 = self.pre_norm2(x2)
        x3 = self.pre_norm3(x3)
        x_1 = self.dw(x_1) #1 32 64 64

        # x2
        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')  # 1 32 64 33
        weight2 = self.complex_weight_x2  # 32 14 8 2
        if not weight2.shape[1:3] == x2.shape[2:4]:
            weight2 = F.interpolate(weight2.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2
        x2 = self.convx2(x2)
        weight2 = torch.view_as_complex(weight2.contiguous())  # 32 64 33

        x2 = x2 * weight2  # 1 32 64 33
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64

        #x3
        x3 = x3.to(torch.float32)
        B, C, a, b = x3.shape
        x3 = torch.fft.rfft2(x3, dim=(2, 3), norm='ortho')  # 1 32 64 33
        weight3 = self.complex_weight_x3  # 32 14 8 2
        if not weight3.shape[1:3] == x3.shape[2:4]:
            weight3 = F.interpolate(weight3.permute(3, 0, 1, 2), size=x3.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2
        x3 = self.convx3(x3)
        weight3 = torch.view_as_complex(weight3.contiguous())  # 32 64 33

        x3 = x3 * weight3  # 1 32 64 33
        x3 = torch.fft.irfft2(x3, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64
        x2 = F.interpolate(x2, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        # b,c,h,w = x_1.size()
        # ampool = nn.AdaptiveAvgPool2d((h, w))
        # x2 = ampool(x2)
        # x3 = ampool(x3)
        x = torch.cat([x_1, x2,x3], dim=1) #1 64 64 64
        # x = torch.cat([x_1, x2,x3], dim=1) #1 64 64 64
        x = self.post_norm3(x) #1 64 64 64
        x = self.conv1(x)
        return x
class DDFA4_2(nn.Module):
    def __init__(self, dim, h=14, w=8,order=3):
        super().__init__()
        self.dims = [128 // 2 ** i for i in range(order)]
        self.dims.reverse()
        # dim1 = self.dims[0]
        # dim2 = self.dims[1]
        # dim3 = self.dims[2]
        self.dw = nn.Conv2d(128 , 128 , kernel_size=3, padding=1, bias=False, groups=128 // 2)
        self.complex_weight_x3 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02) #v5s
        self.complex_weight_x4 = nn.Parameter(torch.randn(64, h, w, 2, dtype=torch.float32) * 0.02)  #v5s
        # self.complex_weight_x3 = nn.Parameter(torch.randn(512, h, w, 2, dtype=torch.float32) * 0.02) #v5l
        # self.complex_weight_x4 = nn.Parameter(torch.randn(128, h, w, 2, dtype=torch.float32) * 0.02)  #v5l
        trunc_normal_(self.complex_weight_x3, std=.02)
        trunc_normal_(self.complex_weight_x4, std=.02)
        self.pre_norm1 = HorLayerNorm(128, eps=1e-6, data_format='channels_first')  #v5s
        # self.pre_norm1 = HorLayerNorm(256, eps=1e-6, data_format='channels_first') #v5l
        self.pre_norm2 = HorLayerNorm(64, eps=1e-6, data_format='channels_first') #v5s
        # self.pre_norm2 = HorLayerNorm(128, eps=1e-6, data_format='channels_first') #v5l
        self.pre_norm3 = HorLayerNorm(256, eps=1e-6, data_format='channels_first')#v5s
        # self.pre_norm3 = HorLayerNorm(512, eps=1e-6, data_format='channels_first')#v5l
        self.pre_norm4 = HorLayerNorm(128, eps=1e-6, data_format='channels_first')#v5s 256_>128
        # self.pre_norm4 = HorLayerNorm(512, eps=1e-6, data_format='channels_first')#v5l
        # self.post_norm3 = HorLayerNorm(1408, eps=1e-6, data_format='channels_first')
        # self.post_norm3 = HorLayerNorm(448, eps=1e-6, data_format='channels_first')
        self.post_norm3 = HorLayerNorm(576, eps=1e-6, data_format='channels_first')#v5s  704_>576
        self.conv1 = nn.Conv2d(576, 128, kernel_size=1, padding=0, bias=False)#v5s
        self.convx2 = ComplexConv2d(64, 64, kernel_size=1, padding=0, bias=False)
        self.convx3 = ComplexConv2d(256, 256, kernel_size=1, padding=0, bias=False)
        # self.conv1 = nn.Conv2d(1408, 256, kernel_size=1, padding=0, bias=False)#v5l
        # self.ATT = GAMAttention(256,256) #v5l
        self.ATT = GAMAttention(128,128) #v5s

        # self.conv1 = nn.Conv2d(448, 128, kernel_size=1, padding=0, bias=False)
    def forward(self, x):
        x_1 = x[0]   #  128
        x2 = x[1][1]  # 64
        x3 = x[2][2]   #256
        x4 = x[3][0]   #256
        x_1 = self.pre_norm1(x_1)
        x2 = self.pre_norm2(x2)
        x3 = self.pre_norm3(x3)
        x4 = self.pre_norm4(x4)
        # x_1 = self.dw(x_1) #1 32 64 64
        #x2
        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')  # 1 32 64 33
        weight4 = self.complex_weight_x4  # 32 14 8 2
        if not weight4.shape[1:3] == x2.shape[2:4]:
            weight4 = F.interpolate(weight4.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2

        weight4 = torch.view_as_complex(weight4.contiguous())  # 32 64 33

        x2 = x2 * weight4  # 1 32 64 33
        x2 = self.convx2(x2)
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64
        # x3
        x3 = x3.to(torch.float32)
        B, C, a, b = x3.shape
        x3 = torch.fft.rfft2(x3, dim=(2, 3), norm='ortho')  # 1 32 64 33
        weight3 = self.complex_weight_x3  # 32 14 8 2
        if not weight3.shape[1:3] == x3.shape[2:4]:
            weight3 = F.interpolate(weight3.permute(3, 0, 1, 2), size=x3.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2

        weight3 = torch.view_as_complex(weight3.contiguous())  # 32 64 33

        x3 = x3 * weight3  # 1 32 64 33
        x3 = self.convx3(x3)
        x3 = torch.fft.irfft2(x3, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64

        #x4
        # x4 = x4.to(torch.float32)
        # B, C, a, b = x4.shape
        # x4 = torch.fft.rfft2(x4, dim=(2, 3), norm='ortho')  # 1 32 64 33
        # weight4 = self.complex_weight_x4  # 32 14 8 2
        # if not weight4.shape[1:3] == x4.shape[2:4]:
        #     weight4 = F.interpolate(weight4.permute(3, 0, 1, 2), size=x4.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2
        #
        # weight4 = torch.view_as_complex(weight4.contiguous())  # 32 64 33
        #
        # x4 = x4 * weight4  # 1 32 64 33
        # x4 = torch.fft.irfft2(x4, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64
        x2 = F.interpolate(x2, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        x3 = F.interpolate(x3, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        x4 = F.interpolate(x4, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        # b,c,h,w = x_1.size()
        # ampool = nn.AdaptiveAvgPool2d((h, w))
        # ampool = nn.AdaptiveMaxPool2d((h, w))
        # x2 = ampool(x2)
        # x3 = ampool(x3)
        # x4 = ampool(x4)
        x = torch.cat([x_1, x2,x3,x4], dim=1) #1 64 64 64
        # x = torch.cat([x_1, x2,x3], dim=1) #1 64 64 64
        x = self.post_norm3(x) #1 64 64 64
        x = self.conv1(x)
        x = self.ATT(x)
        return x
class DDFA4_3(nn.Module):
    def __init__(self, dim, h=14, w=8,order=3):
        super().__init__()
        self.dims = [128 // 2 ** i for i in range(order)]
        self.dims.reverse()
        # dim1 = self.dims[0]
        # dim2 = self.dims[1]
        # dim3 = self.dims[2]
        self.dw = nn.Conv2d(128 , 128 , kernel_size=3, padding=1, bias=False, groups=128 // 2)
        # self.complex_weight_x3 = nn.Parameter(torch.randn(128, h, w, 2, dtype=torch.float32) * 0.02) #v5s
        #xingai
        self.complex_weight_x3 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02) #v5s
        # self.complex_weight_x4 = nn.Parameter(torch.randn(128, h, w, 2, dtype=torch.float32) * 0.02) #v5s
        #xingai
        self.complex_weight_x2 = nn.Parameter(torch.randn(64, h, w, 2, dtype=torch.float32) * 0.02) #v5s
        #
        # self.complex_weight_x3 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02) #v5l
        # self.complex_weight_x4 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02) #v5l

        trunc_normal_(self.complex_weight_x3, std=.02)
        # trunc_normal_(self.complex_weight_x4, std=.02)
        #xingai
        trunc_normal_(self.complex_weight_x2, std=.02)
        self.pre_norm1 = HorLayerNorm(128, eps=1e-6, data_format='channels_first')  #v5s
        # self.pre_norm1 = HorLayerNorm(256, eps=1e-6, data_format='channels_first') #v5l
        self.pre_norm2 = HorLayerNorm(64, eps=1e-6, data_format='channels_first')  #v5s
        # self.pre_norm2 = HorLayerNorm(64, eps=1e-6, data_format='channels_first')  #v5s
        self.pre_norm3 = HorLayerNorm(256, eps=1e-6, data_format='channels_first') #v5s
        # self.pre_norm3 = HorLayerNorm(256, eps=1e-6, data_format='channels_first') #v5l
        # self.pre_norm4 = HorLayerNorm(128, eps=1e-6, data_format='channels_first')#v5s
        # self.pre_norm4 = HorLayerNorm(256, eps=1e-6, data_format='channels_first')#v5l
        # self.post_norm3 = HorLayerNorm(416, eps=1e-6, data_format='channels_first')#v5s
        #xinjia416_>448
        self.post_norm3 = HorLayerNorm(448, eps=1e-6, data_format='channels_first')#v5s
        # self.post_norm3 = HorLayerNorm(832, eps=1e-6, data_format='channels_first')#v5l
        self.conv1 = nn.Conv2d(448, 128, kernel_size=1, padding=0, bias=False)#v5s
        self.convx2 = ComplexConv2d(64, 64, kernel_size=1, padding=0, bias=False)
        self.convx3 = ComplexConv2d(256, 256, kernel_size=1, padding=0, bias=False)
        # self.conv1 = nn.Conv2d(832, 256, kernel_size=1, padding=0, bias=False)#v5l
        self.ATT = GAMAttention(128,128)#v5s
        # self.ATT = GAMAttention(256,256)#v5l
    def forward(self, x):
        x_1 = x[0]   #  128
        #xingai
        x2 = x[1][1]  # 32_>64
        x3 = x[2][2]   #128>256
        # x4 = x[3][0]   #128
        x_1 = self.pre_norm1(x_1)
        x2 = self.pre_norm2(x2)
        x3 = self.pre_norm3(x3)
        # x4 = self.pre_norm4(x4)
        # x_1 = self.dw(x_1) #1 32 64 64

        # x2
        x3 = x3.to(torch.float32)
        B, C, a, b = x3.shape
        x3 = torch.fft.rfft2(x3, dim=(2, 3), norm='ortho')  # 1 32 64 33
        weight3 = self.complex_weight_x3  # 32 14 8 2
        if not weight3.shape[1:3] == x3.shape[2:4]:
            weight3 = F.interpolate(weight3.permute(3, 0, 1, 2), size=x3.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2

        weight3 = torch.view_as_complex(weight3.contiguous())  # 32 64 33

        x3 = x3 * weight3  # 1 32 64 33
        x3 = self.convx3(x3)
        x3 = torch.fft.irfft2(x3, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64

        #x4
        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')  # 1 32 64 33
        weight2 = self.complex_weight_x2  # 32 14 8 2
        if not weight2.shape[1:3] == x2.shape[2:4]:
            weight2 = F.interpolate(weight2.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2

        weight2 = torch.view_as_complex(weight2.contiguous())  # 32 64 33

        x2 = x2 * weight2  # 1 32 64 33
        x2 = self.convx2(x2)
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64

        #
        # x2 = F.interpolate(x2, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        # x3 = F.interpolate(x3, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        # x4 = F.interpolate(x4, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        b,c,h,w = x_1.size()
        # ampool = nn.AdaptiveAvgPool2d((h, w))
        ampool = nn.AdaptiveMaxPool2d((h, w))
        x2 = ampool(x2)
        x3 = ampool(x3)
        # x4 = ampool(x4)
        # x = torch.cat([x_1, x2,x3,x4], dim=1) #1 64 64 64
        #xingai
        x = torch.cat([x_1, x2,x3], dim=1) #1 64 64 64
        # x = torch.cat([x_1, x2,x3], dim=1) #1 64 64 64
        x = self.post_norm3(x) #1 64 64 64
        x = self.conv1(x)
        x = self.ATT(x)
        return x
# class GlobalLocalFilter4_3(nn.Module):
#     def __init__(self, dim, h=14, w=8,order=3):
#         super().__init__()
#         self.dims = [128 // 2 ** i for i in range(order)]
#         self.dims.reverse()
#         # dim1 = self.dims[0]
#         # dim2 = self.dims[1]
#         # dim3 = self.dims[2]
#         self.dw = nn.Conv2d(128 , 128 , kernel_size=3, padding=1, bias=False, groups=128 // 2)
#         self.complex_weight_x3 = nn.Parameter(torch.randn(128, h, w, 2, dtype=torch.float32) * 0.02) #v5s
#         self.complex_weight_x4 = nn.Parameter(torch.randn(128, h, w, 2, dtype=torch.float32) * 0.02) #v5s
#         #
#         # self.complex_weight_x3 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02) #v5l
#         # self.complex_weight_x4 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02) #v5l
#
#         trunc_normal_(self.complex_weight_x3, std=.02)
#         trunc_normal_(self.complex_weight_x4, std=.02)
#         self.pre_norm1 = HorLayerNorm(128, eps=1e-6, data_format='channels_first')  #v5s
#         # self.pre_norm1 = HorLayerNorm(256, eps=1e-6, data_format='channels_first') #v5l
#         self.pre_norm2 = HorLayerNorm(32, eps=1e-6, data_format='channels_first')  #v5s
#         # self.pre_norm2 = HorLayerNorm(64, eps=1e-6, data_format='channels_first')  #v5s
#         self.pre_norm3 = HorLayerNorm(128, eps=1e-6, data_format='channels_first') #v5s
#         # self.pre_norm3 = HorLayerNorm(256, eps=1e-6, data_format='channels_first') #v5l
#         self.pre_norm4 = HorLayerNorm(128, eps=1e-6, data_format='channels_first')#v5s
#         # self.pre_norm4 = HorLayerNorm(256, eps=1e-6, data_format='channels_first')#v5l
#         self.post_norm3 = HorLayerNorm(416, eps=1e-6, data_format='channels_first')#v5s
#         # self.post_norm3 = HorLayerNorm(832, eps=1e-6, data_format='channels_first')#v5l
#         self.conv1 = nn.Conv2d(416, 128, kernel_size=1, padding=0, bias=False)#v5s
#         # self.conv1 = nn.Conv2d(832, 256, kernel_size=1, padding=0, bias=False)#v5l
#         self.ATT = GAMAttention(128,128)#v5s
#         # self.ATT = GAMAttention(256,256)#v5l
#     def forward(self, x):
#         x_1 = x[0]   #  128
#         x2 = x[1][0]  # 32
#         x3 = x[2][1]   #256
#         x4 = x[3][0]   #128
#         x_1 = self.pre_norm1(x_1)
#         x2 = self.pre_norm2(x2)
#         x3 = self.pre_norm3(x3)
#         x4 = self.pre_norm4(x4)
#         # x_1 = self.dw(x_1) #1 32 64 64
#
#         # x2
#         x3 = x3.to(torch.float32)
#         B, C, a, b = x3.shape
#         x3 = torch.fft.rfft2(x3, dim=(2, 3), norm='ortho')  # 1 32 64 33
#         weight3 = self.complex_weight_x3  # 32 14 8 2
#         if not weight3.shape[1:3] == x3.shape[2:4]:
#             weight3 = F.interpolate(weight3.permute(3, 0, 1, 2), size=x3.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2
#
#         weight3 = torch.view_as_complex(weight3.contiguous())  # 32 64 33
#
#         x3 = x3 * weight3  # 1 32 64 33
#         x3 = torch.fft.irfft2(x3, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64
#
#         #x4
#         x4 = x4.to(torch.float32)
#         B, C, a, b = x4.shape
#         x4 = torch.fft.rfft2(x4, dim=(2, 3), norm='ortho')  # 1 32 64 33
#         weight4 = self.complex_weight_x4  # 32 14 8 2
#         if not weight4.shape[1:3] == x4.shape[2:4]:
#             weight4 = F.interpolate(weight4.permute(3, 0, 1, 2), size=x4.shape[2:4], mode='bilinear',align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2
#
#         weight4 = torch.view_as_complex(weight4.contiguous())  # 32 64 33
#
#         x4 = x4 * weight4  # 1 32 64 33
#         x4 = torch.fft.irfft2(x4, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64
#
#         #
#         # x2 = F.interpolate(x2, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
#         # x3 = F.interpolate(x3, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
#         # x4 = F.interpolate(x4, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
#         b,c,h,w = x_1.size()
#         # ampool = nn.AdaptiveAvgPool2d((h, w))
#         ampool = nn.AdaptiveMaxPool2d((h, w))
#         x2 = ampool(x2)
#         x3 = ampool(x3)
#         x4 = ampool(x4)
#         x = torch.cat([x_1, x2,x3,x4], dim=1) #1 64 64 64
#         # x = torch.cat([x_1, x2,x3], dim=1) #1 64 64 64
#         x = self.post_norm3(x) #1 64 64 64
#         x = self.conv1(x)
#         x = self.ATT(x)
#         return x
class DDFA4_4(nn.Module):
    def __init__(self, dim, h=14, w=8,order=3):
        super().__init__()
        self.dims = [256 // 2 ** i for i in range(order)]
        self.dims.reverse()
        # dim1 = self.dims[0]
        # dim2 = self.dims[1]
        # dim3 = self.dims[2]
        self.dw = nn.Conv2d(256 , 256 , kernel_size=3, padding=1, bias=False, groups=256 // 2)
        # self.complex_weight_x3 = nn.Parameter(torch.randn(128, h, w, 2, dtype=torch.float32) * 0.02) #v5s
        #xingai
        self.complex_weight_x3 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02) #v5s
        # self.complex_weight_x4 = nn.Parameter(torch.randn(128, h, w, 2, dtype=torch.float32) * 0.02) #v5s
        #xingai
        self.complex_weight_x2 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02) #v5s
        #
        # self.complex_weight_x3 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02) #v5l
        # self.complex_weight_x4 = nn.Parameter(torch.randn(256, h, w, 2, dtype=torch.float32) * 0.02) #v5l

        trunc_normal_(self.complex_weight_x3, std=.02)
        # trunc_normal_(self.complex_weight_x4, std=.02)
        #xingai
        trunc_normal_(self.complex_weight_x2, std=.02)
        self.pre_norm1 = HorLayerNorm(256, eps=1e-6, data_format='channels_first')  #v5s
        # self.pre_norm1 = HorLayerNorm(256, eps=1e-6, data_format='channels_first') #v5l
        self.pre_norm2 = HorLayerNorm(256, eps=1e-6, data_format='channels_first')  #v5s
        # self.pre_norm2 = HorLayerNorm(64, eps=1e-6, data_format='channels_first')  #v5s
        self.pre_norm3 = HorLayerNorm(256, eps=1e-6, data_format='channels_first') #v5s
        # self.pre_norm3 = HorLayerNorm(256, eps=1e-6, data_format='channels_first') #v5l
        # self.pre_norm4 = HorLayerNorm(128, eps=1e-6, data_format='channels_first')#v5s
        # self.pre_norm4 = HorLayerNorm(256, eps=1e-6, data_format='channels_first')#v5l
        # self.post_norm3 = HorLayerNorm(416, eps=1e-6, data_format='channels_first')#v5s
        #xinjia416_>448
        self.post_norm3 = HorLayerNorm(768, eps=1e-6, data_format='channels_first')#v5s
        # self.post_norm3 = HorLayerNorm(832, eps=1e-6, data_format='channels_first')#v5l
        self.conv1 = nn.Conv2d(768, 512, kernel_size=1, padding=0, bias=False)#v5s
        self.convx2 = ComplexConv2d(256, 256, kernel_size=1, padding=0, bias=False)
        self.convx3 = ComplexConv2d(256, 256, kernel_size=1, padding=0, bias=False)

        # self.conv1 = nn.Conv2d(832, 256, kernel_size=1, padding=0, bias=False)#v5l
        self.ATT = GAMAttention(512,512)#v5s
        # self.ATT = GAMAttention(256,256)#v5l

    def forward(self, x):
        x_1 = x[0]  # 1 256 8 8   #  128
        # xingai
        x2 = x[3][1]  # #1 256 8 8
        x3 = x[2][2]  ##1 256 16 16
        # x4 = x[3][0]   #128
        x_1 = self.pre_norm1(x_1)
        x2 = self.pre_norm2(x2)
        x3 = self.pre_norm3(x3)
        # x4 = self.pre_norm4(x4)
        # x_1 = self.dw(x_1) #1 32 64 64

        # x2
        x3 = x3.to(torch.float32)
        B, C, a, b = x3.shape
        x3 = torch.fft.rfft2(x3, dim=(2, 3), norm='ortho')  # 1 32 64 33
        weight3 = self.complex_weight_x3  # 32 14 8 2
        if not weight3.shape[1:3] == x3.shape[2:4]:
            weight3 = F.interpolate(weight3.permute(3, 0, 1, 2), size=x3.shape[2:4], mode='bilinear',
                                    align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2

        weight3 = torch.view_as_complex(weight3.contiguous())  # 32 64 33

        x3 = x3 * weight3  # 1 32 64 33
        x3 = self.convx3(x3)
        x3 = torch.fft.irfft2(x3, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64

        # x4
        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')  # 1 32 64 33
        weight2 = self.complex_weight_x2  # 32 14 8 2
        if not weight2.shape[1:3] == x2.shape[2:4]:
            weight2 = F.interpolate(weight2.permute(3, 0, 1, 2), size=x2.shape[2:4], mode='bilinear',
                                    align_corners=True).permute(1, 2, 3, 0)  # 32 64 33 2

        weight2 = torch.view_as_complex(weight2.contiguous())  # 32 64 33

        x2 = x2 * weight2  # 1 32 64 33
        x2 = self.convx2(x2)
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')  # 1 32 64 64

        #
        # x2 = F.interpolate(x2, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        # x3 = F.interpolate(x3, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        # x4 = F.interpolate(x4, size=x_1.shape[2:4], mode='bilinear', align_corners=True)
        b, c, h, w = x_1.size()
        # ampool = nn.AdaptiveAvgPool2d((h, w))
        ampool = nn.AdaptiveMaxPool2d((h, w))
        x2 = ampool(x2)
        x3 = ampool(x3)
        # x4 = ampool(x4)
        # x = torch.cat([x_1, x2,x3,x4], dim=1) #1 64 64 64
        # xingai
        x = torch.cat([x_1, x2, x3], dim=1)  # 1 64 64 64
        # x = torch.cat([x_1, x2,x3], dim=1) #1 64 64 64
        x = self.post_norm3(x)  # 1 64 64 64
        x = self.conv1(x)
        x = self.ATT(x)
        return x
