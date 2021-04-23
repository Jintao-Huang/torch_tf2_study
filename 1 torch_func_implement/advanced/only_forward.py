# Author: Jintao Huang
# Date: 

import torch
from torch import Tensor
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import box_area, box_iou
import math


# ------------------------------------ activation

def swish(x: Tensor) -> Tensor:
    """

    :param x: shape(N, In)
    :return: shape = x.shape
    """
    return x * torch.sigmoid(x)


def mish(x: Tensor) -> Tensor:
    """

    :param x: shape(N, In)
    :return: shape = x.shape
    """
    return x * torch.tanh(F.softplus(x))


# ------------------------------------ layers
def depthwise_conv(
        in_channels: int,  # (in_channels == out_channels)
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
) -> nn.Conv2d:
    # 深度的卷积(对单个C，多个点(kernel_size)分开做卷积)
    return nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, groups=in_channels, bias=bias)


def pointwise_conv(
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = False
) -> nn.Conv2d:
    # 点的卷积(对全部C，一个点做卷积)
    return nn.Conv2d(in_channels, out_channels, 1, stride, padding, bias=bias)


# ------------------------------------ loss

def label_smoothing_cross_entropy(pred: Tensor, target: Tensor, smoothing: float = 0.1) -> Tensor:
    """reference: https://github.com/seominseok0429/label-smoothing-visualization-pytorch

    :param pred: shape(N, In). 未过softmax
    :param target: shape(N,)
    :param smoothing: float
    :return: shape()
    """
    pred = F.log_softmax(pred, dim=-1)
    nll_loss = F.nll_loss(pred, target)
    smooth_loss = -torch.mean(pred)
    return (1 - smoothing) * nll_loss + smoothing * smooth_loss


def weighted_binary_focal_loss(pred, target, alpha=0.25, gamma=2.):
    """f(x) = alpha * (1 - x)^a * -ln(pred). 已过sigmoid

    :param pred: shape = (N,)
    :param target: shape = (N,)
    :param alpha: float
        The weight of the negative sample and the positive sample. (alpha * positive + (1 - alpha) * negative)
    :param gamma: float
    :return: shape = ()"""
    # sum / mean
    return torch.sum(alpha * (1 - pred) ** gamma * target * torch.clamp_max(-torch.log(pred), 100) +
                     (1 - alpha) * pred ** gamma * (1 - target) * torch.clamp_max(-torch.log(1 - pred), 100))


# ------------------------------------ iou

def _box_area(boxes):
    """copy from torchvision.ops.boxes.box_area(). """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def _box_iou(boxes1, boxes2):
    """copy from torchvision.ops.boxes.box_iou().
    :return: range: [0., 1.]"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp_min(0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union
    return iou


def box_giou(boxes1, boxes2):
    """Generalized IoU (https://arxiv.org/pdf/1902.09630.pdf).
    Solved the situation where the IoU is zero, Cannot reverse transfer the gradient

    :return: range: [-1., 1.]"""
    # 1. calculate iou
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt_inner = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb_inner = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh_inner = (rb_inner - lt_inner).clamp_min(0)  # [N, M, 2]

    inter = wh_inner[:, :, 0] * wh_inner[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union
    # 2. calculate 惩罚项
    lt_outer = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb_outer = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh_outer = rb_outer - lt_outer  # [N, M, 2]

    enclose = wh_outer[:, :, 0] * wh_outer[:, :, 1]  # the smallest enclosing convex object C for A and B,
    giou = iou - (enclose - union) / enclose
    return giou


def _cal_distance2(dx, dy):
    """欧式距离的平方(Euclidean distance squared)"""
    return dx ** 2 + dy ** 2


def box_diou(boxes1, boxes2):
    """Distance IoU (https://arxiv.org/pdf/1911.08287.pdf).
    the overlapping area and the distance between the center points are considered at the same time

    :return: range: [-1., 1.]"""
    # 1. calculate iou
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt_inner = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb_inner = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh_inner = (rb_inner - lt_inner).clamp_min(0)  # [N, M, 2]

    inter = wh_inner[:, :, 0] * wh_inner[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union
    # 2. calculate 惩罚项(中心点距离)
    lt_outer = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb_outer = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh_outer = rb_outer - lt_outer  # [N, M, 2]
    center_boxes1 = (boxes1[:, 2:] + boxes1[:, :2]) / 2  # [N, 2]
    center_boxes2 = (boxes2[:, 2:] + boxes2[:, :2]) / 2  # [M, 2]
    dx_dy_center = center_boxes1[:, None] - center_boxes2[None, :]

    # dist2_outer: (外边框对角线距离的平方) The square of the diagonal distance of the outer border
    dist2_outer = _cal_distance2(wh_outer[:, :, 0], wh_outer[:, :, 1])
    dist2_center = _cal_distance2(dx_dy_center[:, :, 0], dx_dy_center[:, :, 1])
    diou = iou - dist2_center / dist2_outer
    return diou


def box_ciou(boxes1, boxes2):
    """Complete IoU Loss (https://arxiv.org/pdf/2005.03572.pdf).
    The consistency of the aspect ratio between the anchor boxes and the target boxes is also extremely important

    :return: range: [-1., 1.]"""

    # 1. calculate iou
    wh_boxes1 = boxes1[:, 2:] - boxes1[:, :2]  # shape[N, 2]. 防止重复计算(Prevent double counting)
    wh_boxes2 = boxes2[:, 2:] - boxes2[:, :2]
    area1 = wh_boxes1[:, 0] * wh_boxes1[:, 1]  # shape[N]
    area2 = wh_boxes2[:, 0] * wh_boxes2[:, 1]

    lt_inner = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb_inner = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh_inner = (rb_inner - lt_inner).clamp_min(0)  # [N, M, 2]

    inter = wh_inner[:, :, 0] * wh_inner[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union
    # 2. calculate 惩罚项1(中心点距离)
    lt_outer = torch.min(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb_outer = torch.max(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh_outer = rb_outer - lt_outer  # [N, M, 2]
    center_boxes1 = (boxes1[:, 2:] + boxes1[:, :2]) / 2  # [N, 2]
    center_boxes2 = (boxes2[:, 2:] + boxes2[:, :2]) / 2  # [M, 2]
    dx_dy_center = center_boxes1[:, None] - center_boxes2[None, :]

    # distance2_outer: (外边框对角线距离的平方) The square of the diagonal distance of the outer border
    distance2_outer = _cal_distance2(wh_outer[:, :, 0], wh_outer[:, :, 1])  #
    distance2_center = _cal_distance2(dx_dy_center[:, :, 0], dx_dy_center[:, :, 1])
    # 3. calculate 惩罚项2(aspect_ratios差距). 公式详见论文, 变量同论文
    v = (4 / math.pi ** 2) * \
        (torch.atan(wh_boxes1[:, 0] / wh_boxes1[:, 1])[:, None] -
         torch.atan(wh_boxes2[:, 0] / wh_boxes2[:, 1])[None, :]) ** 2
    alpha = (iou >= 0.5).float() * (v / (1 - iou + v))
    ciou = iou - distance2_center / distance2_outer - alpha * v
    return ciou
