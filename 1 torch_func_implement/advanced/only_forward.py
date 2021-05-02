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
        bias: bool = False
) -> nn.Conv2d:
    # 点的卷积(对全部C，一个点做卷积)
    return nn.Conv2d(in_channels, out_channels, 1, 1, 0, bias=bias)


# ------------------------------------ loss
def label_smoothing_cross_entropy(pred: Tensor, target: Tensor, smoothing: float = 0.1) -> Tensor:
    """reference: https://github.com/seominseok0429/label-smoothing-visualization-pytorch

    :param pred: shape(N, In). 未过softmax
    :param target: shape(N,)
    :param smoothing: float
    :return: shape()
    """
    pred = F.log_softmax(pred, dim=-1)
    ce_loss = F.nll_loss(pred, target)
    smooth_loss = -torch.mean(pred)
    return (1 - smoothing) * ce_loss + smoothing * smooth_loss


# pred = torch.tensor([[1, 0., 0.]])
# pred_prob = torch.softmax(pred, -1)
# target_s = torch.tensor([[0.9 + 1 / 30, 1 / 30, 1 / 30]])
# target = torch.tensor([[1, 0., 0.]])
# print(torch.sum(target * -torch.log(torch.softmax(pred, -1))))
# print(torch.sum(target_s * -torch.log(torch.softmax(pred, -1))))
# print(label_smoothing_cross_entropy(pred, torch.tensor([0]), 0.1))


def binary_focal_loss_with_digits(
        pred: torch.Tensor,
        target: torch.Tensor,
        alpha: float = 0.25,
        gamma: float = 1.5,
        reduction: str = "none",
) -> torch.Tensor:
    """Original implementation from https://github.com/facebookresearch/fvcore/blob/master/fvcore/nn/focal_loss.py .
    Loss used in RetinaNet for dense detection: https://arxiv.org/abs/1708.02002.
    公式：FocalLoss = alpha * (1 - p_t) ^ gamma * ce_loss. CELoss = -log(pred) * target

    :param pred: shape = (N,). 未过sigmoid
    :param target: shape = (N,)
    :param alpha: float. Weighting factor in range (0,1) to balance.
    :param gamma: float
    :param reduction: 'none' | 'mean' | 'sum'
    :return: shape = (N,) or ()
    """
    ce_loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")
    pred_prob = torch.sigmoid(pred)
    p_t = target * pred_prob + (1 - target) * (1 - pred_prob)
    alpha_t = target * alpha + (1 - target) * (1 - alpha)
    loss = alpha_t * ((1 - p_t) ** gamma) * ce_loss

    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()

    return loss


def binary_cross_entropy_blur_with_digits(pred: Tensor, target: Tensor, alpha: float = 0.05,
                                          reduction="mean") -> Tensor:
    """reference: https://github.com/ultralytics/yolov5/blob/master/utils/loss.py

    :param pred: shape(N,). 未过softmax
    :param target: shape(N,)
    :param alpha: float
    :param reduction: mean | sum | none
    :return: shape()
    """
    loss = F.binary_cross_entropy_with_logits(pred, target, reduction="none")  # (N,)
    pred = torch.sigmoid(pred)  # prob from logits
    dx = pred - target  # reduce only missing label effects
    # dx = (pred - target).abs()  # reduce missing label and false label effects
    alpha_factor = 1 - torch.exp((dx - 1) / (alpha + 1e-4))  # (N,)
    loss *= alpha_factor
    if reduction == "mean":
        loss = loss.mean()
    elif reduction == "sum":
        loss = loss.sum()
    return loss


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


# ------------------------------------ reuse
class Conv2dBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, groups, bias,
                 bn_eps=0.1, bn_momentum=1e-5, norm_layer=None):
        norm_layer = norm_layer or nn.BatchNorm2d
        super(Conv2dBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, groups, bias),
            norm_layer(out_channels, bn_eps, bn_momentum),
            nn.ReLU(inplace=True)
        )


class AnchorGenerator(nn.Module):
    def __init__(self, base_scale, scales=None, aspect_ratios=None, pyramid_levels=None):
        """

        :param base_scale: float. 基准尺度(anchor_size / stride)
        :param scales: tuple[float] / tuple[tuple[float]]. scales in single feature.
        :param aspect_ratios: tuple[float] / tuple[tuple[float]].
        :param pyramid_levels: tuple[int]
        """
        super(AnchorGenerator, self).__init__()
        pyramid_levels = pyramid_levels or (3, 4, 5, 6, 7)
        scales = scales or (1., 2 ** (1 / 3.), 2 ** (2 / 3.))
        # both scales and aspect_ratios of each pyramid layer can be controlled
        if not isinstance(scales[0], (list, tuple)):
            self.scales = (scales,) * len(pyramid_levels)
        aspect_ratios = aspect_ratios or (1., 0.5, 2.)
        if not isinstance(aspect_ratios[0], (list, tuple)):
            self.aspect_ratios = (aspect_ratios,) * len(pyramid_levels)
        # strides of Anchor at different pyramid levels
        self.strides = [2 ** i for i in pyramid_levels]
        self.base_scale = base_scale  # anchor_size / stride
        self.image_size = None  # int
        self.anchors = None

    def forward(self, x):
        """

        :param x: (images)Tensor[N, 3, H, W]. need: {.shape, .device, .dtype}
        :return: anchors: Tensor[X(F*H*W*A), 4]. (left, top, right, bottom)
        """
        image_size, dtype, device = x.shape[3], x.dtype, x.device
        if self.image_size == image_size:  # Anchors has been generated
            return self.anchors.to(device, dtype, copy=False)  # default: False
        else:
            self.image_size = image_size

        anchors_all = []
        for stride, scales, aspect_ratios in zip(self.strides, self.scales, self.aspect_ratios):
            anchors_level = []
            for scale in scales:
                for aspect_ratio in aspect_ratios:
                    if image_size % stride != 0:
                        raise ValueError('input size must be divided by the stride.')
                    base_anchor_size = self.base_scale * stride * scale
                    # anchor_h / anchor_w = aspect_ratio
                    anchor_h = base_anchor_size * aspect_ratio[0]
                    anchor_w = base_anchor_size * aspect_ratio[1]
                    shifts_x = torch.arange(stride / 2, image_size, stride, dtype=dtype, device=device)
                    shifts_y = torch.arange(stride / 2, image_size, stride, dtype=dtype, device=device)
                    shift_y, shift_x = torch.meshgrid(shifts_y, shifts_x)
                    shift_x = shift_x.reshape(-1)
                    shift_y = shift_y.reshape(-1)
                    shifts = torch.stack((shift_x, shift_y, shift_x, shift_y), dim=1)  # (X, 4)
                    # left, top, right, bottom. shape(X, 4)
                    anchors = shifts + torch.tensor([-anchor_w / 2, -anchor_h / 2, anchor_w / 2, anchor_h / 2],
                                                    dtype=dtype, device=device)[None]
                    anchors_level.append(anchors)

            anchors_level = torch.stack(anchors_level, dim=1).reshape(-1, 4)  # shape(X, A, 4) -> (-1, 4)
            anchors_all.append(anchors_level)
        self.anchors = torch.cat(anchors_all, dim=0)  # shape(-1, 4)
        return self.anchors
