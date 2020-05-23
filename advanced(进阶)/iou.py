# Author: Jintao Huang
# Time: 2020-5-23

from torchvision.ops.boxes import box_area, box_iou
import torch
import math


def _box_iou(boxes1, boxes2):
    """copy from torchvision.ops.boxes.box_iou()"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union
    return iou


def box_giou(boxes1, boxes2):
    """Generalized IoU (https://arxiv.org/pdf/1902.09630.pdf)

    Solved the situation where the IoU is zero, Cannot reverse transfer the gradient"""
    # 1. calculate iou
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt_inner = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb_inner = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh_inner = (rb_inner - lt_inner).clamp(min=0)  # [N, M, 2]

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


def box_diou(boxes1, boxes2):
    """Distance IoU (https://arxiv.org/pdf/1911.08287.pdf)

    the overlapping area and the distance between the center points
    are considered at the same time"""
    # 1. calculate iou
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt_inner = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb_inner = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh_inner = (rb_inner - lt_inner).clamp(min=0)  # [N, M, 2]

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

    def cal_distance2(dx, dy):
        """欧式距离的平方(Euclidean distance squared)"""
        return dx ** 2 + dy ** 2

    # dist2_outer: (外边框对角线距离的平方) The square of the diagonal distance of the outer border
    dist2_outer = cal_distance2(wh_outer[:, :, 0], wh_outer[:, :, 1])
    dist2_center = cal_distance2(dx_dy_center[:, :, 0], dx_dy_center[:, :, 1])
    diou = iou - dist2_center / dist2_outer
    return diou


def box_ciou(boxes1, boxes2):
    """Complete IoU Loss (https://arxiv.org/pdf/2005.03572.pdf)

    The consistency of the aspect ratio between
    the anchor boxes and the target boxes is also extremely important"""

    # 1. calculate iou
    wh_boxes1 = boxes1[:, 2:] - boxes1[:, :2]  # shape[N, 2]. 防止重复计算(Prevent double counting)
    wh_boxes2 = boxes2[:, 2:] - boxes2[:, :2]
    area1 = wh_boxes1[:, 0] * wh_boxes1[:, 1]  # shape[N]
    area2 = wh_boxes2[:, 0] * wh_boxes2[:, 1]

    lt_inner = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb_inner = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]
    wh_inner = (rb_inner - lt_inner).clamp(min=0)  # [N, M, 2]

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

    def cal_distance2(dx, dy):
        """欧式距离的平方(Euclidean distance squared)"""
        return dx ** 2 + dy ** 2

    # distance2_outer: (外边框对角线距离的平方) The square of the diagonal distance of the outer border
    distance2_outer = cal_distance2(wh_outer[:, :, 0], wh_outer[:, :, 1])  #
    distance2_center = cal_distance2(dx_dy_center[:, :, 0], dx_dy_center[:, :, 1])
    # 3. calculate 惩罚项2(aspect_ratios差距). 公式详见论文, 变量同论文
    v = (4 / math.pi ** 2) * \
        (torch.atan(wh_boxes1[:, 0] / wh_boxes1[:, 1])[:, None] -
         torch.atan(wh_boxes2[:, 0] / wh_boxes2[:, 1])[None, :]) ** 2
    alpha = (iou >= 0.5).float() * (v / (1 - iou + v))
    ciou = iou - distance2_center / distance2_outer - alpha * v
    return ciou
