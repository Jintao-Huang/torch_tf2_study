# Author: Jintao Huang
# Time: 2020-5-23

from torchvision.ops.boxes import box_area, box_iou
import torch


def _box_iou(boxes1, boxes2):
    """copy from torchvision.ops.boxes.box_iou()"""
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp(min=0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union
    return iou


def box_giou(boxes1, boxes2):
    area1 = box_area(boxes1)
    area2 = box_area(boxes2)

    lt_inner = torch.max(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    lt_outer = torch.min(boxes1[:, None, :2], boxes2[:, :2])  # [N, M, 2]
    rb_inner = torch.min(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]
    rb_outer = torch.max(boxes1[:, None, 2:], boxes2[:, 2:])  # [N, M, 2]

    wh_inner = (rb_inner - lt_inner).clamp(min=0)  # [N, M, 2]
    wh_outer = (rb_outer - lt_outer)  # [N, M, 2]
    # Compute intersection/union/complete shape(N, M)
    inter = wh_inner[:, :, 0] * wh_inner[:, :, 1]
    union = area1[:, None] + area2[None, :] - inter
    comp = wh_outer[:, :, 0] * wh_outer[:, :, 1]
    iou = inter / union
    giou = iou - (comp - union) / comp
    return giou
