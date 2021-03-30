# Author: Jintao Huang
# Time: 2020-5-23
import torch


def to_categorical(x, num_classes=None):
    """转热码(已测试)  (./同torch_底层算法实现/)

    :param x: shape = (N,) or (...)
    :param num_classes: 默认 num_classes = max(x) + 1
    :return: shape = (N, num_classes) or (..., num_classes). float32"""

    assert x.dtype in (torch.int32, torch.int64), "x的类型只支持torch.int32与torch.int64"

    x_max = torch.max(x)
    num_classes = num_classes or x_max + 1
    assert num_classes >= x_max + 1, "num_classes 必须 >= max(x) + 1"
    return torch.eye(num_classes, dtype=torch.float, device=x.device)[x]


def focal_loss(y_pred, y_true, gamma=2):
    """f(x) = -(1 - x)^a * ln(x) = (1 - x)^a * CELoss(x)(已测试)

    :param y_pred: shape = (N, num_classes) or (..., num_classes)
    :param y_true: shape = (N,) or (...)"""
    y_pred = torch.clamp_min(torch.softmax(y_pred, dim=-1), 1e-6)
    y_true = to_categorical(y_true, y_pred.shape[-1])

    return torch.mean(torch.sum(y_true * -torch.log(y_pred) * (1 - y_pred) ** gamma, -1))


def binary_focal_loss(y_pred, y_true, gamma=2, with_logits=False):
    """f(x) = -(1 - x)^a * ln(x) = (1 - x)^a * CELoss(x)(已测试)

    :param y_pred: shape = (N,) or (...)
    :param y_true: shape = (N,) or (...)
    :param with_logits: y_pred是否未经过sigmoid"""
    if with_logits:
        y_pred = torch.sigmoid(y_pred)
    y_pred = torch.clamp(y_pred, 1e-6, 1 - 1e-6)
    # 前式与后式关于0.5对称(The former and the latter are symmetric about 0.5)
    return torch.mean(y_true * -torch.log(y_pred) * (1 - y_pred) ** gamma +
                      (1 - y_true) * -torch.log(1 - y_pred) * y_pred ** gamma)


def weighted_binary_focal_loss(y_pred, y_true, alpha=0.25, gamma=2, with_logits=False, reduction="mean"):
    """f(x) = -alpha * (1 - x)^a * ln(x) = alpha * (1 - x)^a * CELoss(x) (已测试)

    :param y_pred: shape = (N,) or (...)
    :param y_true: shape = (N,) or (...)
    :param alpha: 负样本与正样本的权重. The weight of the negative sample and the positive sample
        = alpha * positive + (1 - alpha) * negative
    :param with_logits: y_pred是否未经过sigmoid"""

    if reduction == "mean":
        func = torch.mean
    elif reduction == "sum":
        func = torch.sum
    else:
        raise ValueError("reduction should in ('mean', 'sum')")
    if with_logits:
        y_pred = torch.sigmoid(y_pred)
    y_pred = torch.clamp(y_pred, 1e-6, 1 - 1e-6)
    # 前式与后式关于0.5对称(The former and the latter are symmetric about 0.5)
    # y_true 为-1. 即: 既不是正样本、也不是负样本。
    return func((alpha * y_true * -torch.log(y_pred) * (1 - y_pred) ** gamma +
                 (1 - alpha) * (1 - y_true) * -torch.log(1 - y_pred) * y_pred ** gamma) * (y_true >= 0).float())
