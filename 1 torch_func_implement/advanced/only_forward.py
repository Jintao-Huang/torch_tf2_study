# Author: Jintao Huang
# Date: 

import torch
from torch import Tensor
from torch.nn import Parameter
import torch.nn as nn
import torch.nn.functional as F
from torchvision.ops.boxes import box_area, box_iou
import numpy as np
from torch.nn.init import xavier_uniform_, constant_
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
# ------------------------------------------
# for x in np.linspace(0, 10, 100):
#     pred = torch.tensor([[x, 0., 0., 0., 0., 0., 0.]])
#     pred_prob = torch.softmax(pred, -1)
#     # target_s = torch.tensor([[0.9 + 1 / 30, 1 / 30, 1 / 30]])
#     # target = torch.tensor([[1, 0., 0.]])
#     print(x, pred_prob[0, 0].item(),
#           F.cross_entropy(pred, torch.tensor([0])).item(),
#           label_smoothing_cross_entropy(pred, torch.tensor([0])).item(),
#           label_smoothing_cross_entropy(pred, torch.tensor([0]), 0.01).item())


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
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)

    lt = torch.max(boxes1[:, None, :2], boxes2[None, :, :2])  # [N, M, 2]
    rb = torch.min(boxes1[:, None, 2:], boxes2[None, :, 2:])  # [N, M, 2]

    wh = (rb - lt).clamp_min(0)  # [N, M, 2]
    inter = wh[:, :, 0] * wh[:, :, 1]  # [N, M]
    union = area1[:, None] + area2[None, :] - inter
    iou = inter / union
    return iou


def box_giou(boxes1, boxes2):
    """boxes1, boxes2 计算iou时，一一对应，与torchvision中box_iou计算方式不同
    GIoU: https://arxiv.org/pdf/1902.09630.pdf
    DIoU: https://arxiv.org/pdf/1911.08287.pdf
    CIoU: (https://arxiv.org/pdf/2005.03572.pdf).
    The consistency of the aspect ratio between the anchor boxes and the target boxes is also extremely important

    :param boxes1: Tensor[X, 4]
    :param boxes2: Tensor[X, 4]
    :return: Tensor[X]"""
    # 1. calculate iou
    area1 = _box_area(boxes1)
    area2 = _box_area(boxes2)

    lt_inner = torch.max(boxes1[:, :2], boxes2[:, :2])  # shape[X, 2] 内框
    rb_inner = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # shape[X, 2]
    wh_inner = (rb_inner - lt_inner).clamp_min(0)  # shape[X, 2]

    inter = wh_inner[:, 0] * wh_inner[:, 1]
    union = area1 + area2 - inter
    iou = inter / union  # [X]
    # 2. calculate 惩罚项
    lt_outer = torch.min(boxes1[:, :2], boxes2[:, :2])  # [X, 2] 外框
    rb_outer = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # [X, 2]
    wh_outer = rb_outer - lt_outer  # [X, 2]

    area_outer = wh_outer[:, 0] * wh_outer[:, 1]  # 外框面积  # [X]
    giou = iou - (area_outer - union) / area_outer  # [X]
    return giou


def _cal_distance2(dx, dy):
    """欧式距离的平方(Euclidean distance squared)"""
    return dx ** 2 + dy ** 2


def box_diou(boxes1, boxes2):
    """boxes1, boxes2 计算iou时，一一对应，与torchvision中box_iou计算方式不同
    GIoU: https://arxiv.org/pdf/1902.09630.pdf
    DIoU: https://arxiv.org/pdf/1911.08287.pdf
    CIoU: (https://arxiv.org/pdf/2005.03572.pdf).
    The consistency of the aspect ratio between the anchor boxes and the target boxes is also extremely important

    :param boxes1: Tensor[X, 4]
    :param boxes2: Tensor[X, 4]
    :return: Tensor[X]"""

    def _cal_distance2(dx, dy):
        """欧式距离的平方(Euclidean distance squared)"""
        return dx ** 2 + dy ** 2

    # 1. calculate iou
    wh_boxes1 = boxes1[:, 2:] - boxes1[:, :2]  # shape[X, 2].
    wh_boxes2 = boxes2[:, 2:] - boxes2[:, :2]
    area1 = wh_boxes1[:, 0] * wh_boxes1[:, 1]  # shape[X]
    area2 = wh_boxes2[:, 0] * wh_boxes2[:, 1]

    lt_inner = torch.max(boxes1[:, :2], boxes2[:, :2])  # shape[X, 2] 内框
    rb_inner = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # shape[X, 2]
    wh_inner = (rb_inner - lt_inner).clamp_min(0)  # shape[X, 2]

    inter = wh_inner[:, 0] * wh_inner[:, 1]
    union = area1 + area2 - inter
    iou = inter / union  # [X]

    # 2. calculate 惩罚项1(中心点距离)
    lt_outer = torch.min(boxes1[:, :2], boxes2[:, :2])  # [X, 2]  外框
    rb_outer = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # [X, 2]
    wh_outer = rb_outer - lt_outer  # [X, 2]
    lt_center = (boxes1[:, 2:] + boxes1[:, :2]) / 2  # [X, 2]  中心点框
    rb_center = (boxes2[:, 2:] + boxes2[:, :2]) / 2  # [X, 2]
    wh_center = lt_center - rb_center
    # dist2_outer: (外边框对角线距离的平方) The square of the diagonal distance of the outer border
    dist2_outer = _cal_distance2(wh_outer[:, 0], wh_outer[:, 1])  # [X]
    dist2_center = _cal_distance2(wh_center[:, 0], wh_center[:, 1])

    diou = iou - dist2_center / dist2_outer
    return diou


def box_ciou(boxes1, boxes2):
    """boxes1, boxes2 计算iou时，一一对应，与torchvision中box_iou计算方式不同
    GIoU: https://arxiv.org/pdf/1902.09630.pdf
    DIoU: https://arxiv.org/pdf/1911.08287.pdf
    CIoU: (https://arxiv.org/pdf/2005.03572.pdf).
    The consistency of the aspect ratio between the anchor boxes and the target boxes is also extremely important

    :param boxes1: Tensor[X, 4]
    :param boxes2: Tensor[X, 4]
    :return: Tensor[X]"""

    # 1. calculate iou
    wh_boxes1 = boxes1[:, 2:] - boxes1[:, :2]  # shape[X, 2].
    wh_boxes2 = boxes2[:, 2:] - boxes2[:, :2]
    area1 = wh_boxes1[:, 0] * wh_boxes1[:, 1]  # shape[X]
    area2 = wh_boxes2[:, 0] * wh_boxes2[:, 1]

    lt_inner = torch.max(boxes1[:, :2], boxes2[:, :2])  # shape[X, 2] 内框
    rb_inner = torch.min(boxes1[:, 2:], boxes2[:, 2:])  # shape[X, 2]
    wh_inner = (rb_inner - lt_inner).clamp_min(0)  # shape[X, 2]

    inter = wh_inner[:, 0] * wh_inner[:, 1]
    union = area1 + area2 - inter
    iou = inter / union  # [X]

    # 2. calculate 惩罚项1(中心点距离)
    lt_outer = torch.min(boxes1[:, :2], boxes2[:, :2])  # [X, 2]  外框
    rb_outer = torch.max(boxes1[:, 2:], boxes2[:, 2:])  # [X, 2]
    wh_outer = rb_outer - lt_outer  # [X, 2]
    lt_center = (boxes1[:, 2:] + boxes1[:, :2]) / 2  # [X, 2]  中心点框
    rb_center = (boxes2[:, 2:] + boxes2[:, :2]) / 2  # [X, 2]
    wh_center = lt_center - rb_center
    # dist2_outer: (外边框对角线距离的平方) The square of the diagonal distance of the outer border
    dist2_outer = _cal_distance2(wh_outer[:, 0], wh_outer[:, 1])  # [X]
    dist2_center = _cal_distance2(wh_center[:, 0], wh_center[:, 1])

    # 3. calculate 惩罚项2(aspect_ratios差距). 公式详见论文, 变量同论文
    v = (4 / np.pi ** 2) * \
        (torch.atan(wh_boxes1[:, 0] / wh_boxes1[:, 1]) -
         torch.atan(wh_boxes2[:, 0] / wh_boxes2[:, 1])) ** 2  # [X]
    with torch.no_grad():  # alpha为系数, 无需梯度
        alpha = v / (1 - iou + v)  # [X]
    # diou = iou - dist2_center / dist2_outer
    ciou = iou - dist2_center / dist2_outer - alpha * v  # [X]
    return ciou


# ------------------------------------ transformer
# 此处参考:
# 1. torch.nn.Transformer
# 2. https://arxiv.org/abs/1706.03762
# 3. http://nlp.seas.harvard.edu/2018/04/03/attention.html
# 为了避免与torch.nn中的函数或类混淆，自己复现的函数或类前加上`_`做区分
class _MultiheadAttention(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout_p):
        super(_MultiheadAttention, self).__init__()
        self.num_heads = num_heads
        self.dropout_p = dropout_p
        # project: 映射
        self.in_proj_list = nn.ModuleList([
            nn.Linear(embed_dim, embed_dim, True),  # query_proj
            nn.Linear(embed_dim, embed_dim, True),  # key_proj
            nn.Linear(embed_dim, embed_dim, True)  # value_proj
        ])
        self.out_proj = nn.Linear(embed_dim, embed_dim, True)

    def forward(self, query, key, value,
                key_padding_mask=None,
                need_weights=True, attn_mask=None):
        """

        :param query: shape[TL, N, E]
        :param key: shape[SL, N, E]
        :param value: shape[SL, N, E]
        :param key_padding_mask: shape[N, SL]
        :param need_weights: bool
        :param attn_mask: shape[TL, SL]
        :return: Tuple[output: shape[TL, N, E], output_weight: shape[N, TL, SL]]
        """
        num_heads = self.num_heads
        dropout_p = self.dropout_p
        training = self.training
        tgt_len, batch_size, embed_dim = query.shape
        src_len = key.shape[0]
        head_dim = embed_dim // num_heads  # 需要可以被整除, 此处不进行检查
        # shape[TL, N, E], shape[SL, N, E], shape[SL, N, E]
        query, key, value = self.in_proj_list[0](query), self.in_proj_list[1](key), self.in_proj_list[2](value)
        # shape[N * NH, TL, HD], shape[N * NH, SL, HD], shape[N * NH, SL, HD]
        query = query.contiguous().view(tgt_len, batch_size * num_heads, head_dim).transpose(0, 1)
        key = key.contiguous().view(src_len, batch_size * num_heads, head_dim).transpose(0, 1)
        value = value.contiguous().view(src_len, batch_size * num_heads, head_dim).transpose(0, 1)
        scale = 1 / math.sqrt(head_dim)
        query = query * scale
        # shape[N * NH, TL, SL]. the weights on the values
        attn_output_weights = query @ key.transpose(1, 2)
        if attn_mask is not None:  # TL, SL位置上. decoder层面. [TL, SL] or [N * NH, TL, SL]
            attn_output_weights.masked_fill_(attn_mask, float("-inf"))
        if key_padding_mask is not None:  # key上. 任务层面
            # shape[N, NH, TL, SL]
            attn_output_weights = attn_output_weights.view(batch_size, num_heads, tgt_len, src_len)
            attn_output_weights = attn_output_weights.masked_fill(
                key_padding_mask[:, None, None, :],  # [N, 1, 1, SL]
                float("-inf"),
            )
            # shape[N * NH, TL, SL]
            attn_output_weights = attn_output_weights.view(batch_size * num_heads, tgt_len, src_len)

        attn_output_weights = F.softmax(attn_output_weights, dim=-1)
        attn_output_weights = F.dropout(attn_output_weights, dropout_p, training)
        # shape[N * NH, TL, HD].
        attn_output = attn_output_weights @ value  # [N * NH, TL, SL] @ [N * NH, SL, HD]
        # shape[TL, N, E]
        attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, batch_size, embed_dim)
        attn_output = self.out_proj(attn_output)  # 此处已Concat. 直接全连接
        if need_weights:
            # [N, NH, TL, SL]
            attn_output_weights = attn_output_weights.view(batch_size, num_heads, tgt_len, src_len)
            # [N, TL, SL]
            attn_output_weights = torch.mean(attn_output_weights, 1)
            return attn_output, attn_output_weights
        else:
            return attn_output, None


class _TransformerEncoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout_p=0.1):
        super(_TransformerEncoderLayer, self).__init__()
        # sub_layer1
        self.self_attn = _MultiheadAttention(d_model, num_heads, dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        self.norm1 = nn.LayerNorm(d_model)
        # sub_layer2
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout2 = nn.Dropout(dropout_p)
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src: Tensor, src_mask=None, src_key_padding_mask=None):
        """

        :param src: shape[SL, N, E]
        :param src_mask: shape[SL, SL]
        :param src_key_padding_mask: shape[N, SL]
        :return: shape[SL, N, E]
        """
        # sub_layer1
        src0 = src
        src = self.self_attn(src, src, src, src_key_padding_mask, False, src_mask)[0]
        src = src0 + self.dropout1(src)
        src = self.norm1(src)
        # sub_layer2
        src0 = src
        src = self.ffn(src)
        src = src0 + self.dropout2(src)
        src = self.norm2(src)
        return src


class _TransformerDecoderLayer(nn.Module):
    def __init__(self, d_model, num_heads, dim_feedforward=2048, dropout_p=0.1):
        super(_TransformerDecoderLayer, self).__init__()
        # sub_layer1
        self.self_attn = _MultiheadAttention(d_model, num_heads, dropout_p=dropout_p)
        self.dropout1 = nn.Dropout(dropout_p)
        self.norm1 = nn.LayerNorm(d_model)
        # sub_layer2
        self.multihead_attn = _MultiheadAttention(d_model, num_heads, dropout_p=dropout_p)
        self.dropout2 = nn.Dropout(dropout_p)
        self.norm2 = nn.LayerNorm(d_model)
        # sub_layer3
        self.ffn = nn.Sequential(
            nn.Linear(d_model, dim_feedforward),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout_p),
            nn.Linear(dim_feedforward, d_model),
        )
        self.dropout3 = nn.Dropout(dropout_p)
        self.norm3 = nn.LayerNorm(d_model)

    def forward(self, tgt, memory,
                tgt_mask=None, memory_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """

        :param tgt: shape[TL, N, E]. embedding + positional encoding
        :param memory: shape[SL, N, E]. encoder的输出
        :param tgt_mask: shape[TL, TL]
        :param memory_mask: shape[TL, SL]
        :param tgt_key_padding_mask: shape[N, TL]
        :param memory_key_padding_mask: shape[N, SL]
        :return: shape[TL, N, E]. 未过linear 和 softmax
        """
        # sub_layer1
        tgt0 = tgt
        tgt = self.self_attn(tgt, tgt, tgt, tgt_key_padding_mask, False, tgt_mask)[0]
        tgt = tgt0 + self.dropout1(tgt)
        tgt = self.norm1(tgt)
        # sub_layer2
        tgt0 = tgt
        tgt = self.multihead_attn(tgt, memory, memory, memory_key_padding_mask, False, memory_mask)[0]
        tgt = tgt0 + self.dropout2(tgt)
        tgt = self.norm2(tgt)
        # sub_layer3
        tgt0 = tgt
        tgt = self.ffn(tgt)
        tgt = tgt0 + self.dropout3(tgt)
        tgt = self.norm3(tgt)
        return tgt


class _TransformerBackbone(nn.Module):
    """未加入embedding与positional encoding, 以及最后的Linear和softmax. 同torch.nn.Transformer"""

    def __init__(self, d_model=512, num_heads=8,
                 num_encoder_layers=6, num_decoder_layers=6,
                 dim_feedforward=2048, dropout_p=0.1):
        super(_TransformerBackbone, self).__init__()
        self.num_encoder_layers = num_encoder_layers
        self.num_decoder_layers = num_decoder_layers
        self.encoder_list = nn.ModuleList([
            _TransformerEncoderLayer(d_model, num_heads, dim_feedforward, dropout_p)
            for _ in range(num_encoder_layers)
        ])
        self.norm1 = nn.LayerNorm(d_model)
        self.decoder_list = nn.ModuleList([
            _TransformerDecoderLayer(d_model, num_heads, dim_feedforward, dropout_p)
            for _ in range(num_decoder_layers)
        ])
        self.norm2 = nn.LayerNorm(d_model)

    def forward(self, src, tgt,
                src_mask=None, tgt_mask=None,
                memory_mask=None, src_key_padding_mask=None,
                tgt_key_padding_mask=None, memory_key_padding_mask=None):
        """

        :param src: shape[SL, N, E]
        :param tgt: shape[TL, N, E]
        :param src_mask: shape[SL, SL]
        :param tgt_mask: shape[TL, TL]
        :param memory_mask: shape[TL, SL]
        :param src_key_padding_mask: shape[N, SL]
        :param tgt_key_padding_mask: shape[N, TL]
        :param memory_key_padding_mask: shape[N, SL]
        :return: shape[TL, N, E]. 未过linear 和 softmax
        """
        # encoder
        for i in range(self.num_encoder_layers):
            src = self.encoder_list[i](src, src_mask, src_key_padding_mask)
        memory = self.norm1(src)
        del src
        # decoder
        for i in range(self.num_encoder_layers):
            tgt = self.decoder_list[i](tgt, memory, tgt_mask, memory_mask,
                                       tgt_key_padding_mask, memory_key_padding_mask)
        tgt = self.norm2(tgt)
        return tgt


# torch.manual_seed(0)
# m0 = nn.Transformer()
# m1 = _TransformerBackbone()
# torch.manual_seed(0)
# src = torch.rand(10, 16, 512)
# tgt = torch.rand(20, 16, 512)
# src_mask = (torch.rand(10, 10) * 1.1).floor().bool()
# tgt_mask = (torch.rand(20, 20) * 1.1).floor().bool()
# memory_mask = (torch.rand(20, 10) * 1.1).floor().bool()
# src_key_padding_mask = (torch.rand(16, 10) * 1.1).floor().bool()
# tgt_key_padding_mask = (torch.rand(16, 20) * 1.1).floor().bool()
# memory_key_padding_mask = (torch.rand(16, 10) * 1.1).floor().bool()
# # 注意这里的结果不同，因为参数初始化(顺序)和dropout
# y0 = m0(src, tgt, src_mask, tgt_mask, memory_mask,
#         src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
# y1 = m1(src, tgt, src_mask, tgt_mask, memory_mask,
#         src_key_padding_mask, tgt_key_padding_mask, memory_key_padding_mask)
# print(y0)
# print(y1)


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
