# Author: Jintao Huang
# Time: 2020-5-17

# 为防止与torch中的函数搞混，自己实现的函数前会加上 `_`
import torch
import torch.nn.functional as F
from torch import Tensor
from typing import Tuple, List, Union


# --------------------------------------------------- activation

def _relu(x: Tensor) -> Tensor:
    """(F.relu(inplace=False))

    :param x: shape = (N, In) or (N, C, H, W)
    :return: shape = x.shape"""
    return torch.where(x > 0, x, torch.tensor(0.))
    # or:
    # return x * (x > 0).float()


def _leaky_relu(x: Tensor, negative_slope: float = 0.01) -> Tensor:
    """(F.leaky_relu(inplace=False))"""
    return torch.where(x > 0, x, negative_slope * x)


def _sigmoid(x: Tensor) -> Tensor:
    """sigmoid(F.sigmoid())"""

    return 1 / (1 + torch.exp(-x))


def _tanh(x: Tensor) -> Tensor:
    """(F.tanh())"""
    return (torch.exp(2 * x) - 1) / (torch.exp(2 * x) + 1)
    # or:
    # return (torch.exp(x) - torch.exp(-x)) / (torch.exp(x) + torch.exp(-x))


def _softmax(x: Tensor, dim: int) -> Tensor:
    """(F.softmax())

    :param x: shape = (N, In)
    :param dim: int. 一般dim设为-1(表示输出Tensor的和为1.的dim为哪个)
    :return: shape = x.shape"""
    # shape(N, In) / shape(N, 1), 若dim = -1
    return torch.exp(x) / torch.sum(torch.exp(x), dim, True)


# --------------------------------------------------- loss

def _one_hot(x: Tensor, num_classes: int = -1) -> Tensor:
    """(F.one_hot())

    :param x: shape = (N,), torch.long.
    :param num_classes: int. default: max(x) + 1
    :return: shape = (N, num_classes). torch.long"""
    if num_classes == -1:
        num_classes = torch.max(x) + 1
    return torch.eye(num_classes, dtype=torch.long, device=x.device)[x]


def _nll_loss(pred: Tensor, target: Tensor) -> Tensor:
    """(F.nll_loss())

    :param pred: shape = (N, In).
    :param target: shape = (N,) torch.long.
    :return: shape = ()
    """
    target = _one_hot(target, pred.shape[-1])
    return torch.mean(torch.sum(-pred * target, dim=-1))


def _cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    """交叉熵损失(F.cross_entropy())

    :param pred: shape = (N, In). 未过softmax
    :param target: shape = (N,) torch.long. 未过ont_hot
    :return: shape = ()"""

    pred = F.log_softmax(pred, dim=-1)
    return _nll_loss(pred, target)


def _binary_cross_entropy(pred: Tensor, target: Tensor) -> Tensor:
    """二元交叉熵损失(F.binary_cross_entropy())

    :param pred: shape = (N,)
    :param target: shape = (N,) torch.float32
    :return: shape = ()"""

    return torch.mean(torch.clamp_max(-torch.log(pred), 100) * target +  # 防止inf
                      torch.clamp_max(-torch.log(1 - pred), 100) * (1 - target))


def _binary_cross_entropy_with_logits(pred: Tensor, target: Tensor, pos_weight: Tensor = None) -> Tensor:
    """二元交叉熵损失(F.binary_cross_entropy_with_logits()). 未过sigmoid

    :param pred: shape = (N,)
    :param target: shape = (N,) torch.float32
    :param pos_weight: 正样本的权重. shape = () or num_classes. e.g. shape[20]
    :return: shape = ()"""
    pos_weight = 1. if pos_weight is None else pos_weight
    # F.logsigmoid(- pred)) 即 F.log(1 - F.sigmoid(y_pred))
    return torch.mean(-F.logsigmoid(pred) * target * pos_weight + -F.logsigmoid(-pred) * (1 - target))


# pred = torch.rand(100, 20)
# target = torch.rand(100, 20)
# print(F.binary_cross_entropy_with_logits(pred, target))
# print(F.binary_cross_entropy_with_logits(pred, target, pos_weight=torch.tensor(0.2)))
# print(_binary_cross_entropy_with_logits(pred, target))
# print(_binary_cross_entropy_with_logits(pred, target, pos_weight=torch.tensor(0.2)))


def _mse_loss(pred: Tensor, target: Tensor) -> Tensor:
    """均方误差损失(F.mse_loss())

    :param pred: shape = (N, In)
    :param target: shape = (N, In)
    :return: shape = ()"""

    return torch.mean((target - pred) ** 2)


def _smooth_l1_loss(pred: Tensor, target: Tensor, beta: float = 1.) -> Tensor:
    """(F.smooth_l1_loss())

    :param pred: shape(N, In)
    :param target: shape(N, In)
    :param beta: smooth线
    :return: ()"""

    diff = torch.abs(target - pred)
    return torch.mean(torch.where(diff < beta, 0.5 * diff ** 2 / beta, diff - 0.5 * beta))


# --------------------------------------------------- layers
def _batch_norm(x: Tensor, running_mean: Tensor, running_var: Tensor, weight: Tensor = None, bias: Tensor = None,
                training: bool = False, momentum: float = 0.1, eps: float = 1e-5) -> Tensor:
    """BN(F.batch_norm()). 对NHW做归一化.

    :param x: shape = (N, In) or (N, C, H, W)
    :param running_mean: shape = (In,) 或 (C,) 下同
    :param running_var:
    :param weight:
    :param bias:
    :param training:
    :param momentum: 动量实际为 1 - momentum. (同torch)
    :param eps:
    :return: shape = x.shape"""

    if training:
        if x.dim() == 2:
            _dim = (0,)
        elif x.dim() == 4:
            _dim = (0, 2, 3)
        else:
            raise ValueError("x dim error")
        mean = eval_mean = torch.mean(x, _dim)  # 总体 = 估计. shape = (In,) or (C,)
        eval_var = torch.var(x, _dim, unbiased=True)  # 无偏估计, x作为样本
        var = torch.var(x, _dim, unbiased=False)  # 用于标准化, x作为总体
        running_mean.data = (1 - momentum) * running_mean + momentum * eval_mean
        running_var.data = (1 - momentum) * running_var + momentum * eval_var  # 无偏估计
    else:
        mean = running_mean
        var = running_var
    # 2D时, mean.shape = (In,)
    # 4D时, mean.shape = (C, 1, 1)
    if x.dim() == 4:  # 扩维
        mean, var = mean[:, None, None], var[:, None, None]
        weight, bias = weight[:, None, None], bias[:, None, None]
    return (x - mean) * torch.rsqrt(var + eps) \
           * (weight if weight is not None else 1.) + (bias if bias is not None else 0.)
    # or: 以下为torch中源码实现方式
    # scale = weight * torch.rsqrt(var + eps)
    # bias = bias - mean * scale
    # return x * scale + bias


# x = torch.rand(2, 20, 7, 7)
# running_mean = torch.rand(20)
# running_var = torch.rand(20)
# weight = torch.randn(20)
# bias = torch.randn(20)
# print(F.batch_norm(x, running_mean, running_var, weight, bias, True, 0.1, 1e-5))
# print(_batch_norm(x, running_mean, running_var, weight, bias, True, 0.1, 1e-5))


def _layer_norm(x, normalized_shape, weight=None, bias=None, eps=1e-5):
    """(F.layer_norm()).

    :param x: shape = (*, *normalized_shape)
    :param normalized_shape:
    :param weight: shape = normalized_shape
    :param bias: shape = normalized_shape
    :param eps:
    :return: shape = x.shape"""

    _dim = list(range(x.dim() - len(normalized_shape), x.dim()))  # 与最后维度对齐

    mean = torch.mean(x, _dim, keepdim=True)  # shape = [*, 后补1]. (keep_dim)
    # x作为总体, 无需使用无偏估计. 注: 在BN中计算running_var需要使用无偏估计
    var = torch.var(x, _dim, unbiased=False, keepdim=True)

    return (x - mean) * torch.rsqrt(var + eps) \
           * (weight if weight is not None else 1.) + (bias if bias is not None else 0.)


# import torch.nn as nn
# x = torch.randn(16, 10, 512)
# layer = nn.LayerNorm((10, 512), 1e-5, True)
# weight, bias, eps, normalized_shape = layer.weight, layer.bias, layer.eps, layer.normalized_shape
# y1 = layer(x)
# y2 = F.layer_norm(x, normalized_shape, weight, bias, eps)
# y3 = _layer_norm(x, normalized_shape, weight, bias, eps)
# print(torch.all(torch.abs(y1 - y3) < 1e-5))  # tensor(True)
# print(torch.all(torch.abs(y2 - y3) < 1e-5))  # tensor(True)


def _dropout(x: Tensor, drop_p: float, training: bool) -> Tensor:
    """(F.dropout(inplace=False)).

    :param x: shape = (*)
    :param drop_p: float
    :param training: bool
    :return: shape = x.shape"""
    if not training or not drop_p:
        return x

    keep_p = 1 - drop_p
    keep_tensors = torch.floor(keep_p + torch.rand_like(x))
    x = x * keep_tensors
    return x / keep_p  # 保持均值不变


# import torch
#
# x = torch.randn(16, 10)
# torch.manual_seed(0)
# y1 = F.dropout(x, 0.1, True)
# torch.manual_seed(0)
# y2 = _dropout(x, 0.1, True)
# 结果不同，不知为何
# print(y1)
# print(y2)


def _zero_padding2d(x: Tensor, padding: int) -> Tensor:
    """零填充(F.pad())

    :param x: shape = (N, C, Hin, Win)
    :param padding: int
    :return: shape = (N, C, Hout, Wout)"""

    output = torch.zeros((*x.shape[:2],  # N, C
                          x.shape[-2] + 2 * padding,  # Hout
                          x.shape[-1] + 2 * padding), dtype=x.dtype, device=x.device)  # Wout
    h_out, w_out = output.shape[-2:]
    output[:, :, padding:h_out - padding, padding:w_out - padding] = x
    return output


def _max_pool2d(x: Tensor, kernel_size: int, stride: int = None, padding: int = 0,
                return_indices: bool = False) -> Tensor:
    """最大池化(F.max_pool2d()).

    :param x: shape = (N, C, Hin, Win)
    :param kernel_size: int
    :param stride: int = kernel_size
    :param padding: int
    :param return_indices: bool
    :return: shape = (N, C, Hout, Wout)"""
    stride = stride or kernel_size
    # Out = (In + 2*P − K) // S + 1
    # padding的0.不加入max()运算, 故如此设计
    output_h, output_w = (x.shape[2] + 2 * padding - kernel_size) // stride + 1, \
                         (x.shape[3] + 2 * padding - kernel_size) // stride + 1
    output = torch.empty((*x.shape[:2], output_h, output_w),
                         dtype=x.dtype, device=x.device)
    indices = torch.empty_like(output, dtype=torch.int64, device=x.device)
    for i in range(output.shape[2]):  # Hout
        for j in range(output.shape[3]):  # # Wout
            _h_start_o, _w_start_o = i * stride - padding, j * stride - padding
            _h_start, _w_start = max(0, _h_start_o), max(0, _w_start_o)
            _h_end, _w_end = min(x.shape[-2], _h_start_o + kernel_size), min(x.shape[-1], _w_start_o + kernel_size)
            h_pos, w_pos = slice(_h_start, _h_end), slice(_w_start, _w_end)
            output[:, :, i, j], indices[:, :, i, j] = torch.max(x[:, :, h_pos, w_pos].flatten(2), dim=-1)
            indices[:, :, i, j] = (h_pos.start + indices[:, :, i, j] // (w_pos.stop - w_pos.start)) * x.shape[-2] + \
                                  w_pos.start + indices[:, :, i, j] % (w_pos.stop - w_pos.start)

    return (output, indices) if return_indices else output


def _max_unpool2d(x: Tensor, indices: Tensor,
                  kernel_size: int, stride: int = None, padding: int = 0) -> Tensor:
    """(F.max_unpool2d())

    :param x: shape = (N, C, Hin, Win)
    :param indices: shape = (N, C, Hin, Win)
    :param kernel_size: int
    :param stride: int
    :param padding: int
    :return: shape = (N, C, Hout, Wout)"""
    stride = stride or kernel_size
    # O = S*(In-1) - 2*P + K
    output_h, output_w = stride * (x.shape[2] - 1) - 2 * padding + kernel_size, \
                         stride * (x.shape[3] - 1) - 2 * padding + kernel_size
    output = torch.zeros((*x.shape[:2], output_h, output_w),
                         dtype=x.dtype, device=x.device)
    for i in range(x.shape[2]):  # Hin
        for j in range(x.shape[3]):  # # Win
            output.flatten(2)[:, torch.arange(0, output.shape[1]), indices[:, :, i, j]] = x[:, :, i, j]
    return output


def _avg_pool2d(x: Tensor, kernel_size: int, stride: int = None, padding: int = 0) -> Tensor:
    """平均池化(F.avg_pool2d()).

    :param x: shape = (N, C, Hin, Win)
    :param kernel_size: int
    :param stride: int = kernel_size
    :param padding: int
    :return: shape = (N, C, Hout, Wout)"""

    stride = stride or kernel_size
    if padding:
        x = _zero_padding2d(x, padding)
    # Out = (In + 2*P − K) // S + 1
    # padding的0.不加入mean()运算, 故如此设计
    output_h, output_w = (x.shape[2] - kernel_size) // stride + 1, \
                         (x.shape[3] - kernel_size) // stride + 1
    output = torch.empty((*x.shape[:2], output_h, output_w),
                         dtype=x.dtype, device=x.device)
    for i in range(output.shape[2]):  # Hout
        for j in range(output.shape[3]):  # # Wout
            h_start, w_start = i * stride - padding, j * stride - padding
            h_pos, w_pos = slice(h_start, (h_start + kernel_size)), \
                           slice(w_start, (w_start + kernel_size))
            output[:, :, i, j] = torch.mean(x[:, :, h_pos, w_pos], dim=(-2, -1))
    return output


def _linear(x: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    """全连接层(F.linear())

    :param x: shape = (N, In)
    :param weight: shape = (Out, In)
    :param bias: shape = (Out,)
    :return: shape = (N, Out)"""

    return x @ weight.t() + (bias if bias is not None else 0.)


# x = torch.randn(10, 20)
# w = torch.randn(30, 20)
# print(_linear(x, w))

def _conv2d(x: Tensor, weight: Tensor, bias: Tensor = None, stride: int = 1, padding: int = 0) -> Tensor:
    """2d卷积(F.conv2d()). 点乘

    :param x: shape = (N, Cin, Hin, Win)
    :param weight: shape = (Cout, Cin, KH, KW)
    :param bias: shape = (Cout,)
    :param stride: int
    :param padding: int
    :return: shape = (N, Cout, Hout, Wout)
    """
    if padding:
        x = _zero_padding2d(x, padding)
    kernel_size = weight.shape[-2]
    # Out = (In + 2*P − K) // S + 1
    output_h, output_w = (x.shape[2] - kernel_size) // stride + 1, \
                         (x.shape[3] - kernel_size) // stride + 1
    output = torch.empty((x.shape[0], weight.shape[0], output_h, output_w),
                         dtype=x.dtype, device=x.device)
    for i in range(output.shape[2]):  # Hout
        for j in range(output.shape[3]):  # Wout
            h_start, w_start = i * stride, j * stride
            h_pos, w_pos = slice(h_start, (h_start + kernel_size)), \
                           slice(w_start, (w_start + kernel_size))

            output[:, :, i, j] = torch.sum(
                # N, Cout, Cin, KH, KW
                x[:, None, :, h_pos, w_pos] * weight[None, :, :, :, :], dim=(-3, -2, -1))

    return output + (bias[:, None, None] if bias is not None else 0)  # 后对齐


def _conv2d_2(x: Tensor, weight: Tensor, bias: Tensor = None, stride: int = 1, padding: int = 0) -> Tensor:
    """2d卷积(F.conv2d()). 矩阵乘

    :param x: shape = (N, Cin, Hin, Win)
    :param weight: shape = (Cout, Cin, KH, KW)
    :param bias: shape = (Cout,)
    :param stride: int
    :param padding: int
    :return: shape = (N, Cout, Hout, Wout)
    """
    if padding:
        x = _zero_padding2d(x, padding)
    kernel_size = weight.shape[-2]
    # Out = (In + 2*P − K) // S + 1
    output_h, output_w = (x.shape[2] - kernel_size) // stride + 1, \
                         (x.shape[3] - kernel_size) // stride + 1
    output = torch.empty((x.shape[0], weight.shape[0], output_h, output_w),
                         dtype=x.dtype, device=x.device)
    weight = weight.view(weight.shape[0], -1)
    for i in range(output.shape[2]):  # Hout
        for j in range(output.shape[3]):  # Wout
            h_start, w_start = i * stride, j * stride
            h_pos, w_pos = slice(h_start, (h_start + kernel_size)), \
                           slice(w_start, (w_start + kernel_size))
            data = x[:, :, h_pos, w_pos].contiguous().view(x.shape[0], -1, 1)
            # e.g. [Cout, Cin*KH*KW] @ [N, Cin*KH*KW, 1] -> [N, Cout, 1]
            output[:, :, i, j] = (weight @ data)[:, :, 0]
    return output + (bias[:, None, None] if bias is not None else 0)  # 后对齐


# 为啥那么慢...我也不知道...
# x = torch.randn(8, 64, 24, 40)
# weight = torch.randn(128, 64, 3, 3)
# bias = torch.randn(128)
# import time
# t = time.time()
# y1 = _conv2d(x, weight, bias, 1, 1)
# print(time.time() - t)  # 3.287348508834839
# t = time.time()
# y2 = _conv2d_2(x, weight, bias, 1, 1)
# print(time.time() - t)  # 0.16396212577819824
# t = time.time()
# y3 = F.conv2d(x, weight, bias, 1, 1)
# print(time.time() - t)  # 0.011989116668701172
# print(torch.all(torch.abs(y1 - y3) < 1e-3))  # tensor(True)
# print(torch.all(torch.abs(y2 - y3) < 1e-3))  # tensor(True)


def __conv2d(x: Tensor, weight: Tensor, bias: Tensor = None,
             stride: int = 1, padding: int = 0,
             dilation: int = 1, groups: int = 1) -> Tensor:
    """2d卷积(F.conv2d()) - 复杂版. 点乘

    :param x: shape = (N, Cin, Hin, Win)
    :param weight: shape = (groups * G_Cout, G_Cin, KH, KW).  假设KH=KW
    :param bias: shape = (Cout,)
    :param stride: int
    :param padding: int
    :param dilation: int. 膨胀卷积 / 空洞卷积
    :param groups: int. 分组卷积
    :return: shape = (N, Cout, Hout, Wout).
    """

    if padding:
        x = _zero_padding2d(x, padding)
    kernel_size = dilation * (weight.shape[-2] - 1) + 1

    # O = (I + 2*P - (D*(K-1)+1)) // S + 1
    output_h, output_w = (x.shape[2] - kernel_size) // stride + 1, \
                         (x.shape[3] - kernel_size) // stride + 1
    output = torch.empty((x.shape[0], weight.shape[0], output_h, output_w),
                         dtype=x.dtype, device=x.device)
    for g in range(groups):
        g_cout, g_cin = weight.shape[0] // groups, weight.shape[1]
        cin_pos = slice(g_cin * g, g_cin * (g + 1))
        cout_pos = slice(g_cout * g, g_cout * (g + 1))
        for i in range(output.shape[2]):  # Hout
            for j in range(output.shape[3]):  # # Wout
                h_start, w_start = i * stride, j * stride
                h_pos, w_pos = slice(h_start, (h_start + kernel_size), dilation), \
                               slice(w_start, (w_start + kernel_size), dilation)
                output[:, cout_pos, i, j] = torch.sum(
                    # N, G_Cout, G_Cin, KH, KW
                    x[:, None, cin_pos, h_pos, w_pos] * weight[None, cout_pos, :, :, :], dim=(-3, -2, -1))

    return output + (bias[:, None, None] if bias is not None else 0)  # 后对齐


def __conv2d_2(x: Tensor, weight: Tensor, bias: Tensor = None,
               stride: int = 1, padding: int = 0,
               dilation: int = 1, groups: int = 1) -> Tensor:
    """2d卷积(F.conv2d()) - 复杂版. 矩阵乘

    :param x: shape = (N, Cin, Hin, Win)
    :param weight: shape = (groups * G_Cout, G_Cin, KH, KW).  假设KH=KW
    :param bias: shape = (Cout,)
    :param stride: int
    :param padding: int
    :param dilation: int. 膨胀卷积 / 空洞卷积
    :param groups: int. 分组卷积
    :return: shape = (N, Cout, Hout, Wout).
    """

    if padding:
        x = _zero_padding2d(x, padding)
    kernel_size = dilation * (weight.shape[-2] - 1) + 1

    # O = (I + 2*P - (D*(K-1)+1)) // S + 1
    output_h, output_w = (x.shape[2] - kernel_size) // stride + 1, \
                         (x.shape[3] - kernel_size) // stride + 1
    output = torch.empty((x.shape[0], weight.shape[0], output_h, output_w),
                         dtype=x.dtype, device=x.device)
    g_cout, g_cin = weight.shape[0] // groups, weight.shape[1]
    weight = weight.view(weight.shape[0], -1)
    for g in range(groups):
        cin_pos = slice(g_cin * g, g_cin * (g + 1))
        cout_pos = slice(g_cout * g, g_cout * (g + 1))
        for i in range(output.shape[2]):  # Hout
            for j in range(output.shape[3]):  # # Wout
                h_start, w_start = i * stride, j * stride
                h_pos, w_pos = slice(h_start, (h_start + kernel_size), dilation), \
                               slice(w_start, (w_start + kernel_size), dilation)
                data = x[:, cin_pos, h_pos, w_pos].contiguous().view(x.shape[0], -1, 1)
                # e.g. [G_Cout, G_Cin*KH*KW] @ [N, G_Cin*KH*KW, 1] -> [N, G_Cout, 1]
                output[:, cout_pos, i, j] = (weight[cout_pos] @ data)[:, :, 0]

    return output + (bias[:, None, None] if bias is not None else 0)  # 后对齐


# x = torch.randn(8, 64, 24, 40)
# weight = torch.randn(128, 16, 3, 3)
# bias = torch.randn(128)
# import time
# t = time.time()
# y1 = __conv2d(x, weight, bias, 1, 1, 2, 4)
# print(time.time() - t)  # 1.5747883319854736
# t = time.time()
# y2 = __conv2d_2(x, weight, bias, 1, 1, 2, 4)
# print(time.time() - t)  # 0.2872316837310791
# t = time.time()
# y3 = F.conv2d(x, weight, bias, 1, 1, 2, 4)
# print(time.time() - t)  # 0.002991914749145508
# print(torch.all(torch.abs(y1 - y3) < 1e-3))  # tensor(True)
# print(torch.all(torch.abs(y2 - y3) < 1e-3))  # tensor(True)


def _conv_transpose2d(x: Tensor, weight: Tensor, bias: Tensor = None,
                      stride: int = 1, padding: int = 0) -> Tensor:
    """2d转置卷积(F.conv_transpose2d())
    (在torch底层实现时不采用这种方法，此方法便于学习、效率较低)

    :param x: shape = (N, Cin, Hin, Win)
    :param weight: shape = (Cin, Cout, KH, KW). 其中假设KH=KW
    :param bias: shape = (Cout,)
    :param stride: int
    :param padding: int
    :return: shape = (N, Cout, Hout, Wout)"""
    kernel_size = weight.shape[-2]
    # O = S*(In-1) - 2*P + K
    output_h, output_w = stride * (x.shape[2] - 1) + kernel_size, \
                         stride * (x.shape[3] - 1) + kernel_size
    output = torch.zeros((x.shape[0], weight.shape[1], output_h, output_w),
                         dtype=x.dtype, device=x.device)
    for i in range(x.shape[2]):  # Hin
        for j in range(x.shape[3]):  # # Win
            h_start, w_start = i * stride, j * stride
            h_pos, w_pos = slice(h_start, (h_start + kernel_size)), \
                           slice(w_start, (w_start + kernel_size))
            # N, Cin, Cout, KH, KW
            output[:, :, h_pos, w_pos] += torch.sum(
                x[:, :, None, i:i + 1, j:j + 1] * weight[None, :, :, :, :], dim=1)
    if bias is not None:
        output += bias[:, None, None]
    return output if padding == 0 else output[:, :, padding:-padding, padding:-padding]


def _nearest_interpolate(x: Tensor, size: Tuple[int, int] = None, scale_factor: float = None) -> Tensor:
    """最近邻插值(F.interpolate(mode="nearest")). 与torch实现相同，与cv实现是否相同未知

    :param x: shape = (N, C, Hin, Win) - 像素点当作点来看待(与bilinear不同)
    :param size: Tuple(Hout, Wout)
    :param scale_factor: size和scale_factor必须且只能提供其中的一个参数
    :return: shape = (N, C, Hout, Wout)
    """
    in_size = x.shape[-2:]
    if scale_factor:
        size = int(in_size[0] * scale_factor), int(in_size[1] * scale_factor)  # out_size
    step_h, step_w = in_size[0] / size[0], in_size[1] / size[1]  # 步长
    axis_h = torch.arange(0, in_size[0], step_h, device=x.device).long()  # h坐标轴 floor
    axis_w = torch.arange(0, in_size[1], step_w, device=x.device).long()  # w坐标轴
    grid_h, grid_w = torch.meshgrid(axis_h, axis_w)  # 生成网格
    output = x[:, :, grid_h, grid_w]

    return output


def _bilinear_interpolate(x: Tensor, size: Tuple[int, int] = None, scale_factor: float = None,
                          align_corners: bool = False) -> Tensor:
    """双线性插值(F.interpolate(mode="bilinear"))

    :param x: shape = (N, C, Hin, Win)
    :param size: Tuple[Hout, Wout]
    :param scale_factor: size和scale_factor必须且只能提供其中的一个参数
    :param align_corners: 像素点当作像素方块来看待(与nearest不同)
        False: 输入和输出张量按其角像素的角点对齐。超过边界的值，插值使用边缘值填充
        True(保留角像素的值): 输入和输出张量按其角像素的中心点对齐
    :return: shape = (N, C, Hout, Wout)
    """

    in_size = x.shape[-2:]
    if scale_factor:
        size = int(in_size[0] * scale_factor), int(in_size[1] * scale_factor)  # out_size
    step_h, step_w = in_size[0] / size[0], in_size[1] / size[1]
    if align_corners:  # 角像素的中心点对齐(保留角像素的值)
        axis_h = torch.linspace(0, in_size[0] - 1, size[0], device=x.device)  # h坐标轴
        axis_w = torch.linspace(0, in_size[1] - 1, size[1], device=x.device)  # w坐标轴
    else:  # 角像素的角点对齐
        axis_h = torch.linspace(-0.5 + step_h / 2, - 0.5 + in_size[0] - step_h / 2, size[0], device=x.device)
        axis_w = torch.linspace(-0.5 + step_w / 2, - 0.5 + in_size[1] - step_w / 2, size[1], device=x.device)
    grid_h, grid_w = torch.meshgrid(axis_h, axis_w)  # 生成网格
    # if not align_corners:  # 超过边界的值，插值使用边缘值填充
    # 理论上align_corners == True时不需要截断，但是linespace会有误差，导致有时候过ceil()后索引时会越界，所以都加上
    grid_h.clamp_(0, in_size[0] - 1)
    grid_w.clamp_(0, in_size[1] - 1)
    # 以下6个张量都是2D的, shape(Hout * Wout, Hout * Wout)
    grid_h_f, grid_w_f = grid_h.long(), grid_w.long()  # floor
    grid_h_c, grid_w_c = grid_h.ceil().long(), grid_w.ceil().long()  # ceil
    offset_h, offset_w = grid_h - grid_h_f.float(), grid_w - grid_w_f.float()  # 与floor的偏离量
    # 左上角, 右上角, 左下角, 右下角
    output = (1 - offset_h) * (1 - offset_w) * x[:, :, grid_h_f, grid_w_f] + \
             (1 - offset_h) * offset_w * x[:, :, grid_h_f, grid_w_c] + \
             offset_h * (1 - offset_w) * x[:, :, grid_h_c, grid_w_f] + \
             offset_h * offset_w * x[:, :, grid_h_c, grid_w_c]
    return output


def _adaptive_avg_pool2d(x: Tensor, output_size: int) -> Tensor:
    """自适应的平均池化(F.adaptive_avg_pool2d())

    :param x: shape = (N, Cin, Hin, Win)
    :param output_size: int. 简化
    :return: shape = (N, Cin, output_size, output_size)"""

    # 切成output_size[0]个区间
    split_h = torch.linspace(0, x.shape[-2], output_size + 1)
    split_w = torch.linspace(0, x.shape[-1], output_size + 1)
    output = torch.empty((*x.shape[:-2], output_size, output_size), dtype=x.dtype, device=x.device)

    for i in range(output.shape[-2]):
        for j in range(output.shape[-1]):
            pos_h = slice(split_h[i].int().item(), split_h[i + 1].ceil().int().item())
            pos_w = slice(split_w[j].int().item(), split_w[j + 1].ceil().int().item())
            output[:, :, i, j] = torch.mean(x[:, :, pos_h, pos_w], dim=(-2, -1))
    return output


def _adaptive_max_pool2d(x: Tensor, output_size: int, return_indices: bool = False) -> Tensor:
    """自适应的最大池化(F.adaptive_max_pool2d())

    :param x: shape = (N, Cin, Hin, Win)
    :param output_size: int. 简化
    :param return_indices: bool
    :return: shape = (N, Cin, output_size[0], output_size[1])"""

    # 切成output_size[0]个区间
    split_h = torch.linspace(0, x.shape[-2], output_size + 1)
    split_w = torch.linspace(0, x.shape[-1], output_size + 1)
    output = torch.empty((*x.shape[:-2], output_size, output_size), dtype=x.dtype, device=x.device)
    indices = torch.empty_like(output, dtype=torch.int64, device=x.device)
    for i in range(output.shape[-2]):
        for j in range(output.shape[-1]):
            # 后用ceil()是为了使每个格子都加入运算
            h_pos = slice(split_h[i].int().item(), split_h[i + 1].ceil().int().item())
            w_pos = slice(split_w[j].int().item(), split_w[j + 1].ceil().int().item())
            output[:, :, i, j], indices[:, :, i, j] = torch.max(x[:, :, h_pos, w_pos].flatten(2), dim=-1)
            indices[:, :, i, j] = (h_pos.start + indices[:, :, i, j] // (w_pos.stop - w_pos.start)) * x.shape[-2] + \
                                  w_pos.start + indices[:, :, i, j] % (w_pos.stop - w_pos.start)
    return (output, indices) if return_indices else output


def _rnn_tanh_cell(x: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor,
                   b_ih: Tensor = None, b_hh: Tensor = None) -> Tensor:
    """torch.rnn_tanh_cell()

    :param x: shape = (N, In)
    :param hx: shape = (N, Out)
    :param w_ih: shape = (Out, In)
    :param w_hh: shape = (Out, Out)
    :param b_ih: shape = (Out,)
    :param b_hh: shape = (Out,)
    :return: shape = (N, Out)"""
    # y_i / hx_i+1 = tanh(x_i @ w_ih^T + b_ih + hx_i @ w_hh^T + b_hh)
    if hx is None:
        hx = torch.zeros(x.shape[0], w_ih.shape[0])  # w_ih.shape[0]: Out
    return torch.tanh(x @ w_ih.t() + (b_ih if b_ih is not None else 0.) +
                      hx @ w_hh.t() + (b_hh if b_hh is not None else 0.))


def _rnn_tanh(x: Tensor, hx: Tensor, params: List[Tensor], has_biases: bool) \
        -> Tuple[Tensor, Tensor]:
    """(复现: 类似于torch.rnn_tanh(num_layers=1)) - 已简化(num_layers=1). NL = 1
    规定: dropout: float = 0., train: bool = ..., bidirectional: bool = False, batch_first: bool = False
    使用e.g.: _rnn_tanh(x, hx, [w_ih, w_hh, b_ih, b_hh], True)

    :param x: shape = (L, N, In). Len of Sentence
    :param hx: shape = (NL, N, Out). num of layers
    :param params: [w0_ih, w0_hh, b0_ih, b0_hh] shape = [(Out, In), (Out, Out), (Out,), (Out), (Out, Out)]
    :param has_biases: 若为False: 则len(params)应为2 * NL. True: len(params)应为4 * NL
    :return: (y: shape(L, N, Out), hy: shape(NL, N, Out))
    """

    y = []  # 一层的输出
    hx_l0 = hx[0]
    w_ih, w_hh, b_ih, b_hh = params if has_biases else (*params, None, None)
    for i in range(x.shape[0]):
        hx_l0 = _rnn_tanh_cell(x[i], hx_l0, w_ih, w_hh, b_ih, b_hh)
        y.append(hx_l0)
    y = torch.stack(y)
    return y, y[-1][None]


# x = torch.randn(5, 3, 10)  # (L, N, In)
# hx = torch.randn(1, 3, 20)  # (NL, N, Out)
# rnn = nn.RNN(10, 20, 1)
# output, hn = rnn(x, hx)
# output2, hn2 = _rnn_tanh(x, hx, [rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0], True)
# print(torch.all(torch.abs(output - output2) < 1e-6))  # True
# print(torch.all(torch.abs(hn - hn2) < 1e-6))  # True


def __rnn_tanh(x: Tensor, hx: Tensor, params: List[Tensor], has_biases: bool, num_layers: int,
               dropout: float = 0., train: bool = True,
               bidirectional: bool = False) -> Tuple[Tensor, Tensor]:
    """(复现: torch.rnn_tanh()) - 复杂版
    使用e.g.: y1, hy1 = __rnn_tanh(x, hx, [w_ih, w_hh, b_ih, b_hh,
                               w_ih_r, w_hh_r, b_ih_r, b_hh_r,
                               w_ih1, w_hh1, b_ih1, b_hh1,
                               w_ih1_r, w_hh1_r, b_ih1_r, b_hh1_r
                               ], True, 2, 0.2, True, True, True)

    :param x: shape = (L, N, In)
    :param hx: shape = (NL * ND, N, Out). ND: num of direction
    :param params: [w0_ih, w0_hh, b0_ih, b0_hh, w1_ih, ...] shape = [(Out, In), (Out, Out), (Out,), (Out), (Out, Out), ...]
        or [w0_ih, ..., w0_ih_r, ..., w1_ih,...] shape = [(Out, In), ..., (Out, In), ...(Out, Out), ...]
    :param has_biases: 若为False: 则len(params)应为NL * ND * 2. True: len(params)应为NL * ND * 4
    :param num_layers: NL
    :param dropout: 除最后一层
    :param train:
    :param bidirectional: 若False: ND = 1; True: ND = 2.
    :return: (y: shape(L, N, ND * Out), hy: shape(NL * ND, N, Out))
    """
    hy = []  # 存储每层最后的hy/hy_r
    for i in range(num_layers):
        if bidirectional:
            hx_li, hx_r_li = hx[2 * i:2 * i + 1], hx[2 * i + 1:2 * (i + 1)]
            params_li, params_li_r = (params[8 * i:8 * i + 4], params[8 * i + 4:8 * (i + 1)]) if has_biases \
                else (params[4 * i:4 * i + 2], params[4 * i + 2:4 * (i + 1)])
            y, _ = _rnn_tanh(x, hx_li, params_li, has_biases)
            y_r, _ = _rnn_tanh(torch.flip(x, (0,)), hx_r_li, params_li_r, has_biases)
            x = torch.cat([y, torch.flip(y_r, (0,))], dim=-1)
            hy += [y[-1], y_r[-1]]
        else:
            hx_li = hx[i:i + 1]
            params_li = params[4 * i:4 * (i + 1)] if has_biases else params[2 * i:2 * (i + 1)]
            x, _ = _rnn_tanh(x, hx_li, params_li, has_biases)
            hy.append(x[-1])
        if dropout and i + 1 != num_layers:
            x = F.dropout(x, dropout, train)
    hy = torch.stack(hy)
    return x, hy


# x = torch.randn(5, 3, 10)  # (L, N, In)
# hx = torch.randn(4, 3, 20)  # (NL, N, Out)
# rnn = nn.RNN(10, 20, 2, True, False, 0.2, True)
# torch.manual_seed(0)
# output, hn = rnn(x, hx)
# torch.manual_seed(0)
# output2, hn2 = __rnn_tanh(x, hx, [
#     rnn.weight_ih_l0, rnn.weight_hh_l0, rnn.bias_ih_l0, rnn.bias_hh_l0,
#     rnn.weight_ih_l0_reverse, rnn.weight_hh_l0_reverse, rnn.bias_ih_l0_reverse, rnn.bias_hh_l0_reverse,
#     rnn.weight_ih_l1, rnn.weight_hh_l1, rnn.bias_ih_l1, rnn.bias_hh_l1,
#     rnn.weight_ih_l1_reverse, rnn.weight_hh_l1_reverse, rnn.bias_ih_l1_reverse, rnn.bias_hh_l1_reverse
# ],  True, 2, 0.2, True, True)
# print(torch.all(torch.abs(output - output2) < 1e-6))  # True
# print(torch.all(torch.abs(hn - hn2) < 1e-6))  # True


def _lstm_cell(x: Tensor, hx: Union[Tuple[Tensor, ...], List[Tensor]],
               w_ih: Tensor, w_hh: Tensor,
               b_ih: Tensor = None, b_hh: Tensor = None) -> Tuple[Tensor, Tensor]:
    """torch.lstm_cell()

    :param x: shape = (N, Cin).
    :param hx: Tuple(shape[N, Ch], shape[N, Ch])
    :param w_ih: shape = (Ch * 4, Cin). (i, f, g, o)
    :param w_hh: shape = (Ch * 4, Ch)
    :param b_ih: shape = (Ch * 4,)
    :param b_hh: shape = (Ch * 4,)
    :return: Tuple(y/h_1: shape[N, Ch], c_1: shape[N, Ch])
    """
    # i = sigmoid(x_i @ Wii^T + bii + h_i @ Whi^T + bhi)
    # f = sigmoid(x_i @ Wif^T + bif + h_i @ Whf^T + bhf)
    # g = tanh(x_i @ Wig^T + big + h_i @ Whg^T + bhg)
    # o = sigmoid(x_i @ Wio^T + bio + h_i @ Who^T + bho)
    # c_i+1 = f * c_i + i * g   # Hadamard乘积(点乘)
    # y_i / h_i+1 = o * tanh(c_i+1)
    c_hide = w_ih.shape[0] // 4  # Ch(channels_hide)
    h, c = hx
    if h is None:
        h = torch.zeros(x.shape[0], c_hide)  # weight[0].shape[0]: Ch
    if c is None:
        c = torch.zeros(x.shape[0], c_hide)
    i = torch.sigmoid(x @ w_ih[0:c_hide].t() + (b_ih[0:c_hide] if b_ih is not None else 0) +
                      h @ w_hh[0:c_hide].t() + (b_hh[0:c_hide] if b_hh is not None else 0))
    f = torch.sigmoid(
        x @ w_ih[c_hide:c_hide * 2].t() + (b_ih[c_hide:c_hide * 2] if b_ih is not None else 0) +
        h @ w_hh[c_hide:c_hide * 2].t() + (b_hh[c_hide:c_hide * 2] if b_hh is not None else 0))
    g = torch.tanh(
        x @ w_ih[c_hide * 2:c_hide * 3].t() + (b_ih[c_hide * 2:c_hide * 3] if b_ih is not None else 0) +
        h @ w_hh[c_hide * 2:c_hide * 3].t() + (b_hh[c_hide * 2:c_hide * 3] if b_hh is not None else 0))
    o = torch.sigmoid(
        x @ w_ih[c_hide * 3:c_hide * 4].t() + (b_ih[c_hide * 3:c_hide * 4] if b_ih is not None else 0) +
        h @ w_hh[c_hide * 3:c_hide * 4].t() + (b_hh[c_hide * 3:c_hide * 4] if b_hh is not None else 0))
    c_1 = f * c + i * g
    h_1 = o * torch.tanh(c_1)
    return h_1, c_1


def _gru_cell(x: Tensor, hx: Tensor, w_ih: Tensor, w_hh: Tensor,
              b_ih: Tensor = None, b_hh: Tensor = None) -> Tensor:
    """torch.gru_cell()

    :param x: shape = (N, Cin). 或称为input
    :param hx: shape = (N, Ch)
    :param w_ih: shape = (Ch * 3, Cin). (r, z, n)
    :param w_hh: shape = (Ch * 3, Ch)
    :param b_ih: shape = (Ch * 3,)
    :param b_hh: shape = (Ch * 3,)
    :return: y/hx_1: shape = (N, Ch)
    """
    c_hide = w_ih[0].shape[0] // 3  # Ch(channels_hide)
    if hx is None:
        hx = torch.zeros(x.shape[0], c_hide)  # weight[0].shape[0]: Ch

    # r = sigmoid(x_i @ Wir^T + bir + h_i @ Whr^T + bhr)  
    # z = sigmoid(x_i @ Wiz^T + biz + h_i @ Whz^T + bhz)  
    # n = tanh(x_i @ Win^T + bin + r*(h_i @ Whn^T + bhn))  
    # y_i / h_i+1 = (1 − z) * n + z * h_i
    r = torch.sigmoid(x @ w_ih[0:c_hide].t() + (b_ih[0:c_hide] if b_ih is not None else 0) +
                      hx @ w_hh[0:c_hide].t() + (b_hh[0:c_hide] if b_hh is not None else 0))
    z = torch.sigmoid(x @ w_ih[c_hide:c_hide * 2].t() + (b_ih[c_hide:c_hide * 2] if b_ih is not None else 0) +
                      hx @ w_hh[c_hide:c_hide * 2].t() + (b_hh[c_hide:c_hide * 2] if b_hh is not None else 0))
    n = torch.tanh(
        x @ w_ih[c_hide * 2:c_hide * 3].t() + (b_ih[c_hide * 2:c_hide * 3] if b_ih is not None else 0) +
        r * (hx @ w_hh[c_hide * 2:c_hide * 3].t() + (b_hh[c_hide * 2:c_hide * 3] if b_hh is not None else 0)))
    y = (1 - z) * n + z * hx  # hx_1
    return y


def _embedding(x, weight):
    """(F.embedding())

    :param x: shape[*]
    :param weight: shape[NV, E]. NV: num_vocab, E: embedding_dim
    :return: shape[*, E]
    """
    return weight[x]
