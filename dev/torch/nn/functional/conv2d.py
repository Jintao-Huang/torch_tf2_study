# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from torch import Tensor
from torch.nn.functional import pad
from . import linear
from typing import Tuple


def conv2d(x: Tensor, weight: Tensor, bias: Tensor = None, stride: Tuple[int, int] = (1, 1),
           padding: Tuple[int, int] = (0, 0)) -> Tensor:
    """

    :param x: shape[N, Cin, H, W]
    :param weight: shape[Cout, Cin, KH, KW]
    :param bias: shape[Cout]
    :param stride: Tuple[SH, SW]
    :param padding: Tuple[PH, PW]
    :return: shape[N, Cout, Hout, Wout]
    """
    SH, SW, PH, PW = (*stride, *padding)
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape
    Hout, Wout = (H + 2 * PH - KH) // SH + 1, \
                 (W + 2 * PW - KW) // SW + 1
    #
    if PH != 0 or PW != 0:
        x = pad(x, (PW, PW, PH, PH))  # LTRB
    #
    output = torch.empty((N, Cout, Hout, Wout), dtype=x.dtype, device=x.device)
    weight = weight.view(Cout, Cin * KH * KW)
    #
    for i in range(Hout):  # Hout
        for j in range(Wout):  # Wout
            xij = x[:, :, i * SH:i * SH + KH, j * SW: j * SW + KW]
            #
            xij = xij.contiguous().view(N, Cin * KH * KW)
            # shape[N, Cin*KH*KW] @ shape[Cin*KH*KW, Cout]
            output[:, :, i, j] = linear(xij, weight, bias)

    return output