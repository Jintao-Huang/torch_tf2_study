# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple
from . import linear

"""
- conv2d时间复杂度: Ot(N*Cin*KH*KW*Cout//G * Hout*Wout)
"""


def _conv2d_easy(input: Tensor, weight: Tensor, bias: Tensor = None,
                 stride: Tuple[int, int] = (1, 1),
                 padding: Tuple[int, int] = (0, 0)) -> Tensor:
    """easy to study. no dilation, no groups

    :param input: shape[N, Cin, H, W]
    :param weight: shape[Cout, Cin, KH, KW]
    :param bias: shape[Cout]
    :param stride: Tuple[SH, SW]
    :param padding: Tuple[PH, PW]
    :return: shape[N, Cout, Hout, Wout]
    """
    x = input
    #
    N, Cin, H, W = x.shape
    Cout, _, KH, KW = weight.shape
    (SH, SW), (PH, PW) = stride, padding
    Hout, Wout = (H + 2 * PH - KH) // SH + 1, \
                 (W + 2 * PW - KW) // SW + 1
    #
    if PH != 0 or PW != 0:
        x = F.pad(x, (PW, PW, PH, PH))  # LRTB
    #
    output = torch.empty((N, Cout, Hout, Wout), dtype=x.dtype, device=x.device)
    weight = weight.view(Cout, Cin * KH * KW)
    #
    for i in range(Hout):
        for j in range(Wout):
            xij = x[:, :, i * SH:i * SH + KH, j * SW: j * SW + KW]
            xij = xij.contiguous().view(N, Cin * KH * KW)
            # shape[N, Cin*KH*KW] @ shape[Cin*KH*KW, Cout]
            output[:, :, i, j] = linear(xij, weight, bias)  # Ot(N*Cin*KH*KW*Cout)

    return output


def conv2d(input: Tensor, weight: Tensor, bias: Tensor = None,
           stride: Tuple[int, int] = (1, 1),
           padding: Tuple[int, int] = (0, 0),
           dilation: Tuple[int, int] = (1, 1),
           groups: int = 1) -> Tensor:
    """

    :param input: shape[N, Cin, H, W]
    :param weight: shape[Cout, Cin_G, KH, KW]. Cin_G == Cin // G
    :param bias: shape[Cout]
    :param stride: Tuple[SH, SW]
    :param padding: Tuple[PH, PW]
    :param dilation: Tuple[DH, DW]
    :param groups: Cin % G == 0, Cout % G == 0
    :return: shape[N, Cout, Hout, Wout]
    """
    x = input
    #
    N, Cin, H, W = x.shape
    Cout, Cin_G, KH, KW = weight.shape
    (SH, SW), (PH, PW), (DH, DW) = stride, padding, dilation
    G = groups
    Cout_G = Cout // G
    Hout, Wout = (H + 2 * PH - DH * (KH - 1) - 1) // SH + 1, \
                 (W + 2 * PW - DW * (KW - 1) - 1) // SW + 1
    #
    if PH != 0 or PW != 0:
        x = F.pad(x, (PW, PW, PH, PH))  # LRTB
    #
    output = torch.empty((N, Cout, Hout, Wout), dtype=x.dtype, device=x.device)
    weight = weight.view(G, Cout_G, Cin_G * KH * KW).transpose(1, 2)  # [G, Cin_G*KH*KW, Cout_G]
    #
    for i in range(Hout):
        for j in range(Wout):
            xij = x[:, :,
                  i * SH:i * SH + DH * (KH - 1) + 1:DH,
                  j * SW:j * SW + DW * (KW - 1) + 1:DW]
            xij = xij.contiguous().view(N, G, Cin_G * KH * KW).transpose(0, 1)
            # shape[G, N, Cin_G*KH*KW] @ shape[G, Cin_G*KH*KW, Cout_G].
            # Ot(N*Cin*KH*KW*Cout_G)
            output[:, :, i, j] = (xij @ weight).transpose(0, 1).contiguous().view(N, Cout)
    if bias is not None:
        output += bias[None, :, None, None]
    return output
