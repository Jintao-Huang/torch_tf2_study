# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor
from typing import Tuple

"""
- conv_transpose2d时间复杂度: Ot(N*Cin*Cout*KH*KW * H*W)
"""


def conv_transpose2d(input: Tensor, weight: Tensor, bias: Tensor = None,
                     stride: Tuple[int, int] = (1, 1),
                     padding: Tuple[int, int] = (0, 0)) -> Tensor:
    """

    :param input: shape[N, Cin, H, W]
    :param weight: shape[Cin, Cout, KH, KW]
    :param bias: shape[Cout]
    :param stride:
    :param padding:
    :return: shape[N, Cout, Hout, Wout]"""
    x = input
    #
    N, Cin, H, W = x.shape
    _, Cout, KH, KW = weight.shape
    (SH, SW), (PH, PW) = stride, padding
    Hout, Wout = SH * (H - 1) - 2 * PH + KH, \
                 SW * (W - 1) - 2 * PW + KW,
    #
    output = bias[None, :, None, None].tile([N, 1, Hout + 2 * PH, Wout + 2 * PW])
    weight = weight.view(Cin, Cout * KH * KW)
    #
    for i in range(H):
        for j in range(W):
            output_ij = output[:, :, i * SH:i * SH + KH, j * SW: j * SW + KW]
            # [N, Cin] @ [Cin, Cout*KH*KW]
            output_ij[:] += (x[:, :, i, j] @ weight).view(N, Cout, KH, KW)  # Ot(N*Cin*Cout*KH*KW)
    #
    if PH != 0 or PW != 0:
        output = output[:, :, PH:-PH, PW:-PW]

    return output
