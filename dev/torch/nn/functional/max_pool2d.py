# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple


def max_pool2d(input: Tensor, kernel_size: Tuple[int, int],
               stride: Tuple[int, int] = None, padding: Tuple[int, int] = 0) -> Tensor:
    """return_indices=False

    :param input: shape[N, C, H, W]
    :param kernel_size:
    :param stride: default = kernel_size
    :param padding:
    :return: shape[N, C, Hout, Wout]"""
    x = input
    stride = stride or kernel_size
    #
    N, Cin, H, W = x.shape
    (KH, KW), (SH, SW), (PH, PW) = kernel_size, stride, padding
    Hout, Wout = (H + 2 * PH - KH) // SH + 1, \
                 (W + 2 * PW - KW) // SW + 1
    #
    if PH != 0 or PW != 0:
        x = F.pad(x, (PW, PW, PH, PH), value=-torch.inf)  # LRTB
    #
    output = torch.empty((N, Cin, Hout, Wout), dtype=x.dtype, device=x.device)
    #
    for i in range(Hout):
        for j in range(Wout):
            xij = x[:, :, i * SH:i * SH + KH, j * SW: j * SW + KW]
            xij = xij.contiguous().view(N, Cin, KH * KW)
            #
            output[:, :, i, j] = torch.max(xij, dim=-1)[0]
    return output
