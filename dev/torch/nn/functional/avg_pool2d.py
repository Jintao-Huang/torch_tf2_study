# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor
import torch.nn.functional as F
from typing import Tuple

"""
池化时间复杂度: Ot(N*Cin*KH*KW * Hout*Wout)
"""

def avg_pool2d(x: Tensor, kernel_size: Tuple[int, int],
               stride: Tuple[int, int] = None, padding: Tuple[int, int] = 0) -> Tensor:
    """return_indices=False

    :param x: shape[N, C, H, W]
    :param kernel_size:
    :param stride: default = kernel_size
    :param padding:
    :return: shape[N, C, Hout, Wout]"""
    stride = stride or kernel_size
    #
    (KH, KW), (SH, SW), (PH, PW) = kernel_size, stride, padding
    N, Cin, H, W = x.shape
    Hout, Wout = (H + 2 * PH - KH) // SH + 1, \
                 (W + 2 * PW - KW) // SW + 1
    #
    if PH != 0 or PW != 0:
        x = F.pad(x, (PW, PW, PH, PH))  # LRTB
    #
    output = torch.empty((N, Cin, Hout, Wout), dtype=x.dtype, device=x.device)
    #
    for i in range(Hout):
        for j in range(Wout):
            xij = x[:, :, i * SH:i * SH + KH, j * SW: j * SW + KW]
            output[:, :, i, j] = torch.mean(xij, dim=(2, 3))  # Ot(N*Cin*KH*KW)
    return output
