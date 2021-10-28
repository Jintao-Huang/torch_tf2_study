# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import torch
from torch import Tensor
from . import cov


def corrcoef(input: Tensor) -> Tensor:
    """

    :param input: shape[N, M]
    :return: shape[N, N]. 对称矩阵. 对角线为1
    """
    x_cov = cov(input)  # [N, N]
    x_std = torch.sqrt(torch.diag(x_cov))  # [N]
    x_corr = x_cov / (x_std[:, None] * x_std[None])
    return x_corr
