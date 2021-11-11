# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import torch
from torch import Tensor


def cov(input: Tensor, *, correction: int = 1) -> Tensor:
    """fweights=None, aweights=None

    :param input: shape[N, M]
    :param correction:
    :return: shape[N, N]. 对称矩阵. 对角线为方差
    """
    mean = torch.mean(input, dim=1, keepdim=True)
    x_cov = torch.sum((input - mean)[:, None] *
                      (input - mean)[None], dim=-1) \
            / (input.shape[1] - correction)
    return x_cov
