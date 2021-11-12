# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import torch
from torch import Tensor


def cov_2(input: Tensor, *, correction: int = 1) -> Tensor:
    """fweights=None, aweights=None

    :param input: shape[N, M]
    :param correction:
    :return: shape[N, N]. 对称矩阵. 对角线为方差
    """
    _, M = input.shape
    #
    mean = torch.mean(input, dim=1, keepdim=True)
    diff = input - mean
    x_cov = torch.sum(diff[:, None] * diff[None], dim=-1) / (M - correction)
    return x_cov


def cov(input: Tensor, *, correction: int = 1) -> Tensor:
    """fweights=None, aweights=None. [广播+sum]转矩阵乘

    :param input: shape[N, M]
    :param correction:
    :return: shape[N, N]. 对称矩阵. 对角线为方差
    """
    _, M = input.shape
    #
    mean = torch.mean(input, dim=1, keepdim=True)
    diff = input - mean
    # shape[N, M] @ shape[M, N] -> shape[N, N]
    x_cov = diff @ diff.T / (M - correction)
    return x_cov
