# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor


def var(input: Tensor, unbiased: bool = True) -> Tensor:
    """dim=None

    :param input: shape[M]. 会被flatten
    :param unbiased: True: ddof=1; False: ddof=0
    :return: scalar
    """
    mean = torch.mean(input)
    var = torch.sum((input - mean) ** 2) / (input.numel() - unbiased)
    return var
