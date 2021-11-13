# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

import torch
from torch import Tensor


def relu(input: Tensor) -> Tensor:
    """inplace=False"""
    x = input
    #
    zero = torch.tensor(0., dtype=x.dtype, device=x.device)
    return torch.where(x > 0, x, zero)
