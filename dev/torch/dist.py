# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from . import Number
from torch import Tensor


def dist(input: Tensor, other: Tensor, p: Number = 2) -> Tensor:
    return torch.sum(torch.abs(input - other) ** p) ** (1 / p)
