# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from torch import Tensor
import numpy as np


def gather(input: Tensor, dim: int, index: Tensor) -> Tensor:
    """只支持input, index, return的shape一致

    :param input: shape[...]
    :param dim:
    :param index: Long. shape[...]
    :return: shape[...]
    """
    idxs = [range(length) for length in input.shape]
    idxs = np.ix_(*idxs)
    idxs = [torch.tensor(arr, dtype=torch.long) for arr in idxs]
    idxs[dim] = index
    out = input[idxs]
    return out
