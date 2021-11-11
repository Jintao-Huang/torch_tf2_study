# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from torch import Tensor


def linear(x: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    """

    :param x: shape[N, In]
    :param weight: shape[Out, In]
    :param bias: shape[Out]
    :return: shape[N, Out]"""

    y = x @ weight.T
    if bias is not None:
        y += bias
    return y
