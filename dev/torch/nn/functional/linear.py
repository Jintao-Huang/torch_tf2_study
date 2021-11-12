# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import torch
from torch import Tensor

"""
- 矩阵乘时间复杂度: e.g. [A, B] @ [B, C]. Ot(ABC)
- linear时间复杂度: Ot(N*In*Out)
"""


def linear(x: Tensor, weight: Tensor, bias: Tensor = None) -> Tensor:
    """

    :param x: shape[N, In]
    :param weight: shape[Out, In]
    :param bias: shape[Out]
    :return: shape[N, Out]"""

    y = x @ weight.T  # Ot(N*In*Out)
    if bias is not None:
        y += bias  # Ot(N*Out)
    return y
