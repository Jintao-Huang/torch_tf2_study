# Author: Jintao Huang
# Time: 2020-5-25
import torch
import torch.nn.functional as F


def swish(x):
    """

    :param x: shape(N/..., num_classes)
    :return: shape = x.shape
    """
    return x * torch.sigmoid(x)


def mish(x):
    """

    :param x: shape(N/..., num_classes)
    :return: shape = x.shape
    """
    return x * torch.tanh(F.softplus(x))
