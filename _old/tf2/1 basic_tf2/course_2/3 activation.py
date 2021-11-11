# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-2
import tensorflow as tf
from tensorflow import Tensor
import tensorflow.nn as nn
import torch.nn.init as init


def _relu(x: Tensor) -> Tensor:
    """(nn.relu())

    :param x: shape = (*)
    :return: shape = x.shape"""
    return tf.where(x >= 0, x, 0)


def _leaky_relu(x: Tensor, alpha: float = 0.2) -> Tensor:
    """(nn.leaky_relu())"""
    return tf.where(x >= 0, x, alpha * x)


def _sigmoid(x: Tensor) -> Tensor:
    """(nn.sigmoid())"""
    return 1 / (1 + tf.exp(-x))


def _tanh(x: Tensor) -> Tensor:
    """(nn.tanh())"""
    return (1 - tf.exp(2 * -x)) / (1 + tf.exp(2 * -x))  # 上下同乘(e^-x)
    # return (tf.exp(x) - tf.exp(-x)) / (tf.exp(x) + tf.exp(-x))


def _softmax(x: Tensor, axis: int = -1) -> Tensor:
    """(nn.softmax())

    :param x: shape = (N, In)
    :param axis: int. 一般axis设为-1
    :return: shape = x.shape"""
    return tf.exp(x) / tf.reduce_sum(tf.exp(x), axis, keepdims=True)


def _silu(x):
    """(nn.silu())"""
    return x * tf.sigmoid(x)


x = tf.constant([[-1., 0, 1], [0, 0, 1]])
print(_relu(x))
print(nn.relu(x))
# tf.Tensor(
# [[0. 0. 1.]
#  [0. 0. 1.]], shape=(2, 3), dtype=float32)
# tf.Tensor(
# [[0. 0. 1.]
#  [0. 0. 1.]], shape=(2, 3), dtype=float32)
print(_leaky_relu(x))
print(nn.leaky_relu(x))
# tf.Tensor(
# [[-0.2  0.   1. ]
#  [ 0.   0.   1. ]], shape=(2, 3), dtype=float32)
# tf.Tensor(
# [[-0.2  0.   1. ]
#  [ 0.   0.   1. ]], shape=(2, 3), dtype=float32)
print(_sigmoid(x))
print(nn.sigmoid(x))
# tf.Tensor(
# [[0.26894143 0.5        0.73105854]
#  [0.5        0.5        0.73105854]], shape=(2, 3), dtype=float32)
# tf.Tensor(
# [[0.26894143 0.5        0.73105854]
#  [0.5        0.5        0.7310586 ]], shape=(2, 3), dtype=float32)
print(_tanh(x))
print(nn.tanh(x))
# tf.Tensor(
# [[-0.7615942  0.         0.7615941]
#  [ 0.         0.         0.7615941]], shape=(2, 3), dtype=float32)
# tf.Tensor(
# [[-0.7615942  0.         0.7615942]
#  [ 0.         0.         0.7615942]], shape=(2, 3), dtype=float32)
print(_softmax(x))
print(nn.softmax(x))
# tf.Tensor(
# [[0.09003059 0.24472848 0.66524094]
#  [0.21194156 0.21194156 0.57611686]], shape=(2, 3), dtype=float32)
# tf.Tensor(
# [[0.09003057 0.24472848 0.6652409 ]
#  [0.21194157 0.21194157 0.57611686]], shape=(2, 3), dtype=float32)
print(_silu(x))
print(nn.silu(x))
# tf.Tensor(
# [[-0.26894143  0.          0.73105854]
#  [ 0.          0.          0.7310586 ]], shape=(2, 3), dtype=float32)
# tf.Tensor(
# [[-0.26894143  0.          0.73105854]
#  [ 0.          0.          0.7310586 ]], shape=(2, 3), dtype=float32)
