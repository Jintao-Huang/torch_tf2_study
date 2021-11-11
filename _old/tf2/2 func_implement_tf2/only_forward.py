# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import tensorflow as tf
from tensorflow import Tensor
import tensorflow.nn as nn
import tensorflow.math as tfm
import tensorflow.keras.losses as losses
import numpy as np


# --------------------------------------------------- activation

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


# x = tf.constant([[-1., 0, 1], [0, 0, 1]])
# print(_relu(x))
# print(nn.relu(x))
# print(_leaky_relu(x))
# print(nn.leaky_relu(x))
# print(_sigmoid(x))
# print(nn.sigmoid(x))
# print(_tanh(x))
# print(nn.tanh(x))
# print(_softmax(x))
# print(nn.softmax(x))
# print(_silu(x))
# print(nn.silu(x))


# --------------------------------------------------- losses


def _one_hot(indices, depth) -> Tensor:
    """(tf.one_hot()). 快速

    :param indices: shape = (N,). int32, int64
    :param depth: int.
    :return: shape = (N, depth). float32"""
    indices = tf.convert_to_tensor(indices)
    return tf.gather_nd(tf.eye(depth), indices[:, None])


# print(tf.one_hot([1, 2, 3], 4))
# print(_one_hot([1, 2, 3], 4))


def _mse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """均方误差损失(losses.mse())

    :param y_true: shape = (N, In), float32
    :param y_pred: shape = (N, In), float32
    :return: shape = (N,), float32"""

    return tf.reduce_mean((y_true - y_pred) ** 2, axis=-1)


# tf.random.set_seed(0)
# y_true = tf.random.normal((16, 10))
# y_pred = tf.random.normal((16, 10))
# print(losses.mse(y_true, y_pred))
# print(losses.mean_squared_error(y_true, y_pred))  # 答案同上
# print(_mse(y_true, y_pred))


def _mae(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """均方误差损失(losses.mae())

    :param y_true: shape = (N, In), float32
    :param y_pred: shape = (N, In), float32
    :return: shape = (N,), float32"""

    return tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)


# print(losses.mae(y_true, y_pred))
# print(losses.mean_absolute_error(y_true, y_pred))
# print(_mae(y_true, y_pred))


def _categorical_crossentropy(y_true: Tensor, y_pred: Tensor, from_logits: bool = False,
                              label_smoothing: float = 0) -> Tensor:
    """(losses.categorical_crossentropy()).

    :param y_true: shape = (N, In), float32. ont_hot
    :param y_pred: shape = (N, In), float32.
    :return: shape = (N,), float32"""
    y_pred = nn.log_softmax(y_pred, axis=-1) if from_logits else tfm.log(y_pred, axis=-1)
    n_classes = y_true.shape[-1]
    y_true = y_true if (label_smoothing == 0) else (y_true * (1 - label_smoothing) + label_smoothing / n_classes)
    return tf.reduce_sum(-y_true * y_pred, axis=-1)


# n_classes = 10
# np.random.seed(0)
# y_true = tf.convert_to_tensor(np.random.randint(0, n_classes, (16,)))
# y_true = _one_hot(y_true, n_classes)
# print(losses.categorical_crossentropy(y_true, y_pred, True, 0.1))
# print(_categorical_crossentropy(y_true, y_pred, True, 0.1))


def _binary_crossentropy(y_true: Tensor, y_pred: Tensor, from_logits: bool = False,
                         label_smoothing: float = 0) -> Tensor:
    """(losses.binary_crossentropy())

    :param y_true: shape = (N, X), float32. 一般而言, X=1
    :param y_pred: shape = (N, X), float32
    :return: shape = (N, X), float32"""
    y_pred = tf.clip_by_value(y_pred, 2e-7, 1 - 2e-7)  # 防止inf
    y_true = y_true if (label_smoothing == 0) else (y_true * (1 - label_smoothing) + label_smoothing / 2)
    if from_logits:
        return tf.reduce_mean(-y_true * tfm.log_sigmoid(y_pred) + -(1 - y_true) * tfm.log_sigmoid(- y_pred), axis=-1)
    else:
        return tf.reduce_mean(-y_true * tfm.log(y_pred) + -(1 - y_true) * tfm.log(1 - y_pred), axis=-1)


# np.random.seed(0)
# y_true = tf.convert_to_tensor(np.random.randint(0, 2, (16,)), dtype=tf.float32)[:, None]
# tf.random.set_seed(0)
# y_pred = tf.random.uniform((16, 1))
# print(losses.binary_crossentropy(y_true, y_pred, True, 0.1))
# print(_binary_crossentropy(y_true, y_pred, True, 0.1))
#
# np.random.seed(0)
# y_true = tf.convert_to_tensor(np.random.randint(0, 2, (16, 2)), dtype=tf.float32)
# tf.random.set_seed(0)
# y_pred = tf.random.uniform((16, 2))
# print(losses.binary_crossentropy(y_true, y_pred, True, 0.1))
# print(_binary_crossentropy(y_true, y_pred, True, 0.1))


def _l2_loss(w):
    """l2 正则时.

    :param w:
    :return:
    """
    return tf.reduce_sum(w ** 2 / 2)


# loss_regularization = []
# loss_regularization.append(nn.l2_loss(w1))
# loss_regularization.append(nn.l2_loss(w2))
# loss = loss_ce + sum(loss_regularization) * weight_decay


x = tf.random.normal((10, 20))
print(nn.l2_loss(x))
print(_l2_loss(x))
