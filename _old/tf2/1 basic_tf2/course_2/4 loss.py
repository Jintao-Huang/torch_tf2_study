# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-2

import tensorflow as tf
import tensorflow.nn as nn
from tensorflow import losses  # 等同上句
# losses = keras.losses
# metrics = keras.metrics
# optimizers = keras.optimizers
# initializers = keras.initializers
from tensorflow import Tensor
import numpy as np
import tensorflow.math as tfm


def _one_hot(indices, depth) -> Tensor:
    """(tf.one_hot()). 快速

    :param indices: shape = (N,). int32, int64
    :param depth: int.
    :return: shape = (N, depth). float32"""
    indices = tf.convert_to_tensor(indices)
    return tf.gather_nd(tf.eye(depth), indices[:, None])


print(tf.one_hot([1, 2, 3], 4))
print(_one_hot([1, 2, 3], 4))


# tf.Tensor(
# [[0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]], shape=(3, 4), dtype=float32)
# tf.Tensor(
# [[0. 1. 0. 0.]
#  [0. 0. 1. 0.]
#  [0. 0. 0. 1.]], shape=(3, 4), dtype=float32)
# -----------------------------------

def _mse(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """均方误差损失(losses.mse())

    :param y_true: shape = (N, In), float32
    :param y_pred: shape = (N, In), float32
    :return: shape = (N,), float32"""

    return tf.reduce_mean((y_true - y_pred) ** 2, axis=-1)


tf.random.set_seed(0)
y_true = tf.random.normal((16, 10))
y_pred = tf.random.normal((16, 10))
print(losses.mse(y_true, y_pred))
print(losses.mean_squared_error(y_true, y_pred))  # 答案同上
print(_mse(y_true, y_pred))


# tf.Tensor(
# [1.1952267 3.4243941 2.1024227 2.3010921 2.3643446 1.8302895 1.6360563
#  3.5714912 2.3740485 4.2296114 1.4224513 4.019039  0.7188259 1.5340036
#  1.5875269 2.435854 ], shape=(16,), dtype=float32)
# tf.Tensor(
# [1.1952267 3.4243941 2.1024227 2.3010921 2.3643446 1.8302895 1.6360563
#  3.5714912 2.3740485 4.2296114 1.4224513 4.019039  0.7188259 1.5340036
#  1.5875269 2.435854 ], shape=(16,), dtype=float32)
# tf.Tensor(
# [1.1952267 3.4243941 2.1024227 2.301092  2.3643446 1.8302895 1.6360562
#  3.5714912 2.3740482 4.2296114 1.4224513 4.019039  0.7188258 1.5340036
#  1.587527  2.435854 ], shape=(16,), dtype=float32)
# -----------------------------------
def _mae(y_true: Tensor, y_pred: Tensor) -> Tensor:
    """均方误差损失(losses.mae())

    :param y_true: shape = (N, In), float32
    :param y_pred: shape = (N, In), float32
    :return: shape = (N,), float32"""

    return tf.reduce_mean(tf.abs(y_true - y_pred), axis=-1)


print(losses.mae(y_true, y_pred))
print(losses.mean_absolute_error(y_true, y_pred))
print(_mae(y_true, y_pred))


# tf.Tensor(
# [0.9005594  1.418648   1.1257443  1.1869876  1.2147367  1.0900935
#  1.179163   1.3530084  1.4167826  1.447867   0.95472604 1.6511494
#  0.7503426  0.9016584  1.0450656  1.2433184 ], shape=(16,), dtype=float32)
# tf.Tensor(
# [0.9005594  1.418648   1.1257443  1.1869876  1.2147367  1.0900935
#  1.179163   1.3530084  1.4167826  1.447867   0.95472604 1.6511494
#  0.7503426  0.9016584  1.0450656  1.2433184 ], shape=(16,), dtype=float32)
# tf.Tensor(
# [0.9005594  1.418648   1.1257443  1.1869876  1.2147367  1.0900935
#  1.179163   1.3530084  1.4167826  1.447867   0.95472604 1.6511494
#  0.7503426  0.9016584  1.0450656  1.2433184 ], shape=(16,), dtype=float32)

# -----------------------------------
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


n_classes = 10
np.random.seed(0)
y_true = tf.convert_to_tensor(np.random.randint(0, n_classes, (16,)))
y_true = _one_hot(y_true, n_classes)
print(losses.categorical_crossentropy(y_true, y_pred, True, 0.1))
print(_categorical_crossentropy(y_true, y_pred, True, 0.1))

# tf.Tensor(
# [3.6926932 3.6685426 3.6090984 3.5787313 3.06274   3.329051  2.702104
#  2.618079  2.7482157 4.5264134 1.1691736 3.2064793 2.1327136 2.4106603
#  2.8424053 2.6912308], shape=(16,), dtype=float32)
# tf.Tensor(
# [3.6926935 3.6685426 3.6090987 3.578731  3.0627403 3.329051  2.702104
#  2.618079  2.7482157 4.526413  1.1691736 3.2064793 2.1327136 2.41066
#  2.8424058 2.6912308], shape=(16,), dtype=float32)
# -----------------------------------
n_classes = 10
np.random.seed(0)
y_true = tf.convert_to_tensor(np.random.randint(0, n_classes, (16,)))
print(losses.sparse_categorical_crossentropy(y_true, y_pred, True))
y_true = _one_hot(y_true, n_classes)
print(losses.categorical_crossentropy(y_true, y_pred, True, 0.))


# tf.Tensor(
# [3.780215  3.7131248 3.7176704 3.6711931 3.0742893 3.4138093 2.7306106
#  2.6030748 2.7395084 4.70498   0.9959858 3.1918743 2.0506706 2.3671892
#  2.8741698 2.7103133], shape=(16,), dtype=float32)
# tf.Tensor(
# [3.780215  3.7131248 3.7176704 3.6711931 3.0742893 3.4138093 2.7306106
#  2.6030748 2.7395084 4.70498   0.9959858 3.1918743 2.0506706 2.3671892
#  2.8741698 2.7103133], shape=(16,), dtype=float32)

# -----------------------------------
def _binary_crossentropy(y_true: Tensor, y_pred: Tensor, from_logits: bool = False,
                         label_smoothing: float = 0) -> Tensor:
    """(losses.binary_crossentropy())

    :param y_true: shape = (N, X), float32. 一般而言, X=1
    :param y_pred: shape = (N, X), float32
    :return: shape = (N, X), float32"""
    y_pred = tf.clip_by_value(y_pred, 2e-7, 1 - 2e-7)
    y_true = y_true if (label_smoothing == 0) else (y_true * (1 - label_smoothing) + label_smoothing / 2)
    if from_logits:
        return tf.reduce_mean(-y_true * tfm.log_sigmoid(y_pred) + -(1 - y_true) * tfm.log_sigmoid(- y_pred), axis=-1)
    else:
        return tf.reduce_mean(-y_true * tfm.log(y_pred) + -(1 - y_true) * tfm.log(1 - y_pred), axis=-1)


np.random.seed(0)
y_true = tf.convert_to_tensor(np.random.randint(0, 2, (16,)), dtype=tf.float32)[:, None]
tf.random.set_seed(0)
y_pred = tf.random.uniform((16, 1))
print(losses.binary_crossentropy(y_true, y_pred, True, 0.1))
print(_binary_crossentropy(y_true, y_pred, True, 0.1))
# tf.Tensor(
# [0.83515453 0.60551655 0.4876318  0.984583   0.5271907  0.40907174
#  0.50130147 0.36367166 0.43910736 0.6386926  0.4354303  1.0451092
#  0.99105066 0.54554737 0.90432394 1.0254946 ], shape=(16,), dtype=float32)
# tf.Tensor(
# [0.83515453 0.6055166  0.48763183 0.9845831  0.5271907  0.40907177
#  0.50130147 0.36367163 0.43910736 0.6386926  0.43543026 1.045109
#  0.9910507  0.5455473  0.9043238  1.0254946 ], shape=(16,), dtype=float32)
np.random.seed(0)
y_true = tf.convert_to_tensor(np.random.randint(0, 2, (16, 2)), dtype=tf.float32)
tf.random.set_seed(0)
y_pred = tf.random.uniform((16, 2))
print(losses.binary_crossentropy(y_true, y_pred, True, 0.1))
print(_binary_crossentropy(y_true, y_pred, True, 0.1))


# tf.Tensor(
# [0.72033554 0.7361074  0.4681312  0.43248656 0.5389     0.7402697
#  0.768299   0.96490926 1.1249719  0.9044446  0.7574911  0.77000225
#  0.7721339  0.54734993 0.88563275 0.73239124], shape=(16,), dtype=float32)
# tf.Tensor(
# [0.7203356  0.73610747 0.46813124 0.43248653 0.5389     0.74026966
#  0.768299   0.9649092  1.1249719  0.9044446  0.7574911  0.77000225
# -----------------------------------
def _l2_loss(w):
    return tf.reduce_sum(w ** 2 / 2)


x = tf.random.normal((10, 20))
print(nn.l2_loss(x))
print(_l2_loss(x))
# tf.Tensor(104.98601, shape=(), dtype=float32)
# tf.Tensor(104.98601, shape=(), dtype=float32)
