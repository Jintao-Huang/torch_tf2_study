# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-2

import tensorflow as tf
import torch

# tf.where
def relu_tf(x):
    return tf.where(x > 0, x, 0)


def relu_torch(x):
    return torch.where(x > 0, x, torch.tensor(0, dtype=x.dtype, device=x.device))


x = tf.constant([-1, 0, 1.])
x2 = torch.tensor([-1, 0, 1.])
print(relu_tf(x))
print(relu_torch(x2))
# tf.Tensor([0. 0. 1.], shape=(3,), dtype=float32)
# tensor([0., 0., 1.])

# 2.
a = tf.constant([-1, 0, 1])
b = tf.constant([1, 0, -1])
print(tf.where(a > b, a, b))
print(tf.where(a >= b))
# tf.Tensor([1 0 1], shape=(3,), dtype=int32)
# tf.Tensor(
# [[1]
#  [2]], shape=(2, 1), dtype=int64)
# --------------------------- 随机数
import numpy as np
rdm = np.random.RandomState(seed=1)
print(rdm.rand())
rdm = np.random.RandomState(seed=1)
print(rdm.rand(2))
rdm = np.random.RandomState(seed=1)
print(rdm.rand(2, 2))
# 0.417022004702574
# [0.417022   0.72032449]
# [[4.17022005e-01 7.20324493e-01]
#  [1.14374817e-04 3.02332573e-01]]
