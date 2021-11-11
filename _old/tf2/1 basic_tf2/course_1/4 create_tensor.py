# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-1

import tensorflow as tf
import numpy as np

x = np.arange(0, 5)
y = tf.convert_to_tensor(x, dtype=tf.int64)
print(x, x.dtype)  # [0 1 2 3 4] int32
print(y, y.dtype)  # tf.Tensor([0 1 2 3 4], shape=(5,), dtype=int64) <dtype: 'int64'>

# --------------------------

a = tf.zeros((1, 2))
b = tf.ones((1, 2))
c = tf.fill((1, 2), 2)
d = tf.zeros_like(a)
f = tf.ones_like(a)
print(a, b, c)
print(d, f)
# tf.Tensor([[0. 0.]], shape=(1, 2), dtype=float32) tf.Tensor([[1. 1.]], shape=(1, 2), dtype=float32) tf.Tensor([[2 2]], shape=(1, 2), dtype=int32)
# tf.Tensor([[0. 0.]], shape=(1, 2), dtype=float32) tf.Tensor([[1. 1.]], shape=(1, 2), dtype=float32)
print(tf.eye(3))
print(tf.eye(3, 3))
# tf.Tensor(
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]], shape=(3, 3), dtype=float32)
# tf.Tensor(
# [[1. 0. 0.]
#  [0. 1. 0.]
#  [0. 0. 1.]], shape=(3, 3), dtype=float32)
print(tf.eye(3, 3, (1,)).shape)  # (1, 3, 3)
print(tf.eye(3, 3, (3, 2)).shape)  # (3, 2, 3, 3)

# -------------------------- random

tf.random.set_seed(0)
z = tf.random.uniform((1,))  # 默认[0, 1]
z2 = tf.random.normal((1,))  # 默认[0, 1]
z3 = tf.random.truncated_normal((10,))  # [u-2*std, u+2*std]
print(z, z2)
print(z3)
# tf.Tensor([0.29197514], shape=(1,), dtype=float32) tf.Tensor([1.0668802], shape=(1,), dtype=float32)
# tf.Tensor(
# [-1.8041286  -0.11153453 -0.8455512   0.8489615   0.18171437  0.07833663
#  -0.77281225  0.5105128   1.0920767  -0.6850036 ], shape=(10,), dtype=float32)
zz = tf.random.uniform((1,), 0, 0)  # 前闭后开
print(zz)  # tf.Tensor([0.], shape=(1,), dtype=float32)
