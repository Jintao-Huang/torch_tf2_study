# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-1

import tensorflow as tf

# ----------------- max min sort
x = tf.constant([1, 2])
y = tf.constant([0, 3])
print(tf.reduce_max(x))
print(tf.maximum(x, y))
print(tf.argmax(x))
# tf.Tensor(2, shape=(), dtype=int32)
# tf.Tensor([1 3], shape=(2,), dtype=int32)
# tf.Tensor(1, shape=(), dtype=int64)
print(tf.reduce_min(x))
print(tf.minimum(x, y))
print(tf.argmin(x))
# tf.Tensor(1, shape=(), dtype=int32)
# tf.Tensor([0 2], shape=(2,), dtype=int32)
# tf.Tensor(0, shape=(), dtype=int64)
print(tf.sort(x, direction='DESCENDING'))  # const
print(x)
print(tf.argsort(x))
# tf.Tensor([2 1], shape=(2,), dtype=int32)
# tf.Tensor([1 2], shape=(2,), dtype=int32)
# tf.Tensor([0 1], shape=(2,), dtype=int32)
