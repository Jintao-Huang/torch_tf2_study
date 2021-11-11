# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-1

import tensorflow as tf

# ------------------- cast(), reduce_*()
print(tf.constant(1).shape)  # ()
x = tf.constant([0, 2])
print(tf.reduce_min(x))  # tf.Tensor(0, shape=(), dtype=int32)
print(tf.reduce_max(x).numpy())  # 2
print(tf.reduce_mean(x).numpy())  # 1
print(tf.reduce_sum(x).numpy())  # 2
print(tf.reduce_all(tf.cast(x, dtype=tf.bool)))  # tf.Tensor(False, shape=(), dtype=bool)
print(tf.reduce_any(tf.cast(x, dtype=tf.bool)))  # tf.Tensor(True, shape=(), dtype=bool)
print(x.dtype)  # <dtype: 'int32'>
x = tf.cast(x, dtype=tf.float32)  # 强转
print(x.dtype)  # <dtype: 'float32'>

# ------------------- axis
x = tf.constant([[0, 1], [2, 3]])
print(tf.reduce_min(x).numpy())  # 0
print(tf.reduce_min(x, axis=0).numpy())  # [0 1]
print(tf.reduce_min(x, axis=1).numpy())  # [0 2]

# ------------------- Variable(可学习的)

w = tf.Variable(tf.random.normal((1, 1)))
print(w)
print(w.numpy())
# <tf.Variable 'Variable:0' shape=(1, 1) dtype=float32, numpy=array([[0.7949632]], dtype=float32)>
# [[0.7949632]]


# ------------------ gather gather_nd
import numpy as np
print(tf.range(0, 5, 1, dtype=tf.float32))
print(tf.linspace(0, 5, 5))  # 没有dtype, 请用numpy
# tf.Tensor([0. 1. 2. 3. 4.], shape=(5,), dtype=float32)
# tf.Tensor([0.   1.25 2.5  3.75 5.  ], shape=(5,), dtype=float64)
x = np.linspace(0, 5, 6, dtype=np.float32)
x = tf.reshape(tf.convert_to_tensor(x), (2, 3))
print(x)
# tf.Tensor(
# [[0. 1. 2.]
#  [3. 4. 5.]], shape=(2, 3), dtype=float32)
tf.gather(x, [0, 1])
