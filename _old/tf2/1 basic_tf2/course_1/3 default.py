# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-1
import tensorflow as tf

print(tf.constant(1))  # tf.Tensor(1, shape=(), dtype=int32)
print(tf.constant(1.))  # tf.Tensor(1.0, shape=(), dtype=float32)

a = tf.constant(1)
print(a.dtype)  # <dtype: 'int32'>
print(a.shape)  # ()
print(a.device)  # /job:localhost/replica:0/task:0/device:CPU:0
