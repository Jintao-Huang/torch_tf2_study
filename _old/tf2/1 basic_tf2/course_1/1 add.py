# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-1
import tensorflow as tf

gpu_available = tf.test.is_gpu_available()
print(gpu_available)  # True

a = tf.constant([1.], dtype=tf.float32, shape=[1, 1], name="a")
b = tf.constant([1], dtype=tf.float32, name="b")
print(a, b)  # tf.Tensor([[1.]], shape=(1, 1), dtype=float32) tf.Tensor([1.], shape=(1,), dtype=float32)
result = a + b
print(result)  # tf.Tensor([[2.]], shape=(1, 1), dtype=float32)
