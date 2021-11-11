# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-1
import tensorflow as tf

# ---------------- 四则运算
x, y = tf.constant([1, 2]), tf.constant([3, 4])
print(x.shape, y.shape)  # (2,) (2,)
print(tf.add(x, y))  # tf.Tensor([4 6], shape=(2,), dtype=int32)
print(tf.subtract(x, y))  # tf.Tensor([-2 -2], shape=(2,), dtype=int32)
print(tf.multiply(x, y))  # tf.Tensor([3 8], shape=(2,), dtype=int32)
print(tf.divide(x, y))  # tf.Tensor([0.33333333 0.5       ], shape=(2,), dtype=float64)
print(tf.square(x))  # tf.Tensor([1 4], shape=(2,), dtype=int32)
print(tf.pow(x, 2))  # tf.Tensor([1 4], shape=(2,), dtype=int32)
print(tf.sqrt(tf.cast(x, tf.float32)))  # tf.Tensor([1.        1.4142135], shape=(2,), dtype=float32)
# shape[1, 2] @ shape[2, 1]
print(tf.matmul(x[None], y[:, None]))  # tf.Tensor([[11]], shape=(1, 1), dtype=int32)

# ---------------- 符号表示

x, y = tf.constant([1, 2]), tf.constant([3, 4])
print(x.shape, y.shape)  # (2,) (2,)
print(x + y)  # tf.Tensor([4 6], shape=(2,), dtype=int32)
print(x - y)  # tf.Tensor([-2 -2], shape=(2,), dtype=int32)
print(x * y)  # tf.Tensor([3 8], shape=(2,), dtype=int32)
print(x / y)  # tf.Tensor([0.33333333 0.5       ], shape=(2,), dtype=float64)
print(x // y)  # tf.Tensor([0 0], shape=(2,), dtype=int32)
print(x ** 2)  # tf.Tensor([1 4], shape=(2,), dtype=int32)
print(tf.cast(x, tf.float32) ** (1 / 2))  # tf.Tensor([1.        1.4142135], shape=(2,), dtype=float32)
# shape[1, 2] @ shape[2, 1]
print(x[None] @ y[:, None])  # tf.Tensor([[11]], shape=(1, 1), dtype=int32)
