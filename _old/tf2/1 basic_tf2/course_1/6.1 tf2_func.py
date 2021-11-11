# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-1
import tensorflow as tf

# --------------------------- backward
w = tf.Variable(tf.constant(3.))
with tf.GradientTape() as tape:  # 记录计算过程
    loss = tf.pow(w, 2)  # loss = w ** 2, dloss/dw = 2w = 6
grad = tape.gradient(loss, w)  # 求导
print(grad)  # tf.Tensor(6.0, shape=(), dtype=float32)

# --------------------------- one_hot
labels = tf.constant([1, 2, 0])
print(tf.one_hot(labels, 3))
print(tf.one_hot(1, 3))
# tf.Tensor(
# [[0. 1. 0.]
#  [0. 0. 1.]
#  [1. 0. 0.]], shape=(3, 3), dtype=float32)
# tf.Tensor([0. 1. 0.], shape=(3,), dtype=float32)

# ---------------------------- 激活函数
x = tf.constant([[-1, 0, 1.], [0, 0, 1]])
print(tf.nn.relu(x))
print(tf.nn.leaky_relu(x, alpha=0.2))
print(tf.nn.sigmoid(x))
print(tf.nn.tanh(x))
print(tf.nn.softmax(x))  # 默认axis=-1
print(tf.nn.gelu(x))
print(tf.nn.selu(x))
# tf.Tensor([0. 1. 0.], shape=(3,), dtype=float32)
# tf.Tensor([0. 0. 1.], shape=(3,), dtype=float32)
# tf.Tensor([-0.2  0.   1. ], shape=(3,), dtype=float32)
# tf.Tensor([0.26894143 0.5        0.7310586 ], shape=(3,), dtype=float32)
# tf.Tensor([-0.7615942  0.         0.7615942], shape=(3,), dtype=float32)
# tf.Tensor([0.09003057 0.24472848 0.6652409 ], shape=(3,), dtype=float32)
# tf.Tensor([-0.15865529  0.          0.8413447 ], shape=(3,), dtype=float32)
# tf.Tensor([-1.1113306  0.         1.050701 ], shape=(3,), dtype=float32)
# ---------------------------- 自更新
w = tf.Variable(0)
x = tf.constant(0)
# 只有Variable有, 但Variable不能用 -= (正在的自更新)
# constant没有，但是可以用 -= (虚假的自更新)
w.assign_add(1)  # 可训练参数的自更新. id没变
x += 1  # 本质上是 x = x + 1. id变了
print(w, x)
w.assign_sub(1)
x -= 1
print(w, x)
