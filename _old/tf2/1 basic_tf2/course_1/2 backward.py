# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-1
import tensorflow as tf

w = tf.Variable(tf.constant(5, tf.float32))
lr = 0.2
steps = 40

for i in range(steps):
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
        # loss = (w + 1) ** 2
        # loss = tf.pow(w + 1, 2)
    grads = tape.gradient(loss, w)
    w.assign_sub(lr * grads)  # -= 是错的
    print(i, loss.numpy(), w.numpy())
