# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-1
import tensorflow as tf

# copy from `course_1/2 backward.py`
w = tf.Variable(tf.constant(5, tf.float32))
lr_base = 0.2
lr_decay = 0.99
lr_decay_step = 1
steps = 40

for i in range(steps):
    lr = lr_base * lr_decay ** (i / lr_decay_step)
    with tf.GradientTape() as tape:
        loss = tf.square(w + 1)
        # loss = (w + 1) ** 2
        # loss = tf.pow(w + 1, 2)
    grads = tape.gradient(loss, w)
    w.assign_sub(lr * grads)  # -= 是错的
    print(i, loss.numpy(), w.numpy(), lr)
