# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-2

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import tensorflow.math as tfm

# 控制显存自动增长
gpus = tf.config.list_physical_devices('GPU')  # 必须大写
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

# ------------------------- 超参
epochs = 500
batch_size = 32
lr = 0.1

# ------------------------- dataset
# 读入数据集
from sklearn.datasets import load_iris

x, y = load_iris(True)

# 数据集乱序
np.random.seed(116)
order = np.random.permutation(x.shape[0])
x = tf.cast(x[order], tf.float32)
y = y[order]

x_train = x[:-30]
y_train = y[:-30]
x_test = x[-30:]
y_test = y[-30:]

# .shuffle()
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)
test_db = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# tf.data.Dataset.from_tensor_slices((x_train, y_train))
# <TensorSliceDataset shapes: ((4,), ()), types: (tf.float32, tf.int32)>
# tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(32)
# <BatchDataset shapes: ((None, 4), (None,)), types: (tf.float32, tf.int32)>

# ------------------------- 建立模型

w1 = tf.Variable(tf.random.truncated_normal([4, 3], 0, 0.1, seed=1))  # [In, Out]
b1 = tf.Variable(tf.random.truncated_normal([3], 0, 0.1, seed=1))  # [Out]

# train
for epoch in range(epochs):
    loss_all = 0.
    for x, y in train_db:
        with tf.GradientTape() as tape:
            pred = x @ w1 + b1
            y = tf.one_hot(y, 3)
            # loss = tf.reduce_sum(- y * tfm.log(tfm.softmax(pred, axis=-1)), axis=-1)
            # loss = tf.reduce_sum(- y * tf.nn.log_softmax(pred), axis=-1)
            loss = tf.nn.softmax_cross_entropy_with_logits(y, pred)
            loss = tf.reduce_mean(loss)
        grads = tape.gradient(loss, [w1, b1])
        loss_all += loss.numpy()
        w1.assign_sub(grads[0] * lr)
        b1.assign_sub(grads[1] * lr)

    # test
    correct = 0
    num = 0
    for x, y in test_db:
        pred = x @ w1 + b1
        pred = tfm.softmax(pred, -1)
        pred = tf.argmax(pred, -1, output_type=tf.int32)
        correct += tf.reduce_sum(tf.cast(pred == y, tf.int32)).numpy()
        num += x.shape[0]
    print("Epoch: %d| Loss: %.6f| ACC: %.4f%%" % (epoch, loss_all / len(train_db), correct / num * 100))
