# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import tensorflow as tf
import tensorflow.nn as nn
import tensorflow.keras.losses as losses
import matplotlib.pyplot as plt
import numpy as np

# 正则化只会weight使用，不对bias和bn中的weight使用
# l1正则化大概率会使很多参数变为0，因此此方法可以通过稀疏参数，减少参数的数量，降低复杂度
# l2正则化会使参数很接近零但不为0，因此此方法可通过减少参数值大小降低复杂度

# 控制显存自动增长
gpus = tf.config.list_physical_devices('GPU')  # 必须大写
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
# ------------------------
batch_size = 32
lr = 0.1
epochs = 500
weight_decay = 0  # 1e-3
# ------------------------
# read csv
x_train = []
y_train = []
with open('dot.csv', "r") as f:
    next(f)
    for line in f:
        x1, x2, y_true = line.split(",")
        x1, x2, y_true = float(x1), float(x2), int(y_true)
        x_train.append([x1, x2])
        y_train.append([y_true])

x_train = tf.convert_to_tensor(x_train, dtype=tf.float32)  # [X, 2]
y_train = tf.convert_to_tensor(y_train, dtype=tf.float32)  # [X, 1]
train_db = tf.data.Dataset.from_tensor_slices((x_train, y_train)).batch(batch_size)

# build model
tf.random.set_seed(0)
w1 = tf.Variable(tf.random.truncated_normal((2, 20), 0, 0.1), dtype=tf.float32)
b1 = tf.Variable(tf.constant(0.01, shape=(20,)))
w2 = tf.Variable(tf.random.truncated_normal((20, 1), 0, 0.1), dtype=tf.float32)
b2 = tf.Variable(tf.constant(0.01, shape=(1,)))
params = [w1, b1, w2, b2]
for epoch in range(epochs):
    loss_all = 0.
    for i, (x, y_true) in enumerate(train_db):
        with tf.GradientTape() as tape:
            x = x @ w1 + b1
            x = nn.relu(x)
            y_pred = x @ w2 + b2
            loss_ce = tf.reduce_mean(losses.binary_crossentropy(y_true, y_pred, True))
            loss_regularization = []
            loss_regularization.append(nn.l2_loss(w1))
            loss_regularization.append(nn.l2_loss(w2))
            loss = loss_ce + sum(loss_regularization) * weight_decay
        grads = tape.gradient(loss, params)
        loss_all += loss.numpy()
        for i in range(len(params)):
            params[i].assign_sub(lr * grads[i])
    print("Epoch: %d| Loss: %.6f" % (epoch, loss_all / len(train_db)))

# 预测部分
print("*******predict*******")
_x = np.arange(-3, 3, 0.1)
x_grid, y_grid = np.meshgrid(_x, _x)
x_test = np.stack([x_grid.ravel(), y_grid.ravel()], axis=1)  # shape[3600, 2]
x_test = tf.cast(x_test, tf.float32)

probs = []
for x in x_test:
    x = x[None] @ w1 + b1
    x = nn.relu(x)
    y_pred = x @ w2 + b2
    y_pred = tf.nn.sigmoid(y_pred)
    probs.append(y_pred.numpy())

# 画点
x1, x2 = tf.transpose(x_train[:, :2])
y = np.squeeze(y_train)
plt.scatter(x1, x2, c=y)

# 画线
probs = np.array(probs).reshape(x_grid.shape)
plt.contour(x_grid, y_grid, probs, levels=[0.5])
plt.savefig("images/no.png")
# Epoch: 499| Loss: 0.042338
