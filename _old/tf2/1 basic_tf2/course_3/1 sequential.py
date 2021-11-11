# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-8

# 6步法：
# 1. import
# 2. train test
# 3. model = tf.keras.models.Sequential
# 4. model.compile
# 5. model.fit
# 6. model.summary

# Metrics:
# accuracy: 数值.
# categorical: 独热码.
# sparse_categorical_accuracy: y_true独热码

import tensorflow as tf
import tensorflow.keras as keras
from sklearn.datasets import load_iris
import numpy as np

# 控制显存自动增长
gpus = tf.config.list_physical_devices('GPU')  # 必须大写
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)

x, y = load_iris(True)

np.random.seed(116)
order = np.random.permutation(x.shape[0])
x = tf.cast(x[order], tf.float32)
y = y[order]

np.random.seed(116)
model = tf.keras.models.Sequential([
    tf.keras.layers.Dense(3, activation='softmax', kernel_regularizer=keras.regularizers.L2(0))
])

model.compile(optimizer=keras.optimizers.SGD(0.1),
              loss=keras.losses.SparseCategoricalCrossentropy(False),
              metrics=['sparse_categorical_accuracy'])
model.fit(x, y, 32, 200, validation_split=0.2, validation_freq=20)
model.summary()
