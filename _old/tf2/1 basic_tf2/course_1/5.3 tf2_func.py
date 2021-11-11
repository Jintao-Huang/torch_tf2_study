# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-1

import tensorflow as tf
import numpy as np

x = tf.random.normal((5, 32, 32, 3))
y = np.random.randint(0, 10, (5,))
# y = tf.convert_to_tensor(y, dtype=tf.float32)  # 不需要也可以运行
dataset = tf.data.Dataset.from_tensor_slices((x, y))  # 只能用tuple, 不能有list
print(dataset)
for i, j in dataset:
    print(i.shape, j)
# <TensorSliceDataset shapes: ((32, 32, 3), ()), types: (tf.float32, tf.int32)>
# (32, 32, 3) tf.Tensor(3, shape=(), dtype=int32)
# (32, 32, 3) tf.Tensor(8, shape=(), dtype=int32)
# (32, 32, 3) tf.Tensor(8, shape=(), dtype=int32)
# (32, 32, 3) tf.Tensor(4, shape=(), dtype=int32)
# (32, 32, 3) tf.Tensor(5, shape=(), dtype=int32)
