# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import tensorflow as tf

# --------------- 查看gpu

print("GPUs Available: ", tf.config.list_physical_devices('GPU'))
# GPUs Available:  [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]

# --------------- log设备位置
tf.debugging.set_log_device_placement(True)

a = tf.constant([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
b = tf.constant([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]])
c = tf.matmul(a, b)
# Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
print(a.device)
print(c.device)
# /job:localhost/replica:0/task:0/device:CPU:0
# /job:localhost/replica:0/task:0/device:GPU:0
# --------------- 手动使用什么设备
# Executing op MatMul in device /job:localhost/replica:0/task:0/device:CPU:0
with tf.device('CPU:0'):
    c = tf.matmul(a, b)
# Executing op MatMul in device /job:localhost/replica:0/task:0/device:GPU:0
d = tf.matmul(a, b)
with tf.device('GPU:0'):  # 或 'GPU'
    f = tf.matmul(a, b)
# ---------------- default
# tf.constant: CPU
# random.XXX: GPU
# OP: GPU


