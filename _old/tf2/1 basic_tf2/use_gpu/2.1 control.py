# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import tensorflow as tf

# ---------------- 限制GPU
# 1. 默认将使用所有GPU
# tf.config.list_physical_devices('CPU')
gpus = tf.config.list_physical_devices('GPU')  # 必须大写
# if gpus:
#     tf.config.set_visible_devices([], 'GPU')
#     gpus = tf.config.list_physical_devices('GPU')
#     logical_gpus = tf.config.list_logical_devices('GPU')
#     print("Physical GPUs: %s" % gpus)
#     print("Logical GPU: %s" % logical_gpus)
# # Physical GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# # Logical GPU: []

if gpus:
    tf.config.set_visible_devices([gpus[0]], 'GPU')
    gpus = tf.config.list_physical_devices('GPU')
    logical_gpus = tf.config.list_logical_devices('GPU')
    print("Physical GPUs: %s" % gpus)
    print("Logical GPU: %s" % logical_gpus)
# Physical GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# Logical GPU: [LogicalDevice(name='/device:GPU:0', device_type='GPU')]
