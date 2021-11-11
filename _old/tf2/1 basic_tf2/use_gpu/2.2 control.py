# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import tensorflow as tf

# ---------------- 限制显存增长
# 1. 通过调用来激活内存增长（动态增长)
# 它开始分配很少的内存，随着程序运行，申请更多的GPU内存，
# 注意：不会释放内存，因为这会导致内存碎片
gpus = tf.config.list_physical_devices('GPU')  # 必须大写
if gpus:
    for gpu in gpus:
        tf.config.experimental.set_memory_growth(gpu, True)
    logical_gpus = tf.config.experimental.list_logical_devices('GPU')
    print("Physical GPUs: %s" % gpus)
    print("Logical GPU: %s" % logical_gpus)
# Physical GPUs: [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]
# Logical GPU: [LogicalDevice(name='/device:GPU:0', device_type='GPU')]


# -------------------------
# gpus = tf.config.list_physical_devices('GPU')  # 必须大写
# for gpu in gpus:
#     tf.config.experimental.set_memory_growth(gpu, True)
