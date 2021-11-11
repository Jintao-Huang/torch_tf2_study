# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 
import numpy as np
# ----------------------------- 网格
x = np.array([0, 1, 2])
y = np.array([0, 5, 10])
xi, yi = np.meshgrid(x, y)
print(xi, yi)
# [[0 1 2]
#  [0 1 2]
#  [0 1 2]] [[ 0  0  0]
#  [ 5  5  5]
#  [10 10 10]]
yi, xi = np.mgrid[0:15:5, 0:3:1]
print(xi, yi)
# [[0 1 2]
#  [0 1 2]
#  [0 1 2]] [[ 0  0  0]
#  [ 5  5  5]
#  [10 10 10]]
#  [ 0  5 10]]

print(np.stack([xi.ravel(), yi.ravel()], axis=-1))
# [[ 0  0]
#  [ 1  0]
#  [ 2  0]
#  [ 0  5]
#  [ 1  5]
#  [ 2  5]
#  [ 0 10]
#  [ 1 10]
#  [ 2 10]]
