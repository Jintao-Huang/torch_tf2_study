# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 2021-6-2

import tensorflow as tf
from sklearn.datasets import load_iris
import pandas as pd

x, y = load_iris(True)
print(x.shape, y.shape)  # (150, 4) (150,)
iris = load_iris(False)
print(iris.data.shape)  # (150, 4)
print(iris.feature_names)  # ['sepal length (cm)', 'sepal width (cm)', 'petal length (cm)', 'petal width (cm)']
print(iris.target.shape)  # (150,)
print(iris.target_names)  # ['setosa' 'versicolor' 'virginica']

# ----------------------------
print("----------------------------")
x = pd.DataFrame(x, columns=iris.feature_names)
print("x add index: \n", x)

x['target'] = y
print("x add a column: \n", x)




