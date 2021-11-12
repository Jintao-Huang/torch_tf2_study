# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from dev.torch import dist
from torch import dist as _dist
import torch

# In[0]
torch.manual_seed(42)
x = torch.randn(10)
y = torch.randn(10)
print(dist(x, y))
print(_dist(x, y))
print()
print(dist(x, y, p=1))
print(_dist(x, y, p=1))
"""Out[0]
tensor(3.4248)
tensor(3.4248)

tensor(9.5396)
tensor(9.5396)
"""
