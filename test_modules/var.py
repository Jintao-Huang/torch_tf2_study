# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from dev import var

# In[0]: 1D
x = torch.randn(10)
print(torch.var(x))
print(var(x))
print(torch.var(x, unbiased=False))
print(var(x, unbiased=False))
print()
"""Out[0]
tensor(0.4506)
tensor(0.4506)
tensor(0.4056)
tensor(0.4056)
"""
# In[1]: 2D
x = torch.randn(20, 10)
print(torch.var(x))
print(var(x))
print(torch.var(x, unbiased=False))
print(var(x, unbiased=False))
"""Out[1]
tensor(1.0820)
tensor(1.0820)
tensor(1.0766)
tensor(1.0766)
"""
