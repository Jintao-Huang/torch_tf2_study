# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from dev import cov

# In[0]:
x = torch.randn(10, 20)
print(torch.allclose(torch.cov(x), cov(x)))
print(torch.allclose(torch.cov(x, correction=0), cov(x, correction=0)))
print()
"""Out[0]
True
True
"""
