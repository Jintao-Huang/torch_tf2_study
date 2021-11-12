# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
from dev.torch import corrcoef

# In[0]:
x = torch.randn(10, 20)
print(torch.allclose(torch.corrcoef(x), corrcoef(x)))
print()
"""Out[0]
True
"""
