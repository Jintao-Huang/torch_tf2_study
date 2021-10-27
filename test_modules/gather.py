# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from dev import gather
from torch import gather as _gather
import torch

# In[0]
torch.manual_seed(42)
t = torch.rand(2, 3, 4)
idx = torch.randint(0, 2, (2, 3, 4))
print(torch.allclose(gather(t, 0, idx), _gather(t, 0, idx)))
print(torch.allclose(gather(t, 1, idx), _gather(t, 1, idx)))
print(torch.allclose(gather(t, 2, idx), _gather(t, 2, idx)))
print()
"""Out[0]
True
True
True
"""
