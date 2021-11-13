# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 


import torch.nn.functional as F
from dev.torch.nn.functional import max_pool2d
import torch

# In[0]
x = torch.randn(16, 32, 20, 30, device='cuda')
y1 = max_pool2d(x, (3, 4), (2, 3), (1, 2))
y2 = F.max_pool2d(x, (3, 4), (2, 3), (1, 2))
print(torch.allclose(y1, y2, atol=1e-6))  # True
