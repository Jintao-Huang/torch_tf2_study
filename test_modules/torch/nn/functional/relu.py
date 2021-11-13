# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from dev.torch.nn.functional import relu
import torch.nn.functional as F
import torch

device = 'cuda'
#
rng = torch.Generator(device).manual_seed(42)
x = torch.randn((100,), generator=rng, device=device)
#
y1 = F.relu(x, inplace=True)
y2 = relu(x)
#
print(x is y1)  # True
print(torch.allclose(y1, y2))  # True
