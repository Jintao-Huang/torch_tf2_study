# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from dev.torch.nn.functional import relu
from torch.nn.functional import relu as _relu
import torch

device = 'cuda'
#
rng = torch.Generator(device).manual_seed(42)
x = torch.randn((100,), generator=rng, device=device)
#
y1 = relu(x)
y2 = _relu(x, inplace=True)
#
print(x is y2)  # True
print(torch.allclose(y1, y2))  # True
