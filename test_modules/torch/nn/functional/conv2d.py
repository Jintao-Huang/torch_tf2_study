# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

from torch.nn.functional import conv2d as _conv2d
from dev.torch.nn.functional import conv2d
import torch

# In[0]
device = 'cuda'
N, Cin, H, W = 16, 32, 50, 60
KH, KW, PH, PW = 3, 4, 1, 2
SH, SW = 1, 2
Cout = 64
#
rng = torch.Generator(device).manual_seed(42)
x = torch.randn((N, Cin, H, W), generator=rng, device=device)
weight = torch.randn((Cout, Cin, KH, KW), generator=rng, device=device)
bias = torch.randn(Cout, generator=rng, device=device)
#
y1 = conv2d(x, weight, bias, (SH, SW), (PH, PW))
y2 = _conv2d(x, weight, bias, (SH, SW), (PH, PW))
#
print(torch.allclose(y1, y2, rtol=1e-3, atol=1e-3))  # True
