# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch
import torch.nn.functional as F
from dev.torch.nn.functional import conv_transpose2d

# In[0]
device = 'cuda'
N, Cin, H, W = 16, 32, 50, 60
KH, KW, PH, PW = 3, 4, 2, 3
SH, SW = 1, 2
Cout = 64
#
rng = torch.Generator(device).manual_seed(42)
x = torch.randn((N, Cin, H, W), generator=rng, device=device)
weight = torch.randn((Cin, Cout, KH, KW), generator=rng, device=device)
bias = torch.randn(Cout, generator=rng, device=device)
#
y1 = conv_transpose2d(x, weight, bias, (SH, SW), (PH, PW))
y2 = F.conv_transpose2d(x, weight, bias, (SH, SW), (PH, PW))
#
print(torch.allclose(y1, y2, rtol=1e-3, atol=1e-3))  # True
