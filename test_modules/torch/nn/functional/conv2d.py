# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import torch.nn.functional as F
from dev.torch.nn.functional import conv2d
from dev.torch.nn.functional.conv2d import _conv2d_easy
import torch

#
device = 'cuda'
N, Cin, H, W = 16, 32, 50, 60
KH, KW, PH, PW = 3, 4, 2, 3
SH, SW = 1, 2
DH, DW = 1, 2
G = 32
Cout = 64
# In[0]: test conv2d
rng = torch.Generator(device).manual_seed(42)
x = torch.randn((N, Cin, H, W), generator=rng, device=device)
weight = torch.randn((Cout, Cin // G, KH, KW), generator=rng, device=device)
bias = torch.randn(Cout, generator=rng, device=device)
#
y1 = F.conv2d(x, weight, bias, (SH, SW), (PH, PW), (DH, DW), G)
y2 = conv2d(x, weight, bias, (SH, SW), (PH, PW), (DH, DW), G)
#
print(torch.allclose(y1, y2, rtol=1e-4, atol=1e-4))  # True

# In[1]: test conv2d_easy
rng = torch.Generator(device).manual_seed(42)
x = torch.randn((N, Cin, H, W), generator=rng, device=device)
weight = torch.randn((Cout, Cin, KH, KW), generator=rng, device=device)
bias = torch.randn(Cout, generator=rng, device=device)
#
y1 = F.conv2d(x, weight, bias, (SH, SW), (PH, PW))
y2 = _conv2d_easy(x, weight, bias, (SH, SW), (PH, PW))
#
print(torch.allclose(y1, y2, rtol=1e-4, atol=1e-4))  # True
