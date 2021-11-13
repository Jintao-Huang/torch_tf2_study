# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date:

"""总结
1. FCN将最后 7*7的Cin个特征图 不使用[flatten+全连接]. 而是用7*7核的卷积，输出[N, Cout, 1, 1]
- 下列证明：这是等价的
"""

import torch
import torch.nn.functional as F

N, Cin, H, W = 16, 100, 3, 3
Cout = 200
torch.manual_seed(42)
x = torch.randn(N, Cin, H, W)
weight = torch.randn(Cout, Cin, H, W)
bias = torch.randn(Cout)
"""卷积:
In: shape[N, Cin, H, W]
Out: shape[N, Out, 1, 1]
weight: shape[Out, Cin, H, W]
bias: shape[Out]
"""
y1 = F.conv2d(x, weight, bias)[:, :, 0, 0]
"""全连接:
In: shape[N*H*W, Cin]
Out: shape[N, Out]
weight: shape[Out, Cin*H*W]
bias: shape[Out]
"""
x = x.view(N, Cin * H * W)  # flatten
weight = weight.view(Cout, Cin * H * W)
#
y2 = F.linear(x, weight, bias)
print(torch.allclose(y1, y2, rtol=1e-4, atol=1e-4))  # True
