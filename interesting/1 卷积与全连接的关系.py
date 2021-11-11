# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

"""结论:
1. 卷积和全连接是可以相互转换的
- 卷积等于: [KH * KW * C]上的全连接
"""

import torch

rng = torch.Generator().manual_seed(42)
N, Cin, H, W = 16, 32, 8, 8
Cout = 64
K, P = 1, 0
#
x = torch.randn((N, Cin, H, W), generator=rng)  # [N, Cin, H, W]

# In[0]: 1*1卷积核. 基础
"""卷积权重:
In: shape[N, Cin, H, W]
Out: shape[N, Cout, H, W]
weight: shape[Cout, Cin, 1, 1]
bias: shape[Cout]
"""
torch.manual_seed(42)
conv = torch.nn.Conv2d(Cin, Cout, K, 1, P, bias=True)
"""全连接权重:
In: shape[N*H*W, In]
Out: shape[N*H*W, Out]
weight: shape[Out, In]
bias: shape[Out]
"""
torch.manual_seed(42)
linear = torch.nn.Linear(Cin, Cout, bias=True)

x1 = x
y1 = conv(x1)
x2 = x.permute([0, 2, 3, 1]).contiguous().view(-1, Cin)  # [N, In]
y2 = linear(x2)
y2 = y2.view([N, H, W, Cout]).permute(0, 3, 1, 2)
print(torch.allclose(y1, y2, atol=1e-6))  # True

# In[1]: K*K卷积. 进阶
K, P = 5, 2
"""卷积权重:
In: shape[N, Cin, H, W]
Out: shape[N, Cout, H, W]
weight: shape[Cout, Cin, K, K]
bias: shape[Cout]
"""
torch.manual_seed(42)
conv = torch.nn.Conv2d(Cin, Cout, K, 1, P, bias=True)
"""全连接权重:
In: shape[N*H*W, In*K*K]
Out: shape[N*H*W, Out]
weight: shape[Out, In*K*K]
bias: shape[Out]
"""
torch.manual_seed(42)
linear = torch.nn.Linear(Cin * K * K, Cout, bias=True)

x1 = x
y1 = conv(x1)
#
x2 = torch.nn.ZeroPad2d((P, P, P, P))(x)
x3 = torch.empty((*x.shape, K, K))
for i in range(H):
    for j in range(W):
        x3[:, :, i, j, :, :] = x2[:, :, i:i + K, j:j + K]
#
x2 = x3.permute([0, 2, 3, 1, 4, 5]).contiguous().view(-1, Cin * K * K)
y2 = linear(x2)
y2 = y2.view([N, H, W, Cout]).permute(0, 3, 1, 2)
print(torch.allclose(y1, y2, atol=1e-6))  # True
