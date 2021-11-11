# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 


"""总结
1. 有些[广播+求和]可以优化成矩阵乘，从而加速运算，减少内存消耗
- 求和可以包括多个dim. 详情见`dev>torch>nn>functional>conv2d`
"""
import torch


def matmul_broadcast(a, b):
    return torch.sum(a[..., None] * b[..., None, :, :], dim=-2)


# In[0]: 基础
rng = torch.Generator().manual_seed(42)
a = torch.randn((100, 10), generator=rng)
b = torch.randn((10, 20), generator=rng)
y = a @ b
y2 = matmul_broadcast(a, b)
print(torch.allclose(y, y2, atol=1e-6))  # True
#
a = torch.randn((200, 100, 10), generator=rng)
b = torch.randn((200, 10, 20), generator=rng)
y = a @ b
y2 = matmul_broadcast(a, b)
print(torch.allclose(y, y2, atol=1e-6))  # True

# In[1]. 测速度1
import time

a = torch.randn((1, 100000), generator=rng)
b = torch.randn((100000, 1), generator=rng)
t = time.perf_counter()
c1 = a @ b
print(f"time: {time.perf_counter() - t: .6f}")
#
t = time.perf_counter()
c2 = matmul_broadcast(a, b)
print(f"time: {time.perf_counter() - t: .6f}")
#
print(torch.allclose(c1, c2, rtol=1e-4, atol=1e-4))
"""Out[1]
time:  0.000589
time:  0.000145
True
"""

# In[2]. 测速度2
import time

a = torch.randn((1, 10000), generator=rng)
b = torch.randn((10000, 100), generator=rng)
t = time.perf_counter()
c1 = a @ b
print(f"time: {time.perf_counter() - t: .6f}")
#
t = time.perf_counter()
c2 = matmul_broadcast(a, b)
print(f"time: {time.perf_counter() - t: .6f}")
#
print(torch.allclose(c1, c2, atol=1e-4))
"""Out[2]
time:  0.000168
time:  0.001456
True
"""

# In[3]. 测速度3
import time

a = torch.randn((100, 10000), generator=rng)
b = torch.randn((10000, 100), generator=rng)
t = time.perf_counter()
c1 = a @ b
print(f"time: {time.perf_counter() - t: .6f}")
#
t = time.perf_counter()
c2 = matmul_broadcast(a, b)
print(f"time: {time.perf_counter() - t: .6f}")
#
print(torch.allclose(c1, c2, atol=1e-4))
"""Out[3]
time:  0.000950
time:  0.125832
True
"""
