# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

"""总结
1. n倍最近邻上采样即重复n次
"""

import torch
import torch.nn.functional as F

x = torch.randn(10, 16, 7, 14)
y1 = F.interpolate(x, scale_factor=2, mode="nearest")  # 等价于 `F.upsample()`
y2 = torch.repeat_interleave(x, 2, 2)
y2 = torch.repeat_interleave(y2, 2, 3)
print(torch.allclose(y1, y2))  # True
#
y1 = F.interpolate(x, scale_factor=(2, 3), mode="nearest")  # 等价于 `F.upsample()`
# y1 = F.upsample(x, scale_factor=(2, 3), mode="nearest")  # deprecated
y2 = torch.repeat_interleave(x, 2, 2)
y2 = torch.repeat_interleave(y2, 3, 3)
print(torch.allclose(y1, y2))  # True
