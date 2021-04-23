# Author: Jintao Huang
# Time: 2020-5-17

# 为防止与torch中的函数搞混，自己实现的函数前会加上 `_`
# 调用方式请查看 `/回归_examples/torch_回归_底层.py`
from torch.autograd import Function
import torch


class _Linear(Function):
    """z = x @ w + b"""

    @staticmethod
    def forward(ctx, x, w, b=None):
        assert x.shape[1] == w.shape[0]
        ctx.save_for_backward(x, w, b)
        if b is None:
            return x @ w
        assert w.shape[1] == b.shape[0]
        return x @ w + b

    @staticmethod
    def backward(ctx, z_grad):
        x, w, b = ctx.saved_tensors
        x_grad = z_grad @ w.t()
        w_grad = x.t() @ z_grad / x.shape[0]  # 除去 batch_size!!!
        if b is None:
            return x_grad, w_grad
        b_grad = torch.mean(z_grad, dim=0)
        return x_grad, w_grad, b_grad


class _ReLU(Function):
    """a = relu(z)"""

    @staticmethod
    def forward(ctx, z):
        zero_one_arr = (z > 0).float()
        ctx.save_for_backward(zero_one_arr)

        return z * zero_one_arr

    @staticmethod
    def backward(ctx, a_grad):
        zero_one_arr, = ctx.saved_tensors
        return a_grad * zero_one_arr


class _Sigmoid(Function):
    @staticmethod
    def forward(ctx, z):
        a = 1 / (1 + torch.exp(-z))
        ctx.save_for_backward(a)
        return a

    @staticmethod
    def backward(ctx, a_grad):
        a, = ctx.saved_tensors
        return a_grad * a * (1 - a)


class _MSELoss(Function):
    """loss = torch.mean((y_true - y_pred) ** 2)"""

    @staticmethod
    def forward(ctx, x, target):
        assert x.shape == target.shape
        ctx.save_for_backward(x, target)
        return torch.mean((target - x) ** 2)

    @staticmethod
    def backward(ctx, output_grad):
        x, target = ctx.saved_tensors
        return output_grad * 2 * (x - target), None


class _Max(Function):
    """只反向传递最大的那个值的梯度"""
    pass


class _Mean(Function):
    """梯度平分"""
    pass
