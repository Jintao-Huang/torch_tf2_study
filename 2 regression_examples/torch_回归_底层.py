# Author: Jintao Huang
# Time: 2020-5-10
# 为防止与torch中的函数搞混，自己实现的函数前会加上 `_`

from torch.autograd import Function
import torch
import matplotlib.pyplot as plt
from torch.optim.optimizer import Optimizer


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


class _MSELoss(Function):
    """loss = torch.mean((y_true - y_pred) ** 2)"""

    @staticmethod
    def forward(ctx, x, target):
        assert x.shape == target.shape
        ctx.save_for_backward(x, target)
        return torch.mean((target - x) ** 2)

    @staticmethod
    def backward(ctx, output_grad):
        """output_grad == 1."""
        x, target = ctx.saved_tensors
        return output_grad * 2 * (x - target), None


class _SGD(Optimizer):
    """pure SGD (完整版见`/torch_底层算法实现/optim.py`)"""

    def __init__(self, params, lr):
        defaults = {"lr": lr}
        super(_SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:  # 一般也就一个group
            params = group['params']
            lr = group['lr']
            for param in params:
                param.data -= lr * param.grad


def main():
    lr = 1e-1
    hide_c = 100  # 隐藏层的通道数
    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')
    # 1. 制作数据集
    # input_c: 1, output_c: 1
    x = torch.linspace(-1, 1, 1000, device=device)[:, None]  # shape(1000, 1)
    y_true = x ** 2 + 2 + torch.normal(0, 0.1, x.shape, device=device)

    # 2. 参数初始化
    w1 = torch.normal(0, 0.1, (1, hide_c), requires_grad=True, device=device)
    w2 = torch.normal(0, 0.1, (hide_c, 1), requires_grad=True, device=device)
    b1 = torch.zeros((hide_c,), requires_grad=True, device=device)
    b2 = torch.zeros((1,), requires_grad=True, device=device)

    # 3. 训练
    optim = _SGD([w1, w2, b1, b2], lr)
    # 此处省略batch_size, 方便理解
    # 网络模型: 2层全连接网络
    # loss: mse(mean square error) 均方误差
    # optim: sgd(stochastic gradient descent)  随机梯度下降. (虽然此处为批梯度下降，别在意这些细节)
    for i in range(501):
        # 1.forward
        z = _Linear().apply(x, w1, b1)
        a = _ReLU().apply(z)
        y_pred = _Linear().apply(a, w2, b2)
        # 2. loss
        loss = _MSELoss().apply(y_pred, y_true)
        # 3. backward
        optim.zero_grad()
        loss.backward()
        # 4.update
        params = [w1, w2, b1, b2]
        optim.step()
        if i % 10 == 0:
            print(i, "%.6f" % loss)

    # 4. 作图
    x = x.cpu().numpy()
    y_true = y_true.cpu().numpy()
    y_pred = y_pred.cpu().detach().numpy()
    plt.scatter(x, y_true, s=20)
    plt.plot(x, y_pred, "r-")
    plt.text(0, 2, "loss %.4f" % loss, fontsize=20, color="r")
    plt.show()


if __name__ == '__main__':
    main()
