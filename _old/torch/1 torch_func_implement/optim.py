# Author: Jintao Huang
# Time: 2020-5-23
# 为防止与torch中的函数搞混，自己实现的函数前会加上 `_`

from torch.optim.optimizer import Optimizer


class _SGD(Optimizer):
    """torch.optim.SGD"""

    def __init__(self, params, lr, momentum=0., dampening=0.,
                 weight_decay=0., nesterov=False):
        defaults = {"lr": lr, "momentum": momentum, "dampening": dampening,
                    "weight_decay": weight_decay, "nesterov": nesterov}
        super(_SGD, self).__init__(params, defaults)

    def step(self, closure=None):
        for group in self.param_groups:  # 一般也就一个group
            params = group['params']
            lr = group['lr']
            momentum = group['momentum']
            dampening = group['dampening']
            weight_decay = group['weight_decay']
            nesterov = group['nesterov']

            for param in params:
                if param.grad is None:
                    continue
                p_grad = param.grad.data  # 赋值时需要
                if weight_decay != 0:
                    p_grad += weight_decay * param.data

                if momentum != 0:
                    param_state = self.state[param]  # 唯一性  defaultdict
                    if 'momentum_buffer' not in param_state.keys():
                        buf = param_state['momentum_buffer'] = p_grad.clone()  # no grad
                    else:
                        buf = param_state['momentum_buffer']
                        buf.data = buf * momentum + (1 - dampening) * p_grad
                    if nesterov:
                        p_grad += momentum * buf
                    else:
                        p_grad = buf

                param.data -= lr * p_grad


class _Adam(Optimizer):
    """torch.optim.Adam"""
    pass
