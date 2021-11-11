# Author: Jintao Huang
# Email: hjt_study@qq.com
# Date: 

import tensorflow as tf


# 优化4步
# 1. 计算损失函数关于当前梯度的偏导数
# 2. 计算一阶动量mt和二阶动量vt
# 3. 计算下降梯度
# 4. 计算下一时刻参数
# wt+1 = wt - lr * mt / sqrt(vt)
# 一阶动量：与梯度相关的函数
# 二阶动量：与梯度平方相关的函数
# 不同的优化器只是设置了不同的动量参数


class Optim:
    def __init__(self, params, lr, beta):
        """

        :param params: List[Variable]
        :param lr: float
        :param beta:
        """
        self.params = params
        self.lr = lr
        self.beta = beta

    def step(self, grads):
        """

        :param grads: dloss / dw
        :return:
        """
        raise NotImplementedError

    @staticmethod
    def _step(param, mt, vt, lr):
        """

        :param param: 参数
        :param mt: 一阶动量
        :param vt: 二阶动量
        :param lr: 学习率
        :return: None
        """
        param.assign_sub(lr * mt / tf.sqrt(vt))


class SGD(Optim):
    def __init__(self, params, lr, beta):
        super(SGD, self).__init__(params, lr, beta)
        self.mt = 0

    def step(self, grads):
        for param, grad in zip(self.params, grads):
            self.mt = self.beta * self.mt + (1 - self.beta) * grad
            self._step(param, self.mt, 1, self.lr)


class Adagrad(Optim):
    def __init__(self, params, lr, beta):
        super(Adagrad, self).__init__(params, lr, beta)
        self.vt = 0

    def step(self, grads):
        for param, grad in zip(self.params, grads):
            self.vt += grad ** 2
            self._step(param, grad, self.vt, self.lr)


class RMSprop(Optim):
    def __init__(self, params, lr, beta):
        super(RMSprop, self).__init__(params, lr, beta)
        self.vt = 0

    def step(self, grads):
        for param, grad in zip(self.params, grads):
            self.vt = self.beta * self.vt + (1 - self.beta) * grad ** 2
            self._step(param, grad, self.vt, self.lr)


class Adam(Optim):
    def __init__(self, params, lr, beta):
        super(Adam, self).__init__(params, lr, beta)
        self.mt, self.vt = 0, 0
        self.beta1, self.beta2 = self.beta
        self.global_step = 0

    def step(self, grads):
        self.global_step += 1
        for param, grad in zip(self.params, grads):
            mt = self.beta1 * self.mt + (1 - self.beta1) * grad
            vt = self.beta2 * self.vt + (1 - self.beta2) * grad ** 2
            self.mt = mt / (1 - tf.pow(self.beta1, int(self.global_step)))
            self.vt = vt / (1 - tf.pow(self.beta2, int(self.global_step)))
            self._step(param, self.mt, self.vt, self.lr)
