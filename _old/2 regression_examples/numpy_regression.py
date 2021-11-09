# Author: Jintao Huang
# Time: 2020-5-10

import numpy as np
import matplotlib.pyplot as plt


def fc_forward(x, w, b=None):
    """fc前向

    :param x: shape[N, In]
    :param w: shape[Out, In]
    :param b: shape[Out]
    :return: shape[N, Out]
    """
    if b is None:
        return x @ w.T
    return x @ w.T + b


def fc_backward(x, w, z_grad):
    """fc反向

    :param x: shape[N, In]
    :param w: shape[Out, In]
    :param z_grad: shape[N, Out]
    :return: Tuple[x_grad, w_grad, b_grad]
        x_grad: shape[N, In]
        w_grad: shape[Out, In]
        b_grad: shape[Out]
    """

    x_grad = z_grad @ w
    w_grad = z_grad.T @ x
    b_grad = np.sum(z_grad, axis=0)

    return x_grad, w_grad, b_grad


def relu_forward(z):
    """relu前向

    :param z: shape[Out]
    :return: shape[Out]
    """
    return z * (z > 0)


def relu_backward(z, a_grad):
    """relu反向

    :param z: shape[Out]
    :param a_grad: shape[Out]
    :return: shape[Out]
    """
    return a_grad * (z > 0)


def mse_loss_forward(y_true, y_pred):
    """均方误差前向

    :param y_true: shape[N, Out]
    :param y_pred: shape[N, Out]
    :return: shape[]"""
    return np.mean(np.sum((y_true - y_pred) ** 2, axis=1))


def mse_loss_backward(y_true, y_pred):
    """均方误差反向

    :param y_true: shape[N, Out]
    :param y_pred: shape[N, Out]
    :return: shape[N, Out]"""

    N = y_true.shape[0]  # batch_size
    return 2 * (y_pred - y_true) / N


def sgd(params, grads, lr=1e-2):
    """sgd优化器

    :param params: List[ndarray]
    :param grads: List[ndarray]
    :param lr: float
    """
    for i in range(len(params)):
        params[i] -= lr * grads[i]


def main():
    lr = 1e-1
    hide_c = 100  # 隐藏层的通道数

    # 1. 制作数据集
    # input_c: 1, output_c: 1
    x = np.linspace(-1, 1, 1000)[:, None]  # shape(1000, 1)
    y_true = x ** 2 + 2 + np.random.normal(0, 0.1, x.shape)

    # 2. 参数初始化
    w1 = np.random.normal(0, 0.1, (1, hide_c)).T
    w2 = np.random.normal(0, 0.1, (hide_c, 1)).T
    b1 = np.zeros((hide_c,))
    b2 = np.zeros((1,))

    # 3. 训练
    # 此处省略batch_size, 方便理解
    # 网络模型: 2层全连接网络
    # loss: mse(mean square error) 均方误差
    # optim: sgd(stochastic gradient descent)  随机梯度下降. (虽然此处为批梯度下降，别在意这些细节)
    for i in range(1001):
        # 1.forward
        z = fc_forward(x, w1, b1)
        a = relu_forward(z)
        y_pred = fc_forward(a, w2, b2)

        # 2. loss
        loss = mse_loss_forward(y_true, y_pred)

        # 3. backward
        pred_grad = mse_loss_backward(y_true, y_pred)
        a_grad, w2_grad, b2_grad = fc_backward(a, w2, pred_grad)
        z_grad = relu_backward(z, a_grad)
        _, w1_grad, b1_grad = fc_backward(x, w1, z_grad)

        # 4.update
        params = [w1, w2, b1, b2]
        grads = [w1_grad, w2_grad, b1_grad, b2_grad]
        sgd(params, grads, lr)
        if i % 10 == 0:
            print(i, "%.6f" % loss)

    # 4. 作图
    plt.scatter(x, y_true, s=20)
    plt.plot(x, y_pred, "r-")
    plt.text(0, 2, "loss %.4f" % loss, fontsize=20, color="r")
    plt.show()


if __name__ == '__main__':
    main()
