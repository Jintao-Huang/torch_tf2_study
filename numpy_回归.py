import numpy as np
import matplotlib.pyplot as plt


def fc_forward(x, w, b=None):
    """z = x @ w + b

    全连接层(full connect)的前向传播"""
    assert x.shape[1] == w.shape[0]
    if b is None:
        return np.matmul(x, w)
    assert w.shape[1] == b.shape[0]
    return np.matmul(x, w) + b


def fc_backward(x, w, z_grad):
    """z = x @ w + b

    全连接层的反向传播
    :return: x_grad, w_grad, b_grad"""

    # x @ w = z
    # -> z.shape = x.shape[0], z.shape[1]
    assert z_grad.shape[0] == x.shape[0]
    assert z_grad.shape[1] == w.shape[1]

    x_grad = z_grad @ w.T
    w_grad = x.T @ z_grad / x.shape[0]  # 除去 batch_size!!!
    b_grad = np.mean(z_grad, axis=0)

    return x_grad, w_grad, b_grad


def relu_forward(z):
    """a = relu(z)

    relu前向传播"""
    return z * (z > 0)


def relu_backward(z, a_grad):
    """a = relu(z)

    relu反向传播"""
    return a_grad * (z > 0)


def mse_loss(y_true, y_pred):
    """loss = np.mean((y_true - y_pred) ** 2)

    均方误差

    :param y_true: shape = (batch_size, classes_num)
    :param y_pred: shape = (batch_size, classes_num)
    :return: shape = (batch_size,)"""
    assert y_true.shape == y_pred.shape
    return np.mean((y_true - y_pred) ** 2)


def mse_loss_grad(y_true, y_pred):
    """grad = d_loss/d_y_pred = 2 * (y_true - y_pred) * -1

    :param y_true: shape = (batch_size, classes_num)
    :param y_pred: shape = (batch_size, classes_num)
    :return: shape = (batch_size, classes_num)"""

    assert y_true.shape == y_pred.shape
    return 2 * (y_pred - y_true)


def sgd(params, grads, lr=1e-2):
    """sgd优化器

    :param params: List[ndarray]
    :param grads: List[ndarray]
    :param lr: float
    """
    assert isinstance(params, list)
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
    w1 = np.random.normal(0, 0.1, (1, hide_c))
    w2 = np.random.normal(0, 0.1, (hide_c, 1))
    b1 = np.zeros((hide_c,))
    b2 = np.zeros((1,))

    # 3. 训练
    # 此处省略batch_size, 方便理解
    # 网络模型: 2层全连接网络
    # loss: mse(mean square error) 均方误差
    # optim: sgd(stochastic gradient descent)  随机梯度下降. (虽然此处为批梯度下降，别在意这些细节)
    for i in range(501):
        # 1.forward
        z = fc_forward(x, w1, b1)
        a = relu_forward(z)
        y_pred = fc_forward(a, w2, b2)

        # 2. loss
        loss = mse_loss(y_true, y_pred)

        # 3. backward
        pred_grad = mse_loss_grad(y_true, y_pred)
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
