import torch
import torch.nn as nn
import matplotlib.pyplot as plt


class SimpleNN(nn.Sequential):
    def __init__(self, in_channels, hide_channels, out_channels):
        super().__init__(
            nn.Linear(in_channels, hide_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hide_channels, out_channels)
        )


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

    # 2. 建立网络、损失、优化器
    model = SimpleNN(1, hide_c, 1).to(device)
    loss_fn = nn.MSELoss()
    optim = torch.optim.SGD(model.parameters(), lr)

    # 3. 训练
    # 此处省略batch_size, 方便理解
    # 网络模型: 2层全连接网络
    # loss: mse(mean square error) 均方误差
    # optim: sgd(stochastic gradient descent)  随机梯度下降. (虽然此处为批梯度下降，别在意这些细节)
    for i in range(501):
        # 1. forward
        y_pred = model(x)
        # 2. loss
        loss = loss_fn(y_pred, y_true)
        # 3. backward
        optim.zero_grad()
        loss.backward()
        # 4. update
        optim.step()
        if i % 10 == 0:
            print(i, "%.6f" % loss.item())
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
