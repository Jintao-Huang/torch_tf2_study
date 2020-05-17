# Author: Jintao Huang
# Time: 2020-5-14

from torchvision.datasets.mnist import MNIST
import torchvision.transforms.transforms as trans
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time


class ConvBNReLU(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super(ConvBNReLU, self).__init__(
            nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding, bias=bias),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )


class ConvBlock(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size, stride, padding, bias):
        super().__init__(
            ConvBNReLU(in_channels, out_channels, kernel_size, 1, padding, bias=bias),
            ConvBNReLU(out_channels, out_channels, kernel_size, 1, padding, bias=bias),
            ConvBNReLU(out_channels, out_channels, kernel_size, stride, padding, bias=bias)
        )


class SimpleCNN(nn.Module):
    """VGG like"""

    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            ConvBlock(1, 16, 3, 2, 1, False),  # 14
            ConvBlock(16, 32, 3, 2, 1, False)  # 7
        )
        self.classifier = nn.Sequential(
            nn.Linear(7 * 7 * 32, 256),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # drop_p
            nn.Linear(256, 10),
        )

    def forward(self, x):
        x = self.features(x)
        x = torch.flatten(x, start_dim=1)
        x = self.classifier(x)
        return x


def get_acc(pred, label):
    pred = torch.argmax(pred, dim=1)
    return torch.mean((pred == label).float())


def save_params(model, filename):
    torch.save(model.state_dict(), filename)


def main():
    epoch = 5

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 1. 数据集
    train_mnist = MNIST("./mnist", train=True, transform=trans.ToTensor(), download=True)
    test_mnist = MNIST("./mnist", train=False, transform=trans.ToTensor(), download=True)

    train_loader = DataLoader(train_mnist, 32, True, pin_memory=True, num_workers=1)
    test_loader = DataLoader(test_mnist, 32, True, pin_memory=True, num_workers=1)
    # 若运行报错使用下面的.
    # train_loader = DataLoader(train_mnist, 32, True, pin_memory=True)
    # test_loader = DataLoader(test_mnist, 32, True, pin_memory=True)

    # 2. 建立网络、损失、优化器
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=1e-4)
    # optim = torch.optim.Adam(model.parameters(), 5e-4, weight_decay=1e-4)

    # 3. 训练集训练
    print("---------------------- Train")
    for i in range(epoch):
        loss_total, acc_total, start_time = 0., 0., time.time()
        for j, (data, label) in enumerate(train_loader):
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = loss_fn(pred, label)
            optim.zero_grad()
            loss.backward()
            optim.step()
            # 后续操作
            acc = get_acc(pred, label)
            loss_total += loss.item()
            acc_total += acc.item()
            if j % 10 == 0:
                loss = loss_total / (j + 1)
                acc = acc_total / (j + 1)
                end_time = time.time()
                print("\r>> %d / %d| Loss: %.6f| Acc: %.2f%%| Time: %.4f" %
                      (j + 1, len(train_loader), loss, acc * 100, end_time - start_time), end="")
        else:
            loss = loss_total / (j + 1)
            acc = acc_total / (j + 1)
            end_time = time.time()
            print("\r>> %d / %d| Loss: %.6f| Acc: %.2f%%| Time: %.4f" %
                  (j + 1, len(train_loader), loss, acc * 100, end_time - start_time))

    # 测试集测试
    print("---------------------- Test")

    loss_total, acc_total = 0., 0.
    start_time = time.time()
    model.eval()  # 评估模式. 对dropout、bn有作用
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            pred = model(data)
            loss = loss_fn(pred, label)  # 就看看
            # 后续操作
            acc = get_acc(pred, label)
            loss_total += loss.item()
            acc_total += acc.item()
            if i % 10 == 0:
                loss = loss_total / (i + 1)
                acc = acc_total / (i + 1)
                end_time = time.time()
                print("\r>> %d / %d| Loss: %.6f| Acc: %.2f%%| Time: %.4f" %
                      (i + 1, len(test_loader), loss, acc * 100, end_time - start_time), end="")
        else:
            loss = loss_total / (i + 1)
            acc = acc_total / (i + 1)
            end_time = time.time()
            print("\r>> %d / %d| Loss: %.6f| Acc: %.2f%%| Time: %.4f" %
                  (i + 1, len(test_loader), loss, acc * 100, end_time - start_time))

    save_params(model, "mnist_model.pth")


if __name__ == '__main__':
    main()
