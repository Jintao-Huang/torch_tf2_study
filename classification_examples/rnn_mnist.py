# Author: Jintao Huang
# Time: 2020-5-14

from torchvision.datasets.mnist import MNIST
import torchvision.transforms.transforms as trans
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time
import os
from tensorboardX import SummaryWriter


class SimpleRNN(nn.Module):
    """VGG like"""

    def __init__(self):
        super(SimpleRNN, self).__init__()
        self.features = nn.LSTM(28, 64, 2, True, True, 0., False)
        self.classifier = nn.Sequential(  # 分类器
            nn.Linear(64 * 28, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # drop_p
            nn.Linear(512, 10),
        )

    def forward(self, x):
        x = x.reshape((-1, 28, 28))
        x, _ = self.features(x)  # x: shape(N, T, Ch)
        x = torch.flatten(x, start_dim=1)
        # 或只使用最后一个T, 相应要修改self.classifier第一个linear的in_channels
        # x = x[:, -1, :]
        x = self.classifier(x)
        return x


def save_params(model, filename):
    torch.save(model.state_dict(), filename)


def load_params(model, filepath):
    return model.load_state_dict(torch.load(filepath))


def get_acc(pred, label):
    pred = torch.argmax(pred, dim=1)
    return torch.mean((pred == label).float())


def train(model, train_loader, loss_fn, optim, lr_scheduler, writer, epoch, device):
    for i in range(epoch):
        loss_total, acc_total, start_time = 0., 0., time.time()
        lr_scheduler.step(i)
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
            writer.add_scalar("loss", loss, i * len(train_loader) + j + 1)
            writer.add_scalar("acc", acc, i * len(train_loader) + j + 1)
            if j % 20 == 0:
                loss = loss_total / (j + 1)
                acc = acc_total / (j + 1)
                end_time = time.time()
                print("\r>> Epoch: %d[%d/%d]|| Loss: %.6f| Acc: %.2f%%| Time: %.4f" %
                      (i, j + 1, len(train_loader), loss, acc * 100, end_time - start_time), end="")
        else:
            loss = loss_total / (j + 1)
            acc = acc_total / (j + 1)
            end_time = time.time()
            print("\r>> Epoch: %d[%d/%d]| Loss: %.6f| Acc: %.2f%%| Time: %.4f" %
                  (i, j + 1, len(train_loader), loss, acc * 100, end_time - start_time), flush=True)


def test(model, test_loader, device):
    acc_total = 0.
    start_time = time.time()
    model.eval()  # 评估模式. 对dropout、bn有作用
    with torch.no_grad():
        for i, (data, label) in enumerate(test_loader):
            data, label = data.to(device), label.to(device)
            pred = model(data)
            # 后续操作
            acc = get_acc(pred, label)
            acc_total += acc.item()
            if i % 20 == 0:
                acc = acc_total / (i + 1)
                end_time = time.time()
                print("\r>> %d/%d| Acc: %.2f%%| Time: %.4f" %
                      (i + 1, len(test_loader), acc * 100, end_time - start_time), end="")
        else:
            acc = acc_total / (i + 1)
            end_time = time.time()
            print("\r>> %d/%d| Acc: %.2f%%| Time: %.4f" %
                  (i + 1, len(test_loader), acc * 100, end_time - start_time))


class LRScheduler:
    """学习率控制器"""

    def __init__(self, optim, lr_func):
        self.optim = optim
        self.lr_func = lr_func

    def step(self, epoch):
        lr = self.lr_func(epoch)
        self.optim.param_groups[0]['lr'] = lr
        return lr


def lr_func(epoch):
    if 0 <= epoch < 8:
        return 0.01
    else:
        return 1e-3


def main():
    epoch = 10
    save_model_name = "mnist_rnn_model.pth"  # 保存的模型名

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
    model = SimpleRNN().to(device)
    if os.path.exists(save_model_name):
        load_params(model, save_model_name)
    loss_fn = nn.CrossEntropyLoss()
    optim = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=1e-4)
    writer = SummaryWriter()
    lr_scheduler = LRScheduler(optim, lr_func)
    # optim = torch.optim.Adam(model.parameters(), 5e-4, weight_decay=1e-4)

    # 3. 训练集训练
    print("---------------------- Train")
    train(model, train_loader, loss_fn, optim, lr_scheduler, writer, epoch, device)
    # 测试集测试
    print("---------------------- Test")
    test(model, test_loader, device)
    save_params(model, save_model_name)
    writer.close()


if __name__ == '__main__':
    main()
