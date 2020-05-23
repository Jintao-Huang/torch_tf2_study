# Author: Jintao Huang
# Time: 2020-5-16

from torchvision.datasets.cifar import CIFAR10
import torchvision.transforms.transforms as trans
from torch.utils.data import DataLoader
import torch.nn as nn
import torch
import time
from tensorboardX import SummaryWriter


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
            ConvBlock(3, 16, 3, 2, 1, False),  # 16
            ConvBlock(16, 32, 3, 2, 1, False),  # 8
            ConvBlock(32, 64, 3, 2, 1, False)  # 4
        )
        self.classifier = nn.Sequential(
            nn.Linear(4 * 4 * 64, 512),
            nn.ReLU(inplace=True),
            nn.Dropout(0.2),  # drop_p
            nn.Linear(512, 10),
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


def lr_schedule(optim, epoch, milestones, decline_rate=0.1):
    """学习率下降(Decline in learning rate)"""

    def insert_index(x, arr):
        """arr是一个从小到大的arr, x插入arr应该在的下标(二分查找). 你可以改成顺序查找."""

        lo, hi = 0, len(arr)
        while lo < hi:
            mid = (lo + hi) // 2
            if x < arr[mid]:
                hi = mid
            else:
                lo = mid + 1
        return lo

    default_lr = optim.defaults['lr']
    lr = default_lr * decline_rate ** insert_index(epoch, milestones)
    optim.param_groups[0]['lr'] = lr
    return lr


def train(model, loss_fn, optim, train_loader, epoch, device, writer=None):
    for i in range(epoch):
        loss_total, acc_total, start_time = 0., 0., time.time()
        lr = lr_schedule(optim, i, (10, 15), decline_rate=0.1)
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
                loss_mean = loss_total / (j + 1)
                acc_mean = acc_total / (j + 1)
                end_time = time.time()
                if writer is not None:
                    writer.add_scalar("loss", loss, i * len(train_loader) + j + 1)
                    writer.add_scalar("acc", acc, i * len(train_loader) + j + 1)
                print("\r>> Epoch: %d[%d/%d]| Loss: %.6f| Acc: %.2f%%| Time: %.4f| LR: %g" %
                      (i, j + 1, len(train_loader), loss_mean, acc_mean * 100, end_time - start_time, lr), end="")
        else:
            loss_mean = loss_total / (j + 1)
            acc_mean = acc_total / (j + 1)
            end_time = time.time()
            if writer is not None:
                writer.add_scalar("loss", loss, i * len(train_loader) + j + 1)
                writer.add_scalar("acc", acc, i * len(train_loader) + j + 1)
                writer.flush()
            print("\r>> Epoch: %d[%d/%d]| Loss: %.6f| Acc: %.2f%%| Time: %.4f| LR: %g" %
                  (i, j + 1, len(train_loader), loss_mean, acc_mean * 100, end_time - start_time, lr), flush=True)


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
            if i % 10 == 0:
                acc = acc_total / (i + 1)
                end_time = time.time()
                print("\r>> %d/%d| Acc: %.2f%%| Time: %.4f" %
                      (i + 1, len(test_loader), acc * 100, end_time - start_time), end="")
        else:
            acc = acc_total / (i + 1)
            end_time = time.time()
            print("\r>> %d/%d| Acc: %.2f%%| Time: %.4f" %
                  (i + 1, len(test_loader), acc * 100, end_time - start_time))


def main():
    epoch = 18

    if torch.cuda.is_available():
        device = torch.device('cuda')
    else:
        device = torch.device('cpu')

    # 1. 数据集
    train_cifar10 = CIFAR10("./cifar", train=True, transform=trans.ToTensor(), download=True)
    test_cifar10 = CIFAR10("./cifar", train=False, transform=trans.ToTensor(), download=True)

    train_loader = DataLoader(train_cifar10, 32, True, pin_memory=True, num_workers=1)
    test_loader = DataLoader(test_cifar10, 32, True, pin_memory=True, num_workers=1)
    # 若运行报错使用下面的.
    # train_loader = DataLoader(train_cifar10, 32, True, pin_memory=True)
    # test_loader = DataLoader(test_cifar10, 32, True, pin_memory=True)

    # 2. 建立网络、损失、优化器
    model = SimpleCNN().to(device)
    loss_fn = nn.CrossEntropyLoss()
    # optim = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=2e-5, nesterov=True)
    # optim = torch.optim.SGD(model.parameters(), 1e-2, momentum=0.9, weight_decay=2e-5)
    optim = torch.optim.Adam(model.parameters(), 5e-4, weight_decay=2e-5)
    writer = SummaryWriter()

    # 3. 训练集训练
    print("---------------------- Train")
    train(model, loss_fn, optim, train_loader, epoch, device, writer)
    # 测试集测试
    print("---------------------- Test")
    test(model, test_loader, device)
    save_params(model, "cifar10_model.pth")
    writer.close()


if __name__ == '__main__':
    main()
