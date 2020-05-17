# Author: Jintao Huang
# Time: 2020-5-17

# 为防止与torch中的函数搞混，自己实现的函数前会加上 `_`
import torch


def to_categorical(x, num_classes=None):
    """转热码  (已测试)

    :param x: shape = (N,)
    :param num_classes: 默认 num_classes = max(x) + 1
    :return: shape = (N, class_num)"""

    assert x.dtype in (torch.int32, torch.int64), "x的类型只支持torch.int32与torch.int64"

    x_max = torch.max(x)
    num_classes = num_classes or x_max + 1
    assert num_classes >= x_max + 1, "num_classes 必须 >= max(x) + 1"
    return torch.eye(num_classes, device=x.device)[x]


def _cross_entropy(y_pred, y_true):
    """交叉熵损失函数

    :param y_pred: shape = (N, num_classes). with_logits(未过softmax)
    :param y_true: shape = (N,)
    :return: shape = ()"""

    y_pred = torch.clamp_min(torch.softmax(y_pred, dim=-1), 1e-12)  # 防log(0)
    y_true = to_categorical(y_true, y_pred.shape[1])

    return torch.mean(torch.sum(y_true * -torch.log(y_pred), -1))


def focal_loss(y_pred, y_true, gamma=2):
    """f(x) = -(1 - x)^a ln(x)
    :param y_pred: shape = (N, num_classes)
    :param y_true: shape = (N,)"""

    y_pred = torch.clamp_min(torch.softmax(y_pred, dim=-1), 1e-12)
    y_true = to_categorical(y_true, y_pred.shape[1])

    return torch.mean(torch.sum(y_true * -torch.log(y_pred) * (1 - y_pred) ** gamma, -1))


def _binary_cross_entropy(y_pred, y_true, with_logits=False):
    """交叉熵损失函数(y_pred已过sigmoid)

    :param y_pred: shape = (N,)
    :param y_true: shape = (N,)
    :return: shape = ()"""

    if not with_logits:
        y_pred = torch.sigmoid(y_pred)
    y_pred = torch.clamp_min(y_pred, 1e-12)
    return torch.mean(y_true * -torch.log(y_pred) +
                      (1 - y_true) * -torch.log(1 - y_pred))


def _mse_loss(y_pred, y_true):
    """均方误差损失

    :param y_pred: shape = (N, num)
    :param y_true: shape = (N, num)
    :return: shape = ()"""

    return torch.mean((y_true - y_pred) ** 2)


def _sigmoid(x):
    """sigmoid

    :param x: shape[N, num_classes] or shape[N, C, 1, 1] ...Any
    :return: shape = x.shape"""

    return 1 / (1 + torch.exp(-x))


def _softmax(x):
    """softmax

    :param x: shape[N, num_classes]
    :return: shape = x.shape"""

    return torch.exp(x) / torch.sum(torch.exp(x), -1, True)


def _batch_norm(x, weight, bias, running_mean, running_var,
                training=False, momentum=0.1, eps=1e-5):
    """BN

    :param x: shape = (N, In) or (N, Cin, H, W)
    :param weight: gamma, shape = (In,)
    :param bias: beta, shape = (In,)
    :param running_mean: shape = (In,)
    :param running_var: shape = (In,)

    :param momentum(同torch): 动量实际为 1 - momentum
    :return: shape = x.shape"""

    # training:
    #   归一化x 使用x.mean(), x.var()
    #   running_mean, running_var进行滑动平均 (forward时更新)
    #   weight, bias (backward时更新)

    # not training:
    #   归一化x 使用 running_mean, running_var
    #   running_mean, running_var, weight, bias 不更新

    assert x.dim() in (2, 4)

    dim = 0 if x.dim() == 2 else (0, 2, 3)
    mean = torch.mean(x, dim)  # shape(In,)
    var = torch.var(x, dim, False)  # 不是估计值

    if training:
        # 归一化x 使用x.mean(), x.var()
        # running_mean, running_var进行滑动平均 (forward时更新)
        running_mean[:] = (1 - momentum) * running_mean + momentum * mean
        running_var[:] = (1 - momentum) * running_var + momentum * torch.var(x, dim, True)  # 无偏估计
    else:
        # 归一化x 使用 running_mean, running_var
        mean = running_mean
        var = running_var

    if x.dim() == 4:
        mean, var = mean[:, None, None], var[:, None, None]
        weight, bias = weight[:, None, None], bias[:, None, None]
    return (x - mean) / (torch.sqrt(var + eps)) * weight + bias


def _dropout(x, drop_p, training):
    """如果是training: 返回引用. 如果not training: 返回new的tensor

    :param x: shape = (N, In)
    :return: shape = x.shape"""

    if not training:
        return x

    keep_p = 1 - drop_p
    keep_tensors = torch.floor(keep_p + torch.rand(x.shape, dtype=x.dtype, device=x.device))

    return x / keep_p * keep_tensors  # 只有该步会反向传播


