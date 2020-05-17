# Author: Jintao Huang
# Time: 2020-5-17

# 为防止与torch中的函数搞混，自己实现的函数前会加上 `_`
import torch
import torch.nn.functional as F


def to_categorical(x, num_classes=None):
    """转热码(已测试)

    :param x: shape = (N,)
    :param num_classes: 默认 num_classes = max(x) + 1
    :return: shape = (N, class_num)"""

    assert x.dtype in (torch.int32, torch.int64), "x的类型只支持torch.int32与torch.int64"

    x_max = torch.max(x)
    num_classes = num_classes or x_max + 1
    assert num_classes >= x_max + 1, "num_classes 必须 >= max(x) + 1"
    return torch.eye(num_classes, dtype=x.dtype, device=x.device)[x]


def _cross_entropy(y_pred, y_true):
    """交叉熵损失函数(F.cross_entropy() 只实现了部分功能)

    :param y_pred: shape = (N, num_classes). with_logits(未过softmax)
    :param y_true: shape = (N,)
    :return: shape = ()"""

    y_pred = torch.clamp_min(torch.softmax(y_pred, dim=-1), 1e-12)  # 防log(0)
    y_true = to_categorical(y_true, y_pred.shape[1])

    return torch.mean(torch.sum(y_true * -torch.log(y_pred), -1))


def focal_loss(y_pred, y_true, gamma=2):
    """focal_loss(已测试)

    :param y_pred: shape = (N, num_classes)
    :param y_true: shape = (N,)"""
    # f(x) = -(1 - x)^a * ln(x)
    y_pred = torch.clamp_min(torch.softmax(y_pred, dim=-1), 1e-12)
    y_true = to_categorical(y_true, y_pred.shape[1])

    return torch.mean(torch.sum(y_true * -torch.log(y_pred) * (1 - y_pred) ** gamma, -1))


def _binary_cross_entropy(y_pred, y_true, with_logits=False):
    """交叉熵损失函数(F.binary_cross_entropy() and F.binary_cross_entropy_with_logits())

    :param y_pred: shape = (N,)
    :param y_true: shape = (N,)
    :param with_logits: y_pred是否已过sigmoid
    :return: shape = ()"""

    if not with_logits:
        y_pred = torch.sigmoid(y_pred)
    y_pred = torch.clamp_min(y_pred, 1e-12)
    return torch.mean(y_true * -torch.log(y_pred) +
                      (1 - y_true) * -torch.log(1 - y_pred))


def _mse_loss(y_pred, y_true):
    """均方误差损失(F.mse_loss() 只实现了部分功能)

    :param y_pred: shape = (N, num)
    :param y_true: shape = (N, num)
    :return: shape = ()"""

    return torch.mean((y_true - y_pred) ** 2)


def _sigmoid(x):
    """sigmoid(torch.sigmoid())

    :param x: shape[N, num_classes] or shape[N, C, 1, 1] ...Any
    :return: shape = x.shape"""

    return 1 / (1 + torch.exp(-x))


def _softmax(x, dim):
    """softmax(torch.softmax())

    :param x: shape[N, num_classes]
    :param dim: int. 和为1的轴为哪个
    :return: shape = x.shape"""

    return torch.exp(x) / torch.sum(torch.exp(x), dim, True)


def _batch_norm(x, weight, bias, running_mean, running_var,
                training=False, momentum=0.1, eps=1e-5):
    """BN(torch.batch_norm())

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
    #
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
    """(torch.dropout()). 如果是`not training` / `not drop_p`: 返回引用. 如果training: 返回new的tensor

    :param x: shape = (N, In)
    :return: shape = x.shape"""

    if not training or not drop_p:
        return x

    keep_p = 1 - drop_p
    keep_tensors = torch.floor(keep_p + torch.rand(x.shape, dtype=x.dtype, device=x.device))

    return x / keep_p * keep_tensors  # 只有该步会反向传播


def _zero_padding2d(x, padding):
    """零填充(F.pad()). F.pad()不支持padding为int型

    :param x: shape = (N, Cin, Hin, Win) or (Cin, Hin, Win) or (Hin, Win)
    :param padding: Union[int, tuple(left, right, top, bottom)]
    :return: shape = (..., Hout, Wout)"""
    if padding == 0:
        return x
    assert x.dim() in (2, 3, 4)
    if isinstance(padding, int):
        padding = padding, padding, padding, padding
    elif len(padding) != 4:
        padding = *padding, 0, 0, 0
    output = torch.zeros((*x.shape[:-2],  # N, Cin
                          x.shape[-2] + padding[2] + padding[3],  # Hout
                          x.shape[-1] + padding[0] + padding[1]), dtype=x.dtype, device=x.device)  # Wout

    # output.shape[-2]-padding[3] 书写方式 为防止 padding[3] == 0 的错误
    output[..., padding[2]:output.shape[-2] - padding[3], padding[0]:output.shape[-1] - padding[1]] = x
    return output


def _max_pool2d(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    """最大池化(F.max_pool2d()). padding的0.不加入求max()运算

    :param x: shape = (N, Cin, Hin, Win) or (Cin, Hin, Win). 不允许2D
    :param kernel_size: Union[int, tuple(H, W)]
    :param stride: strides: Union[int, tuple(H, W)] = pool_size
    :param padding: Union[int, tuple(H, W)]
    :param .dilation: 未实现.
    :param .ceil_mode: 未实现.
    :return: shape = (B, Cin, Hout, Wout)"""

    assert x.dim() in (3, 4)
    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size
    stride = stride or kernel_size
    if isinstance(stride, int):
        stride = stride, stride
    if isinstance(padding, int):
        padding = padding, padding

    # Out(H, W) = (In(H, W) + 2 * padding − kernel_size) // stride + 1
    output = torch.empty((*x.shape[:-2],
                          (x.shape[-2] + 2 * padding[0] - kernel_size[0]) // stride[0] + 1,  # Hout
                          (x.shape[-1] + 2 * padding[1] - kernel_size[1]) // stride[1] + 1),  # Wout
                         dtype=x.dtype, device=x.device)
    # ---------------- 算法
    for i in range(output.shape[-2]):
        for j in range(output.shape[-1]):
            h_start, w_start = i * stride[0] - padding[0], j * stride[1] - padding[1]
            # h_start, w_start < 0. 会报错
            h_pos, w_pos = slice(h_start if h_start >= 0 else 0, (h_start + kernel_size[0])), \
                           slice(w_start if w_start >= 0 else 0, (w_start + kernel_size[1]))
            output[..., i, j] = torch.max(torch.max(x[..., h_pos, w_pos], dim=-2)[0], dim=-1)[0]  # dim=(-2, -1)
    return output


def _avg_pool2d(x, kernel_size, stride=None, padding=0, dilation=1, ceil_mode=False):
    """平均池化(F.avg_pool2d()). padding的0.加入求mean()运算

    :param x: shape = (N, Cin, Hin, Win) or (Cin, Hin, Win). 不允许2D
    :param kernel_size: Union[int, tuple(H, W)]
    :param stride: strides: Union[int, tuple(H, W)] = pool_size
    :param padding: Union[int, tuple(H, W)]
    :param .dilation: 未实现.
    :param .ceil_mode: 未实现.
    :return: shape = (B, Cin, Hout, Wout)"""
    assert x.dim() in (3, 4)
    if isinstance(kernel_size, int):
        kernel_size = kernel_size, kernel_size
    stride = stride or kernel_size
    if isinstance(stride, int):
        stride = stride, stride

    # 这里与max_pool2d 有很大区别，avg_pool2d 计算要和pad的0. 一起求平均
    if isinstance(padding, int):
        padding = padding, padding, padding, padding
    else:
        padding = padding[1], padding[1], padding[0], padding[0]
    if padding:
        x = F.pad(x, padding)

    # Out(H, W) = (In(H, W) + 2 * padding − kernel_size) // stride + 1
    output = torch.empty((*x.shape[:-2],
                          (x.shape[-2] - kernel_size[0]) // stride[0] + 1,  # Hout (x已加上padding)
                          (x.shape[-1] - kernel_size[1]) // stride[1] + 1),  # Wout
                         dtype=x.dtype, device=x.device)
    # ---------------- 算法
    for i in range(output.shape[-2]):
        for j in range(output.shape[-1]):
            h_start, w_start = i * stride[0], j * stride[1]
            # h_start, w_start 一定 >= 0
            h_pos, w_pos = slice(h_start, (h_start + kernel_size[0])), \
                           slice(w_start, (w_start + kernel_size[1]))
            output[..., i, j] = torch.mean(x[..., h_pos, w_pos], dim=(-2, -1))
    return output


def _linear(x, weight, bias=None):
    """全连接层(F.linear())

    :param x: shape = (N, In)
    :param weight: shape = (Out, In)
    :param bias: shape = (Out,)
    :return: shape = (N, Out)
    """
    assert x.shape[1] == weight.shape[1], "weight.shape[1] != x.shape[1]"
    if bias is None:
        return x @ weight.t()
    assert weight.shape[0] == bias.shape[0], "weight.shape[0] != bias.shape[0]"
    return x @ weight.t() + bias
