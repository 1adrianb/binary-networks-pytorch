import torch
import torch.nn as nn

from collections import namedtuple

Genotype = namedtuple('Genotype', 'normal normal_concat reduce reduce_concat')

PRIMITIVES = [
    'none',
    'max_pool_3x3',
    'avg_pool_3x3',
    'skip_connect',
    'sep_conv_3x3',
    'sep_conv_5x5',
    'dil_conv_3x3',
    'dil_conv_5x5'
]


def channel_shuffle(x: torch.Tensor, groups: int) -> torch.Tensor:
    batchsize, num_channels, height, width = x.data.size()

    channels_per_group = num_channels // groups

    x = x.view(batchsize, groups, channels_per_group, height, width)
    x = torch.transpose(x, 1, 2).contiguous()

    x = x.view(batchsize, -1, height, width)

    return x


def drop_path(x: torch.Tensor, drop_prob: float) -> torch.Tensor:
    if drop_prob > 0.:
        keep_prob = 1. - drop_prob
        mask = torch.tensor(x.size(0), 1, 1, 1, device=x.device).bernoulli_(keep_prob)
        x.div_(keep_prob)
        x.mul_(mask)
    return x

OPS = {
    'none': lambda C, stride, affine, skip, groups: Zero(stride),
    'avg_pool_3x3': lambda C, stride, affine, skip, groups: nn.AvgPool2d(3, stride=stride, padding=1, count_include_pad=False),
    'max_pool_3x3': lambda C, stride, affine, skip, groups: nn.MaxPool2d(3, stride=stride, padding=1),
    'skip_connect': lambda C, stride, affine, skip, groups: nn.Identity() if stride == 1 else FactorizedReduce(C, C, affine=affine),
    'sep_conv_3x3': lambda C, stride, affine, skip, groups: SepConv(C, C, 3, stride, 1, affine=affine, skip=skip, groups=groups),
    'sep_conv_5x5': lambda C, stride, affine, skip, groups: SepConv(C, C, 5, stride, 2, affine=affine, skip=skip, groups=groups),
    'sep_conv_7x7': lambda C, stride, affine, skip, groups: SepConv(C, C, 7, stride, 3, affine=affine, skip=skip, groups=groups),
    'dil_conv_3x3': lambda C, stride, affine, skip, groups: DilConv(C, C, 3, stride, 2, 2, affine=affine, skip=skip, groups=groups),
    'dil_conv_5x5': lambda C, stride, affine, skip, groups: DilConv(C, C, 5, stride, 4, 2, affine=affine, skip=skip, groups=groups),
    'conv_7x1_1x7': lambda C, stride, affine, skip, groups: FactorizedConv(C, 7, stride, affine=affine, skip=skip),
}


class FactorizedConv(nn.Module):
    def __init__(self, C: int, kernel_size: int, stride: int, affine: bool = True, skip: bool = False) -> None:
        super(FactorizedConv, self).__init__()
        self.skip = skip or True
        self.stride = stride
        self.op = nn.Sequential(
            nn.BatchNorm2d(C, affine=affine),
            nn.Conv2d(C, C, (1, kernel_size), stride=(1, stride), padding=(0, kernel_size // 2), bias=False),
            nn.PReLU(num_parameters=C),

            nn.BatchNorm2d(C, affine=affine),

            nn.Conv2d(C, C, (kernel_size, 1), stride=(stride, 1), padding=(kernel_size // 2, 0), bias=False),
            nn.PReLU(num_parameters=C),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip and self.stride == 1:
            return x + channel_shuffle(self.op(x), 4)
        else:
            return channel_shuffle(self.op(x), 4)


class ReLUConvBN(nn.Module):

    def __init__(
            self,
            C_in: int,
            C_out: int,
            kernel_size: int,
            stride: int,
            padding: int,
            affine: bool = True,
            skip: bool = False) -> None:
        super(ReLUConvBN, self).__init__()
        self.skip = skip or True
        self.stride = stride
        self.C_in = C_in
        self.C_out = C_out

        self.op = nn.Sequential(
            nn.BatchNorm2d(C_in, affine=affine),
            nn.Conv2d(C_in, C_out, kernel_size, stride=stride, padding=padding, bias=False),
            nn.PReLU(num_parameters=C_out),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip and self.stride == 1 and self.C_in == self.C_out:
            return x + self.op(x)
        else:
            return self.op(x)


class DilConv(nn.Module):

    def __init__(
            self,
            C_in: int,
            C_out: int,
            kernel_size: int,
            stride: int,
            padding: int,
            dilation: int,
            affine: bool = True,
            skip: bool = False,
            groups: int = 12) -> None:
        super(DilConv, self).__init__()
        self.skip = skip or True
        self.stride = stride
        self.op = nn.Sequential(
            nn.BatchNorm2d(
                C_in,
                affine=affine),
            nn.Conv2d(
                C_in,
                C_in,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                dilation=dilation,
                groups=groups,
                bias=False),
            nn.PReLU(
                num_parameters=C_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip and self.stride == 1:
            return x + channel_shuffle(self.op(x), 4)
        else:
            return channel_shuffle(self.op(x), 4)


class SepConv(nn.Module):

    def __init__(
            self,
            C_in: int,
            C_out: int,
            kernel_size: int,
            stride: int,
            padding: int,
            affine: bool = True,
            skip: bool = False,
            groups: int = 12) -> None:
        super(SepConv, self).__init__()
        self.skip = skip or True
        self.stride = stride
        self.op = nn.Sequential(
            nn.BatchNorm2d(C_in, affine=affine),
            nn.Conv2d(C_in, C_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=groups, bias=False),
            nn.PReLU(num_parameters=C_in),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.skip and self.stride == 1:
            return x + channel_shuffle(self.op(x), 4)
        else:
            return channel_shuffle(self.op(x), 4)


class Zero(nn.Module):

    def __init__(self, stride: int) -> None:
        super(Zero, self).__init__()
        self.stride = stride

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        n, c, h, w = x.size()
        h //= self.stride
        w //= self.stride
        padding = torch.zeros(n, c, h, w, dtype=torch.float32, device=x.device)
        return padding


class FactorizedReduce(nn.Module):

    def __init__(self, C_in: int, C_out: int, affine: bool = True) -> None:
        super(FactorizedReduce, self).__init__()
        assert C_out % 2 == 0
        self.activation = nn.PReLU(num_parameters=C_out)
        self.conv_1 = nn.Sequential(
            nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        )
        self.conv_2 = nn.Sequential(
            nn.Conv2d(C_in, C_out // 2, 1, stride=2, padding=0, bias=False)
        )
        self.bn = nn.BatchNorm2d(C_in, affine=affine)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.bn(x)
        out = torch.cat([self.conv_1(x), self.conv_2(x[:, :, 1:, 1:])], dim=1)
        out = self.activation(out)

        return out
