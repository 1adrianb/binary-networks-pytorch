from typing import Callable, Optional
import torch
import torch.nn as nn

from .common import conv3x3


class HBlock(nn.Module):
    def __init__(self,
                 inplanes: int,
                 planes: int,
                 stride: int = 1,
                 downsample: Optional[nn.Module] = None,
                 groups: int = 1,
                 base_width: int = 64,
                 dilation: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation=nn.ReLU
                 ) -> None:
        super(HBlock, self).__init__()
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in HBlock")
        if stride > 1:
            raise NotImplementedError("Stride > 1 not supported in HBlock")
        self.bn1 = norm_layer(inplanes)
        self.conv1 = conv3x3(inplanes, int(planes / 2), groups=groups)
        self.bn2 = norm_layer(int(planes / 2))
        self.conv2 = conv3x3(int(planes / 2), int(planes / 4), groups=groups)
        self.bn3 = norm_layer(int(planes / 4))
        self.conv3 = conv3x3(int(planes / 4), int(planes / 4), groups=groups)

        self.act1 = activation(inpace=True) if activation == nn.ReLU else activation(num_parameters=int(planes / 2))
        self.act2 = activation(inpace=True) if activation == nn.ReLU else activation(num_parameters=int(planes / 2))
        self.act3 = activation(inpace=True) if activation == nn.ReLU else activation(num_parameters=int(planes / 4))

        self.downsample = downsample

    def forward(self, x):
        residual = x

        out1 = self.bn1(x)
        out1 = self.act1(out1)
        out1 = self.conv1(out1)

        out2 = self.bn2(out1)
        out2 = self.act2(out2)
        out2 = self.conv2(out2)

        out3 = self.bn3(out2)
        out3 = self.act3(out3)
        out3 = self.conv3(out3)

        out3 = torch.cat((out1, out2, out3), 1)

        if self.downsample is not None:
            residual = self.downsample(residual)

        out3 += residual

        return out3
