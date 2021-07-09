from typing import Union
import torch
import torch.nn as nn

from torch.nn.common_types import _size_1_t, _size_2_t
from .. import BConfig
from .helpers import copy_paramters


class Conv1d(nn.Conv1d):
    _FLOAT_MODULE = nn.Conv1d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_1_t,
        stride: _size_1_t = 1,
        padding: Union[str, _size_1_t] = 0,
        dilation: _size_1_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        bconfig: BConfig = None
    ) -> None:
        super(Conv1d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, bias=bias, padding_mode=padding_mode)
        assert bconfig, 'bconfig is required for a binarized module'
        self.bconfig = bconfig
        self.activation_pre_process = bconfig.activation_pre_process()
        self.activation_post_process = bconfig.activation_post_process(self)
        self.weight_pre_process = bconfig.weight_pre_process()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_proc = self.activation_pre_process(input)
        input_proc = self._conv_forward(input_proc, self.weight_pre_process(self.weight), bias=self.bias)
        
        if isinstance(input_proc, tuple) and len(input_proc) == 1:
            input_proc = input_proc[0]
        
        return self.activation_post_process(
            input_proc,
            input
        )

    @classmethod
    def from_module(cls, mod: nn.Module, bconfig: BConfig = None, update: bool = False):
        assert type(mod) == cls._FLOAT_MODULE or type(mod) == cls, 'bnn.' + cls.__name__ + \
            '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        if not bconfig:
            assert hasattr(mod, 'bconfig'), 'The input modele requires a predifined bconfig'
            assert mod.bconfig, 'The input modele bconfig is invalid'
            bconfig = mod.bconfig
        bnn_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                       groups=mod.groups, bias=mod.bias is not None,
                       padding_mode=mod.padding_mode, bconfig=bconfig)
        bnn_conv.weight = mod.weight
        bnn_conv.bias = mod.bias

        if update:
            copy_paramters(mod, bnn_conv, bconfig)

        return bnn_conv


class Conv2d(nn.Conv2d):
    _FLOAT_MODULE = nn.Conv2d

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: _size_2_t,
        stride: _size_2_t = 1,
        padding: Union[str, _size_2_t] = 0,
        dilation: _size_2_t = 1,
        groups: int = 1,
        bias: bool = True,
        padding_mode: str = 'zeros',
        bconfig: BConfig = None
    ) -> None:
        super(Conv2d, self).__init__(in_channels, out_channels, kernel_size,
                                     stride=stride, padding=padding, dilation=dilation,
                                     groups=groups, bias=bias, padding_mode=padding_mode)
        assert bconfig, 'bconfig is required for a binarized module'
        self.bconfig = bconfig
        self.activation_pre_process = bconfig.activation_pre_process()
        self.activation_post_process = bconfig.activation_post_process(self)
        self.weight_pre_process = bconfig.weight_pre_process()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_proc = self.activation_pre_process(input)
        input_proc = self._conv_forward(input_proc, self.weight_pre_process(self.weight), bias=self.bias)
        
        if isinstance(input_proc, tuple) and len(input_proc) == 1:
            input_proc = input_proc[0]
        
        return self.activation_post_process(
            input_proc,
            input
        )

    @classmethod
    def from_module(cls, mod: nn.Module, bconfig: BConfig = None, update: bool = False):
        assert type(mod) == cls._FLOAT_MODULE or type(mod) == cls, 'bnn.' + cls.__name__ + \
            '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        if not bconfig:
            assert hasattr(mod, 'bconfig'), 'The input modele requires a predifined bconfig'
            assert mod.bconfig, 'The input modele bconfig is invalid'
            bconfig = mod.bconfig
        bnn_conv = cls(mod.in_channels, mod.out_channels, mod.kernel_size,
                       stride=mod.stride, padding=mod.padding, dilation=mod.dilation,
                       groups=mod.groups, bias=mod.bias is not None,
                       padding_mode=mod.padding_mode, bconfig=bconfig)
        bnn_conv.weight = mod.weight
        bnn_conv.bias = mod.bias

        if update:
            copy_paramters(mod, bnn_conv, bconfig)

        return bnn_conv
