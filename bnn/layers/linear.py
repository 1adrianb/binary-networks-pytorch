import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import BConfig
from .helpers import copy_paramters


class Linear(nn.Linear):
    _FLOAT_MODULE = nn.Linear

    def __init__(self, in_features: int, out_features: int, bias: bool = True,
                 bconfig: BConfig = None
                 ) -> None:
        super(Linear, self).__init__(in_features, out_features, bias)
        assert bconfig, 'bconfig is required for a binarized module'
        self.bconfig = bconfig
        self.activation_pre_process = bconfig.activation_pre_process()
        self.activation_post_process = bconfig.activation_post_process(self)
        self.weight_pre_process = bconfig.weight_pre_process()

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        input_proc = self.activation_pre_process(input)
        return self.activation_post_process(
            F.linear(input_proc, self.weight_pre_process(self.weight), self.bias),
            input
        )

    @classmethod
    def from_module(cls, mod: nn.Module, bconfig: BConfig = None, update: bool = False) -> nn.Module:
        assert type(mod) == cls._FLOAT_MODULE or type(mod) == cls, 'bnn.' + cls.__name__ + \
            '.from_float only works for ' + cls._FLOAT_MODULE.__name__
        if not bconfig:
            assert hasattr(mod, 'bconfig'), 'The input modele requires a predifined bconfig'
            assert mod.bconfig, 'The input modele bconfig is invalid'
            bconfig = mod.bconfig
        bnn_linear = cls(mod.in_features, mod.out_features, bias=mod.bias is not None,
                         bconfig=bconfig)
        bnn_linear.weight = mod.weight
        bnn_linear.bias = mod.bias

        if update:
            copy_paramters(mod, bnn_linear, bconfig)
        return bnn_linear
