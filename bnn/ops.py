from functools import partial
from abc import ABCMeta, abstractmethod
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Any, List


def _with_args(cls_or_self: Any, **kwargs: Dict[str, Any]) -> Any:
    r"""Wrapper that allows creation of class factories.
    This can be useful when there is a need to create classes with the same
    constructor arguments, but different instances.
    Source: https://github.com/pytorch/pytorch/blob/b02c932fb67717cb26d6258908541b670faa4e72/torch/quantization/observer.py
    Example::
        >>> Foo.with_args = classmethod(_with_args)
        >>> foo_builder = Foo.with_args(a=3, b=4).with_args(answer=42)
        >>> foo_instance1 = foo_builder()
        >>> foo_instance2 = foo_builder()
        >>> id(foo_instance1) == id(foo_instance2)
        False
    """
    class _PartialWrapper(object):
        def __init__(self, p):
            self.p = p

        def __call__(self, *args, **keywords):
            return self.p(*args, **keywords)

        def __repr__(self):
            return self.p.__repr__()

        with_args = _with_args
    r = _PartialWrapper(partial(cls_or_self, **kwargs))
    return r

ABC = ABCMeta(str("ABC"), (object,), {})


class BinarizerBase(ABC, nn.Module):
    def __init__(self) -> None:
        super(BinarizerBase, self).__init__()

    @abstractmethod
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        pass

    with_args = classmethod(_with_args)


class SignActivation(torch.autograd.Function):
    r"""Applies the sign function element-wise
    :math:`\text{sgn(x)} = \begin{cases} -1 & \text{if } x < 0, \\ 1 & \text{if} x >0  \end{cases}`
    the gradients of which are computed using a STE, namely using :math:`\text{hardtanh(x)}`.
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> input = torch.randn(3)
        >>> output = SignActivation.apply(input)
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        return input.sign()

    @staticmethod
    def backward(ctx, grad_output: torch.Tensor) -> torch.Tensor:
        input, = ctx.saved_tensors
        grad_input = grad_output.clone()
        grad_input.masked_fill_(input.ge(1) | input.le(-1), 0)
        return grad_input


class SignActivationStochastic(SignActivation):
    r"""Binarize the data using a stochastic binarizer
    :math:`\text{sgn(x)} = \begin{cases} -1 & \text{with probablity } p = \sigma(x), \\ 1 & \text{with probablity } 1 - p \end{cases}`
    the gradients of which are computed using a STE, namely using :math:`\text{hardtanh(x)}`.
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> input = torch.randn(3)
        >>> output = SignActivationStochastic.apply(input)
    """
    @staticmethod
    def forward(ctx, input: torch.Tensor) -> torch.Tensor:
        ctx.save_for_backward(input)
        noise = torch.rand_like(input).sub_(0.5)
        return input.add_(1).div_(2).add_(noise).clamp_(0, 1).round_().mul_(2).sub_(1)


class XNORWeightBinarizer(BinarizerBase):
    r"""Binarize the parameters of a given layer using the analytical solution
    proposed in the XNOR-Net paper.
    :math:`\text{out} = \frac{1}{n}\norm{\mathbf{W}}_{\ell} \text{sgn(x)}(\mathbf{W})`
    Shape:
        - Input: :math:`(N, *)` where `*` means, any number of additional
          dimensions
        - Output: :math:`(N, *)`, same shape as the input
    Examples::
        >>> binarizer = XNORWeightBinarizer()
        >>> output = F.conv2d(input, binarizer(weight))
    Args:
        compute_alpha: if True, compute the real-valued scaling factor
        center_weights: make the weights zero-mean
    """

    def __init__(self, compute_alpha: bool = True, center_weights: bool = False) -> None:
        super(XNORWeightBinarizer, self).__init__()
        self.compute_alpha = compute_alpha
        self.center_weights = center_weights

    def _compute_alpha(self, x: torch.Tensor) -> torch.Tensor:
        n = x[0].nelement()
        if x.dim() == 4:
            alpha = x.norm(1, 3, keepdim=True).sum([2, 1], keepdim=True).div_(n)
        elif x.dim() == 2:
            alpha = x.norm(1, 1, keepdim=True).div_(n)
        else:
            raise ValueError(f"Expected ndims equal with 2 or 4, but found {x.dim()}")

        return alpha

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.center_weights:
            mean = x.mean(1, keepdim=True).expand_as(x)
            x = x.sub(mean)

        if self.compute_alpha:
            alpha = self._compute_alpha(x)
            x = SignActivation.apply(x).mul_(alpha.expand_as(x))
        else:
            x = SignActivation.apply(x)

        return x


class BasicInputBinarizer(BinarizerBase):
    r"""Applies the sign function element-wise.
    nn.Module version of SignActivation.
    """

    def __init__(self):
        super(BasicInputBinarizer, self).__init__()

    def forward(self, x: torch.Tensor) -> None:
        return SignActivation.apply(x)


class StochasticInputBinarizer(BinarizerBase):
    r"""Applies the sign function element-wise.
    nn.Module version of SignActivation.
    """

    def __init__(self):
        super(StochasticInputBinarizer, self).__init__()

    def forward(self, x: torch.Tensor):
        return SignActivationStochastic.apply(x)


class AdvancedInputBinarizer(BinarizerBase):
    def __init__(self, derivative_funct=torch.tanh, t: int = 5):
        super(AdvancedInputBinarizer, self).__init__()
        self.derivative_funct = derivative_funct
        self.t = t

    def forward(self, x: torch.tensor) -> torch.Tensor:
        x = self.derivative_funct(x * self.t)
        with torch.no_grad():
            x = torch.sign(x)
        return x


class BasicScaleBinarizer(BinarizerBase):
    def __init__(self, module: nn.Module, shape: List[int] = None) -> None:
        super(BasicScaleBinarizer, self).__init__()

        if isinstance(module, nn.Linear):
            num_channels = module.out_features
        elif isinstance(module, nn.Conv2d):
            num_channels = module.out_channels
        else:
            if hasattr(module, 'out_channels'):
                num_channels = module.out_channels
            else:
                raise Exception('Unknown layer of type {} missing out_channels'.format(type(module)))

        if shape is None:
            alpha_shape = [1, num_channels] + [1] * (module.weight.dim() - 2)
        else:
            alpha_shape = shape
        self.alpha = nn.Parameter(torch.ones(*alpha_shape))

    def forward(self, layer_out: torch.Tensor, layer_in: torch.Tensor) -> torch.Tensor:
        x = layer_out
        return x.mul_(self.alpha)

    def extra_repr(self) -> str:
        return '{}'.format(list(self.alpha.size()))


class XNORScaleBinarizer(BinarizerBase):
    def __init__(self, module: nn.Module) -> None:
        super(BasicScaleBinarizer, self).__init__()
        kernel_size = module.kernel_size
        self.stride = module.stride
        self.padding = module.padding
        self.register_buffer('fixed_weight', torch.ones(*kernel_size).div_(math.prod(kernel_size)), persistent=False)

    def forward(self, layer_out: torch.Tensor, layer_in: torch.Tensor) -> torch.Tensor:
        x = layer_out
        scale = torch.mean(dim=1, keepdim=True)
        scale = F.conv2d(scale, self.fixed_weight, stride=self.stride, padding=self.padding)

        return x.mul_(scale)
