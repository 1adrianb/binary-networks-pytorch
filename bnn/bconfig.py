from dataclasses import dataclass
import torch
import torch.nn as nn


class Identity(nn.Identity):
    def forward(self, layer_out: torch.Tensor, layer_in: torch.Tensor) -> torch.Tensor:
        return layer_out


@dataclass
class BConfig:
    activation_pre_process: nn.Module = nn.Identity
    activation_post_process: nn.Module = Identity
    weight_pre_process: nn.Module = nn.Identity

    def __post_init__(self) -> None:
        if isinstance(
                self.activation_pre_process,
                nn.Module) or isinstance(
                self.activation_post_process,
                nn.Module) or isinstance(
                self.weight_pre_process,
                nn.Module):
            raise ValueError("BConfig received an instance, please pass the class instead.")
