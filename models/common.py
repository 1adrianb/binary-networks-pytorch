import torch
import torch.nn as nn


class LearnableBias(torch.nn.Module):
    def __init__(self, out_chn):
        super(LearnableBias, self).__init__()
        self.bias = nn.Parameter(
            torch.zeros(1, out_chn, 1, 1), requires_grad=True
        )

    def forward(self, x):
        out = x + self.bias.expand_as(x)
        return out
