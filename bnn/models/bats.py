from typing import Tuple, List

import torch
import torch.nn as nn

from .layers.bats_ops import *


class Cell(nn.Module):

    def __init__(
            self,
            genotype: Genotype,
            C_prev_prev: int,
            C_prev: int,
            C: int,
            reduction: bool,
            reduction_prev: bool,
            groups: int = 12,
            use_shake_shake: bool = False) -> None:
        super(Cell, self).__init__()
        self.use_shake_shake = use_shake_shake
        if reduction_prev:
            self.preprocess0 = FactorizedReduce(C_prev_prev, C)
        else:
            self.preprocess0 = ReLUConvBN(C_prev_prev, C, 1, 1, 0)
        self.preprocess1 = ReLUConvBN(C_prev, C, 1, 1, 0)
        if reduction:
            op_names, indices = zip(*genotype.reduce)
            concat = genotype.reduce_concat
        else:
            op_names, indices = zip(*genotype.normal)
            concat = genotype.normal_concat
        self._compile(C, op_names, indices, concat, reduction, groups)

    def _compile(
            self,
            C: int,
            op_names: List[str],
            indices: List[int],
            concat: List[int],
            reduction: bool,
            groups: int) -> None:
        assert len(op_names) == len(indices)
        self._steps = len(op_names) // 2
        self._concat = concat
        self.multiplier = len(concat)

        self._ops = nn.ModuleList()
        for name, index in zip(op_names, indices):
            stride = 2 if reduction and index < 2 else 1
            op = OPS[name](C, stride, True, True, groups)
            self._ops += [op]
        self._indices = indices

    def forward(self, s0: torch.Tensor, s1: torch.Tensor, drop_prob: float) -> torch.Tensor:
        s0 = self.preprocess0(s0)
        s1 = self.preprocess1(s1)

        states = [s0, s1]
        for i in range(self._steps):
            h1 = states[self._indices[2 * i]]
            h2 = states[self._indices[2 * i + 1]]
            op1 = self._ops[2 * i]
            op2 = self._ops[2 * i + 1]
            h1 = op1(h1)
            h2 = op2(h2)
            if self.training and drop_prob > 0.:
                if not isinstance(op1, nn.Identity):
                    h1 = drop_path(h1, drop_prob)
                if not isinstance(op2, nn.Identity):
                    h2 = drop_path(h2, drop_prob)
            s = h1 + h2
            states += [s]

        if self.use_shake_shake:
            if self.training:
                shake = torch.softmax(torch.zeros(len(self._concat)).uniform_(), dim=0)
                return torch.cat([states[i] * shake[j].item() for j, i in enumerate(self._concat)], dim=1)
            else:
                return torch.cat([states[i] * (1 / len(self._concat)) for i in self._concat], dim=1)
        else:
            return torch.cat([states[i] for i in self._concat], dim=1)


class AuxiliaryHead(nn.Module):

    def __init__(self, C: int, num_classes: int, stride: int) -> None:
        super(AuxiliaryHead, self).__init__()
        self.features = nn.Sequential(
            nn.AvgPool2d(5, stride=stride, padding=0, count_include_pad=False),
            nn.BatchNorm2d(C),
            nn.Conv2d(C, 128, 1, bias=False),
            nn.PReLU(num_parameters=128),

            nn.BatchNorm2d(128),
            nn.Conv2d(128, 768, 2, bias=False),
            nn.PReLU(num_parameters=768),
        )
        self.classifier = nn.Linear(768, num_classes)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.features(x)
        x = self.classifier(x.view(x.size(0), -1))
        return x


class BATSNetworkCIFAR(nn.Module):

    def __init__(self, C: int, num_classes: int, layers: int, auxiliary: bool, genotype, groups: int) -> None:
        super(BATSNetworkCIFAR, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary
        stem_multiplier = 3
        C_curr = stem_multiplier * C
        self.stem = nn.Sequential(
            nn.Conv2d(3, C_curr, 3, padding=1, bias=False),
            nn.BatchNorm2d(C_curr),
            nn.ReLU(inplace=True),
        )
        C_prev_prev, C_prev, C_curr = C_curr, C_curr, C
        self.cells = nn.ModuleList()
        reduction_prev = False
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, groups)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev
        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes, 3)
        self.global_pooling = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_aux = None
        s0 = s1 = self.stem(input)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux


class BATSNetworkImageNet(nn.Module):

    def __init__(self, C: int, num_classes: int, layers: int, auxiliary: bool, genotype, groups: int) -> None:
        super(BATSNetworkImageNet, self).__init__()
        self._layers = layers
        self._auxiliary = auxiliary

        self.stem0 = nn.Sequential(
            nn.Conv2d(3, C // 2, kernel_size=3, stride=2, padding=1, bias=False),
            nn.BatchNorm2d(C // 2),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // 2, C, 3, stride=2, padding=1, bias=False, groups=C // 20),
            nn.BatchNorm2d(C),
        )
        self.stem1 = nn.Sequential(
            nn.ReLU(inplace=True),
            nn.Conv2d(C, C, 3, stride=2, padding=1, bias=False, groups=C // 20),
            nn.BatchNorm2d(C),
        )
        C_prev_prev, C_prev, C_curr = C, C, C

        self.cells = nn.ModuleList()
        reduction_prev = True
        for i in range(layers):
            if i in [layers // 3, 2 * layers // 3]:
                C_curr *= 2
                reduction = True
            else:
                reduction = False
            cell = Cell(genotype, C_prev_prev, C_prev, C_curr, reduction, reduction_prev, groups)
            reduction_prev = reduction
            self.cells += [cell]
            C_prev_prev, C_prev = C_prev, cell.multiplier * C_curr
            if i == 2 * layers // 3:
                C_to_auxiliary = C_prev

        if auxiliary:
            self.auxiliary_head = AuxiliaryHead(C_to_auxiliary, num_classes, 2)
        self.global_pooling = nn.AvgPool2d(7)
        self.classifier = nn.Linear(C_prev, num_classes)

    def forward(self, input: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        logits_aux = None
        s0 = self.stem0(input)
        s1 = self.stem1(s0)
        for i, cell in enumerate(self.cells):
            s0, s1 = s1, cell(s0, s1, self.drop_path_prob)
            if i == 2 * self._layers // 3:
                if self._auxiliary and self.training:
                    logits_aux = self.auxiliary_head(s1)
        out = self.global_pooling(s1)
        logits = self.classifier(out.view(out.size(0), -1))
        return logits, logits_aux
