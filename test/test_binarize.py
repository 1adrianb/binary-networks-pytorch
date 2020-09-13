import copy
import unittest

import torch
import torch.nn as nn

from bnn import BConfig, prepare_binary_model
from bnn.layers import Conv2d, Linear
from bnn.ops import (
    BasicInputBinarizer,
    BasicScaleBinarizer,
    XNORWeightBinarizer
)


class Flatten(nn.Module):
    def __init__(self) -> None:
        super(Flatten, self).__init__()

    def forward(self, x):
        return x.view(x.size(0), -1)


class BinarizerTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.linear_layer = nn.Linear(10, 3)
        self.conv_layer = nn.Conv2d(3, 16, 1, 1)
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.Conv2d(16, 16, 1, 1),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
            Flatten(),
            nn.Linear(16, 3)
        )

        self.input1 = torch.rand(1, 10)
        self.input2 = torch.rand(1, 3, 8, 8)

        self.random_bconfig = BConfig(
            activation_pre_process=BasicInputBinarizer,
            activation_post_process=BasicScaleBinarizer,
            weight_pre_process=XNORWeightBinarizer
        )

    def tearDown(self) -> None:
        pass

    def weight_reset(self, m):
        if isinstance(m, nn.Conv2d) or isinstance(m, nn.Linear):
            m.reset_parameters()

    def test_single_linear_layer(self):
        model = copy.copy(self.linear_layer)
        model = prepare_binary_model(model, bconfig=self.random_bconfig)

        self.assertEqual(type(model), Linear)

    def test_single_conv2d_layer(self):
        model = copy.copy(self.conv_layer)
        model = prepare_binary_model(model, bconfig=self.random_bconfig)

        self.assertEqual(type(model), Conv2d)

    def test_many_layers(self):
        model = copy.copy(self.linear_layer)
        model = prepare_binary_model(model, bconfig=self.random_bconfig)

        self.assertEqual(type(model), Linear)

    def test_skip_binarization(self):
        model = copy.copy(self.net)

        fp32_config = BConfig(
            activation_pre_process=nn.Identity,
            activation_post_process=nn.Identity,
            weight_pre_process=nn.Identity
        )
        model = prepare_binary_model(model, bconfig=self.random_bconfig, custom_config_layers_name={'8': fp32_config})

        cnt_conv, cnt_linear = 0, 0
        for module in model.modules():
            if isinstance(module, Conv2d):
                cnt_conv += 1
            elif isinstance(module, Linear):
                if isinstance(module.activation_pre_process, nn.Identity):
                    cnt_linear += 1

        self.assertEqual(cnt_conv, 2)
        self.assertEqual(cnt_linear, 1)

    def test_save_load_state_dict(self):
        model = copy.deepcopy(self.net)
        x = self.input2.clone()

        model = prepare_binary_model(model, bconfig=self.random_bconfig)
        out1 = model(x)

        binary_state_dict = model.state_dict()

        model = copy.deepcopy(self.net)
        model.apply(self.weight_reset)
        model = prepare_binary_model(model, bconfig=self.random_bconfig)
        model.load_state_dict(binary_state_dict)
        out2 = model(x)

        self.assertTrue(torch.equal(out1, out2))


class OpsTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.input = torch.tensor([0.3, 0.1, -2, -0.001, 0.01])
        self.conv_layer = nn.Conv2d(3, 16, 1, 1)

    def test_basic_input_binarizer(self):
        funct = BasicInputBinarizer()
        self.assertTrue(torch.equal(funct(self.input.clone()), torch.sign(self.input.clone())))

    def test_BasicScaleBinarizer(self):
        funct = BasicScaleBinarizer(copy.copy(self.conv_layer))

    def test_XNORWeightBinarizer(self):
        funct = XNORWeightBinarizer()

if __name__ == '__main__':
    unittest.main()
