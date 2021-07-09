import copy
import unittest

import torch
import torch.nn as nn

from bnn import BConfig, prepare_binary_model
from bnn.ops import (
    BasicInputBinarizer,
    BasicScaleBinarizer,
    XNORWeightBinarizer
)


class BinaryLayersTestCase(unittest.TestCase):
    def setUp(self) -> None:
        self.test_bconfig = BConfig(
            activation_pre_process=BasicInputBinarizer,
            activation_post_process=BasicScaleBinarizer,
            weight_pre_process=XNORWeightBinarizer
        )
        self.data = torch.tensor([-0.05263, -0.05068, -0.03849, 0.03104, 0.0772, 0.03038, -0.06640, 0.05894,
                                  0.13059, 0.03433, -0.25811, 0.13785]).view(1, 3, 2, 2)
        self.weights = torch.tensor([-0.0252, 0.0084, -0.0676, 0.0891, -0.0010, 0.0518, 0.0380, 0.2866,
                                     -0.0050])

    def tearDown(self) -> None:
        pass

    def test_linear_layer(self):
        layer = nn.Linear(3, 3, bias=False)
        layer.weight.data.copy_(self.weights.view(3, 3))
        x = self.data[:, :, 0, 0].view(1, 3)
        layer = prepare_binary_model(layer, bconfig=self.test_bconfig)

        output = layer(x)
        expected = torch.tensor([[0.0337, -0.0473, -0.1099]])
        self.assertTrue(torch.allclose(expected, output, atol=1e-4))

    def test_conv1d_layer(self):
        layer = nn.Conv1d(3, 3, 1, bias=False)
        layer.weight.data.copy_(self.weights.view(3, 3, 1))
        x = self.data[:, :, :, 0].view(1, 3, 2)
        layer = prepare_binary_model(layer, bconfig=self.test_bconfig)

        output = layer(x)
        expected = torch.tensor([[[0.0337, 0.0337],
                                  [-0.0473, -0.0473],
                                  [-0.1099, -0.1099]]])
        self.assertTrue(torch.allclose(expected, output, atol=1e-4))

    def test_conv2d_layer(self):
        layer = nn.Conv2d(3, 3, 1, bias=False)
        layer.weight.data.copy_(self.weights.view(3, 3, 1, 1))
        x = self.data
        layer = prepare_binary_model(layer, bconfig=self.test_bconfig)

        output = layer(x)
        expected = torch.tensor([[[[0.0337, 0.0337],
                                   [0.0337, -0.0337]],

                                  [[-0.0473, -0.0473],
                                   [-0.0473, 0.0473]],

                                  [[-0.1099, -0.1099],
                                   [-0.1099, 0.1099]]]])
        self.assertTrue(torch.allclose(expected, output, atol=1e-4))


if __name__ == '__main__':
    unittest.main()
