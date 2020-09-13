import copy
import unittest

import torch
import torch.nn as nn

from bnn.engine import BinaryChef
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


class EngineTestCase(unittest.TestCase):
    def setUp(self) -> None:
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

        self.asset_path = 'test/assets/test.yaml'

    def test_step_length(self):
        chef = BinaryChef(self.asset_path)

        self.assertEqual(len(chef), 3)

    def test_engine_function(self):
        model = copy.copy(self.net)
        chef = BinaryChef(self.asset_path)

        # Step1
        model = chef.next(model)
        # self.assertFalse(hasattr(model[0], 'bconfig'))
        self.assertTrue(hasattr(model[3], 'bconfig'))
        self.assertIsInstance(model[3].weight_pre_process, nn.Identity)

        # Step2
        model = chef.next(model)
        # self.assertFalse(hasattr(model[0], 'bconfig'))
        self.assertTrue(hasattr(model[3], 'bconfig'))
        self.assertIsInstance(model[3].weight_pre_process, XNORWeightBinarizer)
        w_alpha = model[3].activation_post_process.alpha

        # Step3
        model = chef.next(model)
        # self.assertTrue(hasattr(model[0], 'bconfig'))
        self.assertTrue(hasattr(model[3], 'bconfig'))
        self.assertIsInstance(model[3].weight_pre_process, XNORWeightBinarizer)
        self.assertTrue(torch.equal(w_alpha, model[3].activation_post_process.alpha))


if __name__ == '__main__':
    unittest.main()
