from typing import Type, OrderedDict

import torch

import unittest
from parameterized import parameterized
from fltk.nets import Cifar10CNN, Cifar10ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, Cifar100ResNet, \
    Cifar100VGG, FashionMNISTCNN, FashionMNISTResNet, SimpleMnist, SimpleNet
from fltk.nets.util.reproducability import init_reproducibility

models = [
    (Cifar10CNN),
    (Cifar10ResNet),
    (ResNet18),
    (ResNet34),
    (ResNet50),
    (ResNet101),
    (ResNet152),
    (Cifar100ResNet),
    (Cifar100VGG),
    (FashionMNISTCNN),
    (FashionMNISTResNet),
    (SimpleMnist),
    (SimpleNet)
]

class TestReproducibleNet(unittest.TestCase):

        @parameterized.expand(
            map(lambda x: [x], models)
        )
        def test_reproducible_initialization(self, network_class: Type[torch.nn.Module]): # pylint: disable=missing-function-docstring
            init_reproducibility(seed=42)
            param_1: OrderedDict[str, torch.nn.Module] = network_class().state_dict()
            init_reproducibility(seed=42)
            param_2: OrderedDict[str, torch.nn.Module] = network_class().state_dict()

            for key, value in param_1.items():
                assert torch.equal(value, param_2.get(key)) # pylint: disable=no-member

            del param_1, param_2
