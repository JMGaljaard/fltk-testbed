from typing import Type, OrderedDict

import pytest
import torch

from fltk.nets import Cifar10CNN, Cifar10ResNet, ResNet18, ResNet34, ResNet50, ResNet101, ResNet152, Cifar100ResNet, \
    Cifar100VGG, FashionMNISTCNN, FashionMNISTResNet, SimpleMnist, SimpleNet
from fltk.nets.util import init_reproducibility

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

@pytest.mark.parametrize('network_class', models)
def test_reproducible_initialization(network_class: Type[torch.nn.Module]):
    init_reproducibility()
    param_1: OrderedDict[str, torch.nn.Module] = network_class().state_dict()
    init_reproducibility()
    param_2: OrderedDict[str, torch.nn.Module] = network_class().state_dict()

    for key, value in param_1.items():
        assert torch.equal(value, param_2.get(key))

    del param_1, param_2