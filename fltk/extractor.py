from argparse import Namespace

from torchvision.datasets import FashionMNIST, CIFAR10, CIFAR100, MNIST

from fltk.util.config import BareConfig


def download_datasets(args: Namespace, config: BareConfig):
    # Prepare MNIST
    mnist = MNIST(root=config.get_data_path(), download=True)
    # Prepare Fashion MNIST
    mnist = FashionMNIST(root=config.get_data_path(), download=True)
    del mnist
    # Prepare CIFAR10
    cifar10 = CIFAR10(root=config.get_data_path(), download=True)
    del cifar10
    # Prepare CIFAR100
    cifar100 = CIFAR100(root=config.get_data_path(), download=True)
