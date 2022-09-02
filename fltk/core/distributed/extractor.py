from __future__ import annotations
import os
from argparse import Namespace

from torchvision.datasets import FashionMNIST, CIFAR10, CIFAR100, MNIST

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fltk.util.config import DistributedConfig


def download_datasets(args: Namespace, config: DistributedConfig):
    """
    Function to Download datasets to a system. This is currently meant to be run (using the extractor mode of FLTK) to
    download all datasets into the `data` directory and include it in the Docker image that is build for the project.
    (This to prevent unnecessary load on the services that provide the datasets, and decrease the energy footprint of
    using the FLTK framework).
    @param args: Namespace object.
    @type args: Namespace
    @param config: FLTK configuration file, for finding the path where the datasets should be stored.
    @type config: DistributedConfig
    @return: None
    @rtype: None
    """
    data_path = config.get_data_path()
    root = str(data_path)

    if not data_path.is_dir():
        os.mkdirs(root, exist_ok=True) # pylint: disable=no-member

    # Prepare MNIST
    MNIST(root=root, download=True)
    # Prepare Fashion MNIST
    FashionMNIST(root=root, download=True)
    # Prepare CIFAR10
    CIFAR10(root=root, download=True)
    # Prepare CIFAR100
    CIFAR100(root=root, download=True)
