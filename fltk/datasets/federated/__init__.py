from __future__ import annotations

import typing

from .cifar10 import FedCIFAR10Dataset
from .cifar100 import FedCIFAR100Dataset
from .fashion_mnist import FedFashionMNISTDataset
from .mnist import FedMNISTDataset

if typing.TYPE_CHECKING:
    import fltk.util.config.definitions as defs

def available_fed_datasets():
    import fltk.util.config.definitions as defs

    return {
        defs.Dataset.cifar10: FedCIFAR10Dataset,
        defs.Dataset.cifar100: FedCIFAR100Dataset,
        defs.Dataset.fashion_mnist: FedFashionMNISTDataset,
        defs.Dataset.mnist: FedMNISTDataset
    }


def get_fed_dataset(name: defs.Dataset):
    return available_fed_datasets()[name]
