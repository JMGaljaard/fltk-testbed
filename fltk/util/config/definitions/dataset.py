from typing import Type

from aenum import unique, Enum

from fltk import datasets
from fltk.datasets import CIFAR10Dataset, CIFAR100Dataset, FashionMNISTDataset, MNIST


@unique
class Dataset(Enum):
    """Enum for provided dataset Types."""
    cifar10 = 'cifar10'
    cifar100 = 'cifar100'
    fashion_mnist = 'fashion-mnist'
    mnist = 'mnist'

    @classmethod
    def _missing_name_(cls, name: str) -> "Dataset":
        """Helper function in case name could not be looked up (to support older configurations).

        Args:
            name (str): Name of Type to be looked up.

        Returns:
            Dataset: Corresponding Enum instance, if name is recognized from lower case.

        """
        for member in cls:
            if member.name.lower() == name.lower():
                return member


def get_dist_dataset(name: Dataset) -> Type[datasets.Dataset]:
    """Function to retrieve (distributed) dataset, for Distributed Learning Experiments.

    Args:
      name (Dataset): Definition (Enum) of the dataset configurated.

    Returns:
        Type[datasets.Dataset]: Class reference to requested dataset.

    """
    __lookup = {
        Dataset.cifar10: CIFAR10Dataset,
        Dataset.cifar100: CIFAR100Dataset,
        Dataset.fashion_mnist: FashionMNISTDataset,
        Dataset.mnist: MNIST
    }
    return __lookup[name]
