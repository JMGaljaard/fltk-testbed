from aenum import unique, Enum
from re import T

from fltk.datasets import CIFAR10Dataset, CIFAR100Dataset, FashionMNISTDataset, MNIST


@unique
class Dataset(Enum):
    cifar10 = 'cifar10'
    cifar100 = 'cifar100'
    fashion_mnist = 'fashion-mnist'
    mnist = 'mnist'

    @classmethod
    def _missing_name_(cls, name: str) -> T:
        for member in cls:
            if member.name.lower() == name.lower():
                return member


def get_dist_dataset(name: Dataset):
    """
    Function to retrieve distributed dataset (Distributed Learning Experiment).
    @param name: Definition name of the datset.
    @type name: Dataset
    @return:
    @rtype:
    """
    __lookup = {
        Dataset.cifar10: CIFAR10Dataset,
        Dataset.cifar100: CIFAR100Dataset,
        Dataset.fashion_mnist: FashionMNISTDataset,
        Dataset.mnist: MNIST
    }
    return __lookup[name]
