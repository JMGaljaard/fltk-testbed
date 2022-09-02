from .cifar10 import FedCIFAR10Dataset
from .cifar100 import FedCIFAR100Dataset
from .fashion_mnist import FedFashionMNISTDataset
from .mnist import FedMNISTDataset
from .dataset import FedDataset
from ...util.config.definitions import Dataset


def available_fed_datasets():
    return {
        Dataset.cifar10: FedCIFAR10Dataset,
        Dataset.cifar100: FedCIFAR100Dataset,
        Dataset.fashion_mnist: FedFashionMNISTDataset,
        Dataset.mnist: FedMNISTDataset
    }


def get_fed_dataset(name: Dataset):
    return available_fed_datasets()[name]
