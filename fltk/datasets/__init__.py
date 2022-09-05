from fltk.datasets.cifar10 import CIFAR10Dataset
from fltk.datasets.cifar100 import CIFAR100Dataset
from fltk.datasets.fashion_mnist import  FashionMNISTDataset
from fltk.datasets.mnist import MNIST
from fltk.util.config.definitions import Dataset

def available_dataparallel_datasets():
    return {
        Dataset.cifar10: CIFAR10Dataset,
        Dataset.cifar100: CIFAR100Dataset,
        Dataset.fashion_mnist: FashionMNISTDataset,
        Dataset.mnist: MNIST
    }


def get_train_loader_path(name: Dataset) -> str:
    paths = {
        Dataset.cifar10: 'data_loaders/cifar10/train_data_loader.pickle',
        Dataset.fashion_mnist: 'data_loaders/fashion-mnist/train_data_loader.pickle',
        Dataset.cifar100: 'data_loaders/cifar100/train_data_loader.pickle',
        Dataset.mnist: 'data_loaders/mnist/train_data_loader.pickle',
    }
    return paths[name]


def get_test_loader_path(name: Dataset) -> str:
    paths = {
        Dataset.cifar10: 'data_loaders/cifar10/test_data_loader.pickle',
        Dataset.fashion_mnist: 'data_loaders/fashion-mnist/test_data_loader.pickle',
        Dataset.cifar100: 'data_loaders/cifar100/test_data_loader.pickle',
        Dataset.mnist: 'data_loaders/mnist/test_data_loader.pickle',
    }
    return paths[name]


def get_dist_dataset(name: Dataset):
    """
    Function to retrieve distributed dataset (Distributed Learning Experiment).
    @param name: Definition name of the datset.
    @type name: Dataset
    @return:
    @rtype:
    """
    return available_dataparallel_datasets()[name]
