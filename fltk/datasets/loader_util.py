from fltk.datasets.distributed import DistMNISTDataset, DistFashionMNISTDataset, DistCIFAR100Dataset, DistCIFAR10Dataset
from fltk.util.definitions import Dataset


def available_datasets():
    return {
        Dataset.cifar10: DistCIFAR10Dataset,
        Dataset.cifar100: DistCIFAR100Dataset,
        Dataset.fashion_mnist: DistFashionMNISTDataset,
        Dataset.mnist: DistMNISTDataset
    }


def get_dataset(name: Dataset):
    return available_datasets()[name]


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
