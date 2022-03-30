import pathlib
import os
import logging

from fltk.datasets.distributed import DistCIFAR10Dataset, DistCIFAR100Dataset, DistFashionMNISTDataset
# from fltk.datasets import CIFAR10Dataset, FashionMNISTDataset, CIFAR100Dataset
from fltk.util.arguments import Arguments
from fltk.util.data_loader_utils import generate_train_loader, generate_test_loader, save_data_loader_to_file

logging.basicConfig(level=logging.DEBUG)


if __name__ == '__main__':
    args = Arguments(logging)

    # ---------------------------------
    # ------------ CIFAR10 ------------
    # ---------------------------------
    dataset = DistCIFAR10Dataset(args)
    TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/train_data_loader.pickle"
    TEST_DATA_LOADER_FILE_PATH = "data_loaders/cifar10/test_data_loader.pickle"

    if not os.path.exists("data_loaders/cifar10"):
        pathlib.Path("data_loaders/cifar10").mkdir(parents=True, exist_ok=True)

    train_data_loader = generate_train_loader(args, dataset)
    test_data_loader = generate_test_loader(args, dataset)

    with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(train_data_loader, f)

    with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(test_data_loader, f)

    # ---------------------------------
    # --------- Fashion-MNIST ---------
    # ---------------------------------
    dataset = DistFashionMNISTDataset(args)
    TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/fashion-mnist/train_data_loader.pickle"
    TEST_DATA_LOADER_FILE_PATH = "data_loaders/fashion-mnist/test_data_loader.pickle"

    if not os.path.exists("data_loaders/fashion-mnist"):
        pathlib.Path("data_loaders/fashion-mnist").mkdir(parents=True, exist_ok=True)

    train_data_loader = generate_train_loader(args, dataset)
    test_data_loader = generate_test_loader(args, dataset)

    with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(train_data_loader, f)

    with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(test_data_loader, f)

    # ---------------------------------
    # ------------ CIFAR100 -----------
    # ---------------------------------
    dataset = DistCIFAR100Dataset(args)
    TRAIN_DATA_LOADER_FILE_PATH = "data_loaders/cifar100/train_data_loader.pickle"
    TEST_DATA_LOADER_FILE_PATH = "data_loaders/cifar100/test_data_loader.pickle"

    if not os.path.exists("data_loaders/cifar100"):
        pathlib.Path("data_loaders/cifar100").mkdir(parents=True, exist_ok=True)

    train_data_loader = generate_train_loader(args, dataset)
    test_data_loader = generate_test_loader(args, dataset)

    with open(TRAIN_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(train_data_loader, f)

    with open(TEST_DATA_LOADER_FILE_PATH, "wb") as f:
        save_data_loader_to_file(test_data_loader, f)
