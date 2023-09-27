from __future__ import annotations
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

import fltk.datasets.federated.dataset as dataset
import fltk.samplers as samplers
import typing
if typing.TYPE_CHECKING:

    import fltk.util.config as learn_config


class FedCIFAR10Dataset(dataset.FedDataset):
    """
    CIFAR10 Dataset implementation for Distributed learning experiments.
    """

    def __init__(self, args: learn_config.FedLearnerConfig):
        super(FedCIFAR10Dataset, self).__init__(args)
        self.init_train_dataset()
        self.init_test_dataset()

    def init_train_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' CIFAR10 train data")
        # self.get_args().get_logger().debug(f"Loading '{dist_loader_text}' CIFAR10 train data")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])
        self.train_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=True, download=True,
                                              transform=transform)
        self.train_sampler = samplers.get_sampler(self.train_dataset, self.args)
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.args.batch_size, sampler=self.train_sampler)
        self.logger.info(f"this client gets {len(self.train_sampler)} samples")
        # logging.info("this client gets {} samples".format(len(self.train_sampler)))

    def init_test_dataset(self):
        self.logger.debug("Loading CIFAR10 test data")
        # self.get_args().get_logger().debug("Loading CIFAR10 test data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        self.test_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=False, download=True,
                                             transform=transform)
        self.test_sampler = samplers.get_sampler(self.test_dataset, self.args)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.test_batch_size, sampler=self.test_sampler)
