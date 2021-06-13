import logging

from torch.utils.data import DataLoader
import logging
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from fltk.datasets.distributed.dataset import DistDataset
from fltk.strategy.data_samplers import get_sampler
from fltk.util.poison.poisonpill import PoisonPill


class DistCIFAR10Dataset(DistDataset):

    def __init__(self, args, pill: PoisonPill = None):
        super(DistCIFAR10Dataset, self).__init__(args, pill)
        self.get_args().get_logger().debug(f"Instantiated CIFAR10 train data, with pill: {pill}")
        self.init_train_dataset()
        self.init_test_dataset()


    def init_train_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.get_args().get_logger().debug(f"Loading '{dist_loader_text}' CIFAR10 train data")
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])

        self.train_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=True, download=True,
                                              transform=transform,
                                              target_transform=None if not self.pill else self.pill.poison_targets())

        self.train_sampler = get_sampler(self.train_dataset, self.args)
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, sampler=self.train_sampler)
        logging.info("this client gets {} samples".format(len(self.train_sampler)))

    def init_test_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 test data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        # TODO: decide on whether to poison test or not target_transform=None if not self.pill else self.pill.poison_targets()
        self.test_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=False, download=True,
                                             transform=transform)
        self.test_sampler = get_sampler(self.test_dataset, self.args)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, sampler=self.test_sampler)

    def __del__(self):
        del self.train_dataset
        del self.train_sampler
        del self.train_loader
        del self.test_dataset
        del self.test_sampler
        del self.test_loader