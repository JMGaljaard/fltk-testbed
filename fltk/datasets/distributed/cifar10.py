from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler

from fltk.datasets.distributed.dataset import DistDataset


class DistCIFAR10Dataset(DistDataset):

    def __init__(self, args):
        super(DistCIFAR10Dataset, self).__init__(args)
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
                                         transform=transform)
        self.train_sampler = DistributedSampler(self.train_dataset, rank=self.args.get_rank(),
                                     num_replicas=self.args.get_world_size()) if self.args.get_distributed() else None
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, sampler=self.train_sampler)
        # self.train_loader = DataLoader(self.train_dataset, batch_size=len(self.train_dataset), sampler=self.train_sampler)

    def init_test_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 test data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        self.test_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=False, download=True,
                                        transform=transform)
        self.test_sampler = DistributedSampler(self.test_dataset, rank=self.args.get_rank(),
                                     num_replicas=self.args.get_world_size()) if self.args.get_distributed() else None
        # self.test_sampler = None
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, sampler=self.test_sampler)

    def load_train_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 train data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=True, download=True, transform=transform)
        sampler = DistributedSampler(train_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size()) if self.args.get_distributed() else None
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset), sampler=sampler)
        self.args.set_sampler(sampler)

        train_data = self.get_tuple_from_data_loader(train_loader)
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.get_args().get_logger().debug(f"Finished loading '{dist_loader_text}' CIFAR10 train data")

        return train_data

    def load_test_dataset(self):
        self.get_args().get_logger().debug("Loading CIFAR10 test data")

        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_dataset = datasets.CIFAR10(root=self.get_args().get_data_path(), train=False, download=True, transform=transform)
        sampler = DistributedSampler(test_dataset, rank=self.args.get_rank(), num_replicas=self.args.get_world_size()) if self.args.get_distributed() else None
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), sampler=sampler)
        self.args.set_sampler(sampler)

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.get_args().get_logger().debug("Finished loading CIFAR10 test data")

        return test_data
