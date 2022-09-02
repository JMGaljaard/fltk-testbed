from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from torchvision import transforms

from fltk.datasets.dataset import Dataset


class CIFAR10Dataset(Dataset):
    """
    CIFAR10 Dataset implementation for Distributed learning experiments.
    """

    def __init__(self, config, learning_param, rank: int = 0, world_size: int = None):
        super(CIFAR10Dataset, self).__init__(config, learning_param, rank, world_size)

    def load_train_dataset(self, rank: int = 0, world_size: int = None):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.RandomCrop(32, 4),
            transforms.ToTensor(),
            normalize
        ])

        train_dataset = datasets.CIFAR10(root=self.config.get_data_path(),
                                         train=True,
                                         download=True,
                                         transform=transform)
        if not self.world_size:
            sampler = None
        else:
            sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=self.world_size)
        train_loader = DataLoader(train_dataset, batch_size=self.learning_params.batch_size, sampler=sampler,
                                  shuffle=(sampler is None))

        return train_loader

    def load_test_dataset(self):
        normalize = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        transform = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])
        test_dataset = datasets.CIFAR10(root=self.config.get_data_path(), train=False, download=True,
                                        transform=transform)
        sampler = DistributedSampler(test_dataset, rank=self.rank,
                                     num_replicas=self.world_size) if self.world_size else None
        test_loader = DataLoader(test_dataset, batch_size=self.learning_params.test_batch_size, sampler=sampler)
        return test_loader
