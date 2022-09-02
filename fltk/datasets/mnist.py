# pylint: disable=missing-class-docstring,invalid-name,missing-function-docstring
from torch.utils.data import DataLoader, DistributedSampler
from torchvision import datasets
from torchvision import transforms

from fltk.datasets.dataset import Dataset


class MNIST(Dataset):
    """
    MNIST Dataset implementation for Distributed learning experiments.
    """

    DEFAULT_TRANSFORM = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])

    def __init__(self, config, learning_param, rank: int = 0, world_size: int = None):
        super(MNIST, self).__init__(config, learning_param, rank, world_size)

    def load_train_dataset(self, rank: int = 0, world_size: int = None):
        train_dataset = datasets.FashionMNIST(root=self.config.get_data_path(), train=True, download=True,
                                              transform=self.DEFAULT_TRANSFORM)
        sampler = DistributedSampler(train_dataset, rank=rank,
                                     num_replicas=self.world_size) if self.world_size else None
        train_loader = DataLoader(train_dataset, batch_size=self.learning_params.batch_size, sampler=sampler,
                                  shuffle=(sampler is None))

        return train_loader

    def load_test_dataset(self):
        test_dataset = datasets.FashionMNIST(root=self.config.get_data_path(), train=False, download=True,
                                             transform=self.DEFAULT_TRANSFORM)
        sampler = DistributedSampler(test_dataset, rank=self.rank,
                                     num_replicas=self.world_size) if self.world_size else None
        test_loader = DataLoader(test_dataset, batch_size=self.learning_params.test_batch_size, sampler=sampler)
        return test_loader
