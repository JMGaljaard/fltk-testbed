# pylint: disable=missing-function-docstring,missing-class-docstring,invalid-name
from fltk.datasets.dataset import Dataset
from torchvision import datasets
from torchvision import transforms
from torch.utils.data import DataLoader, DistributedSampler


class FashionMNISTDataset(Dataset):
    """
    FashionMNIST Dataset implementation for Distributed learning experiments.
    """

    def __init__(self, config, learning_param, rank: int = 0, world_size: int = None):
        super(FashionMNISTDataset, self).__init__(config, learning_param, rank, world_size)

    def load_train_dataset(self, rank: int = 0, world_size: int = None):
        train_dataset = datasets.FashionMNIST(root=self.config.get_data_path(), train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
        sampler = DistributedSampler(train_dataset, rank=rank,
                                     num_replicas=self.world_size) if self.world_size else None
        train_loader = DataLoader(train_dataset, batch_size=self.learning_params.batch_size, sampler=sampler,
                                  shuffle=(sampler is None))

        return train_loader

    def load_test_dataset(self):
        test_dataset = datasets.FashionMNIST(root=self.config.get_data_path(), train=False, download=True,
                                             transform=transforms.Compose([transforms.ToTensor()]))
        sampler = DistributedSampler(test_dataset, rank=self.rank,
                                     num_replicas=self.world_size) if self.world_size else None
        test_loader = DataLoader(test_dataset, batch_size=self.learning_params.test_batch_size, sampler=sampler)
        return test_loader
