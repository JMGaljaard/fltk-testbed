# pylint: disable=missing-function-docstring,missing-class-docstring,invalid-name
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from fltk.datasets.distributed.dataset import DistDataset
from fltk.samplers import get_sampler


class DistFashionMNISTDataset(DistDataset):

    def __init__(self, args):
        super(DistFashionMNISTDataset, self).__init__(args)
        self.init_train_dataset()
        self.init_test_dataset()

    def init_train_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' Fashion MNIST train data")

        self.train_dataset = datasets.FashionMNIST(root=self.get_args().get_data_path(), train=True, download=True,
                                                   transform=transforms.Compose([transforms.ToTensor()]))
        self.train_sampler = get_sampler(self.train_dataset, self.args)
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, sampler=self.train_sampler)

    def init_test_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' Fashion MNIST test data")
        self.test_dataset = datasets.FashionMNIST(root=self.get_args().get_data_path(), train=False, download=True,
                                                  transform=transforms.Compose([transforms.ToTensor()]))
        self.test_sampler = get_sampler(self.test_dataset, self.args)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, sampler=self.test_sampler)

    def load_train_dataset(self):
        self.logger.debug("Loading Fashion MNIST train data")

        train_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.logger.debug("Finished loading Fashion MNIST train data")

        return train_data

    def load_test_dataset(self):
        self.logger.debug("Loading Fashion MNIST test data")

        test_dataset = datasets.FashionMNIST(self.get_args().get_data_path(), train=False, download=True,
                                             transform=transforms.Compose([transforms.ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.logger.debug("Finished loading Fashion MNIST test data")

        return test_data
