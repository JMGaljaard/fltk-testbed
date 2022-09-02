# pylint: disable=missing-function-docstring,missing-class-docstring,invalid-name
from torch.utils.data import DataLoader
from torchvision import datasets
from torchvision import transforms

from fltk.datasets.federated.dataset import FedDataset
from fltk.samplers import get_sampler


class FedFashionMNISTDataset(FedDataset):

    def __init__(self, args):
        super(FedFashionMNISTDataset, self).__init__(args)
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
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.args.test_batch_size, sampler=self.test_sampler)
