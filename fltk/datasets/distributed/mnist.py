from __future__ import annotations
from fltk.datasets.distributed.dataset import DistDataset
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# from fltk.strategy import get_sampler, get_augmentations, get_augmentations_tensor, UnifyingSampler
from random import choice
from PIL import Image


# typing:
from typing import TYPE_CHECKING, Tuple, Any, List

from fltk.samplers import get_sampler

if TYPE_CHECKING:
    from fltk.util import BareConfig

# class MNIST(datasets.MNIST):
#     def __init__(self,  root:str, transform, augment:bool=False):
#         super().__init__(root=root, train=True, download=True, transform=transform)
#         if augment:
#             self.augmentation_transforms = get_augmentations()
#             self.tensor_augmentations = get_augmentations_tensor()
#
#     def __getitem__(self, index: int) -> Tuple[Any, Any]:
#         augment = False
#         if isinstance(index, str):
#             target = int(index)
#             index = choice(self.ordedered_by_label[target])
#             augment = True
#
#         img, target = self.data[index], int(self.targets[index])
#
#         img = img.numpy()
#         if augment:
#             img = self.augmentation_transforms(image=img)['image']
#             img = Image.fromarray(img, mode='L')
#             img = self.tensor_augmentations(img)
#
#         if self.transform is not None:
#             img = self.transform(img)
#
#         return img, target
#
#     def set_available_indices(self, ordedered_by_label:List[int]):
#         self.ordedered_by_label = ordedered_by_label
#
# class DistMNISTDataset_2(DistDataset):
#
#     def __init__(self, args:BareConfig):
#         super(DistMNISTDataset_2, self).__init__(args)
#         self.augment = args.augment
#         self.augmented_emd = args.augmented_emd
#         self.init_train_dataset(args)
#         self.init_test_dataset()
#
#     def init_train_dataset(self, args:BareConfig):
#         dist_loader_text = "distributed" if self.args.get_distributed() else ""
#         self.get_args().get_logger().debug(f"Loading '{dist_loader_text}' MNIST train data")
#
#         self.train_dataset = MNIST(root=self.get_args().get_data_path(), transform=transforms.ToTensor(), augment=self.augment)
#         self.train_sampler = get_sampler(self.train_dataset, self.args)
#         self.train_dataset.set_available_indices(self.train_sampler.order_by_label(self.train_dataset))
#         if self.augment:
#             self.train_sampler = UnifyingSampler(self.train_dataset, args, self.train_sampler, self.augmented_emd)
#         self.train_loader = DataLoader(self.train_dataset, batch_size=16, sampler=self.train_sampler)
#
#     def init_test_dataset(self):
#         dist_loader_text = "distributed" if self.args.get_distributed() else ""
#         self.get_args().get_logger().debug(f"Loading '{dist_loader_text}' MNIST test data")
#         self.test_dataset = datasets.MNIST(root=self.get_args().get_data_path(), train=False, download=True,
#                                              transform=transforms.Compose([transforms.ToTensor()]))
#         self.test_sampler = get_sampler(self.test_dataset, self.args)
#         self.test_loader = DataLoader(self.test_dataset, batch_size=16, sampler=self.test_sampler)


class DistMNISTDataset(DistDataset):

    def __init__(self, args):
        super(DistMNISTDataset, self).__init__(args)
        self.init_train_dataset()
        self.init_test_dataset()

    def init_train_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' MNIST train data")

        self.train_dataset = datasets.MNIST(root=self.get_args().get_data_path(), train=True, download=True,
                                              transform=transforms.Compose([transforms.ToTensor()]))
        self.train_sampler = get_sampler(self.train_dataset, self.args)
        self.train_loader = DataLoader(self.train_dataset, batch_size=16, sampler=self.train_sampler)

    def init_test_dataset(self):
        dist_loader_text = "distributed" if self.args.get_distributed() else ""
        self.logger.debug(f"Loading '{dist_loader_text}' MNIST test data")
        self.test_dataset = datasets.MNIST(root=self.get_args().get_data_path(), train=False, download=True,
                                             transform=transforms.Compose([transforms.ToTensor()]))
        self.test_sampler = get_sampler(self.test_dataset, self.args)
        self.test_loader = DataLoader(self.test_dataset, batch_size=16, sampler=self.test_sampler)

    def load_train_dataset(self):
        self.logger.debug("Loading MNIST train data")

        train_dataset = datasets.MNIST(self.get_args().get_data_path(), train=True, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        train_loader = DataLoader(train_dataset, batch_size=len(train_dataset))

        train_data = self.get_tuple_from_data_loader(train_loader)

        self.logger.debug("Finished loading MNIST train data")

        return train_data

    def load_test_dataset(self):
        self.logger.debug("Loading MNIST test data")

        test_dataset = datasets.MNIST(self.get_args().get_data_path(), train=False, download=True, transform=transforms.Compose([transforms.ToTensor()]))
        test_loader = DataLoader(test_dataset, batch_size=len(test_dataset))

        test_data = self.get_tuple_from_data_loader(test_loader)

        self.logger.debug("Finished loading MNIST test data")

        return test_data