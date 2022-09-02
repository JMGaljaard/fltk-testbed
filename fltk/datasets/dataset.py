from __future__ import annotations

import abc
from abc import abstractmethod

import torch
from torch.utils.data import DataLoader
from torch.utils.data import TensorDataset

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fltk.util.config import DistLearningConfig


class Dataset(abc.ABC):
    """
    Dataset implementation for Distributed learning experiments.
    """

    def __init__(self, config, learning_params: DistLearningConfig, rank: int, world_size: int):
        self.config = config
        self.learning_params = learning_params

        self.rank = rank
        self.world_size = world_size

        self.train_loader = self.load_train_dataset()
        self.test_loader = self.load_test_dataset()

    def get_train_dataset(self):
        """
        Returns the train dataset.

        :return: tuple
        """
        return self.train_loader

    def get_test_dataset(self):
        """
        Returns the test dataset.

        :return: tuple
        """
        return self.test_loader

    def get_train_loader(self, **kwargs):
        """
        Return the data loader for the train dataset.

        :param batch_size: batch size of data loader.
        :type batch_size: int
        :return: torch.utils.data.DataLoader.
        """
        return self.train_loader

    def get_test_loader(self, **kwargs):
        """
        Return the data loader for the test dataset.

        :param batch_size: batch size of data loader.
        :type batch_size: int
        :return: torch.utils.data.DataLoader
        """
        return self.test_loader

    @abstractmethod
    def load_train_dataset(self):
        """
        Loads & returns the training dataset.

        :return: tuple
        """
        raise NotImplementedError("load_train_dataset() isn't implemented")

    @abstractmethod
    def load_test_dataset(self):
        """
        Loads & returns the test dataset.

        :return: tuple
        """
        raise NotImplementedError("load_test_dataset() isn't implemented")

    def get_data_loader_from_data(batch_size, X, Y, **kwargs):
        """
        Get a data loader created from a given set of data.

        :param batch_size: batch size of data loader.
        :type batch_size: int
        :param X: data features,
        :type X: numpy.Array()
        :param Y: data labels.
        :type Y: numpy.Array()
        :return: torch.utils.data.DataLoader
        """
        X_torch = torch.from_numpy(X).float() # pylint: disable=no-member

        if "classification_problem" in kwargs and kwargs["classification_problem"] == False:
            Y_torch = torch.from_numpy(Y).float()  # pylint: disable=no-member
        else:
            Y_torch = torch.from_numpy(Y).long() # pylint: disable=no-member
        dataset = TensorDataset(X_torch, Y_torch)

        kwargs.pop("classification_problem", None)

        return DataLoader(dataset, batch_size=batch_size, **kwargs)
