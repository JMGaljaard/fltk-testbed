from __future__ import annotations
from abc import abstractmethod

import typing

from fltk.util import getLogger

if typing.TYPE_CHECKING:
    from fltk.util.config import FedLearnerConfig


class FedDataset:

    def __init__(self, args: FedLearnerConfig):
        self.args = args
        self.train_sampler = None
        self.test_sampler = None
        self.train_dataset = None
        self.test_dataset = None
        self.train_loader = None
        self.test_loader = None
        self.logger = getLogger(__name__)

    def get_args(self):
        """
        Returns the arguments.

        :return: Arguments
        """
        return self.args

    def get_train_loader(self):
        return self.train_loader

    def get_test_loader(self):
        return self.test_loader

    def get_train_sampler(self):
        return self.train_sampler

    def get_test_sampler(self):
        return self.test_sampler

    @abstractmethod
    def init_train_dataset(self):
        pass

    @abstractmethod
    def init_test_dataset(self):
        pass

class ContinualFedDataset(FedDataset):
    """FIXME: Implement advancing of rounds (either at ConintualLearning dataset, or concrete instances).
    Training round aware dataset allowing to simulate continual learning during the experiment as round_id's advance
    during training.
    """
    def get_train_loader(self, round_id: int = -1):
        """Retrieve training data loader for task(s) that are available during the federated round round_id.
        @param round_id: Round identifier passed by the Federator, used as proxy for passing of time.
        @type round_id: int
        @return: Train data loader with shifted task if requested.
        @rtype: DataLoader
        """
        assert round_id > -1
        return self.train_loader

    def get_test_loader(self, round_id: int = -1):
        """Retrieve test data loader for task(s) that are available during the federated round round_id.
        @param round_id: Round identifier passed by the Federator, used as proxy for passing of time.
        @type round_id: int
        @return: Test data loader with shifted task if requested.
        @rtype: DataLoader
        """
        assert round_id > -1
        return self.test_loader

    def get_train_sampler(self, round_id: int = -1):
        """Retrieve training data (sub)sampler for training data during the indicated round `round_id`.
        @param round_id: Round identifier passed by the Federator, used as proxy for passing of time.
        @type round_id: int
        @return: Data subsampler for data subsampling by the testing dataloader.
        @rtype: Sampler
        """
        assert round_id > -1
        return self.train_sampler

    def get_test_sampler(self, round_id: int = -1):
        """Retrieve testing data (sub)sampler for training data during the indicated round `round_id`.
        @param round_id: Round identifier passed by the Federator, used as proxy for passing of time.
        @type round_id: int
        @return: Data subsampler for data subsampling by the training dataloader.
        @rtype: Sampler
        """
        assert round_id > -1
        return self.test_sampler

