from __future__ import annotations
from abc import abstractmethod

from torch.utils.data.dataloader import DataLoader
from torch.utils.data.dataset import Dataset
import typing

from fltk.samplers.continuous_sampler import ContinuousSampler
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


class ContinousFedDataset(FedDataset):
    """Wrapper Dataset for Federeated Learning Dataset, to allow to arbitrary Federated datasets to be mapped to
     the Continuous Learning domain. Leveraging a Wrapper around non-IID data sampling and a Task sampler for the
     continuous domain, allows for the clients to retrieve the correct data, without having to do bookkeeping
     of which samples correspond to which class.
    """

    def __init__(self, federated_dataset: FedDataset, train_task_sampler: ContinuousSampler,
                 test_task_sampler: ContinuousSampler, args: FedLearnerConfig):
        super().__init__(args)
        self.wrapped_dataset = federated_dataset
        self.train_task_sampler = train_task_sampler
        self.test_task_sampler = test_task_sampler
        self.train_task_to_loader = dict()
        self.test_task_to_loader = dict()

    def _get_loader(self, task_id, dataset: Dataset, sampler: ContinuousSampler):
        """Private shared logic for updating a task specific loader and instantiating the loader.
        """
        sampler = sampler.set_task(task_id)
        task_train_loader = DataLoader(
            dataset,
            batch_size=self.args.batch_size,
            sampler=sampler
        )
        return task_train_loader

    def get_train_loader(self, task_id: int = -1):
        """Retrieve training data loader for task(s) that are available during the federated round round_id.
        @param task_id: Task identifier passed computed by the Client, used as proxy for passing of time.
        @type task_id: int
        @return: Train data loader with shifted task if requested.
        @rtype: DataLoader
        """

        if task_id in self.train_task_to_loader:
            return self.train_task_to_loader[task_id]
        loader = self._get_loader(task_id, self.wrapped_dataset.train_dataset, self.train_task_sampler)
        self.train_task_to_loader[task_id] = loader
        return loader

    def get_test_loader(self, task_id: int = -1):
        """Retrieve test data loader for task(s) that are available during the federated round round_id.
        @param task_id: Task identifier computed by the Client, used as proxy for passing of time.
        @type task_id: int
        @return: Test data loader with shifted task if requested.
        @rtype: DataLoader
        """
        task_id = task_id
        if task_id in self.test_task_to_loader:
            return self.test_task_to_loader[task_id]
        loader = self._get_loader(task_id, self.wrapped_dataset.test_dataset, self.test_task_sampler)
        self.test_task_to_loader[task_id] = loader
        return loader

    def get_train_sampler(self, task_id: int = -1):
        """Retrieve training data (sub)sampler for training data during the indicated round `round_id`.
        @param task_id: Task identifier computed by the Client, used as proxy for passing of time.
        @type task_id: int
        @return: Data subsampler for data subsampling by the testing dataloader.
        @rtype: Sampler
        """
        return self.train_task_sampler

    def get_test_sampler(self, task_id: int = -1):
        """Retrieve testing data (sub)sampler for training data during the indicated round `round_id`.
        @param task_id: Task identifier computed by the Client, used as proxy for passing of time.
        @type task_id: int
        @return: Data subsampler for data subsampling by the training dataloader.
        @rtype: Sampler
        """
        return self.test_task_sampler

