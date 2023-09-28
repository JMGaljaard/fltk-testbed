from __future__ import annotations
import abc
from typing import Dict, List

import numpy as np


import typing
from fltk.samplers import DistributedSamplerWrapper

if typing.TYPE_CHECKING:
    from fltk.datasets.federated import FedDataset


def sparse2coarse(targets: np.ndarray):
    """Convert Pytorch CIFAR100 sparse targets to coarse targets.

    Usage:
        trainset = torchvision.datasets.CIFAR100(path)
        trainset.targets = sparse2coarse(trainset.targets)
    """
    coarse_labels = np.array([ 4,  1, 14,  8,  0,  6,  7,  7, 18,  3,  
                               3, 14,  9, 18,  7, 11,  3,  9,  7, 11,
                               6, 11,  5, 10,  7,  6, 13, 15,  3, 15,  
                               0, 11,  1, 10, 12, 14, 16,  9, 11,  5, 
                               5, 19,  8,  8, 15, 13, 14, 17, 18, 10, 
                               16, 4, 17,  4,  2,  0, 17,  4, 18, 17, 
                               10, 3,  2, 12, 12, 16, 12,  1,  9, 19,  
                               2, 10,  0,  1, 16, 12,  9, 13, 15, 13, 
                              16, 19,  2,  4,  6, 19,  5,  5,  8, 19, 
                              18,  1,  2, 15,  6,  0, 17,  8, 14, 13])
    return coarse_labels[targets]


class TaskIndexSampler(abc.ABC):
    """Abstract base class for Continous task sampling, which stores the pre-computed ordering of tasks over time
    within a sampler object. Concrete instances implement the availability of tasks over time.
    """

    def __init__(self, task_order):
        self.task_order = task_order

    @abc.abstractmethod
    def task_indices(self, task_index):
        pass

class ExpandingWindowSampler(TaskIndexSampler):
    """Continual Learning Task sampler for an expanding window of the data without providing feedback out of *which*
    task a sample came. I.e., returns the list of task identifiers that were seen up until a requested task_index.
    """

    def task_indices(self, task_index):
        return self.task_order[:task_index+1]


class SlidingWindowSampler(TaskIndexSampler):
    """Continual Learning Task sampler for a Sliding window of the data without providing feedback out of *which*
    task a sample came. I.e., returns the list of length 1 of task identifiers that were seen up until a requested task_index.
    """
    def __init__(self, task_order, window_size=1):
        """Sliding or Jumping window TaskIndexSampler, to allow for the selection of different types of...

        @param window_size:
        @type window_size:
        """
        super().__init__(task_order)
        self.window_size = window_size

    def task_indices(self, task_index):
        """Returns a sliding (/jumping) window of tasks, depending on the window size that has been provided to the
        learner.
        @param task_index:
        @type task_index:
        @return:
        @rtype:
        """
        return self.task_order[max(0, task_index-self.window_size+1):task_index+1]


class ContinuousSampler(DistributedSamplerWrapper):
    """Wrapper class to wrap as abstraction around (optional) label non-IID sampler for Federated Learning. Thereby,
    allowing for FederatedDataset implementations to be 'mapped' into continual learning datasets while still
    benefiting form non-IID data loading through separate abstractions."""

    def __init__(
            self,
            dataset: FedDataset,
            num_replicas,
            rank,
            sampler: DistributedSamplerWrapper,
            task_indices_sampler: TaskIndexSampler,
            task_to_label: Dict[int, List[int]],
            args=(5, 42),
            indices: List[int] = None,
            labels_to_indices: List[int, List[int]] = None,
            task_to_indices: List[int, List[int]] = None,
            train: bool = False,
            test: bool = False,
            valid: bool = False,
        ):
        """
        @param dataset: Underlying dataset object (required for labels).
        @type dataset:
        @param sampler: Distributed sampler wrapper for (potential) non-IID datasets during training.
        @type sampler:
        @param task_indices_sampler: Sampler for task indices, to allow for different types of task availability over
            time.
        @type task_indices_sampler:
        @param task_to_label: Mapping from task to a list of labels corresponding to a class. Depending on coarse or
            fine labels, this will eighter be a list of length 1, or of length #classes in a task.
        @type task_to_label:
        """
        if train:
            super_dataset = dataset.train_dataset
        elif test:
            super_dataset = dataset.test_dataset
        elif valid:
            raise Exception('Validation splits are WIP in Freddie.')

        super().__init__(super_dataset, num_replicas=num_replicas, rank=rank)
        self.limit, self.seed = args

        self.dataset = dataset
        self.subsampler = sampler
        self.task_indices_sampler = task_indices_sampler
        self.task_to_labels = task_to_label

        self.labels_to_indices = dict() if labels_to_indices is None else labels_to_indices
        self.task_to_indices = dict() if task_to_indices is None else task_to_indices
        self.indices = [] if indices is None else indices
        self.train, self.validation, self.test = train, valid, test
        if labels_to_indices is None and task_to_indices is None:
            self.build_indices()

    def build_indices(self) -> None:
        """Helper function to pre-compute indices to construct tasks given a task index provided a mapping of tasks to
        labels for a supervised FedDataset.
        @return: None
        @rtype: None
        """
        # random = np.random.RandomState(self.seed)
        if self.train:
            dataset = self.dataset.train_dataset
        elif self.test:
            dataset = self.dataset.test_dataset
        else:
            raise NotImplementedError("Validation datasets are WIP.")

        label_list = np.array(dataset.targets)
        unique_labels = np.unique(label_list)
        for target in unique_labels:
            self.labels_to_indices[target] = np.where(label_list == target)

        for task, task_labels in self.task_to_labels.items():
            task_sample_indices = np.concatenate([
                self.labels_to_indices[target] for target in task_labels
            ])
            # Get the intersection between the labels of the task and those 'available' to the client (following the
            # clients (potential) non-IID sampler).
            task_sub_samples = np.in1d(task_sample_indices, np.array(self.subsampler.indices), assume_unique=True)
            # Store build indices.
            self.task_to_indices[task] = task_sub_samples


    def set_task_(self, task_idx: int) -> None:
        """Inplace method to set the task_index of a client. This will leverage the provided Task Indices subsampler
        to ensure that all the corresponding labels of a task are loaded. Note that this function will change the
        indicises in-place, and must thus be used *before* passing to a DataLoader.
        @param task_idx: Current task to be 'masked' out by the sampler.
        @type task_idx: int
        @return: None
        @rtype:
        """
        tasks_till_task_idx = self.task_indices_sampler.task_indices(task_idx)
        self.indices = []
        for task in tasks_till_task_idx:
            task_labels = self.task_to_labels[task]
            self.task_indices = np.concatenate([
                self.labels_to_indices[target] for target in task_labels
            ])
            # Get fast intersection leveraging
            subsample_intersection = self.task_indices[np.in1d(self.task_indices, np.array(self.subsampler.indices), assume_unique=True)[None,:]]
            self.indices.extend(subsample_intersection)

    def copy(self) -> ContinuousSampler:
        """Helper method to create copy to prevent the need to re-compute indices for a dataset."""
        return ContinuousSampler(
            self.dataset, self.num_replicas, self.rank, self.subsampler, self.task_indices_sampler, self.task_to_labels,
            args=(self.limit, self.seed), indices=[], labels_to_indices=self.labels_to_indices,
            task_to_indices=self.task_to_indices)

    def set_task(self, task_idx) -> ContinuousSampler:
        """Method to set the task_index of a client. This will leverage the provided Task Indices subsampler to ensure
        that all the corresponding labels of a task are loaded. Note that this returns a new instance with the same
        pre-computed lookup directories. However, with different task_index.
        @param task_idx: Task index that is requested by the callee.
        @type task_idx: int
        @return: Copy of the current sampler, with updated indices for the next set of
        @rtype: ContinuousSampler
        """
        new_continual_sampler = self.copy()
        new_continual_sampler.set_task_(task_idx)
        return new_continual_sampler
