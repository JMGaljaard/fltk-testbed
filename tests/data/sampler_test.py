import unittest

import numpy as np
import torch

import fltk.datasets.federated as fed_dataset
import fltk.util.config.definitions as defs
import fltk.util.config.learner_config as learner_config
from fltk.datasets.federated.dataset import ContinousFedDataset
from fltk.samplers.continuous_sampler import sparse2coarse, ContinuousSampler, \
    ExpandingWindowSampler, SlidingWindowSampler
from fltk.util.config.definitions import Optimizations


class ContinualLearningSmokeTest(unittest.TestCase):

    def setUp(self) -> None:
        self.fed_conf = learner_config.FedLearnerConfig(batch_size=128, cuda=False, min_lr=.001, optimizer=Optimizations.sgd,
                                                   replication=1, scheduler_gamma=0.5, scheduler_step_size=50,
                                                   test_batch_size=128)
        self.fed_conf.dataset_name = defs.Dataset.cifar100
        # Step 1 create dataset (ordinary federated ataset)
        self.cifar100 = fed_dataset.get_fed_dataset(self.fed_conf.dataset_name)(self.fed_conf)

        # (Optional) Step 2 map labels to new labels, specific for coarse dataset
        self.cifar100.train_dataset.targets = sparse2coarse(self.cifar100.train_dataset.targets)
        self.cifar100.test_dataset.targets = sparse2coarse(self.cifar100.test_dataset.targets)

    def test_sliding_window_batch_and_label(self):

        # FIXME: Port code to construct task order and mapping psuedo-randomly
        # FIXME: Port code to calculate task_ID based on round.
        # Create a mapping from task IDs to labels in the dataset
        task_to_label = {i: [i] for i in range(20)}
        # Create ordering of task ids' over time
        task_order = list(range(20))

        # Create continual sampler, sliding for train, expanding for test, which restricts a dataloader to the subset of
        # indices that a client may access.
        train_task_sampler = SlidingWindowSampler(task_order=task_order, window_size=1)
        task_sampler = ExpandingWindowSampler(task_order=task_order)

        # Create a continual learning sampler that combines the task-ids with label-nonIIDness samplers. (Takes the inter
        # section of client data and task samples).
        train_task_sampler = ContinuousSampler(
            dataset=self.cifar100,
            num_replicas=2,
            rank=1,
            sampler=self.cifar100.train_sampler,
            task_indices_sampler=train_task_sampler,
            task_to_label=task_to_label,
            train=True,
            test=False)
        # Similarly, create a test-time sampler
        test_task_sampler = ContinuousSampler(self.cifar100, 2, 0, self.cifar100.test_sampler, task_sampler, task_to_label,
                                              train=False, test=True)

        # Lastly, create a Wrapped dataset that has access to the original federated dataset, and samplers that wrap
        # around non-IID samplers.
        continuous_dataset = ContinousFedDataset(self.cifar100, train_task_sampler, test_task_sampler, self.cifar100.args)

        # Cliets then can retrieve the 'correct' dataset following the original interface, with as added benefit that
        # they can define the task-id.
        task_id = 2
        loader = continuous_dataset.get_train_loader(task_id=task_id)

        for x, y in loader:
            self.assertTrue(torch.all(y == task_id))

        self.assertTrue(len(loader) == 20)

    def test_expanding_window_batch_and_label(self):

        # FIXME: Port code to construct task order and mapping psuedo-randomly
        # FIXME: Port code to calculate task_ID based on round.
        # Create a mapping from task IDs to labels in the dataset
        task_to_label = {i: [i] for i in range(20)}
        # Create ordering of task ids' over time
        task_order = list(range(20))

        # Create continual sampler, sliding for train, expanding for test, which restricts a dataloader to the subset of
        # indices that a client may access.
        train_task_sampler = SlidingWindowSampler(task_order=task_order, window_size=1)
        task_sampler = ExpandingWindowSampler(task_order=task_order)

        # Create a continual learning sampler that combines the task-ids with label-nonIIDness samplers. (Takes the inter
        # section of client data and task samples).
        train_task_sampler = ContinuousSampler(
            dataset=self.cifar100,
            num_replicas=2,
            rank=1,
            sampler=self.cifar100.train_sampler,
            task_indices_sampler=train_task_sampler,
            task_to_label=task_to_label,
            train=True,
            test=False)
        # Similarly, create a test-time sampler
        test_task_sampler = ContinuousSampler(self.cifar100, 2, 0, self.cifar100.test_sampler, task_sampler, task_to_label,
                                              train=False, test=True)

        # Lastly, create a Wrapped dataset that has access to the original federated dataset, and samplers that wrap
        # around non-IID samplers.
        continuous_dataset = ContinousFedDataset(self.cifar100, train_task_sampler, test_task_sampler, self.cifar100.args)

        # Cliets then can retrieve the 'correct' dataset following the original interface, with as added benefit that
        # they can define the task-id.
        for task_id in range(20):
            loader = continuous_dataset.get_test_loader(task_id=task_id)

            for x, y in loader:
                self.assertTrue(torch.all(y <= task_id))

            # Cifar is not neatly divisible in the chosen batch sizes
            self.assertTrue( 4 * (task_id + 1) - 1 <= len(loader) <=  4 * (task_id + 1))

