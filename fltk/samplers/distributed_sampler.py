import random
from typing import Iterator, List

import numpy as np
from torch.utils.data import DistributedSampler, Dataset


class DistributedSamplerWrapper(DistributedSampler):
    """
    Wrapper class wrap DistributedSampler to make it possible to sample from the underlying data.
    """
    indices = []
    epoch_size = 1.0

    def __init__(self, dataset: Dataset, num_replicas=None,
                 rank=None, seed=0) -> None:
        super().__init__(dataset, num_replicas=num_replicas, rank=rank)

        self.client_id = rank - 1
        self.n_clients = num_replicas - 1
        self.n_labels = len(dataset.classes)
        self.seed = seed

    def order_by_label(self, dataset):
        # order the indices by label
        ordered_by_label: List[List[int]] = [[] for _ in range(len(dataset.classes))]
        for index, target in enumerate(dataset.targets):
            ordered_by_label[target].append(index)

        return ordered_by_label

    def set_epoch_size(self, epoch_size: float) -> None:
        """ Sets the epoch size as relative to the local amount of data.
        1.5 will result in the __iter__ function returning the available
        indices with half appearing twice.

        Args:
            epoch_size (float): relative size of epoch
        """
        self.epoch_size = epoch_size

    def __iter__(self) -> Iterator[int]:
        random.seed(self.rank + self.epoch)
        epochs_todo = self.epoch_size
        indices = []
        while (epochs_todo > 0.0):
            random.shuffle(self.indices)
            if epochs_todo >= 1.0:
                indices.extend(self.indices)
            else:
                end_index = int(round(len(self.indices) * epochs_todo))
                indices.extend(self.indices[:end_index])

            epochs_todo = epochs_todo - 1

        ratio = len(indices) / float(len(self.indices))
        np.testing.assert_almost_equal(ratio, self.epoch_size, decimal=2)

        return iter(indices)

    def __len__(self) -> int:
        return len(self.indices)
