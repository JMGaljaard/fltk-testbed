import logging
import random
from collections import Counter

import numpy as np
from torch.utils.data import Dataset

from fltk.samplers import DistributedSamplerWrapper


class DirichletSampler(DistributedSamplerWrapper):
    """ Generates a (non-iid) data distribution by sampling the dirichlet distribution. Dirichlet constructs a
    vector of length num_clients, that sums to one. Decreasing alpha results in a more non-iid data set.
    This distribution method results in both label and quantity skew.
    """
    def __init__(self, dataset: Dataset, num_replicas = None,
                 rank = None, args = (0.5, 42)) -> None:
        alpha, seed = args
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, seed=seed)

        np.random.seed(seed)
        indices = []
        ordered_by_label = self.order_by_label(dataset)
        for labels in ordered_by_label:
            n_samples = len(labels)
            # generate an allocation by sampling dirichlet, which results in how many samples each client gets
            allocation = np.random.dirichlet([alpha] * self.n_clients) * n_samples
            allocation = allocation.astype(int)
            start_index = allocation[0:self.client_id].sum()
            end_index = 0
            if self.client_id + 1 == self.n_clients:  # last client
                end_index = n_samples
            else:
                end_index = start_index + allocation[self.client_id]

            selection = labels[start_index:end_index]
            indices.extend(selection)

        labels = [dataset.targets[i] for i in indices]
        logging.info(f"nr of samplers in client with rank {rank}: {len(indices)}")
        logging.info(f"distribution in client with rank {rank}: {Counter(labels)}")

        random.seed(seed + self.client_id)  # give each client a unique shuffle
        random.shuffle(indices)  # shuffle indices to spread the labels

        self.indices = indices
