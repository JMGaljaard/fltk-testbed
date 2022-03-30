from fltk.samplers import DistributedSamplerWrapper
from torch.utils.data import DistributedSampler, Dataset
import numpy as np
import logging
import random
from collections import Counter


class UniformSampler(DistributedSamplerWrapper):
    def __init__(self, dataset, num_replicas=None, rank=None, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, seed=seed)
        indices = list(range(len(self.dataset)))
        self.indices = indices[self.rank:self.total_size:self.n_clients]