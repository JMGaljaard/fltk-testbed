from fltk.samplers import DistributedSamplerWrapper


class UniformSampler(DistributedSamplerWrapper):
    """
    Distributed Sampler implementation that samples uniformly from the available datapoints, assuming all clients
    have an equal distribution over the data (following the original random seed).
    """
    def __init__(self, dataset, num_replicas=None, rank=None, seed=0):
        super().__init__(dataset, num_replicas=num_replicas, rank=rank, seed=seed)
        indices = list(range(len(self.dataset)))
        self.indices = indices[self.rank:self.total_size:self.n_clients]
