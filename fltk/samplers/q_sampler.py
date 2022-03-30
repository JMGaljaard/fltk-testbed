from fltk.samplers import DistributedSamplerWrapper
from torch.utils.data import DistributedSampler, Dataset
import numpy as np
import logging
import random
from collections import Counter


class Probability_q_Sampler(DistributedSamplerWrapper):
    """
    Clients are divided among M groups, with M being the number of labels.
    A sample with label m is than given to a member of group m with probability q,
    and to any other group with probability (1-q)/(m-1)

    side effect of this method is that the reported loss on the test dataset becomes somewhat meaningless...logging.info("distribution in client with rank {}: {}".format(rank, Counter(labels)))
    """

    def __init__(self, dataset, num_replicas, rank, args=(0.5, 42)):
        q, seed = args
        super().__init__(dataset, num_replicas, rank, seed)

        if self.n_clients % self.n_labels != 0:
            logging.error(
                "multiples of {} clients are needed for the 'probability-q-sampler' data distribution method, {} does not work".format(
                    self.n_labels, self.n_clients))
            return

        # divide data among groups
        counter = 0  # for dividing data within a group
        group_id = self.client_id % self.n_labels
        group_clients = [client for client in range(self.n_clients) if client % self.n_labels == group_id]
        indices = []
        random.seed(seed)
        ordered_by_label = self.order_by_label(dataset)
        for group, label_list in enumerate(ordered_by_label):
            for sample_idx in label_list:
                rnd_val = random.random()
                if rnd_val < q:
                    if group == group_id:
                        if group_clients[counter] == self.client_id:
                            indices.append(sample_idx)
                        counter = (counter + 1) % len(group_clients)
                else:
                    others = [grp for grp in range(self.n_labels) if grp != group]
                    if random.choice(others) == group_id:
                        if group_clients[counter] == self.client_id:
                            indices.append(sample_idx)
                        counter = (counter + 1) % len(group_clients)

        labels = [dataset.targets[i] for i in indices]
        logging.info("nr of samplers in client with rank {}: {}".format(rank, len(indices)))
        logging.info("distribution in client with rank {}: {}".format(rank, Counter(labels)))

        random.seed(seed + self.client_id)  # give each client a unique shuffle
        random.shuffle(indices)  # shuffle indices to spread the labels

        self.indices = indices