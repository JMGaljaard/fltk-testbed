from fltk.samplers import DistributedSamplerWrapper
from torch.utils.data import DistributedSampler, Dataset
import numpy as np
import logging
import random
from collections import Counter


class LimitLabelsSamplerFlex(DistributedSamplerWrapper):
    """
    A sampler that limits the number of labels per client
    The number of clients must <= than number of labels
    """

    def __init__(self, dataset, num_replicas, rank, args=(5, 42)):
        limit, seed = args
        super().__init__(dataset, num_replicas, rank, seed)

        labels_per_client = int(np.floor(self.n_labels / self.n_clients))
        remaining_labels = self.n_labels - labels_per_client
        labels = list(range(self.n_labels))  # list of labels to distribute
        clients = list(range(self.n_clients))  # keeps track of which clients should still be given a label
        client_labels = [set() for n in range(self.n_clients)]  # set of labels given to each client
        random.seed(seed)  # seed, such that the same result can be obtained multiple times
        print(client_labels)

        label_order = random.sample(labels, len(labels))
        client_label_dict = {}
        for client_id in clients:
            client_label_dict[client_id] = []
            for _ in range(labels_per_client):
                chosen_label = label_order.pop()
                client_label_dict[client_id].append(chosen_label)
                client_labels[client_id].add(chosen_label)
        client_label_dict['rest'] = label_order

        indices = []
        ordered_by_label = self.order_by_label(dataset)
        labels = client_label_dict[self.client_id]
        for label in labels:
            n_samples = int(len(ordered_by_label[label]))
            clients = [c for c, s in enumerate(client_labels) if label in s]  # find out which clients have this label
            index = clients.index(self.client_id)  # find the position of this client
            start_index = index * n_samples  # inclusive
            if rank == self.n_clients:
                end_index = len(ordered_by_label[label])  # exclusive
            else:
                end_index = start_index + n_samples  # exclusive

            indices += ordered_by_label[label][start_index:end_index]

        # Last part is uniform sampler
        rest_indices = []
        for l in client_label_dict['rest']:
            rest_indices += ordered_by_label[l]
        filtered_rest_indices = rest_indices[self.rank:self.total_size:self.num_replicas]
        indices += filtered_rest_indices
        random.seed(seed + self.client_id)  # give each client a unique shuffle
        random.shuffle(indices)  # shuffle indices to spread the labels

        self.indices = indices