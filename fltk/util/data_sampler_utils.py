
from torch.utils.data import Dataset
import random
import logging
from torch.utils.data import DistributedSampler
from typing import Iterator



class LimitLabelsSampler(DistributedSampler):
    """
    A sampler that limits the number of labels per client
    """
    def __init__(self, dataset, rank, world_size, limit=5, seed=42): 
        super(LimitLabelsSampler, self).__init__(
            dataset, world_size, rank, False)
        
        client_id = rank - 1
        n_clients = world_size - 1
        # order the indices by label
        ordered_by_label = [[] for i in range(len(dataset.classes))]
        for index, target in enumerate(dataset.targets):
            ordered_by_label[target].append(index)

        n_labels = len(ordered_by_label)
        
        if n_clients % n_labels != 0:
            logging.error("multiples of {} clients are needed for the 'limiting-labels' data distribution method, {} does not work".format(n_labels, n_clients))
            return

        n_occurrences = limit * int(n_clients/n_labels)         # number of occurrences of each label
        counters = [n_occurrences] * n_clients                  # keeps track of which labels still can be given out
        labels = list(range(n_labels))                          # list of labels to distribute
        clients = list(range(n_clients))                        # keeps track of which clients should still be given a label
        client_labels = [set() for n in range(n_clients)]       # set of labels given to each client
        random.seed(seed)                                       # seed, such that the same result can be obtained multiple times

        while labels:
            # pick a random label
            label = random.choice(labels)
            counters[label] -= 1        # decrement counter of this label
            if counters[label] == 0:    # if needed, remove label
                labels.remove(label)
            
            # check which clients the label can be given to
            selectable = [i for i in clients if not label in client_labels[i]]
            client = None
            
            if not selectable:
                # poor choice, let's fix this -> swap two labels
                # conditions for swapping: 
                #   sets of labels A, B, with B incomplete, remaining label l that is not possible to give to B, s.t.:
                #       (1) l not in A
                #       (2) exists label l' in A but not in B
                #   l, l' can be swapped

                client = random.choice(clients) # label can not be given to this client
                for c, s in enumerate(client_labels):
                    if len(s) == limit:     # this a completed set
                        if label not in s:  # label can be given to this client (1)
                            subset = s.difference(client_labels[client]) # remove labels client already has (2...)
                            if subset:  # subset is not empty (2 continued):
                                l = min(subset) # get a swappable label (in a deterministic way), and swap labels
                                client_labels[c].remove(l)  
                                client_labels[c].add(label)
                                client_labels[client].add(l)
                                break
            else:   # normal operation, pick a rondom selectable client
                client = random.choice(selectable)
                client_labels[client].add(label)
            
            # check if this client has been given the maximum number of labels
            if len(client_labels[client]) == limit:
                clients.remove(client)

        # now we have a set of labels for each client
        # client with rank=rank now needs to be given data
        # all clients get the same amount of data, the first portion is given to client with rank 1, the second to rank 2, etc

        labels = client_labels[client_id]
        logging.info("Client {} gets labels {}".format(rank, client_labels[client_id]))
        indices = []
        for label in labels:
            n_samples = int( len(ordered_by_label[label])/n_occurrences ) 
            clients = [c for c, s in enumerate(client_labels) if label in s]    # find out which clients have this label
            index = clients.index(client_id)           # find the position of this client
            start_index = index*n_samples           # inclusive
            if rank == n_clients:
                end_index = len(ordered_by_label[label])     # exclusive
            else:
                end_index = start_index + n_samples # exclusive

            indices += ordered_by_label[label][start_index:end_index]

        random.seed(seed+client_id)  # give each client a unique shuffle
        random.shuffle(indices) # shuffle indices to spread the labels

        self.indices = indices

    def __iter__(self) -> Iterator[int]:
        return iter(self.indices)

    def __len__(self) -> int:
        return len(self.indices)
