import random

import numpy as np

from fltk.samplers import DistributedSamplerWrapper


class N_Labels(DistributedSamplerWrapper):
    """
    A sampler that limits the number of labels per client
    The number of clients must <= than number of labels
    """

    def __init__(self, dataset, num_replicas, rank, args=(5, 42)):
        limit, seed = args
        super().__init__(dataset, num_replicas, rank, seed)

        num_copies = np.ceil((args[0] * self.n_clients) / self.n_labels)
        label_dict = {}
        for l in range(self.n_labels):
            label_dict[l] = num_copies

        def get_least_used_labels(l_dict: dict):
            label_list = [[k, v] for k, v in label_dict.items()]
            label_list[-1][1] = 0
            sorted_list = sorted(label_list, key=lambda x: x[1], reverse=True)
            # print('d')
            # label_list.sort(lambda x:x)

        def choice_n(l_dict: dict, n, seed_offset = 0):
            # get_least_used_labels(l_dict)
            labels = [k for k, v in label_dict.items() if v]
            # summed = sum([int(v) for k, v in label_dict.items() if v])
            # amounts = [float(v) / float(summed) for k, v in label_dict.items() if v]
            # # p = amounts / summed
            print(f'Available labels: {labels} choose {n}')
            # # np.random.seed(seed + seed_offset)
            # # @TODO: Error is in this section!
            # print(f'n={n}, labels={labels}, p={amounts}')
            # print(amounts)

            selected = np.random.choice(labels, n, replace=False)
            # print(selected)
            for k, v in l_dict.items():
                if k in selected:
                    # v -= 1
                    l_dict[k] -= 1
            return selected


        # print(f'N Clients={self.n_clients}')
        # print(f'Num_buckets={num_copies}')

        clients = list(range(self.n_clients))  # keeps track of which clients should still be given a label
        client_label_dict = {}
        ordered_list = list(range(self.n_labels)) * int(num_copies)

        # Old code
        # for idx, client_id in enumerate(clients):
        #     # client_label_dict[client_id] = []
        #     label_set = choice_n(label_dict, args[0], idx)
        #     client_label_dict[client_id] = label_set

        # Now code
        for idx, client_id in enumerate(clients):
            label_set = []
            for _ in range(args[0]):
                label_set.append(ordered_list.pop())
            client_label_dict[client_id] = label_set

        client_label_dict['rest'] = []
        # New code
        if len(ordered_list):
            client_label_dict['rest'] = ordered_list

        #     Old code
        # client_label_dict['rest'] = labels = [k for k, v in label_dict.items() if v]
        # for k, v in label_dict.items():
        #     for x in range(int(v)):
        #         client_label_dict['rest'].append(int(k))

        # Order data by label; split into N buckets and select indices based on the order found in the client-label-dict

        reverse_label_dict = {}
        for l in range(self.n_labels):
            reverse_label_dict[l] = []

        for k, v in client_label_dict.items():
            # print(f'client {k} has labels {v}')
            for l_c in v:
                reverse_label_dict[l_c].append(k)

        indices = []
        ordered_by_label = self.order_by_label(dataset)
        indices_per_client = {}
        for c in clients:
            indices_per_client[c] = []

        rest_indices = []
        for group, label_list in enumerate(ordered_by_label):
            splitted = np.array_split(label_list, num_copies)
            client_id_to_distribute = reverse_label_dict[group]
            for split_part in splitted:
                client_key = client_id_to_distribute.pop()
                if client_key == 'rest':
                    rest_indices.append(split_part)
                else:
                    indices_per_client[client_key].append(split_part)
            # for split_part in splitted:
        # @TODO: Fix this part in terms of code cleanness. Could be written more cleanly
        if len(rest_indices):
            rest_indices = np.concatenate(rest_indices)
            rest_splitted = np.array_split(rest_indices, len(indices_per_client))

            for k, v in indices_per_client.items():
                v.append(rest_splitted.pop())
                indices_per_client[k] = np.concatenate(v)
        else:
            rest_indices = np.ndarray([])
            for k, v in indices_per_client.items():
                indices_per_client[k] = np.concatenate(v)

        indices = indices_per_client[self.client_id]
        random.seed(seed + self.client_id)  # give each client a unique shuffle
        random.shuffle(indices)  # shuffle indices to spread the labels

        self.indices = indices

        # labels_per_client = int(np.floor(self.n_labels / self.n_clients))
        # remaining_labels = self.n_labels - labels_per_client
        # labels = list(range(self.n_labels))  # list of labels to distribute
        # clients = list(range(self.n_clients))  # keeps track of which clients should still be given a label
        # client_labels = [set() for n in range(self.n_clients)]  # set of labels given to each client
        # random.seed(seed)  # seed, such that the same result can be obtained multiple times
        # print(client_labels)
        #
        # label_order = random.sample(labels, len(labels))
        # client_label_dict = {}
        # for client_id in clients:
        #     client_label_dict[client_id] = []
        #     for _ in range(labels_per_client):
        #         chosen_label = label_order.pop()
        #         client_label_dict[client_id].append(chosen_label)
        #         client_labels[client_id].add(chosen_label)
        # client_label_dict['rest'] = label_order
        #
        #
        #
        # indices = []
        # ordered_by_label = self.order_by_label(dataset)
        # labels = client_label_dict[self.client_id]
        # for label in labels:
        #     n_samples = int(len(ordered_by_label[label]))
        #     clients = [c for c, s in enumerate(client_labels) if label in s]  # find out which clients have this label
        #     index = clients.index(self.client_id)  # find the position of this client
        #     start_index = index * n_samples  # inclusive
        #     if rank == self.n_clients:
        #         end_index = len(ordered_by_label[label])  # exclusive
        #     else:
        #         end_index = start_index + n_samples  # exclusive
        #
        #     indices += ordered_by_label[label][start_index:end_index]
        #
        # # Last part is uniform sampler
        # rest_indices = []
        # for l in client_label_dict['rest']:
        #     rest_indices += ordered_by_label[l]
        # filtered_rest_indices = rest_indices[self.rank:self.total_size:self.num_replicas]
        # indices += filtered_rest_indices
        # random.seed(seed + self.client_id)  # give each client a unique shuffle
        # random.shuffle(indices)  # shuffle indices to spread the labels
        #
        # self.indices = indices