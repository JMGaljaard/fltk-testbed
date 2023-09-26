import random

import numpy as np

from fltk.samplers import DistributedSamplerWrapper


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



class ContinuousSampler(DistributedSamplerWrapper):
    """
    A sampler that limits the number of labels per client
    The number of clients must <= than number of labels
    """

    def __init__(self, dataset, num_replicas: int, rank: int, partitioning: str, num_tasks: int, args=(5, 42)):
        limit, seed = args
        self.num_tasks = num_tasks
        self.partition = partitioning
        self.task_idx = 0
        super().__init__(dataset, num_replicas, rank, seed)

        # Column based sampling
        # We assume that the dataset has labels


        tasks = np.random.permutation(20)[:args.task]


        # Coarse labels:
        coarse_labels = sparse2coarse(self.dataset.targets)


        
        # if(self.labelType == "finecoarse"):
        #     zipped = zip(self.ldata, self.targets, self.extra_targets)
        # else:
        #     zipped = zip(self.ldata, self.targets) #ldata contains the images, targets contains the corresponding fine label (label from 0-99 for cifar-100)


        # self.sort_zipped = sorted(zipped, key=lambda x: x[1]) #sorts the zipped data and labels based on the label number
        # self.task_datasets = []
        # samples = len(self.data) // task_num
        # for i in range(task_num):
        #     task_dataset = []
        #     for j in range(samples):
        #         if(self.labelType == "finecoarse"):
        #             fine_label_orig = self.sort_zipped[i * samples + j][2]
        #             # fine_label_fixed = labelMapper(fine_label_orig, i)
        #             # task_dataset.append((self.sort_zipped[i * samples + j][0], fine_label_fixed))
        #             task_dataset.append((self.sort_zipped[i * samples + j][0], self.sort_zipped[i * samples + j][2]))
        #         else:
        #             task_dataset.append(self.sort_zipped[i * samples + j])
        #     self.task_datasets.append(task_dataset)

        # if(self.labelType == "coarse"):
        #     self.targets.extend(entry['coarse_labels'])
        # elif self.labelType == "finecoarse": #case where we want to group based on coarse labels but use fine labels for training
        #     self.targets.extend(entry["coarse_labels"])
        #     self.extra_targets.extend(entry['fine_labels'])
        # else:
        #     self.targets.extend(entry['fine_labels'])




        # # This is the code block for the column partitioning
        # for task in tasks:
        #     task_dataset = dataset[task].data
        #     chunk_size = len(task_dataset)//num_users
        #     # remainder = len(task_dataset)%num_users #discard the remainder
        #     client_id = 0
        #     for i in range(0, len(task_dataset), chunk_size):
        #         dict_users[client_id].append(task_dataset[i:i + chunk_size])
        #         client_id = (client_id + 1)%num_users
        #     # if(remainder != 0):
        #     #     dict_users[0].append(task_dataset[-remainder:])



        # labels_per_client = int(np.floor(self.n_labels / self.n_clients))
        # remaining_labels = self.n_labels - labels_per_client
        # labels = list(range(self.n_labels))  # list of labels to distribute
        # clients = list(range(self.n_clients))  # keeps track of which clients should still be given a label
        # client_labels = [set() for n in range(self.n_clients)]  # set of labels given to each client
        # random.seed(seed)  # seed, such that the same result can be obtained multiple times
        # print(client_labels)

        # label_order = random.sample(labels, len(labels))
        # client_label_dict = {}
        # for client_id in clients:
        #     client_label_dict[client_id] = []
        #     for _ in range(labels_per_client):
        #         chosen_label = label_order.pop()
        #         client_label_dict[client_id].append(chosen_label)
        #         client_labels[client_id].add(chosen_label)
        # client_label_dict['rest'] = label_order

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

        #     indices += ordered_by_label[label][start_index:end_index]

        # # Last part is uniform sampler
        # rest_indices = []
        # for l in client_label_dict['rest']:
        #     rest_indices += ordered_by_label[l]
        # filtered_rest_indices = rest_indices[self.rank:self.total_size:self.num_replicas]
        # indices += filtered_rest_indices
        # random.seed(seed + self.client_id)  # give each client a unique shuffle
        # random.shuffle(indices)  # shuffle indices to spread the labels

        # self.indices = indices


        # Currently the wrong implementation but it works in essence.
        indices = list(range(len(self.dataset)))
        chunk_size = self.total_size // args.tasks
        task_indices = indices[self.task_idx*chunk_size: (self.task_idx+1)* chunk_size]
        total_task_size = self.total_size // args.tasks
        self.indices = task_indices[self.rank:total_task_size:self.n_clients]

    def next_task(self):
        self.task_idx += 1

    def set_task(self, idx: int):
        self.task_idx = idx
        