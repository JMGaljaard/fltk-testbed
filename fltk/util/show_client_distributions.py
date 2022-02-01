import logging
import sys

import pandas as pd
from tqdm import tqdm

from fltk.client import Client
from fltk.datasets import DistCIFAR10Dataset, DistCIFAR100Dataset, DistFashionMNISTDataset, DistDataset
import logging

from fltk.util.base_config import BareConfig

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
)
# dist_settings = {
#     'uniform':{},
#     'limit labels': {'seed': 1, 'range':[0.1, 1, 0.1]},
#     'q sampler': {'seed': 1, 'range':[0.1, 1, 0.1]},
#     'dirichlet': {'seed': 1, 'range':[0.1, 1, 0.1]},
# }

dist_settings = {
    # 'uniform':{},
    # 'limit labels flex': {'seed': 1, 'range':[0.1, 1, 0.1]},
    'n labels': {'seed': 1, 'range':[0.1, 1, 0.1]},
    # 'q sampler': {'seed': 1, 'range':[0.1, 1, 0.1]},
    # 'dirichlet': {'seed': 1, 'range':[0.1, 1, 0.1]},
}

num_clients = 10
class dummy_args:
    net = 'Cifar10CNN'
    dataset_name = 'cifar10'
    # data_sampler = "uniform" #s = "dirichlet"  # "limit labels" || "q sampler" || "dirichlet" || "uniform" (default)
    # data_sampler = "limit labels flex"
    data_sampler = "n labels"
    # sampler: "uniform" # "limit labels" || "q sampler" || "dirichlet" || "uniform" (default)
    # data_sampler_args = [0.07, 42]  # random seed || random seed || random seed || unused
    data_sampler_args = [2 , 42]  # random seed || random seed || random seed || unused
    DistDatasets = {
        'cifar10': DistCIFAR10Dataset,
        'cifar100': DistCIFAR100Dataset,
        'fashion-mnist': DistFashionMNISTDataset,
    }
    distributed = True
    rank = 0
    world_size = 2
    logger = logging.Logger(__name__)
    data_path = 'data'
    cuda = False

    def get_net(self):
        return self.net

    def init_logger(self, logger):
        self.logger = logger

    def get_distributed(self):
        return self.distributed

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def get_sampler(self):
        return self.data_sampler

    def get_sampler_args(self):
        return tuple(self.data_sampler_args)
    def get_logger(self):
        return self.logger

    def get_data_path(self):
        return self.data_path

def gen_distribution(name, params):
    world_size = num_clients + 1
    datasets = []
    idx2class = None
    distributions = {}
    for rank in range(world_size):
        if rank == 0:
            continue
        print(f'node {rank}')
        args = BareConfig()
        args.init_logger(logging)
        args.data_sampler = name


        # args.set_net_by_name('MNISTCNN')
        # args.dataset_name = 'mnist'
        args.set_net_by_name('FashionMNISTCNN')
        args.dataset_name = 'fashion-mnist'
        # data_sampler = "uniform" #s = "dirichlet"  # "limit labels" || "q sampler" || "dirichlet" || "uniform" (default)
        # data_sampler = "limit labels flex"
        args.data_sampler = "n labels"
        args.data_sampler_args = [2 , 42]
        args.world_size = world_size
        args.rank = rank
        dataset: DistDataset = args.DistDatasets[args.dataset_name](args)
        datasets.append((args, dataset))
        # test_loader = dataset.get_test_loader()
        # train_loader = dataset.get_train_loader()
        # class_dict = dataset.train_dataset.class_to_idx
        print('Iterating over all items')
        batch_size = 16
        # for i, (inputs, labels) in enumerate(dataset.get_train_loader(), 0):
        #     print(labels)
        # print('d')
        client = Client("test", None, rank, args.world_size, args)
        client.init_dataloader()
        train_loader = client.dataset.get_train_loader()
        train_loader2 = dataset.get_train_loader()
        test_loader = client.dataset.get_test_loader()
        test_loader2 = dataset.get_test_loader()
        idx2class = {v: k for k, v in train_loader.dataset.class_to_idx.items()}

        count_dict = {k: 0 for k, v in train_loader.dataset.class_to_idx.items()}
        for (inputs, labels) in tqdm(train_loader):
            for element in labels.numpy():
                # y_lbl = element[1]
                y_lbl = idx2class[element]
                count_dict[y_lbl] += 1
        if rank not in distributions:
            distributions[rank] = {}
        distributions[rank]['train'] = count_dict
        count_dict = {k: 0 for k, v in train_loader.dataset.class_to_idx.items()}
        for (inputs, labels) in tqdm(test_loader):
            for element in labels.numpy():
                # y_lbl = element[1]
                y_lbl = idx2class[element]
                count_dict[y_lbl] += 1
        if rank not in distributions:
            distributions[rank] = {}
        distributions[rank]['test'] = count_dict

        # return count_dict

    label_data = []

    for i, data_ in distributions.items():
        for k, v in data_['train'].items():
            label_data.append([i, k, v, 'train', name])
        for k, v in data_['test'].items():
            label_data.append([i, k, v, 'test', name])
    return label_data

def get_client_distributions():

    prefix = f'{num_clients}_clients'
    all_label_data = []
    for key, value in dist_settings.items():
        print(key, value)
        all_label_data += gen_distribution(key, value)

    #
    # world_size = num_clients + 1
    # datasets = []
    # idx2class = None
    # distributions = {}
    # for rank in range(world_size):
    #     if rank == 0:
    #         continue
    #     print(f'node {rank}')
    #     args = dummy_args()
    #     args.world_size = world_size
    #     args.rank = rank
    #     dataset : DistDataset = args.DistDatasets[args.dataset_name](args)
    #     datasets.append((args, dataset))
    #     test_loader = dataset.get_test_loader()
    #     train_loader = dataset.get_train_loader()
    #     class_dict = dataset.train_dataset.class_to_idx
    #     print('Iterating over all items')
    #     batch_size = 16
    #     # for i, (inputs, labels) in enumerate(dataset.get_train_loader(), 0):
    #     #     print(labels)
    #     # print('d')
    #     train_loader = dataset.get_train_loader()
    #     test_loader = dataset.get_test_loader()
    #     idx2class = {v: k for k, v in train_loader.dataset.class_to_idx.items()}
    #
    #     count_dict = {k: 0 for k, v in train_loader.dataset.class_to_idx.items()}
    #     for (inputs, labels) in tqdm(train_loader):
    #         for element in labels.numpy():
    #             # y_lbl = element[1]
    #             y_lbl = idx2class[element]
    #             count_dict[y_lbl] += 1
    #     if rank not in distributions:
    #         distributions[rank] = {}
    #     distributions[rank]['train'] = count_dict
    #     count_dict = {k: 0 for k, v in train_loader.dataset.class_to_idx.items()}
    #     for (inputs, labels) in tqdm(test_loader):
    #         for element in labels.numpy():
    #             # y_lbl = element[1]
    #             y_lbl = idx2class[element]
    #             count_dict[y_lbl] += 1
    #     if rank not in distributions:
    #         distributions[rank] = {}
    #     distributions[rank]['test'] = count_dict
    #
    #     # return count_dict
    #
    # label_data = []
    #
    # for i,  data_ in distributions.items():
    #     for k,v in data_['train'].items():
    #       label_data.append([i, k, v, 'train'])
    #     for k,v in data_['test'].items():
    #       label_data.append([i, k, v, 'test'])

    df = pd.DataFrame(all_label_data, columns=['node', 'key', 'value', 'type', 'sampler'])

    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure()
    # g = sns.FacetGrid(df, col='node', row='type')
    # g = sns.FacetGrid(df, row='node', col='type')
    g = sns.FacetGrid(df, col='node', row='sampler', hue='type')
    g.map(sns.barplot, 'key', 'value')
    plt.savefig(f'{prefix}_dist_plot.png')
    # sns.barplot(data=df, x='key', y='value')
    plt.show()

    # print(distributions)
    print('Train distribution per client:')
    print(df.groupby('node')['value'].sum().reset_index())
    sampler = 1
    distributed = True
    rank = 1
    world_size = 2
    data_set = 'cifar10'

    net = 'Cifar10CNN'
    dataset = 'cifar10'
    sampler = "dirichlet"  # "limit labels" || "q sampler" || "dirichlet" || "uniform" (default)
    # sampler: "uniform" # "limit labels" || "q sampler" || "dirichlet" || "uniform" (default)
    sampler_args = [0.07, 42]  # random seed || random seed || random seed || unused

    dist_datasets = {
        'cifar10': DistCIFAR10Dataset,
        'cifar100': DistCIFAR100Dataset,
        'fashion-mnist': DistFashionMNISTDataset,
    }

    # args = dummy_args()
    # ddataset: DistDataset = args.DistDatasets[args.dataset_name](args)

    # print(len(list(ddataset.get_train_loader())))
    # print('Done')




if __name__ == '__main__':
    # if len(sys.argv) <= 1:
    #     print('Missing arguments')
    #     exit(1)
    # num_clients = sys.argv[1]
    # print(f'Calculating client distributions for {num_clients} number of clients')
    get_client_distributions()

