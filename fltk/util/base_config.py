from datetime import datetime

import torch
import json

from fltk.datasets.distributed import DistCIFAR10Dataset, DistCIFAR100Dataset, DistFashionMNISTDataset
from fltk.datasets.distributed.mnist import DistMNISTDataset
from fltk.nets import Cifar10CNN, FashionMNISTCNN, Cifar100ResNet, FashionMNISTResNet, Cifar10ResNet, Cifar100VGG
from fltk.nets.mnist_cnn import MNIST_CNN
from fltk.strategy.optimization import FedProx, FedNova
from fltk.util.definitions import Optimizations, DataSampler, Nets, Dataset

SEED = 1
torch.manual_seed(SEED)


class BareConfig:

    def __init__(self):
        # self.logger = logger

        self.batch_size = 1
        self.test_batch_size = 1000
        self.epochs = 1
        # self.lr = 0.001
        self.lr = 0.0001
        # self.momentum = 0.9
        self.momentum = 0.1
        self.cuda = False
        self.shuffle = False
        self.log_interval = 10
        self.kwargs = {}
        self.contribution_measurement_round = 1
        self.contribution_measurement_metric = 'Influence'
        self.epochs_per_round = 1

        self.scheduler_step_size = 50
        self.scheduler_gamma = 0.5
        self.min_lr = 1e-10

        self.loss_function = torch.nn.CrossEntropyLoss
        self.optimizer = torch.optim.SGD

        self.optimizers = {
            Optimizations.sgd: torch.optim.SGD,
            Optimizations.fedprox: FedProx,
            Optimizations.fednova: FedNova
        }

        self.optimizer_args = {
            'lr': self.lr,
            'momentum': self.momentum
        }

        self.round_worker_selection_strategy = None
        self.round_worker_selection_strategy_kwargs = None

        self.save_model = False
        self.save_temp_model = False
        self.save_epoch_interval = 1
        self.save_model_path = "models"
        self.epoch_save_start_suffix = "start"
        self.epoch_save_end_suffix = "end"
        self.get_poison_effort = 'half'
        self.num_workers = 50
        # self.num_poisoned_workers = 10

        self.offload_strategy = 'vanilla'
        self.profiling_size = 30
        self.deadline = 400
        self.first_deadline = 400
        self.deadline_threshold = 10
        self.warmup_round = False

        # FLTK options
        self.node_groups = None

        # Termination policy data
        self.termination_percentage = 1

        self.federator_host = '0.0.0.0'
        self.rank = 0
        self.world_size = 0
        self.data_sampler = DataSampler.uniform
        self.data_sampler_args = None
        self.distributed = False

        self.available_nets = {
            Nets.cifar100_resnet: Cifar100ResNet,
            Nets.cifar100_vgg: Cifar100VGG,
            Nets.cifar10_cnn: Cifar10CNN,
            Nets.cifar10_resnet: Cifar10ResNet,
            Nets.fashion_mnist_cnn: FashionMNISTCNN,
            Nets.fashion_mnist_resnet: FashionMNISTResNet,
            Nets.mnist_cnn: MNIST_CNN,

        }

        self.nets_split_point = {
            Nets.cifar100_resnet: 48,
            Nets.cifar100_vgg: 28,
            Nets.cifar10_cnn: 15,
            Nets.cifar10_resnet: 39,
            Nets.fashion_mnist_cnn: 7,
            Nets.fashion_mnist_resnet: 7,
            Nets.mnist_cnn: 2,
        }
        self.net = None
        self.net_name = Nets.cifar10_cnn
        self.set_net_by_name(self.net_name.value)
        # self.dataset_name = 'cifar10'
        self.dataset_name = Dataset.cifar10

        self.DistDatasets = {
            Dataset.cifar10: DistCIFAR10Dataset,
            Dataset.cifar100: DistCIFAR100Dataset,
            Dataset.fashion_mnist: DistFashionMNISTDataset,
            Dataset.mnist: DistMNISTDataset
        }
        self.train_data_loader_pickle_path = {
            Dataset.cifar10: 'data_loaders/cifar10/train_data_loader.pickle',
            Dataset.fashion_mnist: 'data_loaders/fashion-mnist/train_data_loader.pickle',
            Dataset.cifar100: 'data_loaders/cifar100/train_data_loader.pickle',
            Dataset.mnist: 'data_loaders/mnist/train_data_loader.pickle',
        }

        self.test_data_loader_pickle_path = {
            Dataset.cifar10: 'data_loaders/cifar10/test_data_loader.pickle',
            Dataset.fashion_mnist: 'data_loaders/fashion-mnist/test_data_loader.pickle',
            Dataset.cifar100: 'data_loaders/cifar100/test_data_loader.pickle',
            Dataset.mnist: 'data_loaders/mnist/test_data_loader.pickle',

        }
        self.loss_function = torch.nn.CrossEntropyLoss
        self.default_model_folder_path = "default_models"
        self.data_path = "data"

        # For freezing effect experiment
        self.freeze_clients = []

    ###########
    # Methods #
    ###########

    def merge_yaml(self, cfg = {}):
        """
        total_epochs: 20
        epochs_per_cycle: 2
        wait_for_clients: true
        net: Cifar10CNN
        dataset: cifar10
        experiment_prefix: 'experiment'
        output_location: 'output'
        tensor_board_active: true
        :param yaml_config:
        :return:
        """
        if 'total_epochs' in cfg:
            self.epochs = cfg['total_epochs']
        if 'epochs_per_cycle' in cfg:
            self.epochs_per_cycle = cfg['epochs_per_cycle']
        if 'wait_for_clients' in cfg:
            self.wait_for_clients = cfg['wait_for_clients']
        if 'net' in cfg:
            self.net_name = Nets(cfg['net'])
            self.set_net_by_name(cfg['net'])
        if 'dataset' in cfg:
            self.dataset_name = Dataset(cfg['dataset'])
        if 'offload_stategy' in cfg:
            self.offload_strategy = cfg['offload_stategy']
        if 'profiling_size' in cfg:
            self.profiling_size = cfg['profiling_size']
        if 'deadline' in cfg:
            self.deadline = cfg['deadline']
        if 'deadline_threshold' in cfg:
            self.deadline_threshold = cfg['deadline_threshold']
        if 'first_deadline' in cfg:
            self.first_deadline = cfg['first_deadline']
        if 'warmup_round' in cfg:
            self.warmup_round = cfg['warmup_round']
        if 'experiment_prefix' in cfg:
            self.experiment_prefix = cfg['experiment_prefix']
        else:
            self.experiment_prefix = f'{datetime.now()}'
        if 'output_location' in cfg:
            self.output_location = cfg['output_location']
        if 'tensor_board_active' in cfg:
            self.tensor_board_active = cfg['tensor_board_active']
        if 'clients_per_round' in cfg:
            self.clients_per_round = cfg['clients_per_round']
        if 'system' in cfg:
            if 'clients' in cfg['system']:
                if 'amount' in cfg['system']['clients']:
                    self.world_size = cfg['system']['clients']['amount'] + 1

        if 'system' in cfg:
            if 'federator' in cfg['system']:
                if 'hostname' in cfg['system']['federator']:
                    self.federator_host = cfg['system']['federator']['hostname']
        if 'cuda' in cfg:
            if cfg['cuda']:
                self.cuda = True
            else:
                self.cuda = False
        if 'optimizer' in cfg:
            self.optimizer = self.optimizers[Optimizations(cfg['optimizer'])]
        if 'optimizer_args' in cfg:
            for k, v in cfg['optimizer_args'].items():
                self.optimizer_args[k] = v
        if 'sampler' in cfg:
            self.data_sampler = DataSampler(cfg['sampler'])
        if 'sampler_args' in cfg:
            self.data_sampler_args = cfg['sampler_args']

        if 'node_groups' in cfg:
            self.node_groups = cfg['node_groups']
        if 'termination_percentage' in cfg:
            self.termination_percentage = cfg['termination_percentage']
        
        if 'epochs_per_round' in cfg:
            self.epochs_per_round = cfg['epochs_per_round']
        if 'freeze_clients' in cfg:
            self.freeze_clients = cfg['freeze_clients']
            


    def init_logger(self, logger):
        self.logger = logger

    def get_distributed(self):
        return self.distributed

    def get_rank(self):
        return self.rank

    def get_world_size(self):
        return self.world_size

    def set_sampler(self, sampler):
        self.data_sampler = sampler

    def get_sampler(self):
        return self.data_sampler

    def get_optimizer(self):
        return self.optimizer
    
    def get_sampler_args(self):
        return tuple(self.data_sampler_args)

    def get_round_worker_selection_strategy(self):
        return self.round_worker_selection_strategy

    def get_round_worker_selection_strategy_kwargs(self):
        return self.round_worker_selection_strategy_kwargs

    def set_round_worker_selection_strategy_kwargs(self, kwargs):
        self.round_worker_selection_strategy_kwargs = kwargs

    def set_client_selection_strategy(self, strategy):
        self.round_worker_selection_strategy = strategy

    def get_data_path(self):
        return self.data_path

    def get_epoch_save_start_suffix(self):
        return self.epoch_save_start_suffix

    def get_epoch_save_end_suffix(self):
        return self.epoch_save_end_suffix

    def get_dataloader_list(self):
        return list(self.train_data_loader_pickle_path.keys())

    def get_nets_list(self):
        return list(map(lambda c: c.value, Nets))

    def set_train_data_loader_pickle_path(self, path, name='cifar10'):
        self.train_data_loader_pickle_path[name] = path

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path[self.dataset_name]

    def set_test_data_loader_pickle_path(self, path, name='cifar10'):
        self.test_data_loader_pickle_path[name] = path

    def get_test_data_loader_pickle_path(self):
        return self.test_data_loader_pickle_path[self.dataset_name]

    def set_net_by_name(self, name: str):
        self.net_name = Nets(name)
        self.net = self.available_nets[self.net_name]

    def get_cuda(self):
        return self.cuda

    def get_scheduler_step_size(self):
        return self.scheduler_step_size

    def get_scheduler_gamma(self):
        return self.scheduler_gamma

    def get_min_lr(self):
        return self.min_lr

    def get_default_model_folder_path(self):
        return self.default_model_folder_path

    def get_num_epochs(self):
        return self.epochs

    def set_num_poisoned_workers(self, num_poisoned_workers):
        self.num_poisoned_workers = num_poisoned_workers

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def set_model_save_path(self, save_model_path):
        self.save_model_path = save_model_path

    def get_logger(self):
        return self.logger

    def get_loss_function(self):
        return self.loss_function

    def get_net(self):
        return self.net

    def get_num_workers(self):
        return self.num_workers

    def get_num_poisoned_workers(self):
        return self.num_poisoned_workers

    def get_poison_effort(self):
        return self.get_poison_effort

    def get_learning_rate(self):
        return self.lr

    def get_momentum(self):
        return self.momentum

    def get_shuffle(self):
        return self.shuffle

    def get_batch_size(self):
        return self.batch_size

    def get_test_batch_size(self):
        return self.test_batch_size

    def get_log_interval(self):
        return self.log_interval

    def get_save_model_folder_path(self):
        return self.save_model_path

    def get_learning_rate_from_epoch(self, epoch_idx):
        lr = self.lr * (self.scheduler_gamma ** int(epoch_idx / self.scheduler_step_size))

        if lr < self.min_lr:
            self.logger.warning("Updating LR would place it below min LR. Skipping LR update.")

            return self.min_lr

        self.logger.debug("LR: {}".format(lr))

        return lr

    def get_contribution_measurement_round(self):
        return self.contribution_measurement_round

    def get_contribution_measurement_metric(self):
        return self.contribution_measurement_metric

    def should_save_model(self, epoch_idx):
        """
        Returns true/false models should be saved.

        :param epoch_idx: current training epoch index
        :type epoch_idx: int
        """
        if not self.save_model:
            return False

        if epoch_idx == 1 or epoch_idx % self.save_epoch_interval == 0:
            return True

    def log(self):
        """
        Log this arguments object to the logger.
        """
        self.logger.debug("Arguments: {}", str(self))

    def __str__(self):
        return "\nBatch Size: {}\n".format(self.batch_size) + \
               "Test Batch Size: {}\n".format(self.test_batch_size) + \
               "Epochs: {}\n".format(self.epochs) + \
               "Learning Rate: {}\n".format(self.lr) + \
               "Momentum: {}\n".format(self.momentum) + \
               "CUDA Enabled: {}\n".format(self.cuda) + \
               "Shuffle Enabled: {}\n".format(self.shuffle) + \
               "Log Interval: {}\n".format(self.log_interval) + \
               "Scheduler Step Size: {}\n".format(self.scheduler_step_size) + \
               "Scheduler Gamma: {}\n".format(self.scheduler_gamma) + \
               "Scheduler Minimum Learning Rate: {}\n".format(self.min_lr) + \
               "Client Selection Strategy: {}\n".format(self.round_worker_selection_strategy) + \
               "Client Selection Strategy Arguments: {}\n".format(
                   json.dumps(self.round_worker_selection_strategy_kwargs, indent=4, sort_keys=True)) + \
               "Model Saving Enabled: {}\n".format(self.save_model) + \
               "Model Saving Interval: {}\n".format(self.save_epoch_interval) + \
               "Model Saving Path (Relative): {}\n".format(self.save_model_path) + \
               "Epoch Save Start Prefix: {}\n".format(self.epoch_save_start_suffix) + \
               "Epoch Save End Suffix: {}\n".format(self.epoch_save_end_suffix) + \
               "Number of Clients: {}\n".format(self.num_workers) + \
               "Number of Poisoned Clients: {}\n".format(self.num_poisoned_workers) + \
               "NN: {}\n".format(self.net) + \
               "Train Data Loader Path: {}\n".format(self.train_data_loader_pickle_path) + \
               "Test Data Loader Path: {}\n".format(self.test_data_loader_pickle_path) + \
               "Loss Function: {}\n".format(self.loss_function) + \
               "Default Model Folder Path: {}\n".format(self.default_model_folder_path) + \
               "Data Path: {}\n".format(self.data_path) + \
               "Dataset Name: {}\n".format(self.dataset_name)