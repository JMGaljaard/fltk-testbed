from dataclasses import dataclass
from typing import Dict

import torch
from dataclasses_json import dataclass_json

SEED = 1
torch.manual_seed(SEED)

@dataclass
@dataclass_json
class ExecutionConfig():
    cuda: bool = False
    save_model: bool = False
    save_temp_model: bool = False
    save_epoch_interval: int = 1
    save_model_path: str = "models"
    epoch_save_start_suffix: str = "start"
    epoch_save_end_suffix = "end"

@dataclass
@dataclass_json
class BareConfig(object):
    # Configuration parameters for PyTorch and models that are generated.
    execution_config = ExecutionConfig()

    def __init__(self):

        # TODO: Move to external class/object
        self.train_data_loader_pickle_path = {
            'cifar10': 'data_loaders/cifar10/train_data_loader.pickle',
            'fashion-mnist': 'data_loaders/fashion-mnist/train_data_loader.pickle',
            'cifar100': 'data_loaders/cifar100/train_data_loader.pickle',
        }

        self.test_data_loader_pickle_path = {
            'cifar10': 'data_loaders/cifar10/test_data_loader.pickle',
            'fashion-mnist': 'data_loaders/fashion-mnist/test_data_loader.pickle',
            'cifar100': 'data_loaders/cifar100/test_data_loader.pickle',
        }

        # TODO: Make part of different configuration
        self.loss_function = torch.nn.CrossEntropyLoss

        self.default_model_folder_path = "default_models"
        self.data_path = "data"

    def merge_yaml(self, cfg: Dict[str, str] = {}):
        """
        @deprecated This function will become redundant after using dataclasses_json to load the values into the object.

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
            self.set_net_by_name(cfg['net'])
        if 'dataset' in cfg:
            self.dataset_name = cfg['dataset']
        if 'experiment_prefix' in cfg:
            self.experiment_prefix = cfg['experiment_prefix']
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
        if 'sampler' in cfg:
            self.data_sampler = cfg['sampler']
        if 'sampler_args' in cfg:
            self.data_sampler_args = cfg['sampler_args']

        if 'poison' in cfg:
            self.poison = cfg['poison']

        if 'antidote' in cfg:
            self.antidote = cfg['antidote']

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
        return list(self.available_nets.keys())

    def set_train_data_loader_pickle_path(self, path, name='cifar10'):
        self.train_data_loader_pickle_path[name] = path

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path[self.dataset_name]

    def set_test_data_loader_pickle_path(self, path, name='cifar10'):
        self.test_data_loader_pickle_path[name] = path

    def get_test_data_loader_pickle_path(self):
        return self.test_data_loader_pickle_path[self.dataset_name]

    def set_net_by_name(self, name: str):
        self.net = self.available_nets[name]

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


    def should_save_model(self, epoch_idx):
        """
        Returns true/false models should be saved.

        :param epoch_idx: current training epoch index
        :type epoch_idx: int
        """
        return self.save_model and (epoch_idx == 1 or epoch_idx % self.save_epoch_interval == 0)

    def log(self):
        """
        Log this arguments object to the logger.
        """
        self.logger.debug("Arguments: {}", str(self))

