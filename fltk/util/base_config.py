from dataclasses import dataclass
from typing import Dict

import torch
from dataclasses_json import dataclass_json


# TODO: Move reproducability settings towards a different part of the codebase.
# SEED = 1
# torch.manual_seed(SEED)

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

    def get_dataloader_list(self):
        """
        @deprecated
        @return:
        @rtype:
        """
        return list(self.train_data_loader_pickle_path.keys())

    def get_nets_list(self):
        """
        @deprecated
        @return:
        @rtype:
        """
        return list(self.available_nets.keys())

    def set_train_data_loader_pickle_path(self, path, name='cifar10'):
        self.train_data_loader_pickle_path[name] = path

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path[self.dataset_name]

    def set_test_data_loader_pickle_path(self, path, name='cifar10'):
        self.test_data_loader_pickle_path[name] = path

    def get_test_data_loader_pickle_path(self):
        return self.test_data_loader_pickle_path[self.dataset_name]

    def get_save_model_folder_path(self):
        return self.save_model_path

    def should_save_model(self, epoch_idx):
        """
        Returns true/false models should be saved.

        :param epoch_idx: current training epoch index
        :type epoch_idx: int
        """
        return self.save_model and (epoch_idx == 1 or epoch_idx % self.save_epoch_interval == 0)
