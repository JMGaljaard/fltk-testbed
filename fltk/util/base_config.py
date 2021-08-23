from dataclasses import dataclass

import torch
from dataclasses_json import dataclass_json


# TODO: Move reproducability settings towards a different part of the codebase.
# SEED = 1
# torch.manual_seed(SEED)


@dataclass
@dataclass_json
class GeneralNetConfig:
    save_model: bool = False
    save_temp_model: bool = False
    save_epoch_interval: int = 1
    save_model_path: str = "models"
    epoch_save_start_suffix: str = "start"
    epoch_save_end_suffix = "end"


@dataclass(frozen=True)
@dataclass_json
class ReproducabilityConfig:
    torch_seed: int
    arrival_seed: int


@dataclass
@dataclass_json
class ExecutionConfig:
    general_net: GeneralNetConfig
    reproducability: ReproducabilityConfig
    experiment_prefix: str = "experiment"
    tensorboard_active: str = True
    cuda: bool = False


@dataclass
@dataclass_json
class OrchestratorConfig:
    service: str
    nic: str


@dataclass
@dataclass_json
class ClientConfig:
    prefix: str


@dataclass
@dataclass_json
class ClusterConfig:
    orchestrator: OrchestratorConfig
    client: ClientConfig
    wait_for_clients: bool = True


@dataclass
@dataclass_json
class BareConfig(object):
    # Configuration parameters for PyTorch and models that are generated.
    execution_config: ExecutionConfig
    cluster_config: ClusterConfig

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
