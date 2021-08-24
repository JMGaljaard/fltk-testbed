from dataclasses import dataclass, field
from pathlib import Path

from dataclasses_json import config, dataclass_json


@dataclass_json
@dataclass
class GeneralNetConfig:
    save_model: bool = False
    save_temp_model: bool = False
    save_epoch_interval: int = 1
    save_model_path: str = 'models'
    epoch_save_start_suffix: str = 'cloud_experiment'
    epoch_save_end_suffix: str = 'cloud_experiment'


@dataclass_json
@dataclass(frozen=True)
class ReproducibilityConfig:
    torch_seed: int
    arrival_seed: int


@dataclass_json
@dataclass(frozen=True)
class TensorboardConfig:
    active: bool
    record_dir: str

    def prepare_log_dir(self, working_dir: Path = None):
        """
        Function to create logging directory used by TensorBoard. When running in a cluster, this function should not be
        used, as the TensorBoard instance that is started simultaneously with the Orchestrator.
        @param working_dir: Current working directory, by default PWD is assumed at which the Python interpreter is
        started.
        @type working_dir: pathlib.Path
        @return: None
        @rtype: None
        """
        dir_to_check = Path(self.record_dir)
        if working_dir:
            dir_to_check = working_dir.joinpath(dir_to_check)
        if not dir_to_check.exists() and dir_to_check.parent.is_dir():
            dir_to_check.mkdir()


@dataclass_json
@dataclass
class ExecutionConfig:
    general_net: GeneralNetConfig = field(metadata=config(field_name="net"))
    reproducibility: ReproducibilityConfig
    tensorboard: TensorboardConfig

    experiment_prefix: str = "experiment"
    cuda: bool = False


@dataclass_json
@dataclass
class OrchestratorConfig:
    service: str
    nic: str


@dataclass_json
@dataclass
class ClientConfig:
    prefix: str
    tensorboard_active: bool


@dataclass_json
@dataclass
class ClusterConfig:
    orchestrator: OrchestratorConfig
    client: ClientConfig
    wait_for_clients: bool = True


@dataclass_json
@dataclass
class BareConfig(object):
    # Configuration parameters for PyTorch and models that are generated.
    execution_config: ExecutionConfig
    cluster_config: ClusterConfig = field(metadata=config(field_name="cluster"))

    #     # TODO: Move to external class/object
    #     self.train_data_loader_pickle_path = {
    #         'cifar10': 'data_loaders/cifar10/train_data_loader.pickle',
    #         'fashion-mnist': 'data_loaders/fashion-mnist/train_data_loader.pickle',
    #         'cifar100': 'data_loaders/cifar100/train_data_loader.pickle',
    #     }
    #
    #     self.test_data_loader_pickle_path = {
    #         'cifar10': 'data_loaders/cifar10/test_data_loader.pickle',
    #         'fashion-mnist': 'data_loaders/fashion-mnist/test_data_loader.pickle',
    #         'cifar100': 'data_loaders/cifar100/test_data_loader.pickle',
    #     }
    #
    #     # TODO: Make part of different configuration
    #     self.loss_function = torch.nn.CrossEntropyLoss
    #
    #     self.default_model_folder_path = "default_models"
    #     self.data_path = "data"

    def get_dataloader_list(self):
        """
        @deprecated
        @return:
        @rtype:
        """
        return list(self.train_data_loader_pickle_path.keys())

    def set_train_data_loader_pickle_path(self, path, name='cifar10'):
        self.train_data_loader_pickle_path[name] = path

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path[self.dataset_name]

    def set_test_data_loader_pickle_path(self, path, name='cifar10'):
        self.test_data_loader_pickle_path[name] = path

    def get_test_data_loader_pickle_path(self):
        return self.test_data_loader_pickle_path[self.dataset_name]

    def should_save_model(self, epoch_idx):
        """
        Returns true/false models should be saved.

        :param epoch_idx: current training epoch index
        :type epoch_idx: int
        """
        return self.execution_config.general_net.save_model and (
                epoch_idx == 1 or epoch_idx % self.execution_config.general_net.save_epoch_interval == 0)
