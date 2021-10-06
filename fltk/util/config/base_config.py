import os
from dataclasses import dataclass, field
from pathlib import Path

from dataclasses_json import config, dataclass_json

from fltk.nets.util.reproducability import init_reproducibility


@dataclass_json
@dataclass
class GeneralNetConfig:
    save_model: bool = False
    save_temp_model: bool = False
    save_epoch_interval: int = 1
    save_model_path: str = 'models'
    epoch_save_start_suffix: str = 'cloud_experiment'
    epoch_save_end_suffix: str = 'cloud_experiment'
    scheduler_step_size = 50
    scheduler_gamma = 0.5
    min_lr = 1e-10


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
        @type working_dir: Path
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

    duration: int
    experiment_prefix: str = "experiment"
    cuda: bool = False
    default_model_folder_path = "default_models"
    epoch_save_end_suffix = "epoch_end"
    save_model_path = "models"
    data_path = "data"
    log_path = "logging"


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
    namespace: str = 'test'
    image: str = 'fltk:latest'

    def load_incluster_namespace(self):
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
            current_namespace = f.read()
            self.namespace = current_namespace

    def load_incluster_image(self):
        """
        Function to load the in-cluster image. The fltk-values.yaml file in charts is expected to have (at least) the
        following contents. The default Helm chart contains the necessary options to set this correctly.

        provider:
            domain: gcr.io
            projectName: <your-project-name>
            imageName: fltk:latest

        @return: None
        @rtype: None
        """
        self.image = os.environ.get('IMAGE_NAME')


@dataclass_json
@dataclass
class BareConfig(object):
    execution_config: ExecutionConfig
    cluster_config: ClusterConfig = field(metadata=config(field_name="cluster"))
    config_path: Path = None

    def set_seed(self) -> None:
        """
        Set seeds for better reproducibility, and prevent testing random initialization of the model,
        i.e. 'lucky draws' in network initialization.
        @return: None
        @rtype: None
        """
        init_reproducibility(
                torch_seed=self.execution_config.reproducibility.torch_seed,
                cuda=self.execution_config.cuda,
                numpy_seed=self.execution_config.reproducibility.arrival_seed
        )

    def get_duration(self) -> int:
        """
        Function to get execution duration of an experiment.
        @return: Integer representation of seconds for which the experiments must be run.
        @rtype: int
        """
        return self.execution_config.duration

    def get_log_dir(self):
        """
        Function to get the logging directory from the configuration.
        @return: path object to the logging directory.
        @rtype: Path
        """
        return self.execution_config.log_path

    def get_log_path(self, experiment_id: str, client_id: int, network_name: str) -> Path:
        """
        Function to get the logging path that corresponds to a specific experiment, client and network that has been
        deployed as learning task.
        @param experiment_id: Unique experiment ID (should be provided by the Orchestrator).
        @type experiment_id: str
        @param client_id: Rank of the client.
        @type client_id: int
        @param network_name: Name of the network that is to be trained.
        @type network_name: str
        @return: Path representation of the directory/path should be logged by the training process.
        @rtype: Path
        """
        base_log = Path(self.execution_config.tensorboard.record_dir)
        experiment_dir = Path(f"{self.execution_config.experiment_prefix}_{client_id}_{network_name}_{experiment_id}")
        return base_log.joinpath(experiment_dir)

    def get_scheduler_step_size(self) -> int:
        """
        Function to get the step_size of the Learning Rate decay scheduler/
        @return: Learning rate scheduler step-size.
        @rtype: int
        """
        return self.execution_config.general_net.scheduler_step_size

    def get_scheduler_gamma(self) -> float:
        """
        Function to get multiplication factor for LR update from config.
        @return: Multiplication factor for LR update
        @rtype: float
        """
        return self.execution_config.general_net.scheduler_gamma

    def get_min_lr(self) -> float:
        """
        Function to get the minimum learning rate from config.
        @return: Minimum learning rate of training process.
        @rtype: float
        """
        return self.execution_config.general_net.min_lr

    def get_data_path(self) -> Path:
        """
        Function to get the data path from config.
        @return: Path representation to where data can be written.
        @rtype: Path
        """
        return Path(self.execution_config.data_path)

    def get_default_model_folder_path(self) -> Path:
        """
        @deprecated Function to get the default model folder path from Config, needed for non-default training in the
        FLTK framework.
        @return: Path representation of model path.
        @rtype: Path
        """
        return Path(self.execution_config.default_model_folder_path)

    def cuda_enabled(self) -> bool:
        """
        Function to check CUDA availability independent of BareConfig structure.
        @return: True when CUDA should be used, False otherwise.
        @rtype: bool
        """
        return self.execution_config.cuda

    def should_save_model(self, epoch_idx) -> bool:
        """
        @deprecated Returns true/false models should be saved.

        @param epoch_idx: current training epoch index
        @type epoch_idx: int
        @return: Boolean indication of whether the model should be saved
        @rtype: bool
        """
        return self.execution_config.general_net.save_model and (
                epoch_idx == 1 or epoch_idx % self.execution_config.general_net.save_epoch_interval == 0)

    def get_epoch_save_end_suffix(self) -> str:
        """
        Function to gather the end suffix for saving after running an epoch.
        @return: Suffix for saving epoch data.
        @rtype: str
        """
        return self.execution_config.epoch_save_end_suffix

    def get_save_model_folder_path(self) -> Path:
        """
        Function to get save path for a model.
        @return: Path to where the model should be saved.
        @rtype: Path
        """
        return Path(self.execution_config.save_model_path)
