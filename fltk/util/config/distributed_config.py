from __future__ import annotations
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List

from dataclasses_json import config, dataclass_json

from fltk.util.config.definitions import OrchestratorType

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fltk.util.config import DistLearnerConfig


@dataclass_json
@dataclass
class GeneralNetConfig:
    """ """
    save_model: bool = False
    save_temp_model: bool = False
    save_epoch_interval: int = 1
    save_model_path: str = 'models'
    epoch_save_start_suffix: str = 'cloud_experiment'
    epoch_save_end_suffix: str = 'cloud_experiment'


@dataclass_json
@dataclass(frozen=True)
class ReproducibilityConfig:
    """Dataclass object to hold experiment configuration settings related to reproducibility of experiments."""
    seeds: List[int]


@dataclass_json
@dataclass(frozen=True)
class TensorboardConfig:
    """ """
    active: bool
    record_dir: str

    def prepare_log_dir(self, working_dir: Path = None):
        """Function to create logging directory used by TensorBoard. When running in a cluster, this function should not be
        used, as the TensorBoard instance that is started simultaneously with the Orchestrator.

        Args:
          working_dir(Path): Current working directory, by default PWD is assumed at which the Python interpreter is
        started.
          working_dir: Path:  (Default value = None)

        Returns:
          None: None

        """
        dir_to_check = Path(self.record_dir)
        if working_dir:
            dir_to_check = working_dir.joinpath(dir_to_check)
        if not dir_to_check.exists() and dir_to_check.parent.is_dir():
            dir_to_check.mkdir()


@dataclass_json
@dataclass
class ExecutionConfig:
    """ """
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
    """ """
    orchestrator_type: OrchestratorType
    parallel_execution: bool = True


@dataclass_json
@dataclass
class ClientConfig:
    """ """
    prefix: str
    tensorboard_active: bool


@dataclass_json
@dataclass
class ClusterConfig:
    """ """
    orchestrator: OrchestratorConfig
    client: ClientConfig
    namespace: str = 'test'
    image: str = 'fltk:latest'

    def load_incluster_namespace(self):
        """Function to retrieve information from teh cluster itself provided by K8s.

        Args:

        Returns:
          None: None

        """
        with open("/var/run/secrets/kubernetes.io/serviceaccount/namespace") as f:
            current_namespace = f.read()
            self.namespace = current_namespace

    def load_incluster_image(self):
        """Function to load the in-cluster image. The fltk-values.yaml file in charts is expected to have (at least) the
        following contents. The default Helm chart contains the necessary options to set this correctly.
        
        provider:
            domain: gcr.io
            projectName: <your-project-name>
            imageName: fltk:latest

        Args:

        Returns:
          None: None

        """
        self.image = os.environ.get('IMAGE_NAME')


@dataclass_json
@dataclass
class DistributedConfig:
    """Configuration Dataclass for shared configurations between experiments. This regards your general setup, describing
    elements like the utilization of CUDA accelerators, format of logging file names, whether to save experiment data
    and the likes.

    Args:

    Returns:

    """
    execution_config: ExecutionConfig
    cluster_config: ClusterConfig = field(metadata=config(field_name="cluster"))
    config_path: Optional[Path] = None

    def get_duration(self) -> int:
        """Function to get execution duration of an experiment.

        Args:

        Returns:
          int: Integer representation of seconds for which the experiments must be run.

        """
        return self.execution_config.duration

    def get_log_dir(self):
        """Function to get the logging directory from the configuration.

        Args:

        Returns:
          Path: path object to the logging directory.

        """
        return self.execution_config.log_path

    def get_log_path(self, experiment_id: str, client_id: int, learn_params: DistLearnerConfig) -> Path:
        """Function to get the logging path that corresponds to a specific experiment, client and network that has been
        deployed as learning task.

        Args:
          experiment_id(str): Unique experiment ID (should be provided by the Orchestrator).
          client_id(int): Rank of the client.
          experiment_id: str: 
          client_id: int: 
          learn_params: DistLearnerConfig: 

        Returns:
          Path: Path representation of the directory/path should be logged by the training process.

        """
        base_log = Path(self.execution_config.tensorboard.record_dir)
        model, dataset, replication = learn_params.model, learn_params.dataset, learn_params.replication
        experiment_dir = Path(f"{replication}/{self.execution_config.experiment_prefix}_{experiment_id}/{client_id}/{model}_{dataset}")
        return base_log.joinpath(experiment_dir)

    def get_data_path(self) -> Path:
        """Function to get the data path from config.

        Args:

        Returns:
          Path: Path representation to where data can be written.

        """
        return Path(self.execution_config.data_path)

    def get_default_model_folder_path(self) -> Path:
        """@deprecated Function to get the default model folder path from FedLearningConfig, needed for non-default training in the
        FLTK framework.

        Args:

        Returns:
          Path: Path representation of model path.

        """
        return Path(self.execution_config.default_model_folder_path)

    def cuda_enabled(self) -> bool:
        """Function to check CUDA availability independent of BareConfig structure.

        Args:

        Returns:
          bool: True when CUDA should be used, False otherwise.

        """
        return self.execution_config.cuda

    def should_save_model(self, epoch_idx) -> bool:
        """@deprecated Returns true/false models should be saved.

        Args:
          epoch_idx(int): current training epoch index

        Returns:
          bool: Boolean indication of whether the model should be saved

        """
        return self.execution_config.general_net.save_model and (
                epoch_idx == 1 or epoch_idx % self.execution_config.general_net.save_epoch_interval == 0)

    def get_epoch_save_end_suffix(self) -> str:
        """Function to gather the end suffix for saving after running an epoch.

        Args:

        Returns:
          str: Suffix for saving epoch data.

        """
        return self.execution_config.epoch_save_end_suffix

    def get_save_model_folder_path(self) -> Path:
        """Function to get save path for a model.

        Args:

        Returns:
          Path: Path to where the model should be saved.

        """
        return Path(self.execution_config.save_model_path)
