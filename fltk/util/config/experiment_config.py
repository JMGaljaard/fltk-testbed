import logging
from dataclasses import dataclass, field
from pathlib import Path
# noinspection PyUnresolvedReferences
from typing import Optional, Union, Tuple, Dict, Any, MutableMapping, Type, OrderedDict, List, T

from dataclasses_json import dataclass_json, LetterCase, config

from fltk.util.config.definitions import Optimizations, Nets, Dataset, DataSampler, Aggregations, ExperimentType, Loss


def _none_factory() -> None:
    """Helper function to construct a default 'None' value.

    Args:

    Returns:
      None: Default None value.

    """
    return None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class OptimizerConfig:
    """Dataclass containing learning Optimizer parameters for learning tasks."""
    type: Optional[Optimizations] = None
    momentum: Optional[Union[float, Tuple[float]]] = None
    betas: Optional[Union[float, Tuple[float]]] = None
    lr: Optional[float] = field(metadata=config(field_name="learningRate"), default_factory=_none_factory)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class SchedulerConfig:
    """Dataclass containing learning rate scheduler configuration."""
    scheduler_step_size: Optional[int] = None
    scheduler_gamma: Optional[float] = None
    min_lr: Optional[float] = field(metadata=config(field_name="minimumLearningRate"), default_factory=_none_factory)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class HyperParameterConfiguration:
    """Dataclass containing training hyper parameters for learning tasks."""
    optimizer_config: Optional[OptimizerConfig] = field(metadata=config(field_name="optimizerConfig"),
                                                        default_factory=_none_factory)
    scheduler_config: Optional[SchedulerConfig] = field(metadata=config(field_name="schedulerConfig"),
                                                        default_factory=_none_factory)
    bs: Optional[int] = field(metadata=config(field_name="batchSize"),
                              default_factory=_none_factory)
    test_bs: Optional[int] = field(metadata=config(field_name="testBatchSize"), default_factory=_none_factory)
    lr_decay: Optional[float] = field(metadata=config(field_name="learningRateDecay"), default_factory=_none_factory)
    total_epochs: int = None

    def merge_default(self, other: Dict[str, Any]) -> "HyperParameterConfiguration":
        """Function to merge a HyperParameterConfiguration object with a default configuration.

        Args:
          other (Dict[str, Any]): Non-default configuration to overwrite default HyperParameter configuration object.

        Returns:
          HyperParameterConfiguration: Merged configuration objects, allowing defaults to be overwritten

        """
        return HyperParameterConfiguration.from_dict({**self.__dict__, **other})  # pylint: disable=no-member


def merge_optional(default_dict: Dict[str, Any], update_dict: Dict[str, Any], tpe: str) -> Dict[str, Any]:
    """Function to merge dictionaries to add set parameters from update dictionary into default dictionary.

    Args:
      default_dict (Dict[str, Any]): Default configuration dictionary to be overwritten by update_dict.
      update_dict ((Dict[str, Any])): Update configuration to be merged into default configurations.
      tpe (str): String representation of type of learner.

    Returns:
      Dict[str, Any]: Result of merged dictionaries.

    """
    default_copy = default_dict.copy()
    for k, v in default_copy.items():  # pylint: disable=invalid-name
        if k in update_dict:
            if all(isinstance(e, MutableMapping) for e in (v, update_dict[k])):
                update_dict[k] = merge_optional(v, update_dict[k], tpe)
        else:
            logging.warning(f"Gotten unknown alternative mapping {k}:{v} for {tpe}")

    # Base case
    update = list(filter(lambda item: item[1] is not None, update_dict.items()))
    for k, v in update:  # pylint: disable=invalid-name
        if not isinstance(v, dict):
            logging.info(f'Updating {k} from {default_copy[k]} to {v} for {tpe}')
        default_copy[k] = v
    return default_copy


def merge_optional_dataclass(default: T, update: T, data_type: Type[T], learner_type: str) -> T:
    """Function to merge two dataclasses of same type to update a default object with an update dataclass containing
    only a few set parameters.

    Args:
      default (T): Default dataclass.
      update (T): Update dataclass to merge into default.
      data_type (Type[T]): Type of the two dataclasses, to return Type T of dataclass.
      learner_type (str): String representation of learner type.

    Returns:
      T: Instance of the passed data_type.

    """
    if isinstance(update, default.__class__):
        merged = data_type.from_dict(merge_optional(default.to_dict(), update.to_dict(), learner_type))  # pylint: disable=no-member
        return merged
    raise Exception(f"Cannot merge dataclasses of different type: {default.__class__} and {update.__class__}")


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class HyperParameters:
    """Learning HyperParameters.
    
    bs: Number of images that are used during each forward/backward phase.
    max_epoch: Number of times epochs are executed.
    lr: Learning rate parameter, limiting the step size in the gradient update.
    lr_decay: How fast the learning rate 'shrinks'.
    """
    default: HyperParameterConfiguration
    configurations: OrderedDict[str, Optional[HyperParameterConfiguration]]

    def __post_init__(self):
        """
        Post init function that populates the hyperparameters of optionally configured elements of a HyperParam
        Configuration.
        """

        for learner_type in self.configurations.keys():
            conf: HyperParameterConfiguration
            if not (conf := self.configurations.get(learner_type, self.default)):
                self.configurations[learner_type] = self.default
            else:
                updated_conf = merge_optional_dataclass(self.default, conf, HyperParameterConfiguration, learner_type)

                self.configurations[learner_type] = updated_conf

    def get(self, tpe: str) -> "HyperParameterConfiguration":
        """Get function to retrieve HyperParameterConfiguration corresponding to a type of learner.

        Args:
            tpe (str): String representation of type of learner.

        Returns:
            HyperParameterConfiguration: Configuration corresponding to learner.
        """
        return self.configurations[tpe]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class Priority:
    """Job class priority, indicating the presedence of one arrival over another."""
    priority: int
    probability: float


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class SystemResources:
    """Dataclass representing SystemResources for Pods to be spawned in K8s."""
    cores: str
    memory: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class SystemParameters:
    """System parameters to spawn pods with.

    data_parallelism: Number of pods (distributed) that will work together on training the network.
    executor_cores: Number of cores assigned to each executor.
    executor_memory: Amount of RAM allocated to each executor.
    action: Indicating whether it regards 'inference' or 'train'ing time.
    """
    data_parallelism: int
    configurations: OrderedDict[str, SystemResources]

    def get(self, tpe: str):
        """Get function to retrieve SystemParameters for Type. Currently only supports Master and Client per Kubeflow.

        Args:
          tpe (str): Type of Pod/Learner to retrieve SystemResources for.

        Returns:
            SystemResources: Configuration object for system resources corresponding to the requested type.

        """
        if tpe in self.configurations:
            return self.configurations[tpe]
        # Fallback to default for alternative declaration.
        return self.configurations['default']


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class NetworkConfiguration:
    """Dataclass describing the network and dataset that is 'trained' for a task."""
    network: Nets
    dataset: Dataset
    loss_function: Optional[Loss]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class SamplerConfiguration:
    """Dataclass containing configuration for datasampler to be used by learners."""
    type: DataSampler
    q_value: str
    seed: int
    shuffle: bool


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class LearningParameters:
    """Dataclass containing configuration parameters for the learning process itself. This includes the Federated learning"""
    cuda: bool
    rounds: Optional[int] = None
    epochs_per_round: Optional[int] = None
    clients_per_round: Optional[int] = None
    aggregation: Optional[Aggregations] = None
    data_sampler: Optional[SamplerConfiguration] = None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class JobClassParameter:
    """Dataclass describing the job specific parameters (system and hyper)."""
    network_configuration: NetworkConfiguration
    system_parameters: SystemParameters
    hyper_parameters: HyperParameters
    class_probability: Optional[float] = None
    learning_parameters: Optional[LearningParameters] = None
    priorities: Optional[List[Priority]] = None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class JobDescription:
    """Dataclass describing the characteristics of a Job type, as well as the corresponding arrival statistic.
    Currently, the arrival statistics is the lambda value used in a Poisson arrival process.
    
    preemtible_jobs: indicates whether the jobs can be pre-emptively rescheduled by the scheduler. This is currently
    not implemented in FLTK, but could be added as a project (advanced).
    """
    experiment_type: ExperimentType = field(metadata=config(field_name='type'))
    job_class_parameters: List[JobClassParameter]
    preemtible_jobs: Optional[bool] = field(metadata=config(field_name='preemptJobs'), default_factory=_none_factory)
    arrival_statistic: Optional[float] = field(metadata=config(field_name='lambda'), default_factory=_none_factory)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class ExperimentConfig:
    """Dataclass object containing the configuration of an entire experiment, including
    configurations for repetitions. See also JobDescription to define types of jobs, and
    ExperimentConfiguration to set the configuration of experiments.
    """
    train_tasks: List[JobDescription]


class ExperimentParser:  # pylint: disable=too-few-public-methods
    """Simple parser to load experiment configuration into a programmatic objects."""

    def __init__(self, config_path: Path):
        self._config_path = config_path

    def parse(self) -> ExperimentConfig:
        """Parse function to load JSON conf into JobDescription objects. Any changes to the JSON file format
        should be reflected by the classes used. For more information refer to the dataclasses JSON
        documentation https://pypi.org/project/dataclasses-json/.
        """
        with open(self._config_path, 'r') as config_file:
            experiment = ExperimentConfig.from_json(config_file.read())
        return experiment
