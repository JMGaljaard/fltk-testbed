import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Optional, OrderedDict, Any, Union, Tuple, Type

import torch
from dataclasses_json import dataclass_json, LetterCase, config
from torch.nn.modules.loss import _Loss

from fltk.util.definitions import Aggregations, DataSampler, Optimizations, Dataset, Nets


def _none_factory():
    return None


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class OptimizerConfig:
    type: Optimizations
    momentum: Optional[Union[float, Tuple[float]]]
    lr: Optional[float] = field(metadata=config(field_name="learningRate"), default_factory=_none_factory)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class SchedulerConfig:
    scheduler_step_size: int
    scheduler_gamma: float
    min_lr: float = field(metadata=config(field_name="minimumLearningRate"))


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class HyperParameterConfiguration:
    optimizer_config: Optional[OptimizerConfig] = field(metadata=config(field_name="optimizerConfig"), default_factory=_none_factory)
    scheduler_config: Optional[SchedulerConfig] = field(metadata=config(field_name="schedulerConfig"), default_factory=_none_factory)
    bs: Optional[int] = field(metadata=config(field_name="batchSize"), default_factory=_none_factory)
    test_bs: Optional[int] = field(metadata=config(field_name="testBatchSize"), default_factory=_none_factory)
    lr_decay: Optional[float] = field(metadata=config(field_name="learningRateDecay"), default_factory=_none_factory)

    def merge_default(self, other: dict[str, Any]):
        """
        Function to merge a HyperParameterConfiguration object with a default configuration
        @param other:
        @type other:
        @return:
        @rtype:
        """
        return HyperParameterConfiguration.from_dict({**self.__dict__, **other})


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class HyperParameters:
    """
    Learning HyperParameters.

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
        @return:
        @rtype:
        """
        for config_type in self.configurations.keys():
            if not (conf := self.configurations.get(config_type, self.default)):
                conf = self.default

            merge = {k: v for k, v in conf.__dict__.items() if v is not None}
            self.configurations[config_type] = self.default.merge_default(merge)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class Priority:
    """
    Job class priority, indicating the presedence of one arrival over another.
    """
    priority: int
    probability: float


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class SystemResources:
    cores: str
    memory: str


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class SystemParameters:
    """
    System parameters to spawn pods with.
    data_parallelism: Number of pods (distributed) that will work together on training the network.
    executor_cores: Number of cores assigned to each executor.
    executor_memory: Amount of RAM allocated to each executor.
    action: Indicating whether it regards 'inference' or 'train'ing time.
    """
    data_parallelism: Optional[int]
    configurations: OrderedDict[str, SystemResources]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class NetworkConfiguration:
    """
    Dataclass describing the network and dataset that is 'trained' for a task.
    """
    network: Nets
    dataset: Dataset
    loss_function: Optional[Type[_Loss]]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class SamplerConfiguration:
    type: str
    q_value: str
    seed: int
    shuffle: bool


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class LearningParameters:
    total_epochs: int
    rounds: int
    epochs_per_round: int
    cuda: bool
    clients_per_round: int
    aggregation: Optional[Aggregations]
    data_sampler: Optional[SamplerConfiguration]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class ExperimentConfiguration:
    random_seed: List[int]
    worker_replication: OrderedDict[str, int]


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class JobClassParameter:
    """
    Dataclass describing the job specific parameters (system and hyper).
    """
    network_configuration: NetworkConfiguration
    system_parameters: SystemParameters
    hyper_parameters: HyperParameters
    learning_parameters: LearningParameters
    experiment_configuration: ExperimentConfiguration
    class_probability: Optional[float] = field(default_factory=_none_factory)
    priorities: Optional[List[Priority]] = field(default_factory=_none_factory)


@dataclass_json(letter_case=LetterCase.CAMEL)
@dataclass(frozen=True)
class JobDescription:
    """
    Dataclass describing the characteristics of a Job type, as well as the corresponding arrival statistic.
    Currently, the arrival statistics is the lambda value used in a Poisson arrival process.

    preemtible_jobs: indicates whether the jobs can be pre-emptively rescheduled by the scheduler. This is currently
    not implemented in FLTK, but could be added as a project (advanced)
    """
    job_class_parameters: JobClassParameter
    preemtible_jobs: Optional[float] = field(default_factory=_none_factory)
    arrival_statistic: Optional[float] = field(default_factory=_none_factory)
    priority: Optional[Priority] = None

    def get_experiment_configuration(self) -> ExperimentConfiguration:
        return self.job_class_parameters.experiment_configuration


@dataclass(order=True)
class TrainTask:
    """
    Training description used by the orchestrator to generate tasks. Contains 'transposed' information of the
    configuration file to make job generation easier and cleaner by using a 'flat' data class.

    Dataclass is ordered, to allow for ordering of arrived tasks in a PriorityQueue (for scheduling).
    """
    priority: int
    network_configuration: NetworkConfiguration = field(compare=False)
    experiment_configuration: ExperimentConfiguration = field(compare=False)
    system_parameters: SystemParameters = field(compare=False)
    hyper_parameters: HyperParameters = field(compare=False)
    learning_parameters: LearningParameters = field(compare=False)
    identifier: str = field(compare=False)

    def __init__(self, identity: str, job_parameters: JobClassParameter, priority: Priority = None,
                 experiment_config: ExperimentConfiguration = None):
        """
        Overridden init method for dataclass, to allow for 'exploding' a JobDescription object to a flattened object.
        @param job_parameters:
        @type job_parameters:
        @param job_description:
        @type job_description:
        @param priority:
        @type priority:
        """
        self.identifier = identity
        self.network_configuration = job_parameters.network_configuration
        self.system_parameters = job_parameters.system_parameters
        self.hyper_parameters = job_parameters.hyper_parameters
        if priority:
            self.priority = priority.priority
        self.experiment_configuration = experiment_config
        self.learning_parameters = job_parameters.learning_parameters


class ExperimentParser(object):

    def __init__(self, config_path: Path):
        self.__config_path = config_path

    def parse(self) -> List[JobDescription]:
        """
        Parse function to load JSON conf into JobDescription objects. Any changes to the JSON file format
        should be reflected by the classes used. For more information refer to the dataclasses JSON
        documentation https://pypi.org/project/dataclasses-json/.
        """
        with open(self.__config_path, 'r') as config_file:
            config_dict = json.load(config_file)
            job_list = [JobDescription.from_dict(job_description) for job_description in config_dict]
        return job_list
