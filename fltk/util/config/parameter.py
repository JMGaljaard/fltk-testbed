import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List

from dataclasses_json import config, dataclass_json


@dataclass_json
@dataclass(frozen=True)
class HyperParameters:
    """
    Learning HyperParameters.

    batch_size: Number of images that are used during each forward/backward phase.
    max_epoch: Number of times epochs are executed.
    learning_rate: Learning rate parameter, limiting the step size in the gradient update.
    learning_rate_decay: How fast the learning rate 'shrinks'.
    """
    batch_size: int = field(metadata=config(field_name="batchSize"))
    max_epoch: int = field(metadata=config(field_name="maxEpoch"))
    learning_rate: str = field(metadata=config(field_name="learningRate"))
    learning_rate_decay: str = field(metadata=config(field_name="learningrateDecay"))


@dataclass_json
@dataclass(frozen=True)
class Priority:
    """
    Job class priority, indicating the presedence of one arrival over another.
    """
    priority: int = field(metadata=config(field_name="priority"))
    probability: float = field(metadata=config(field_name="probability"))


@dataclass_json
@dataclass(frozen=True)
class SystemParameters:
    """
    System parameters to spawn pods with.
    data_parallelism: Number of pods (distributed) that will work together on training the network.
    executor_cores: Number of cores assigned to each executor.
    executor_memory: Amount of RAM allocated to each executor.
    action: Indicating whether it regards 'inference' or 'train'ing time.
    """
    data_parallelism: int = field(metadata=config(field_name="dataParallelism"))
    executor_cores: int = field(metadata=config(field_name="executorCores"))
    executor_memory: str = field(metadata=config(field_name="executorMemory"))
    action: str


@dataclass_json
@dataclass(frozen=True)
class JobClassParameter:
    """
    Dataclass describing the job specific parameters (system and hyper).
    """
    system_parameters: SystemParameters = field(metadata=config(field_name="systemParameters"))
    hyper_parameters: HyperParameters = field(metadata=config(field_name="hyperParameters"))
    class_probability: float = field(metadata=config(field_name="classProbability"))


@dataclass_json
@dataclass(frozen=True)
class JobDescription:
    """
    Dataclass describing the characteristics of a Job type, as well as the arrival statistics.
    Currently, the arrival statistics is the lambda value used in a Poisson arrival process.
    """
    job_class_parameters: List[JobClassParameter] = field(metadata=config(field_name="jobClassParameters"))
    arrival_statistic: float = field(metadata=config(field_name="lambda"))


class ExperimentParser(object):

    def __init__(self, config_path: Path):
        self.__config_path = config_path

    def parse(self):
        """
        Parse function to load JSON config into JobDescription objects. Any changes to the JSON file format
        should be reflected by the classes used. For more information refer to the dataclasses JSON
        documentation https://pypi.org/project/dataclasses-json/.
        """
        with open(self.__config_path, 'r') as config_file:
            config_dict = json.load(config_file)
            job_list = [JobDescription.from_dict(job_description) for job_description in config_dict]
        return job_list
