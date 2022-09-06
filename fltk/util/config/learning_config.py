# pylint: disable=missing-function-docstring,invalid-name
from dataclasses import dataclass, field
from logging import getLogger
from pathlib import Path
# noinspection PyUnresolvedReferences
from typing import Type, List, Dict, Any, T, Union

import re

import torch
import torch.nn
import torch.optim
import yaml
from dataclasses_json import config, dataclass_json
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss

from fltk.util.config.definitions import (DataSampler, Loss, Aggregations, Dataset, LogLevel, Nets,
                                          Optimizations)
from fltk.util.config.definitions.loss import get_loss_function


def _eval_decoder(obj: Union[str, T]) -> Union[Any, T]:
    """
    Decoder function to help decoding string objects to objects using the Python interpeter.
    If a non-string object is passed it will return the argument

    """
    if isinstance(obj, str):
        return eval(obj)
    return obj


def get_safe_loader() -> Type[yaml.SafeLoader]:
    """
    Function to get a yaml SafeLoader that is capable of properly parsing yaml compatible floats.

    The default yaml loader would parse a value such as `1e-10` as a string, rather than a float.

    @return: SafeLoader capable of parsing scientificly notated yaml values.
    @rtype: yaml.SafeLoader
    """
    # Current version of yaml does not parse numbers like 1e-10 correctly, resulting in a str type.
    # Credits to https://stackoverflow.com/a/30462009/14661801
    safe_loader = yaml.SafeLoader
    safe_loader.add_implicit_resolver(
            u'tag:yaml.org,2002:float',
            re.compile(u'''^(?:
             [-+]?(?:[0-9][0-9_]*)\\.[0-9_]*(?:[eE][-+]?[0-9]+)?
            |[-+]?(?:[0-9][0-9_]*)(?:[eE][-+]?[0-9]+)
            |\\.[0-9_]+(?:[eE][-+][0-9]+)?
            |[-+]?[0-9][0-9_]*(?::[0-5]?[0-9])+\\.[0-9_]*
            |[-+]?\\.(?:inf|Inf|INF)
            |\\.(?:nan|NaN|NAN))$''', re.X),
            list(u'-+0123456789.'))
    return safe_loader


# fixme: With python 3.10, this can be done with the dataclass kw_only kwarg.
@dataclass_json
@dataclass
class LearningConfig:
    replication: int = field(metadata=dict(required=False, missing=-1))
    batch_size: int = field(metadata=dict(required=False, missing=128))
    test_batch_size: int = field(metadata=dict(required=False, missing=128))
    cuda: bool = field(metadata=dict(required=False, missing=False))
    scheduler_step_size: int = field(metadata=dict(required=False, missing=50))
    scheduler_gamma: float = field(metadata=dict(required=False, missing=0.5))
    min_lr: float = field(metadata=dict(required=False, missing=1e-10))
    optimizer: Optimizations = field(metadata=dict(required=False, missing=Optimizations.sgd))


@dataclass_json
@dataclass
class FedLearningConfig(LearningConfig):
    loss_function: Loss = Loss.cross_entropy_loss
    # Number of communication epochs.
    rounds: int = 2
    # Number of epochs to perform per ROUND
    epochs: int = 1
    lr: float = 0.01
    momentum: float = 0.1
    shuffle: bool = False
    log_interval: int = 10
    rng_seed = 0

    # Enum
    optimizer_args = {
        'lr': lr,
        'momentum': momentum
    }
    # Enum
    log_level: LogLevel = LogLevel.DEBUG

    num_clients: int = 10
    clients_per_round: int = 2
    distributed: bool = True
    single_machine: bool = False
    # Enum
    aggregation: Aggregations = Aggregations.fedavg
    # Enum
    dataset_name: Dataset = Dataset.mnist
    # Enum
    net_name: Nets = Nets.mnist_cnn
    default_model_folder_path: str = "default_models"
    data_path: str = "data"
    # Enum
    data_sampler: DataSampler = DataSampler.uniform
    data_sampler_args: List[float] = field(default_factory=list)

    # Set by Node upon argument
    rank: int = 0
    world_size: int = 0

    replication_id: int = None
    experiment_prefix: str = ''

    real_time: bool = False

    # Save data in append mode. Thereby flushing on every append to file.
    # This could be useful when a system is likely to crash midway an experiment
    save_data_append: bool = False
    output_path: Path = field(metadata=config(encoder=str, decoder=Path), default=Path('logging'))

    def update_rng_seed(self):
        torch.manual_seed(self.rng_seed)

    def get_default_model_folder_path(self):
        return self.default_model_folder_path

    def get_distributed(self):
        return self.distributed

    def get_sampler(self):
        return self.data_sampler

    def get_world_size(self):
        return self.world_size

    def get_rank(self):
        return self.rank

    def get_sampler_args(self):
        return tuple(self.data_sampler_args)

    def get_data_path(self):
        return self.data_path

    def get_loss_function(self) -> Type[_Loss]:
        return get_loss_function(self.loss_function)

    @staticmethod
    def from_yaml(path: Path):
        """
        Parse yaml file to dataclass. Re-implemented to rely on dataclasses_json to load data with tested library.

        Alternatively, running the followign code would result in loading a JSON formatted configuration file, in case
        you prefer to create json based configuration files.

        >>> with open("configs/example.json") as f:
        >>>     FedLearningConfig.from_json(f.read())

        @param path: Path pointing to configuration yaml file.
        @type path: Path
        @return: Configuration dataclass representation of the configuration file.
        @rtype: FedLearningConfig
        """
        getLogger(__name__).debug(f'Loading yaml from {path.absolute()}')
        safe_loader = get_safe_loader()
        with open(path) as file:
            content = yaml.load(file, Loader=safe_loader)
            conf = FedLearningConfig.from_dict(content)
        return conf


@dataclass_json
@dataclass
class DistLearningConfig(LearningConfig):  # pylint: disable=too-many-instance-attributes
    """
    Class encapsulating LearningParameters, for now used under DistributedLearning.
    """
    optimizer_args: Dict[str, Any]
    model: Nets
    dataset: Dataset
    max_epoch: int
    learning_rate: float
    learning_rate_decay: float
    seed: int
    loss: Loss = Loss.cross_entropy_loss

    @staticmethod
    def from_yaml(path: Path) -> "DistLearningConfig":
        """
        Parse yaml file to dataclass. Re-implemented to rely on dataclasses_json to load data with tested library.

        Alternatively, running the followign code would result in loading a JSON formatted configuration file, in case
        you prefer to create json based configuration files.

        >>> with open("configs/example.json") as f:
        >>>     DistLearningConfig.from_json(f.read())

        @param path: Path pointing to configuration yaml file.
        @type path: Path
        @return: Configuration dataclass representation of the configuration file.
        @rtype: FedLearningConfig
        """
        getLogger(__name__).debug(f'Loading yaml from {path.absolute()}')
        safe_loader = get_safe_loader()
        with open(path) as file:
            content = yaml.load(file, Loader=safe_loader)
            conf = DistLearningConfig.from_dict(content)
        return conf


    def get_loss_function(self) -> Type[_Loss]:
        """
        Helper function to get loss_function based on definition _or_ string.
        """
        return get_loss_function(self.loss)
