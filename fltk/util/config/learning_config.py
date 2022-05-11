# pylint: disable=missing-function-docstring,invalid-name
import logging
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from logging import getLogger
from pathlib import Path
from typing import Type, List, Dict, Any, T

import re

import torch
import torch.nn
import torch.optim
import yaml
from dataclasses_json import config, dataclass_json
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss

from fltk.util.config.definitions import DataSampler, Nets, Dataset
from fltk.util.config.definitions.aggregate import Aggregations
from fltk.util.config.definitions.dataset import Dataset
from fltk.util.config.definitions.logging import LogLevel
from fltk.util.config.definitions.net import Nets
from fltk.util.config.definitions.optim import Optimizations


def get_safe_loader() -> yaml.SafeLoader:
    """
    Function to get a yaml SafeLoader that is capable of properly parsing yaml compatible floats.

    By default otherwise loading a value such as `1e-10` will result in in being parsed as a string.

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
    batch_size: int = field(metadata=dict(required=False, missing=128))
    test_batch_size: int = field(metadata=dict(required=False, missing=128))
    cuda: bool = field(metadata=dict(required=False, missing=False))
    scheduler_step_size: int = field(metadata=dict(required=False, missing=50))
    scheduler_gamma: float = field(metadata=dict(required=False, missing=0.5))


@dataclass_json
@dataclass
class FedLearningConfig(LearningConfig):
    rounds: int = 2
    epochs: int = 1
    lr: float = 0.01
    momentum: float = 0.1
    shuffle: bool = False
    log_interval: int = 10
    min_lr: float = 1e-10
    rng_seed = 0

    # Enum
    optimizer: Optimizations = Optimizations.sgd
    optimizer_args = {
        'lr': lr,
        'momentum': momentum
    }
    loss_function: Type[_Loss] = torch.nn.CrossEntropyLoss
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
        return self.loss_function

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


_available_loss = {
    "CROSSENTROPYLOSS": torch.nn.CrossEntropyLoss,
    "HUBERLOSS" : torch.nn.HuberLoss
}
_available_optimizer: Dict[str, Type[torch.optim.Optimizer]] = {
    "SGD": torch.optim.SGD,
    "ADAM": torch.optim.Adam,
    "ADAMW": torch.optim.AdamW
}


@dataclass_json
@dataclass
class DistLearningConfig(LearningConfig):  # pylint: disable=too-many-instance-attributes
    """
    Class encapsulating LearningParameters, for now used under DistributedLearning.
    """
    model: Nets
    dataset: Dataset
    batch_size: int
    test_batch_size: int
    max_epoch: int
    learning_rate: float
    learning_decay: float
    loss: str
    optimizer: Optimizations
    optimizer_args: Dict[str, Any]

    min_lr: float
    seed: int

    @staticmethod
    def __safe_get(lookup: Dict[str, T], keyword: str) -> T:
        """
        Static function to 'safe' get elements from a dictionary, to prevent issues with Capitalization in the code.
        @param lookup: Lookup dictionary to 'safe get' from.
        @type lookup: dict
        @param keyword: Keyword to 'get' from the Lookup dictionary.
        @type keyword: str
        @return: Lookup value from 'safe get' request.
        @rtype: T
        """
        safe_keyword = str.upper(keyword)
        if safe_keyword not in lookup:
            logging.fatal(f"Cannot find configuration parameter {keyword} in dictionary.")
        return lookup.get(safe_keyword)

    def get_loss(self) -> Type:
        """
        Function to obtain the loss function Type that was given via commandline to be used during the training
        execution.
        @return: Type corresponding to the loss function that was passed as argument.
        @rtype: Type
        """
        return self.__safe_get(_available_loss, self.loss)

