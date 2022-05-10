# pylint: disable=missing-function-docstring,invalid-name
from dataclasses import dataclass, field
from enum import Enum, EnumMeta
from logging import getLogger
from pathlib import Path
from typing import Type, List

import re

import torch
import yaml
from dataclasses_json import config, dataclass_json
# noinspection PyProtectedMember
from torch.nn.modules.loss import _Loss

from fltk.util.config.definitions import DataSampler
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

@dataclass_json
@dataclass
class Config:
    batch_size: int = 1
    test_batch_size: int = 1000
    rounds: int = 2
    epochs: int = 1
    lr: float = 0.01
    momentum: float = 0.1
    cuda: bool = False
    shuffle: bool = False
    log_interval: int = 10
    scheduler_step_size: int = 50
    scheduler_gamma: float = 0.5
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
        >>>     Config.from_json(f.read())

        @param path: Path pointing to configuration yaml file.
        @type path: Path
        @return: Configuration dataclass representation of the configuration file.
        @rtype: Config
        """
        getLogger(__name__).debug(f'Loading yaml from {path.absolute()}')
        safe_loader = get_safe_loader()
        with open(path) as file:
            content = yaml.load(file, Loader=safe_loader)
            conf = Config.from_dict(content)
        return conf