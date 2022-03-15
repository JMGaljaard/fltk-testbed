import copy
from dataclasses import dataclass
from enum import Enum, EnumMeta
from pathlib import Path
from typing import Type

import torch
import yaml

from fltk.util.definitions import Dataset, Nets, DataSampler, Optimizations, LogLevel, Aggregations


@dataclass
class Config:
    # optimizer: Optimizations
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
    # Enum
    optimizer: Optimizations = Optimizations.sgd
    optimizer_args = {
        'lr': lr,
        'momentum': momentum
    }
    loss_function = torch.nn.CrossEntropyLoss
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
    data_sampler_args = []

    rank: int = 0
    world_size: int = 0

    replication_id: int = None
    experiment_prefix: str = ''

    real_time : bool = False

    # Save data in append mode. Thereby flushing on every append to file.
    # This could be useful when a system is likely to crash midway an experiment
    save_data_append: bool = False
    output_path: Path = Path('output_test_2')

    def __init__(self, **kwargs) -> None:
        enum_fields = [x for x in self.__dataclass_fields__.items() if isinstance(x[1].type, Enum) or isinstance(x[1].type, EnumMeta)]
        for name, field in enum_fields:
            if name in kwargs and isinstance(kwargs[name], str):
                kwargs[name] = field.type(kwargs[name])
        for name, value in kwargs.items():
            self.__setattr__(name, value)
            if name == 'output_location':
                self.output_path = Path(value)

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

    def get_loss_function(self):
        return self.loss_function

    @classmethod
    def FromYamlFile(cls, path: Path):
        print(f'Loading yaml from {path.absolute()}')
        with open(path) as file:
            content = yaml.safe_load(file)
            for k, v in content.items():
                print(f'Inserting key "{k}" into config')
            return cls(**content)
