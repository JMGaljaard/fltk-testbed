from dataclasses import dataclass

import torch

from fltk.util.definitions import Dataset, Nets, DataSampler, Optimizations, LogLevel


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
    optimizer = Optimizations.sgd
    optimizer_args = {
        'lr': lr,
        'momentum': momentum
    }
    loss_function = torch.nn.CrossEntropyLoss

    log_level: LogLevel = LogLevel.DEBUG

    num_clients: int = 10
    clients_per_round: int = 2
    distributed: bool = True
    single_machine: bool = False

    dataset_name: Dataset = Dataset.mnist
    net_name: Nets = Nets.mnist_cnn
    default_model_folder_path: str = "default_models"
    data_path: str = "data"
    data_sampler: DataSampler = DataSampler.uniform
    data_sampler_args = []

    rank: int = 0
    world_size: int = 0

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
