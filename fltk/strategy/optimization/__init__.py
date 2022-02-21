import torch
from .fedprox import FedProx
from .FedNova import FedNova
from ...util.definitions import Optimizations


def get_optimizer(name: Optimizations):
    optimizers = {
            Optimizations.sgd: torch.optim.SGD,
            Optimizations.fedprox: FedProx,
            Optimizations.fednova: FedNova
        }
    return optimizers[name]
