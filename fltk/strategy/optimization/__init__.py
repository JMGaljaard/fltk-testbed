from typing import Type

import torch

from fltk.util.definitions import Optimizations
from .fedprox import FedProx
from .FedNova import FedNova


def get_optimizer(name: Optimizations) -> Type[torch.optim.Optimizer]:
    """
    Helper function to get specific Optimization class references.
    @param name: Optimizer class reference.
    @type name: Optimizations
    @return: Class reference corresponding to the requested Optimizations definition.
    @rtype: Type[torch.optim.Optimizer]
    """
    optimizers = {
            Optimizations.sgd: torch.optim.SGD,
            Optimizations.fedprox: FedProx,
            Optimizations.fednova: FedNova
        }
    return optimizers[name]
