from typing import Type

import torch

from fltk.util.config.definitions.optim import Optimizations
from .fed_prox import FedProx
from .fed_nova import FedNova


def get_optimizer(name: Optimizations, federated: bool = True) -> Type[torch.optim.Optimizer]:
    """
    Helper function to get specific Optimization class references.
    @param name: Optimizer class reference.
    @type name: Optimizations
    @return: Class reference corresponding to the requested Optimizations definition. Requires instantiation with
    pre-defined args and kwargs, depending on the Type of Optimizer.
    @rtype: Type[torch.optim.Optimizer]
    """
    optimizers = {
            Optimizations.adam: torch.optim.Adam,
            Optimizations.adam_w: torch.optim.AdamW,
            Optimizations.sgd: torch.optim.SGD,
        }
    if federated:
        optimizers.update({
            Optimizations.fedprox: FedProx,
            Optimizations.fednova: FedNova
        })
    return optimizers[name]
