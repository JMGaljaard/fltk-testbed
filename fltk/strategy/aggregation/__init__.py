from typing import Callable

import torch

from fltk.util.definitions import Aggregations
from .FedAvg import fed_avg
from .aggregation import average_nn_parameters, average_nn_parameters_simple


def get_aggregation(name: Aggregations) -> Callable[[...], torch.Tensor]:
    """
    Helper function to get specific Aggregation class references.
    @param name: Aggregation class reference.
    @type name: Aggregations
    @return: Class reference corresponding to the requested Aggregation definition.
    @rtype: Type[torch.optim.Optimizer]
    """
    enum_type = Aggregations(name.value)
    aggregations_dict = {
            Aggregations.fedavg: fed_avg,
            Aggregations.sum: lambda x: x,
            Aggregations.avg: lambda x: x*2
        }
    return aggregations_dict[enum_type]
