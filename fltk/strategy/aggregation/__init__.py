from fltk.util.definitions import Aggregations
from .FedAvg import fed_avg
from .aggregation import average_nn_parameters, average_nn_parameters_simple


def get_aggregation(name: Aggregations):
    enum_type = Aggregations(name.value)
    aggregations_dict = {
            Aggregations.fedavg: fed_avg,
            Aggregations.sum: lambda x: x,
            Aggregations.avg: lambda x: x*2
        }
    return aggregations_dict[enum_type]