from fltk.util.config.definitions.aggregate import Aggregations
from .fed_avg import fed_avg


def get_aggregation(name: Aggregations):
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
