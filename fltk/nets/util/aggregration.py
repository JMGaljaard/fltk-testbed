import collections
from typing import Union, List
import torch

def average_nn_parameters(parameters):
    """
    Takes unweighted average of a list of Tensor weights. Averages passed parameters.

    :param parameters: nn model named parameters
    :type parameters: list
    """
    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)
    return new_params


def drop_local_weights(
        new_params: collections.OrderedDict[str, torch.Tensor],
        local_params: collections.OrderedDict[str, torch.Tensor],
        exclude: Union[str, List[str]]):
    """Helper function to remove (partial) parameters from a shared set of global parameters. Intended for Federated
    Continual Learning experiments where local models of clients are expected to have a local component in addition
    to global only parameters.
    @param new_params: New (global) parameters send by the Federator.
    @type new_params: collections.OrderedDict[str, torch.Tensor]
    @param local_params: Old (local) parameters with local only parameters.
    @type local_params: collections.OrderedDict[str, torch.Tensor]
    @param exclude: Substring or set of named parameters not to be updated from the global model.
    @type exclude: str|List[str]
    @return: Updated ordered dictionary to be used as new parameters.
    @rtype: collections.OrderedDict[str, torch.Tensor]
    """
    if isinstance(exclude, str):
        partial_params = collections.OrderedDict()
        for name, layer_param in local_params:
            if exclude not in name:
                partial_params[partial_params] = layer_param
        new_params = partial_params
    if isinstance(exclude, list):
        for exclude in exclude:
            new_params.pop(exclude)
    return new_params
