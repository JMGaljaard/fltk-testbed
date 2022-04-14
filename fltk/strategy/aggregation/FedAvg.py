# pylint: disable=invalid-name
from typing import Dict

import torch


def fed_avg(parameters: Dict[str, Dict[str, torch.Tensor]], sizes: Dict[str, int]) -> Dict[str, torch.Tensor]:
    """
    Function to perform FederatedAveraging with on a list of parameters.
    @param parameters: Dictionary of per-client provided Parameters.
    @type parameters:  Dict[str, Dict[str, torch.Tensor]]
    @param sizes: Dictionary of size descriptions for volume of data on client (to weight accordingly).
    @type sizes: Dict[str, int]
    @return: New parameters for next round.
    @rtype: Dict[str, torch.Tensor]
    """
    new_params = {}
    sum_size = 0
    # For each client
    for client in parameters:
        # For each module in the client
        for name in parameters[client].keys():
            try:
                new_params[name].data += (parameters[client][name].data * sizes[client])
            except:
                new_params[name] = (parameters[client][name].data * sizes[client])
        sum_size += sizes[client]

    for name in new_params:
        # @TODO: Is .long() really required?
        new_params[name].data = new_params[name].data.long() / sum_size

    return new_params
