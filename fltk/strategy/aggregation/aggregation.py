

def average_nn_parameters_simple(parameters):
    """
    Averages passed parameters.
    :param parameters: nn model named parameters
    :type parameters: list
    """
    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)

    return new_params


def average_nn_parameters(parameters, sizes):
    """
    @deprecated Federated Average passed parameters.
    :param parameters: nn model named parameters
    :type parameters: list
    :param sizes:
    :type sizes:
    """
    new_params = {}
    sum_size = 0
    for client in parameters:
        for key, _ in parameters[client].items():
            try:
                new_params[key].data += (parameters[client][key].data * sizes[client])
            except Exception: # pylint: disable=broad-except
                new_params[key] = (parameters[client][key].data * sizes[client])
        sum_size += sizes[client]

    for key, _ in new_params.items():
        new_params[key].data /= sum_size

    return new_params
