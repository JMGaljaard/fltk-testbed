def average_nn_parameters(parameters):
    """
    @deprecated Average passed parameters.
    @param parameters: nn model named parameters
    @type parameters: list
    """
    new_params = {}
    for name in parameters[0].keys():
        new_params[name] = sum([param[name].data for param in parameters]) / len(parameters)

    return new_params


def fed_average_nn_parameters(parameters, sizes):
    """
    @deprecated Federated Average passed parameters.
    @param parameters: nn model named parameters
    @type parameters: list
    @param sizes:
    @type sizes:
    """
    new_params = {}
    sum_size = 0
    for client in parameters:
        for name in parameters[client].keys():
            try:
                new_params[name].data += (parameters[client][name].data * sizes[client])
            except Exception as e:
                new_params[name] = (parameters[client][name].data * sizes[client])
        sum_size += sizes[client]

    for name in new_params:
        new_params[name].data /= sum_size

    return new_params
