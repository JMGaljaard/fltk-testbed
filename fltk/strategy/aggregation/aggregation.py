

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


def average_nn_parameters(parameters):
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
    new_params = {}
    sum_size = 0
    for client in parameters:
        for name in parameters[client].keys():
            try:
                new_params[name].data += (parameters[client][name].data * sizes[client])
            except:
                new_params[name] = (parameters[client][name].data * sizes[client])
        sum_size += sizes[client]

    for name in new_params:
        new_params[name].data /= sum_size

    return new_params