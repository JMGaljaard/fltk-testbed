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
