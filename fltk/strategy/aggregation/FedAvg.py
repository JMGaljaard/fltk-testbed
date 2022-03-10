

def fed_avg(parameters, sizes):
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
        # @TODO: Is .long() really required?
        new_params[name].data = new_params[name].data.long() / sum_size

    return new_params