from collections import OrderedDict
from typing import Union

import torch

from fltk.util.base_config import BareConfig
import os

def flatten_params(model_description: Union[torch.nn.Module, OrderedDict]):
    """
    flattens all parameters into a single column vector. Returns the dictionary to recover them
    :param: parameters: a generator or list of all the parameters
    :return: a dictionary: {"params": [#params, 1],
    "indices": [(start index, end index) for each param] **Note end index in uninclusive**
    """
    if isinstance(model_description, torch.nn.Module):
        parameters = model_description.parameters()
    else:
        parameters = model_description.values()
    l = [torch.flatten(p) for p in parameters]
    flat = torch.cat(l).view(-1, 1)
    return flat


def recover_flattened(flat_params, model):
    """
    Gives a list of recovered parameters from their flattened form
    :param flat_params: [#params, 1]
    :param indices: a list detaling the start and end index of each param [(start, end) for param]
    :param model: the model that gives the params with correct shapes
    :return: the params, reshaped to the ones in the model, with the same order as those in the model
    """
    indices = []
    s = 0
    for p in model.parameters():
        size = p.shape[0]
        indices.append((s, s+size))
        s += size
    l = [flat_params[s:e] for (s, e) in indices]
    for i, p in enumerate(model.parameters()):
        l[i] = l[i].view(*p.shape)
    return l

def initialize_default_model(config: BareConfig, model_class) -> torch.nn.Module:
    """
    Load a default model dictionary into a torch model.
    @param model:
    @type model:
    @param config:
    @type config:
    @return:
    @rtype:
    """
    model = model_class()
    default_model_path = f"{config.get_default_model_folder_path()}/{model_class.__name__}.model"
    model.load_state_dict(torch.load(default_model_path))
    return model


def save_model(model: torch.nn.Module, directory: str, epoch, config: BareConfig, ratio):
    """
    Saves the model if necessary.
    """
    config.get_logger().debug(f"Saving model to flat file storage. Save #{model.__class__}")

    full_save_path = f"./{directory}/{ratio}_{config.get_net()}_{epoch}.pth"
    torch.save(model.state_dict(), full_save_path)