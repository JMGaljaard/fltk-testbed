import copy
import os
from typing import Callable, Any

import torch

# from fltk.core.rpc_util import _remote_method_direct
from torch.distributed import rpc

from fltk.datasets.loader_util import get_dataset
from fltk.nets import get_net
from fltk.util.config import Config
from fltk.util.log import getLogger

global_vars = {}


def _remote_method_direct(method, other_node: str, *args, **kwargs):
    # Client example
    #  ret = rpc.rpc_async(self.client_to_offload_to, Client.offload_receive_endpoint, args=([model_weights, i, self.id, local_updates_left]))

    args = [method, other_node] + list(args)
    # return rpc.rpc_sync(other_node, _call_method, args=args, kwargs=kwargs)
    return rpc.rpc_sync(other_node, method, args=args, kwargs=kwargs)

class Node:
    id: int
    rank: int
    world_size: int
    counter = 0
    real_time = False
    distributed = True
    cuda = False
    finished_init: bool = False

    device = torch.device("cpu")
    net: Any
    dataset: Any
    logger = getLogger(__name__)


    # _address_book = {}

    def __init__(self, id: int, rank: int, world_size: int, config: Config):
        self.config = config
        self.id = id
        self.rank = rank
        self.world_size = world_size
        self.real_time = config.real_time
        global global_vars
        global_vars['self'] = self
        self._config(config)

    def _config(self, config: Config):
        self.logger.setLevel(config.log_level.value)
        self.config.rank = self.rank
        self.config.world_size = self.world_size
        self.cuda = config.cuda
        self.device = self.init_device()
        self.distributed = config.distributed
        self.set_net(self.load_default_model())

    def init_dataloader(self, world_size: int = None):
        config = copy.deepcopy(self.config)
        if world_size:
            config.world_size = world_size
        self.logger.info(f'world size = {config.world_size} with rank={config.rank}')
        self.dataset = get_dataset(config.dataset_name)(config)
        self.finished_init = True
        self.logger.info('Done with init')

    def is_ready(self):
        return self.finished_init

    # def _add_address(self, node_name: str, ref: Any):
    #     self._address_book[node_name] = ref

    @staticmethod
    def _receive(method: Callable, sender: str, *args, **kwargs):
        global global_vars
        # print('_receive')
        # print(global_vars)
        global_self = global_vars['self']
        # print(type(method))
        # print(type(global_self))
        if type(method) is str:
            # print(f'Retrieving method from string: "{method}"')
            method = getattr(global_self, method)
            return method(*args, **kwargs)
        else:
            # print(method)
            # print(global_self, *args, kwargs)
            return method(global_self, *args, **kwargs)

    # def _lookup_reference(self, node_name: str):

    def init_device(self):
        if self.cuda and not torch.cuda.is_available():
            self.logger.warning('Unable to configure device for GPU because cuda.is_available() == False')
        if self.cuda and torch.cuda.is_available():
            self.logger.info("Configure device for GPU (Cuda)")
            return torch.device("cuda:0")
        else:
            self.logger.info("Configure device for CPU")
            return torch.device("cpu")

    def set_net(self, net):
        self.net = net
        self.net.to(self.device)

    # def load_model_from_file(self):
    #     model_class = self.args.get_net()
    #     default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")
    #     return self.load_model_from_file(default_model_path)

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = get_net(self.config.net_name)
        default_model_path = os.path.join(self.config.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.

        :param model_file_path: string
        """
        model_class = get_net(self.config.net_name)
        model = model_class()

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.logger.warning("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.logger.warning("Could not find model: {}".format(model_file_path))
        return model


    def update_nn_parameters(self, new_params, is_offloaded_model = False):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        if is_offloaded_model:
            pass
            # self.offloaded_net.load_state_dict(copy.deepcopy(new_params), strict=True)
        else:
            self.net.load_state_dict(copy.deepcopy(new_params), strict=True)
        # self.logger.info(f'Weights of the model are updated')

    def message(self, other_node: str, method: Callable, *args, **kwargs) -> torch.Future:
        if self.real_time:
            func = Node._receive
            args_list = [method, self.id] + list(args)
            return rpc.rpc_sync(other_node, func, args=args_list,  kwargs=kwargs)
        return method(other_node, *args, **kwargs)

    def message_async(self, other_node: str, method: Callable, *args, **kwargs) -> torch.Future:
        if self.real_time:
            func = Node._receive
            args_list = [method, self.id] + list(args)
            return rpc.rpc_async(other_node, func, args=args_list,  kwargs=kwargs)
        # Wrap inside a future to keep the logic the same
        future = torch.futures.Future()
        future.set_result(method(other_node, *args, **kwargs))
        return future

    # def register_client(self, client_name, rank):
    #     print(f'self={self}')
    #     self.logger.info(f'[Default Implementation!] Got new client registration from client {client_name}')

    def ping(self, sender: str, be_weird=False):
        self.logger.info(f'Pong from {self.id}, got call from {sender} [{self.counter}]')
        # print(f'Pong from {self.id}, got call from {sender} [{self.counter}]')
        self.counter += 1
        if be_weird:
            return 'AAAAAAAAAAAAAAAAAAAAAAHHHH!!!!'
        else:
            return f'Pong {self.counter}'

    def __repr__(self):
        return str(self.id)
