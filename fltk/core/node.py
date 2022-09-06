from __future__ import annotations

import abc
import copy
import os
from typing import Callable, Any, Union
import torch
from torch.distributed import rpc
from fltk.datasets.federated import get_fed_dataset
from fltk.nets import get_net
from typing import TYPE_CHECKING

from fltk.util.log import getLogger

if TYPE_CHECKING:
    from fltk.util.config import FedLearningConfig

# Global dictionary to enable peer to peer communication between clients
global_vars = {}


class Node(abc.ABC):
    """
    Implementation of any participating node.
    It handles communication and the basic functions for Deep Learning.
    """
    id: str
    rank: int
    world_size: int
    real_time: bool = False
    distributed: bool = True
    cuda: bool = False
    finished_init: bool = False
    device = torch.device("cpu") # pylint: disable=no-member
    net: Any
    dataset: Any
    logger = getLogger(__name__)

    def __init__(self, identifier: str, rank: int, world_size: int, config: FedLearningConfig):
        self.config = config
        self.id = identifier # pylint: disable=invalid-name
        self.rank = rank
        self.world_size = world_size
        self.real_time = config.real_time
        global global_vars
        global_vars['self'] = self
        self._config(config)

    def _config(self, config: FedLearningConfig):
        self.logger.setLevel(config.log_level.value)
        self.config.rank = self.rank
        self.config.world_size = self.world_size
        self.cuda = config.cuda
        self.device = self.init_device()
        self.distributed = config.distributed
        self.set_net(self.load_default_model())

    def init_dataloader(self, world_size: int = None):
        """
        Function for nodes to initialize the datalaoders used for training.
        @param world_size: Worldsize of all training entities.
        @type world_size: int
        @return: None
        @rtype: None
        """
        config = copy.deepcopy(self.config)
        if world_size:
            config.world_size = world_size
        self.logger.info(f'world size = {config.world_size} with rank={config.rank}')
        self.dataset = get_fed_dataset(config.dataset_name)(config)
        self.finished_init = True
        self.logger.info('Done with init')

    def is_ready(self):
        """
        Helper function to establish whether a training Node has finished its initialization.
        @return: Boolean indicating whether the training Node has finished
        @rtype:
        """
        return self.finished_init

    @staticmethod
    def _receive(method: Callable, sender: str, *args, **kwargs):
        """
        Communication utility function.
        This is the entry points for all incoming RPC communication.
        The class object (self) will be loaded from the global space
        and the callable method is executed within the context of self.
        :param method: Function to execute provided by remote client.
        :param sender: Name of other Client that has request a function to be called.
        :param args: Arguments to pass to function to execute.
        :param kwargs: Keyword arguments to pass to function to execute.
        :return: Method executed on Client object.
        """
        global global_vars
        global_self = global_vars['self']
        if isinstance(method, str):
            method = getattr(global_self, method)
            return method(*args, **kwargs)
        return method(global_self, *args, **kwargs)

    def init_device(self):
        """
        Function to initialize learning accelerator, effectively performs nothing when the device to learn with is not
        an Nvidia accelerator.
        @return: Torch device corresponding to the device that was set to be initialized in the set configuration of
        the Client.
        @rtype: torch.device
        """
        if self.cuda and not torch.cuda.is_available():
            self.logger.warning('Unable to configure device for GPU because cuda.is_available() == False')
        if self.cuda and torch.cuda.is_available():
            self.logger.info("Configure device for GPU (Cuda)")
            return torch.device("cuda:0") # pylint: disable=no-member

        self.logger.info("Configure device for CPU")
        return torch.device("cpu") # pylint: disable=no-member

    def set_net(self, net):
        """
        Update the local parameters of self.net with net.
        This method also makes sure that the parameters are configured for the correct device (CPU or GPU/CUDA)
        :param net:
        """
        self.net = net
        self.net.to(self.device)

    def get_nn_parameters(self):
        """
        Return the DNN parameters.
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
            self.logger.warning(f"Could not find model: {model_file_path}")
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

    def message(self, other_node: str, method: Union[Callable, str], *args, **kwargs) -> torch.Future: # pylint: disable=no-member
        """
        All communication with other nodes should go through this method.
        The attribute real_time determines if the communication should use RPC or if it is a direct object call.
        :return: (resolved) torch.Future
        """
        if self.real_time:
            func = Node._receive
            args_list = [method, self.id] + list(args)
            return rpc.rpc_sync(other_node, func, args=args_list,  kwargs=kwargs)
        return method(other_node, *args, **kwargs)

    def message_async(self, other_node: str, method: Union[Callable, str], *args, **kwargs) -> torch.Future: # pylint: disable=no-member
        """
        This is the async version of 'message'.
        All communication with other nodes should go through this method.
        The attribute real_time determines if the communication should use RPC or if it is a direct object call.
        :return: torch.Future
        """
        if self.real_time:
            func = Node._receive
            args_list = [method, self.id] + list(args)
            return rpc.rpc_async(other_node, func, args=args_list,  kwargs=kwargs)
        # Wrap inside a future to keep the logic the same
        future = torch.futures.Future()
        future.set_result(method(other_node, *args, **kwargs))
        return future

    def ping(self, sender: str):
        """
        Utility function that can be used to test the connectivity between nodes.
        :param sender: str
        :return: str
        """
        self.logger.info(f'{self.id} got a ping from {sender}')
        return 'Pong'

    def __repr__(self):
        return str(self.id)
