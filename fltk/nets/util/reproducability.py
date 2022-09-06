import os
from typing import Optional

import numpy as np
import torch

from fltk.util.config.distributed_config import ExecutionConfig
from fltk.util.config import DistLearningConfig


def cuda_reproducible_backend(cuda: bool) -> None:
    """
    Function to set the CUDA backend to reproducible (i.e. deterministic) or to default configuration (per PyTorch
    1.9.1).
    @param cuda: Parameter to set or unset the reproducability of the PyTorch CUDA backend.
    @type cuda: bool
    @return: None
    @rtype: None
    """
    if cuda:
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False


def init_reproducibility(config: Optional[ExecutionConfig] = None, seed: Optional[int] = None) -> None:
    """
    Function to pre-set all seeds for libraries used during training. Allows for re-producible network initialization,
    and non-deterministic number generation. Allows to prevent 'lucky' draws in network initialization.
    @param config: Execution configuration for the experiments to be run on the remote cluster.
    @type config: ExecutionConfig
    @return: None
    @rtype: None
    """
    torch_seed, rand_seed = seed, seed
    if not seed:
        torch_seed, rand_seed = config.reproducibility.seeds[0], config.reproducibility.seeds[0]


    torch.manual_seed(torch_seed)
    if seed or (config and config.cuda):
        torch.cuda.manual_seed_all(torch_seed)
        cuda_reproducible_backend(True)
    np.random.seed(rand_seed)
    os.environ['PYTHONHASHSEED'] = str(rand_seed)



def init_learning_reproducibility(params: DistLearningConfig) -> None:
    """
    Function to pre-set all seeds for libraries used during training. Allows for re-producible network initialization,
    and non-deterministic number generation. Allows to prevent 'lucky' draws in network initialization.
    @param params: Execution parameters for the experiments to be run on the remote cluster.
    @type params: DistLearningConfig
    @return: None
    @rtype: None
    """
    random_seed = torch_seed = params.seed
    torch.manual_seed(torch_seed)
    if params.cuda:
        torch.cuda.manual_seed_all(torch_seed)
        cuda_reproducible_backend(True)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)
