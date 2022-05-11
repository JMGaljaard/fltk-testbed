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


def init_reproducibility(config: Optional[ExecutionConfig] = None, test_seed: int = 42) -> None:
    """
    Function to pre-set all seeds for libraries used during training. Allows for re-producible network initialization,
    and non-deterministic number generation. Allows to prevent 'lucky' draws in network initialization.
    @param config: Execution configuration for the experiments to be run on the remote cluster.
    @type config: ExecutionConfig
    @return: None
    @rtype: None
    """
    if config:
        random_seed = config.reproducibility.arrival_seed
        torch_seed = config.reproducibility.torch_seed
        cuda = config.cuda
    else:
        random_seed = test_seed
        torch_seed = test_seed
        cuda = torch.cuda.is_available()

    torch.manual_seed(torch_seed)
    if cuda:
        torch.cuda.manual_seed_all(torch_seed)
        cuda_reproducible_backend(True)
    np.random.seed(random_seed)
    os.environ['PYTHONHASHSEED'] = str(random_seed)



def init_learning_reproducibility(params: DistLearningConfig) -> None:
    """
    Function to pre-set all seeds for libraries used during training. Allows for re-producible network initialization,
    and non-deterministic number generation. Allows to prevent 'lucky' draws in network initialization.
    @param config: Execution configuration for the experiments to be run on the remote cluster.
    @type config: ExecutionConfig
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
