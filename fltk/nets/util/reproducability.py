import os

import numpy as np
import torch


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


def init_reproducibility(torch_seed: int = 42, cuda: bool = False, numpy_seed: int = 43, hash_seed: int = 44) -> None:
    """
    Function to pre-set all seeds for libraries used during training. Allows for re-producible network initialization,
    and non-deterministic number generation. Allows to prevent 'lucky' draws in network initialization.
    @param torch_seed: Integer seed to use for the PyTorch PRNG and CUDA PRNG.
    @type torch_seed: int
    @param cuda: Flag to indicate whether the CUDA backend needs to be
    @type cuda: bool
    @param numpy_seed: Integer seed to use for NumPy's PRNG.
    @type numpy_seed: int
    @param hash_seed: Integer seed to use for Pythons Hash function PRNG, will set the
    @type hash_seed: int

    @return: None
    @rtype: None
    """
    torch.manual_seed(torch_seed)
    if cuda:
        torch.cuda.manual_seed_all(torch_seed)
        cuda_reproducible_backend(True)
    np.random.seed(numpy_seed)
    os.environ['PYTHONHASHSEED'] = str(hash_seed)
