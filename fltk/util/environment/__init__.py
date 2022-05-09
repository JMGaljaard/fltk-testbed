import os
from typing import Optional

__RETRIEVE_PARAMS = ['RANK', 'WORLD_SIZE', 'MASTER_PORT']


def retrieve_or_init_env(nic: Optional[str] = None, host: Optional[str] = None) -> (str, str, str):
    """
    Function to initialize required environmental variables or initialize them when they have not been
    properly set. Note that this function should only be used to run in a Docker Compose setup. For initialization
    and retrieval in a K8s cluster, refer to `retrieve_env_config`.
    @param nic: Network Interface Card to use during execution, required for PyTorch Distributed/RPC.
    @type nic: Optional[str]
    @param host: Hostname to connect to, currently assumes master node.
    @type host: Optional[str]
    @return: None
    @rtype: None
    """
    if host:
        os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = '5000'
    if nic:
        os.environ['GLOO_SOCKET_IFNAME'] = nic
        os.environ['TP_SOCKET_IFNAME'] = nic


def retrieve_env_config() -> (int, int, int):
    """
    Helper function to get environmental configuration variables. These are provided in case the experiment is run
    in a K8s cluster.
    @return: Tuple containing the parsed rank, world_size, and port that may have been set in the environment.
    @rtype: Tuple[int, int, int]
    """
    rank, world_size, port = (int(os.environ.get('RANK')), int(os.environ.get('WORLD_SIZE')),
                              int(os.environ["MASTER_PORT"]))
    return rank, world_size, port
