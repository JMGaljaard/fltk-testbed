# pylint: disable=unused-argument
import logging
import os
from typing import Callable, Optional, NewType

from argparse import Namespace
from multiprocessing.pool import ThreadPool
from pathlib import Path

import torch.distributed as dist
from kubernetes import config
from torch.distributed import rpc
from fltk.core.distributed import DistClient, download_datasets
from fltk.util.config.definitions.orchestrator import get_orchestrator, get_arrival_generator
from fltk.core import Client, Federator
from fltk.nets.util.reproducability import init_reproducibility, init_learning_reproducibility
from fltk.util.cluster.client import ClusterManager

from fltk.util.cluster.worker import should_distribute
from fltk.util.config import DistributedConfig, FedLearningConfig, retrieve_config_network_params, get_learning_param_config, \
    DistLearningConfig

from fltk.util.environment import retrieve_or_init_env, retrieve_env_config

# Define types for clarity in execution

Rank = NewType('Rank', int)
NIC = NewType('NIC', str)
Host = NewType('Host', int)
Prefix = NewType('Prefix', str)
launch_signature = Callable[[Path, Path, Optional[Rank], Optional[NIC], Optional[Host], Optional[Prefix],
                             Optional[Namespace], Optional[DistributedConfig]], None]


def exec_distributed_client(task_id: str, conf: DistributedConfig = None,
                            learning_params: DistLearningConfig = None,
                            namespace: Namespace = None):
    """
    Helper function to start the execution of the distributed client training loop.

    @param task_id: String representation (should be unique) corresponding to a client.
    @type task_id: str
    @param conf: Configuration for components, needed for spinning up components of the Orchestrator.
    @type conf: DistributedConfig
    @param learning_params: Parsed configuration of Hyper-Parameters for learning.
    @type: LearningParameters
    @return: None
    @rtype: None
    """
    logging.info(f'Starting with host={os.environ["MASTER_ADDR"]} and port={os.environ["MASTER_PORT"]}')
    rank, world_size = 0, None
    distributed = should_distribute()
    if distributed:
        logging.info(f'Initializing backend for training process: {namespace.backend}')
        dist.init_process_group(namespace.backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    logging.info(f'Starting Creating client with {rank}')

    client = DistClient(rank, task_id, world_size, conf, learning_params)
    client.prepare_learner(distributed)
    epoch_data = client.run_epochs()
    print(epoch_data)


def exec_orchestrator(args: Namespace = None, conf: DistributedConfig = None, replication: int = 1):
    """
    Default runner for the Orchestrator that is based on KubeFlow
    @param args: Commandline arguments passed to the execution. Might be removed in a future commit.
    @type args: Namespace
    @param conf: Configuration for execution of Orchestrators components, needed for spinning up components of the
    Orchestrator.
    @type conf: Optional[DistributedConfig]
    @param replication: Replication index of the experiment.
    @type replication: int
    @return: None
    @rtype: None
    """
    logging.info('Starting as Orchestrator')
    logging.info("Starting Orchestrator, initializing resources....")
    if args.local:
        logging.info("Loading local configuration file")
        config.load_kube_config()
    else:
        logging.info("Loading in cluster configuration file")
        config.load_incluster_config()
        logging.info("Pointing configuration to in cluster configuration.")
        conf.cluster_config.load_incluster_namespace()
        conf.cluster_config.load_incluster_image()

    # TODO: Move ClusterManager one level up, to allow for re-use
    cluster_manager = ClusterManager()
    arrival_generator = get_arrival_generator(conf, args.experiment)
    orchestrator = get_orchestrator(conf, cluster_manager, arrival_generator)

    pool = ThreadPool(3)

    logging.info("Starting cluster manager")
    pool.apply(cluster_manager.start)

    logging.info("Starting arrival generator")
    pool.apply_async(arrival_generator.start, args=[conf.get_duration()])
    logging.info("Starting orchestrator")
    pool.apply(orchestrator.run, kwds={"experiment_replication": replication})

    pool.close()

    logging.info(f"Stopped execution of Orchestrator replication: {replication}.")


def launch_extractor(arg_path: Path, conf_path: Path, rank: Rank, nic: Optional[NIC] = None,
                     host: Optional[Host] = None,
                     prefix: Optional[Prefix] = None, args: Optional[Namespace] = None,
                     conf: Optional[DistributedConfig] = None):
    """
    Extractor launch function, will only download all models and quit execution.

    @param arg_path: Base argument path for Local/Federated execution.
    @type arg_path: Path
    @param conf_path: Configuration file path for local/Federated execution.
    @type conf_path: Path
    @param rank: (Optional) rank of the worker/Node for the algorithm.
    @type rank: Rank
    @param nic: (Optional) Name of the Network Interface Card to use in Docker execution.
    @type nic: NIC
    @param host: (Optional) Name of the (Master) host name to use in Docker execution.
    @type host: Host
    @param prefix: (Optional) Experiment name prefix to use in Local execution.
    @type prefix: Prefix
    @param args: (Optional) Parsed argument from arg parse containing arguments passed to the program.
    @type args: Namespace
    @param conf: (Optional) Distributed configuration object for running in a Kubernetes cluster.
    @type conf: DistributedConfig
    @return: None
    @rtype: None
    """
    download_datasets(args, conf)


def launch_client(arg_path: Path, conf_path: Path, rank: Rank, nic: Optional[NIC] = None, host: Optional[Host] = None,
                  prefix: Optional[Prefix] = None, args: Optional[Namespace] = None,
                  conf: Optional[DistributedConfig] = None):
    """
    Client launch function.

    @param arg_path: Base argument path for Local/Federated execution.
    @type arg_path: Path
    @param conf_path: Configuration file path for local/Federated execution.
    @type conf_path: Path
    @param rank: (Optional) rank of the worker/Node for the algorithm.
    @type rank: Rank
    @param nic: (Optional) Name of the Network Interface Card to use in Docker execution.
    @type nic: NIC
    @param host: (Optional) Name of the (Master) host name to use in Docker execution.
    @type host: Host
    @param prefix: (Optional) Experiment name prefix to use in Local execution.
    @type prefix: Prefix
    @param args: (Optional) Parsed argument from arg parse containing arguments passed to the program.
    @type args: Namespace
    @param conf: (Optional) Distributed configuration object for running in a Kubernetes cluster.
    @type conf: DistributedConfig
    @return: None
    @rtype: None
    """
    logging.info("Starting in client mode")

    learning_params = get_learning_param_config(args)

    # Set the seed for PyTorch, numpy seed is mostly ignored. Set the `torch_seed` to a different value
    # for each repetition that you want to run an experiment with.
    init_learning_reproducibility(learning_params)
    task_id = args.task_id
    exec_distributed_client(task_id, conf=conf, learning_params=learning_params, namespace=args)
    logging.info("Stopping client...")


def launch_single(arg_path: Path, conf_path: Path, rank: Rank, nic: Optional[NIC] = None, host: Optional[Host] = None,
                  prefix: Optional[Prefix] = None, args: Optional[Namespace] = None,
                  conf: Optional[DistributedConfig] = None):
    """
    Single runner launch function.

    @param arg_path: Base argument path for Local/Federated execution.
    @type arg_path: Path
    @param conf_path: Configuration file path for local/Federated execution.
    @type conf_path: Path
    @param rank: (Optional) rank of the worker/Node for the algorithm.
    @type rank: Rank
    @param nic: (Optional) Name of the Network Interface Card to use in Docker execution.
    @type nic: NIC
    @param host: (Optional) Name of the (Master) host name to use in Docker execution.
    @type host: Host
    @param prefix: (Optional) Experiment name prefix to use in Local execution.
    @type prefix: Prefix
    @param args: (Optional) Parsed argument from arg parse containing arguments passed to the program.
    @type args: Namespace
    @param conf: (Optional) Distributed configuration object for running in a Kubernetes cluster.
    @type conf: DistributedConfig
    @return: None
    @rtype: None
    """
    # We can iterate over all the experiments in the directory and execute it, as long as the system remains the same!
    # System = machines and its configuration
    print(conf_path)
    s_conf = FedLearningConfig.from_yaml(conf_path)
    s_conf.world_size = conf.num_clients + 1
    s_conf.replication_id = prefix
    federator_node = Federator('federator', 0, conf.world_size, s_conf)
    federator_node.run()


def launch_remote(arg_path: Path, conf_path: Path, rank: Rank, nic: Optional[NIC] = None, host: Optional[Host] = None,
                  prefix: Optional[Prefix] = None, args: Optional[Namespace] = None,
                  conf: Optional[DistributedConfig] = None):
    """
    Function to launch a remote experiment. When launched in K8s, configuration will be set by KubeFlow, meaning that
    many parameters will be None. When launched in docker, parameters are provided by the callee through passed argument
    flags.

    @param arg_path: Base argument path for Local/Federated execution.
    @type arg_path: Path
    @param conf_path: Configuration file path for local/Federated execution.
    @type conf_path: Path
    @param rank: (Optional) rank of the worker/Node for the algorithm.
    @type rank: Rank
    @param nic: (Optional) Name of the Network Interface Card to use in Docker execution.
    @type nic: NIC
    @param host: (Optional) Name of the (Master) host name to use in Docker execution.
    @type host: Host
    @param prefix: (Optional) Experiment name prefix to use in Local execution.
    @type prefix: Prefix
    @param args: (Optional) Parsed argument from arg parse containing arguments passed to the program.
    @type args: Namespace
    @param conf: (Optional) Distributed configuration object for running in a Kubernetes cluster.
    @type conf: DistributedConfig
    @return: None
    @rtype: None
    """
    r_conf = FedLearningConfig.from_yaml(conf_path)
    r_conf.world_size = r_conf.num_clients + 1
    r_conf.replication_id = prefix
    if rank and not (nic and host):
        print("Getting parameters from configuration file")
        nic, host = retrieve_config_network_params(r_conf, nic, host)
        retrieve_or_init_env(nic, host)
    elif not rank:
        print("Retrieving environmental configurations!")
        rank, world_size, master_port = retrieve_env_config()
        print(f"Retrieved: rank {rank} w_s {world_size} m_p {master_port}")
        r_conf.world_size = world_size
    else:
        raise Exception('Missing rank, host, world-size, checking environment!')

    msg = f'Starting with host={host} and port={os.environ["MASTER_PORT"]} and interface={nic}'
    logging.log(logging.INFO, msg)
    options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            rpc_timeout=0,  # infinite timeout
            init_method='env://',
            _transports=["uv"]  # Use LibUV backend for async/IO interaction
    )
    if rank != 0:
        print(f'Starting worker-{rank} with world size={r_conf.world_size}')
        rpc.init_rpc(
                f"client{rank}",
                rank=rank,
                world_size=r_conf.world_size,
                rpc_backend_options=options,
        )
        client_node = Client(f'client{rank}', rank, r_conf.world_size, r_conf)
        client_node.remote_registration()
    else:
        print(f'Starting the PS (Fed) with world size={r_conf.world_size}')
        rpc.init_rpc(
                "federator",
                rank=rank,
                world_size=r_conf.world_size,
                rpc_backend_options=options

        )
        federator_node = Federator('federator', 0, r_conf.world_size, r_conf)
        federator_node.run()
        federator_node.stop_all_clients()
    print('Ending program')
    exit(0)


def launch_cluster(arg_path: Path, conf_path: Path, rank: Rank, nic: Optional[NIC] = None, host: Optional[Host] = None,
                   prefix: Optional[Prefix] = None, args: Optional[Namespace] = None,
                   conf: Optional[DistributedConfig] = None) -> None:
    """
    Function to launch Orchestrator for execution with provided configurations. Currently, this assumes that a single
    Orchestrator is started that manages all the training jobs withing the K8s cluster.

    @param arg_path: Base argument path for Local/Federated execution.
    @type arg_path: Path
    @param conf_path: Configuration file path for local/Federated execution.
    @type conf_path: Path
    @param rank: (Optional) rank of the worker/Node for the algorithm.
    @type rank: Rank
    @param nic: (Optional) Name of the Network Interface Card to use in Docker execution.
    @type nic: NIC
    @param host: (Optional) Name of the (Master) host name to use in Docker execution.
    @type host: Host
    @param prefix: (Optional) Experiment name prefix to use in Local execution.
    @type prefix: Prefix
    @param args: (Optional) Parsed argument from arg parse containing arguments passed to the program.
    @type args: Namespace
    @param conf: (Optional) Distributed configuration object for running in a Kubernetes cluster.
    @type conf: DistributedConfig
    @return: None
    @rtype: None
    """
    logging.info(f"Starting in cluster mode{' (locally)' if args.local else ''}.")
    logging.basicConfig(level=logging.DEBUG,
                        datefmt='%m-%d %H:%M')


    # Set the seed for arrivals, torch seed is mostly ignored. Set the `arrival_seed` to a different value
    # for each repetition that you want to run an experiment with.
    for replication, experiment_seed in enumerate(conf.execution_config.reproducibility.seeds):
        try:
            logging.info(f"Starting with experiment replication: {replication} with seed: {experiment_seed}")
            init_reproducibility(conf.execution_config)
            exec_orchestrator(args=args, conf=conf, replication=replication+1)
        except Exception as e:
            logging.info(f"Execution of replication {replication} with seed {experiment_seed} failed."
                         f"Reason: {e}")
