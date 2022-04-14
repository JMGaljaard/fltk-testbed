# pylint: disable=unused-argument
import logging
import os
import sys
from argparse import Namespace
from multiprocessing.pool import ThreadPool
from pathlib import Path

import torch.distributed as dist
from kubernetes import config
from torch.distributed import rpc

from fltk.core.client import Client
from fltk.core.distributed import DistClient
from fltk.core.distributed import Orchestrator
from fltk.core.distributed.extractor import download_datasets
from fltk.core.federator import Federator
from fltk.util.cluster.client import ClusterManager
from fltk.util.config import DistributedConfig, Config
from fltk.util.config.arguments import LearningParameters, extract_learning_parameters
from fltk.util.task.generator.arrival_generator import DistributedExperimentGenerator, FederatedArrivalGenerator


def should_distribute() -> bool:
    """
    Function to check whether distributed execution is needed.

    Note: the WORLD_SIZE environmental variable needs to be set for this to work (larger than 1).
    PytorchJobs launched from KubeFlow automatically set this property.
    @return: Indicator for distributed execution.
    @rtype: bool
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    return dist.is_available() and world_size > 1


def launch_distributed_client(task_id: str, conf: DistributedConfig = None,
                              learning_params: LearningParameters = None,
                              namespace: Namespace = None):
    """
    @param task_id: String representation (should be unique) corresponding to a client.
    @type task_id: str
    @param conf: Configuration for components, needed for spinning up components of the Orchestrator.
    @type conf: BareConfig
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


def launch_orchestrator(args: Namespace = None, conf: DistributedConfig = None, simulate_arrivals=False):
    """
    Default runner for the Orchestrator that is based on KubeFlow
    @param args: Commandline arguments passed to the execution. Might be removed in a future commit.
    @type args: Namespace
    @param config: Configuration for execution of Orchestrators components, needed for spinning up components of the
    Orchestrator.
    @type config: BareConfig
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
    arrival_generator = DistributedExperimentGenerator() if simulate_arrivals else FederatedArrivalGenerator()
    cluster_manager = ClusterManager()

    orchestrator = Orchestrator(cluster_manager, arrival_generator, conf)

    pool = ThreadPool(3)

    logging.info("Starting cluster manager")
    pool.apply(cluster_manager.start)

    logging.info("Starting arrival generator")
    pool.apply_async(arrival_generator.start, args=[conf.get_duration()])
    logging.info("Starting orchestrator")
    pool.apply(orchestrator.run if simulate_arrivals else orchestrator.run_federated)

    pool.close()
    pool.join()

    logging.info("Stopped execution of Orchestrator...")


def launch_extractor(base_path: Path, config_path: Path, args: Namespace = None, conf: DistributedConfig = None,
                     **kwargs):
    """
    Extractor launch function, will only download all models and quit execution.
    @param args: Arguments passed from CLI.
    @type args: Namespace
    @param conf: Parsed configuration file passed from the CLI.
    @type conf: BareConfig
    @return: None
    @rtype: None
    """
    download_datasets(args, conf)


def launch_client(arg_path, conf_path, args: Namespace = None, configuration: DistributedConfig = None, **kwargs):
    """
    Client launch function.
    @param arg_path:
    @type arg_path:
    @param conf_path:
    @type conf_path:
    @param args:
    @type args:
    @param configuration:
    @type configuration:
    @param kwargs:
    @type kwargs:
    """
    logging.info("Starting in client mode")

    learning_params = extract_learning_parameters(args)
    # Set the seed for PyTorch, numpy seed is mostly ignored. Set the `torch_seed` to a different value
    # for each repetition that you want to run an experiment with.
    configuration.set_seed()
    task_id = args.task_id
    launch_distributed_client(task_id, conf=configuration, learning_params=learning_params, namespace=args)
    logging.info("Stopping client...")


def launch_single(base_path: Path, config_path: Path, prefix: str = None, **kwargs):
    """
    Single runner launch function.
    @param base_path:
    @type base_path:
    @param config_path:
    @type config_path:
    @param prefix:
    @type prefix:
    @param kwargs:
    @type kwargs:
    """
    # We can iterate over all the experiments in the directory and execute it, as long as the system remains the same!
    # System = machines and its configuration
    print(config_path)
    conf = Config.FromYamlFile(config_path)
    conf.world_size = conf.num_clients + 1
    conf.replication_id = prefix
    federator_node = Federator('federator', 0, conf.world_size, conf)
    federator_node.run()


def _retrieve_or_init_env(nic=None, host=None):
    """
    Function
    @param nic:
    @type nic:
    @param host:
    @type host:
    @return:
    @rtype:
    """
    if host:
        os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = '5000'
    if nic:
        os.environ['GLOO_SOCKET_IFNAME'] = nic
        os.environ['TP_SOCKET_IFNAME'] = nic


def _retrieve_env_config():
    """
    Helper function to get environmental configuration variables. These are provided in case the experiment is run
    in an K8s cluster.
    @return: Tuple containing the parsed rank, world_size, and port that may have been set in the environment.
    @rtype: Tuple[int, int, int]
    """
    rank, world_size, port = (int(os.environ.get('RANK')), int(os.environ.get('WORLD_SIZE')),
                              int(os.environ["MASTER_PORT"]))
    return rank, world_size, port


def _retrieve_network_params_from_config(conf: Config, nic=None, host=None):
    if hasattr(conf, 'system'):
        system_attr = getattr(conf, 'system')
        if 'federator' in system_attr:
            if 'hostname' in system_attr['federator'] and not host:
                host = system_attr['federator']['hostname']
            if 'nic' in system_attr['federator'] and not nic:
                nic = system_attr['federator']['nic']
    return nic, host


def launch_remote(base_path: Path, config_path: Path, rank: int, parser, nic=None, host=None, prefix: str = None,
                  **kwargs):
    """
    Function to launch a remote experiment. When launched in K8s, configuration will be set by KubeFlow, meaning that
    many parameters will be None. When launched in docker, parameters are provided by the callee through passed argument
    flags.
    @param base_path:
    @type base_path:
    @param config_path:
    @type config_path:
    @param rank:
    @type rank:
    @param parser:
    @type parser:
    @param nic:
    @type nic:
    @param host:
    @type host:
    @param prefix:
    @type prefix:
    @param kwargs:
    @type kwargs:
    @return:
    @rtype:
    """
    conf = Config.FromYamlFile(config_path)
    conf.world_size = conf.num_clients + 1
    conf.replication_id = prefix
    if rank and not (nic and host):
        print("Getting parameters from configuration file")
        nic, host = _retrieve_network_params_from_config(conf, nic, host)
        _retrieve_or_init_env(nic, host)
    elif not rank:
        print("Retrieving environmental configurations!")
        rank, world_size, master_port = _retrieve_env_config()
        print(f"Retrieved: rank {rank} w_s {world_size} m_p {master_port}")
        conf.world_size = world_size
    else:
        print('Missing rank, host, world-size, checking environment!')
        parser.print_help()
        sys.exit(1)

    msg = f'Starting with host={host} and port={os.environ["MASTER_PORT"]} and interface={nic}'
    logging.log(logging.INFO, msg)
    options = rpc.TensorPipeRpcBackendOptions(
            num_worker_threads=16,
            rpc_timeout=0,  # infinite timeout
            init_method='env://',
            _transports=["uv"]  # Use LibUV backend for async/IO interaction
    )
    if rank != 0:
        print(f'Starting worker-{rank} with world size={conf.world_size}')
        rpc.init_rpc(
                f"client{rank}",
                rank=rank,
                world_size=conf.world_size,
                rpc_backend_options=options,
        )
        client_node = Client(f'client{rank}', rank, conf.world_size, conf)
        client_node.remote_registration()
    else:
        print(f'Starting the PS (Fed) with world size={conf.world_size}')
        rpc.init_rpc(
                "federator",
                rank=rank,
                world_size=conf.world_size,
                rpc_backend_options=options

        )
        federator_node = Federator('federator', 0, conf.world_size, conf)
        federator_node.run()
        federator_node.stop_all_clients()
    print('Ending program')
    exit(0)


def launch_cluster(arg_path, conf_path, args: Namespace = None, config: DistributedConfig = None, **kwargs):
    """
    Function to launch Orchestrator for execution with provided configurations. Currently, this assumes that a single
    Orchestrator is started that manages all the training jobs withing the K8s cluster.
    """
    logging.info("Starting in cluster mode.")
    logging.basicConfig(level=logging.DEBUG,
                        datefmt='%m-%d %H:%M')
    # Set the seed for arrivals, torch seed is mostly ignored. Set the `arrival_seed` to a different value
    # for each repetition that you want to run an experiment with.
    config.set_seed()
    launch_orchestrator(args=args, conf=config)
