import logging
import os
from argparse import Namespace
from multiprocessing.pool import ThreadPool
from pathlib import Path

import torch.distributed as dist
from kubernetes import config

from fltk.core.distributed.client import Client
from fltk.core.distributed.extractor import download_datasets
from fltk.core.distributed.orchestrator import Orchestrator
from fltk.util.cluster.client import ClusterManager
from fltk.util.config.arguments import LearningParameters
from fltk.util.config import DistributedConfig
from fltk.util.task.generator.arrival_generator import ExperimentGenerator


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


def launch_federated_client(task_id: str, config: ...):
    """
    Function to launch a federated client within Kubernetes. This differs from Docker-Compose, as this intended to
    be variable launch during runtime rather than pre-generated through compose. As a result, some information needs
    to be looked-up durign execution, rather than it being provided from the get-go. As such, make sure that pods
    have the permission to interact with the ConfigMaps and Pods within their working Namespace on your cluster.
    @param task_id:
    @type task_id:
    @param config:
    @type config:
    @return:
    @rtype:
    """


def launch_distributed_client(task_id: str, config: DistributedConfig = None, learning_params: LearningParameters = None,
                              namespace: Namespace = None):
    """
    @param task_id: String representation (should be unique) corresponding to a client.
    @type task_id: str
    @param config: Configuration for components, needed for spinning up components of the Orchestrator.
    @type config: BareConfig
    @param learning_params: Parsed configuration of Hyper-Parameters for learning.
    @type: LearningParameters
    @return: None
    @rtype: None
    """
    logging.info(f'Starting with host={os.environ["MASTER_ADDR"]} and port={os.environ["MASTER_PORT"]}')
    rank, world_size, backend = 0, None, None
    distributed = should_distribute()
    if distributed:
        logging.info(f'Initializing backend for training process: {namespace.backend}')
        dist.init_process_group(namespace.backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()

    logging.info(f'Starting Creating client with {rank}')

    client = Client(rank, task_id, world_size, config, learning_params)
    client.prepare_learner(distributed)
    epoch_data = client.run_epochs()
    print(epoch_data)


def launch_orchestrator(args: Namespace = None, conf: DistributedConfig = None):
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

    arrival_generator = ExperimentGenerator()
    cluster_manager = ClusterManager()

    orchestrator = Orchestrator(cluster_manager, arrival_generator, conf)

    pool = ThreadPool(3)
    logging.info("Starting cluster manager")
    pool.apply(cluster_manager.start)
    logging.info("Starting arrival generator")
    pool.apply_async(arrival_generator.start, args=[conf.get_duration()])
    logging.info("Starting orchestrator")
    pool.apply(orchestrator.run)
    pool.join()

    logging.info("Stopped execution of Orchestrator...")


def launch_extractor(base_path: Path, config_path: Path, args: Namespace=None, conf: DistributedConfig=None, **kwargs):
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
