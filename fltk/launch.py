import logging
import os
from argparse import Namespace
from multiprocessing.pool import ThreadPool

import torch.distributed as dist

from fltk.client import Client
from fltk.extractor import download_datasets
from fltk.orchestrator import Orchestrator
from fltk.util.cluster.client import ClusterManager
from fltk.util.config.arguments import LearningParameters
from fltk.util.config.base_config import BareConfig
from fltk.util.task.generator.arrival_generator import ExperimentGenerator

logging.basicConfig(level=logging.INFO)


def is_distributed() -> bool:
    """
    Function to check whether distributed execution is needed.

    Note: the WORLD_SIZE environmental variable needs to be set for this to work (larger than 1).
    PytorchJobs launched from KubeFlow automatically set this property.
    @return: Indicator for distributed execution.
    @rtype: bool
    """
    world_size = int(os.environ.get('WORLD_SIZE', 1))
    leader_port = int(os.environ.get('MASTER_PORT', 5000))
    leader_address = os.environ.get('MASTER_ADDR', 'localhost')
    logging.info(f"Training with WS: {world_size} connecting to: {leader_address}:{leader_port}")
    return dist.is_available() and world_size > 1


def launch_client(task_id: str, config: BareConfig = None, learning_params: LearningParameters = None):
    """
    @param task_id:
    @type task_id:
    @param config: Configuration for components, needed for spinning up components of the Orchestrator.
    @type config: BareConfig
    @param learning_params:
    @type: LearningParameters
    @return: None
    @rtype: None
    """
    logging.info(f'Starting with host={os.environ["MASTER_ADDR"]} and port={os.environ["MASTER_PORT"]}')
    rank, world_size, backend = 0, None, None
    distributed = is_distributed()
    if distributed:
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        backend = dist.get_backend()
    logging.info(f'Starting Creating client with {rank}')
    client = Client(rank, task_id, world_size, config, learning_params)
    client.prepare_learner(distributed, backend)
    epoch_data = client.run_epochs()
    print(epoch_data)


def launch_orchestrator(args: Namespace = None, config: BareConfig = None):
    """
    Default runner for the Orchestrator that is based on KubeFlow
    @param args: Commandline arguments passed to the execution. Might be removed in a future commit.
    @type args: Namespace
    @param config: Configuration for components, needed for spinning up components of the Orchestrator.
    @type config: BareConfig
    @return: None
    @rtype: None
    """
    logging.info('Starting as Orchestrator')
    logging.info("Starting Orchestrator, initializing resources....")

    arrival_generator = ExperimentGenerator()
    # cluster_manager = ClusterManager()
    #
    # orchestrator = Orchestrator(cluster_manager, arrival_generator, config)

    pool = ThreadPool(3)
    logging.info("Starting cluster manager")
    # pool.apply_async(cluster_manager.start)
    # logging.info("Starting arrival generator")
    pool.apply(arrival_generator.start, args=[config.get_duration()])
    # logging.info("Starting orchestrator")
    # pool.apply(orchestrator.run)
    pool.join()

    logging.info("Stopped execution of Orchestrator...")


def launch_extractor(args: Namespace, config: BareConfig):
    download_datasets(args, config)
