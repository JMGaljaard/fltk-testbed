import logging
import os
from argparse import Namespace
from multiprocessing.pool import ThreadPool

import torch.distributed as dist

from fltk.client import Client
from fltk.orchestrator import run_orchestrator, Orchestrator
from fltk.util.cluster.client import ClusterManager
from fltk.util.config.base_config import BareConfig
from fltk.util.task.generator.arrival_generator import ExperimentGenerator

logging.basicConfig(level=logging.INFO)


def is_distributed():
    return dist.is_available() and dist.is_initialized()


def launch_client(task_id=None, rank=None, options=None, args: Namespace = None, config: BareConfig = None):
    """

    @param host:
    @type host:
    @param rank:
    @type rank:
    @param options:
    @type options:
    @param args:
    @type args:
    @param config: Configuration for components, needed for spinning up components of the Orchestrator.
    @type config: BareConfig
    @return:
    @rtype:
    """

    # prepare_environment(host, nic)

    logging.info(f'Starting with host={os.environ["MASTER_ADDR"]} and port={os.environ["MASTER_PORT"]}')
    distributed = is_distributed()
    if distributed:
        rank, world_size = dist.get_rank(), dist.get_world_size()
        logging.info(f'Starting with rank={rank} and world size={world_size}')

        client = Client(rank, task_id, config)
        backend = dist.get_backend()
        client.prepare_learner(distributed, backend)
        client.run_epochs()
    else:
        """
        Currently on only DistributedDataParallel is supported. If you want, you can implement a different 
        approach, although it is advised to tinker with the DistributedDataParallel execution, as this 
        greatly simplifies the forward and backward computations using AllReduce under the hood.
        
        For more information, refer to the Kubeflow PyTorch-Operator and PyTorch Distributed documentation.
        """
        print("Non DistributedDataParallel execution is not (yet) supported!")


def launch_orchestrator(host=None, rank=None, options=None, args: Namespace = None, config: BareConfig = None):
    """
    Default runner for the Orchestrator that is based on KubeFlow
    @param host:
    @type host:
    @param rank:
    @type rank:
    @param options:
    @type options:
    @param args:
    @type args:
    @param config: Configuration for components, needed for spinning up components of the Orchestrator.
    @type config: BareConfig
    @return:
    @rtype:
    """
    logging.info('Starting as Orchestrator')
    logging.info("Starting Orchestrator, initializing resources....")
    orchestrator = Orchestrator(config)
    cluster_manager = ClusterManager()
    arrival_generator = ExperimentGenerator()

    pool = ThreadPool(3)
    logging.info("Starting cluster manager")
    pool.apply_async(cluster_manager.start)
    logging.info("Starting arrival generator")
    pool.apply_async(arrival_generator.run)
    logging.info("Starting orchestrator")
    pool.apply(orchestrator.run)
    pool.join()

    logging.info("Stopped execution of Orchestrator...")
