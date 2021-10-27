import logging
import os
from argparse import Namespace
from multiprocessing.pool import ThreadPool

import torch.distributed as dist
from kubernetes import config

from fltk.client import Client
from fltk.extractor import download_datasets
from fltk.orchestrator import Orchestrator
from fltk.util.cluster.client import ClusterManager
from fltk.util.config.arguments import LearningParameters
from fltk.util.config.base_config import BareConfig
from fltk.util.task.generator.arrival_generator import ExperimentGenerator
from elasticsearch import Elasticsearch
from datetime import datetime


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


def launch_client(task_id: str, config: BareConfig = None, learning_params: LearningParameters = None,
                  namespace: Namespace = None):
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

    es = Elasticsearch(["3.137.193.160:9200"])
    logging.info(f'Starting with host={os.environ["MASTER_ADDR"]} and port={os.environ["MASTER_PORT"]}')
    rank, world_size, backend = 0, None, None
    distributed = should_distribute()
    if distributed:
        dist.init_process_group(namespace.backend)
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        backend = dist.get_backend()
    logging.info(f'Starting Creating client with {rank}')
    client = Client(rank, task_id, world_size, config, learning_params)
    client.prepare_learner(distributed)
    epoch_data = client.run_epochs()
    end_time = datetime.now()
    print(epoch_data)
    doc_end = {
        'task_id': task_id,
        'event': 'stop',
        'timestamp': datetime.now()
    }
    res = es.index(index=learning_params.elastic_index, id=task_id, document=doc_end)


def launch_orchestrator(args: Namespace = None, conf: BareConfig = None):
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


def launch_extractor(args: Namespace, config: BareConfig):
    download_datasets(args, config)
