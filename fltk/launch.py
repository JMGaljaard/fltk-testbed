import logging
import os

import torch.distributed.rpc as rpc
import torch.multiprocessing as mp

from fltk.orchestrator import run_ps
from fltk.util.env.learner_environment import prepare_environment

logging.basicConfig(level=logging.INFO)


def await_assigned_orchestrator():
    # TODO: Implement await function for client

    """
    1. Setup everything correctly according to provided configuration files.
    2. Register to cleint
    3. Start working on task description provided by orchestrator
    4. Send heartbeats? (Alternatively use Kubernetes for this)
    5. Send completed data
    6. Terminate/complete pod execution.
    """
    pass


def run_single(rank, world_size, host=None, args=None, nic=None):
    logging.info(f'Starting with rank={rank} and world size={world_size}')
    prepare_environment(host, nic)

    logging.info(f'Starting with host={os.environ["MASTER_ADDR"]} and port={os.environ["MASTER_PORT"]}')
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=20,
        rpc_timeout=0,
        init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
    )
    if rank != 0:

        logging.info(f'Starting worker {rank}')
        rpc.init_rpc(
            f"client{rank}",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options,
        )
    else:
        logging.info('Starting the ps')
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options
        )
        run_ps([(f"client{r}", r, world_size) for r in range(1, world_size)], args)

    # block until all rpc finish
    rpc.shutdown()


# def run_single(rank, world_size, host = None, args = None, nic = None):

def run_spawn(config):
    world_size = config.world_size
    master_address = config.federator_host
    mp.spawn(
        run_single,
        args=(world_size, master_address, config),
        nprocs=world_size,
        join=True
    )
