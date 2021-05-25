import os
import sys
import torch.distributed.rpc as rpc
import logging

import yaml
import argparse

import torch.multiprocessing as mp
from fltk.federator import Federator
from fltk.strategy.attack import Attack
from fltk.util.base_config import BareConfig
from fltk.util.env.learner_environment import prepare_environment

logging.basicConfig(level=logging.DEBUG)


def run_ps(rpc_ids_triple, args, attack=Attack):
    print(f'Starting the federator...')
    fed = Federator(rpc_ids_triple, config=args, attack=Attack)
    fed.run(attack)

def run_single(rank, world_size, host = None, args = None, nic = None, attack=None):
    logging.info(f'Starting with rank={rank} and world size={world_size}')
    prepare_environment(host, nic)

    logging.info(f'Starting with host={os.environ["MASTER_ADDR"]} and port={os.environ["MASTER_PORT"]}')
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16, # TODO: Retrieve number of cores from system
        rpc_timeout=0,  # infinite timeout
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
        # trainer passively waiting for ps to kick off training iterations
    else:
        logging.info('Starting the ps')
        rpc.init_rpc(
            "ps",
            rank=rank,
            world_size=world_size,
            rpc_backend_options=options

        )
        run_ps([(f"client{r}", r, world_size) for r in range(1, world_size)], args, attack)
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