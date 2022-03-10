# import os
# import random
# import sys
# import time
#
# import torch.distributed.rpc as rpc
# import logging
#
# import yaml
# import argparse
#
# import torch.multiprocessing as mp
# from fltk.federator import Federator
# from fltk.launch import run_single, run_spawn
# from fltk.util.base_config import BareConfig
#
# logging.basicConfig(level=logging.DEBUG)
#
# def add_default_arguments(parser):
#     parser.add_argument('--world_size', type=str, default=None,
#                         help='Number of entities in the world. This is the number of clients + 1')
#
# def main():
#     parser = argparse.ArgumentParser(description='Experiment launcher for the Federated Learning Testbed')
#
#     subparsers = parser.add_subparsers(dest="mode")
#
#     single_parser = subparsers.add_parser('single')
#     single_parser.add_argument('config', type=str)
#     single_parser.add_argument('--rank', type=int)
#     single_parser.add_argument('--nic', type=str, default=None)
#     single_parser.add_argument('--host', type=str, default=None)
#     add_default_arguments(single_parser)
#
#     spawn_parser = subparsers.add_parser('spawn')
#     spawn_parser.add_argument('config', type=str)
#     add_default_arguments(spawn_parser)
#
#     remote_parser = subparsers.add_parser('remote')
#     remote_parser.add_argument('--rank', type=int)
#     remote_parser.add_argument('--nic', type=str, default=None)
#     remote_parser.add_argument('--host', type=str, default=None)
#     add_default_arguments(remote_parser)
#     args = parser.parse_args()
#     if args.mode == 'remote':
#         if args.rank is None or args.host is None or args.world_size is None or args.nic is None:
#             print('Missing rank, host, world-size, or nic argument when in \'remote\' mode!')
#             parser.print_help()
#             exit(1)
#         world_size = int(args.world_size)
#         master_address = args.host
#         nic = args.nic
#         rank = int(args.rank)
#         if rank == 0:
#             print('Remote mode only supports ranks > 0!')
#             exit(1)
#         print(f'rank={args.rank}, world_size={world_size}, host={master_address}, args=None, nic={nic}')
#         run_single(rank=args.rank, world_size=world_size, host=master_address, args=None, nic=nic)
#     else:
#         with open(args.config) as file:
#             sleep_time = random.uniform(0, 5.0)
#             time.sleep(sleep_time)
#             cfg = BareConfig()
#             yaml_data = yaml.load(file, Loader=yaml.FullLoader)
#             cfg.merge_yaml(yaml_data)
#             if args.mode == 'single':
#                 if args.rank is None:
#                     print('Missing rank argument when in \'single\' mode!')
#                     parser.print_help()
#                     exit(1)
#                 world_size = args.world_size
#                 master_address = args.host
#                 nic = args.nic
#
#                 if not world_size:
#                     world_size = yaml_data['system']['clients']['amount'] + 1
#                 if not master_address:
#                     master_address = yaml_data['system']['federator']['hostname']
#                 if not nic:
#                     nic = yaml_data['system']['federator']['nic']
#                 print(f'rank={args.rank}, world_size={world_size}, host={master_address}, args=cfg, nic={nic}')
#                 run_single(rank=args.rank, world_size=world_size, host=master_address, args=cfg, nic=nic)
#             else:
#                 run_spawn(cfg)
#
# if __name__ == "__main__":
#     main()
import os
import sys
from pathlib import Path

from torch.distributed import rpc

from fltk.core.client import Client

print(sys.path)
# from fltk.core.federator import Federator as Fed
print(list(Path.cwd().iterdir()))
import argparse
from enum import Enum
from pathlib import Path

from fltk.core.federator import Federator
from fltk.util.config import Config
from fltk.util.definitions import Aggregations, Optimizations

def run_single(config_path: Path):

    # We can iterate over all the experiments in the directory and execute it, as long as the system remains the same!
    # System = machines and its configuration

    print(config_path)
    config = Config.FromYamlFile(config_path)
    config.world_size = config.num_clients + 1
    config.replication_id = 1
    federator_node = Federator('federator', 0,  config.world_size, config)
    federator_node.run()


def retrieve_env_params(nic=None, host=None):
    if host:
        os.environ['MASTER_ADDR'] = host
    os.environ['MASTER_PORT'] = '5000'
    if nic:
        os.environ['GLOO_SOCKET_IFNAME'] = nic
        os.environ['TP_SOCKET_IFNAME'] = nic

def retrieve_network_params_from_config(config: Config, nic=None, host=None):
    if hasattr(config, 'system'):
        system_attr = getattr(config, 'system')
        if 'federator' in system_attr:
            if 'hostname' in system_attr['federator'] and not host:
                host = system_attr['federator']['hostname']
            if 'nic' in system_attr['federator'] and not nic:
                nic = system_attr['federator']['nic']
    return nic, host

def run_remote(config_path: Path, rank: int, nic=None, host=None):
    print(config_path, rank)
    config = Config.FromYamlFile(config_path)
    config.world_size = config.num_clients + 1
    nic, host = retrieve_network_params_from_config(config, nic, host)
    if not nic or not host:
        print('Missing rank, host, world-size, or nic argument when in \'remote\' mode!')
        parser.print_help()
        exit(1)
    retrieve_env_params(nic, host)
    print(f'Starting with host={os.environ["MASTER_ADDR"]} and port={os.environ["MASTER_PORT"]} and interface={nic}')
    options = rpc.TensorPipeRpcBackendOptions(
        num_worker_threads=16,
        rpc_timeout=0,  # infinite timeout
        init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
    )
    if rank != 0:
        print(f'Starting worker {rank}')
        rpc.init_rpc(
            f"client{rank}",
            rank=rank,
            world_size=config.world_size,
            rpc_backend_options=options,
        )
        client_node = Client(f'client{rank}', rank, config.world_size, config)
        client_node.remote_registration()

        # trainer passively waiting for ps to kick off training iterations
    else:
        print(f'Starting the ps with world size={config.world_size}')
        rpc.init_rpc(
            "federator",
            rank=rank,
            world_size=config.world_size,
            rpc_backend_options=options

        )
        federator_node = Federator('federator', 0, config.world_size, config)
        # federator_node.create_clients()
        federator_node.run()
        federator_node.stop_all_clients()
    print('Ending program')
    # if rank == 0:
    #     print('FEDERATOR!')
    # else:
    #     print(f'CLIENT {rank}')

def main():
    pass


def add_default_arguments(parser):
    parser.add_argument('config', type=str,
                        help='')

if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='fltk', description='Experiment launcher for the Federated Learning Testbed (fltk)')
    subparsers = parser.add_subparsers(dest="action", required=True)

    launch_parser = subparsers.add_parser('launch-util')
    remote_parser = subparsers.add_parser('remote')
    single_machine_parser = subparsers.add_parser('single')
    add_default_arguments(launch_parser)
    add_default_arguments(remote_parser)
    add_default_arguments(single_machine_parser)

    remote_parser.add_argument('rank', type=int)
    remote_parser.add_argument('--nic', type=str, default=None)
    remote_parser.add_argument('--host', type=str, default=None)

    # single_parser = subparsers.add_parser('single', help='single help')
    # single_parser.add_argument('config')
    # util_parser = subparsers.add_parser('util', help='util help')
    # util_parser.add_argument('action')
    # print(sys.argv)
    args = parser.parse_args()
    if args.action == 'launch-util':
        pass
        # run_single(Path(args.config))
    if args.action == 'remote':
        run_remote(Path(args.config), args.rank, args.nic, args.host)
    else:
        # Run single machine mode
        run_single(Path(args.config))

    # if args.mode == 'single':
    #     print('Single')
    #     c = Config(optimizer=Optimizations.fedprox)
    #     print(isinstance(Config.aggregation, Enum))
    #     config = Config.FromYamlFile(args.config)
    #
    #     auto = config.optimizer
    #     print(config)
    #     print('Parsed')

    # print(args)