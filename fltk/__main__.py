import os
from torch.distributed import rpc
from fltk.core.client import Client
import argparse
from pathlib import Path
from fltk.core.federator import Federator
from fltk.util.config import Config
from fltk.util.generate_experiments import generate, run


def run_single(config_path: Path, prefix: str = None):
    # We can iterate over all the experiments in the directory and execute it, as long as the system remains the same!
    # System = machines and its configuration
    print(config_path)
    config = Config.FromYamlFile(config_path)
    config.world_size = config.num_clients + 1
    config.replication_id = prefix
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


def run_remote(config_path: Path, rank: int, nic=None, host=None, prefix: str=None):
    print(config_path, rank)
    config = Config.FromYamlFile(config_path)
    config.world_size = config.num_clients + 1
    config.replication_id = prefix
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
        # init_method=f'tcp://{os.environ["MASTER_ADDR"]}:{os.environ["MASTER_PORT"]}'
        init_method='env://',
        _transports=["uv"]
    )
    if rank != 0:
        print(f'Starting worker {rank}  with world size={config.world_size}')
        rpc.init_rpc(
            f"client{rank}",
            rank=rank,
            world_size=config.world_size,
            rpc_backend_options=options,
        )
        client_node = Client(f'client{rank}', rank, config.world_size, config)
        client_node.remote_registration()
    else:
        print(f'Starting the ps with world size={config.world_size}')
        rpc.init_rpc(
            "federator",
            rank=rank,
            world_size=config.world_size,
            rpc_backend_options=options

        )
        federator_node = Federator('federator', 0, config.world_size, config)
        federator_node.run()
        federator_node.stop_all_clients()
    print('Ending program')


def add_default_arguments(parser):
    parser.add_argument('config', type=str, help='')
    parser.add_argument('--prefix', type=str, default=None)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='fltk', description='Experiment launcher for the Federated Learning Testbed (fltk)')
    subparsers = parser.add_subparsers(dest="action", required=True)

    util_docker_parser = subparsers.add_parser('util-docker')
    util_docker_parser.add_argument('name', type=str)
    util_docker_parser.add_argument('--clients', type=int)
    util_generate_parser = subparsers.add_parser('util-generate')
    util_generate_parser.add_argument('path', type=str)
    util_run_parser = subparsers.add_parser('util-run')
    util_run_parser.add_argument('path', type=str)

    remote_parser = subparsers.add_parser('remote')
    single_machine_parser = subparsers.add_parser('single')
    add_default_arguments(remote_parser)
    add_default_arguments(single_machine_parser)

    remote_parser.add_argument('rank', type=int)
    remote_parser.add_argument('--nic', type=str, default=None)
    remote_parser.add_argument('--host', type=str, default=None)

    args = parser.parse_args()
    if args.action == 'util-docker':
        print('Unimplemented!')
    elif args.action == 'util-generate':
        path = Path(args.path)
        print(f'generate for {path}')
        generate(path)
    elif args.action == 'util-run':
        run(Path(args.path))
    elif args.action == 'remote':
        run_remote(Path(args.config), args.rank, args.nic, args.host, args.prefix)
    else:
        # Run single machine mode
        run_single(Path(args.config), args.prefix)
