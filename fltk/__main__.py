import argparse
import json
import logging
import os
from argparse import Namespace
from pathlib import Path

from torch.distributed import rpc

from fltk.core.client import Client
from fltk.core.federator import Federator
from fltk.launch import launch_distributed_client, launch_orchestrator, launch_extractor
from fltk.util.config import DistributedConfig, Config
from fltk.util.config.arguments import create_all_subparsers, extract_learning_parameters
from fltk.util.generate_experiments import generate, run


def cluster_start(arg_path, conf_path, args: Namespace = None, configuration: DistributedConfig = None, **kwargs):
    """
    Function to launch Orchestrator for execution with provided configurations. Currently
    this assumes that a single Orchestrator is started that manages all the resources in the cluster.
    """
    logging.info("Starting in cluster mode.")
    logging.basicConfig(level=logging.DEBUG,
                        datefmt='%m-%d %H:%M')
    # Set the seed for arrivals, torch seed is mostly ignored. Set the `arrival_seed` to a different value
    # for each repetition that you want to run an experiment with.
    configuration.set_seed()
    launch_orchestrator(args=args, conf=configuration)


def client_start(arg_path, conf_path, args: Namespace = None, configuration: DistributedConfig = None, **kwargs):
    logging.info("Starting in client mode")

    learning_params = extract_learning_parameters(args)
    # Set the seed for PyTorch, numpy seed is mostly ignored. Set the `torch_seed` to a different value
    # for each repetition that you want to run an experiment with.
    configuration.set_seed()
    task_id = args.task_id
    launch_distributed_client(task_id, config=configuration, learning_params=learning_params, namespace=args)
    logging.info("Stopping client...")


def run_single(base_path: Path, config_path: Path, prefix: str = None, **kwargs):
    # We can iterate over all the experiments in the directory and execute it, as long as the system remains the same!
    # System = machines and its configuration
    print(config_path)
    config = Config.FromYamlFile(config_path)
    config.world_size = config.num_clients + 1
    config.replication_id = prefix
    federator_node = Federator('federator', 0, config.world_size, config)
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


def run_remote(base_path: Path, config_path: Path, rank: int, parser, nic=None, host=None, prefix: str = None,
               **kwargs):
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


__run_op_dict = {
    'util-generate': generate,
    'util-run': run,
    'remote': run_remote,
    'single': run_single,
    'cluster': cluster_start,
    'client': client_start,
    'extractor': launch_extractor
}


def __main__():
    parser = argparse.ArgumentParser(prog='fltk',
                                     description='Experiment launcher for the Federated Learning Testbed (fltk)')
    subparsers = parser.add_subparsers(dest="action", required=True)
    create_all_subparsers(subparsers)
    """
    To create your own parser mirror the construction in the 'client_parser' object.
    Or refer to the ArgumentParser library documentation.
    """

    args = parser.parse_args()

    with open(args.config, 'r') as config_file:
        config: DistributedConfig = DistributedConfig.from_dict(json.load(config_file))
        config.config_path = Path(args.config)

    arg_path = Path(args.path)
    conf_path = Path(args.config)

    # Lookup execution mode and launch
    __run_op_dict[args.action](arg_path, conf_path, rank=args.rank, parser=parser, nic=args.nic, host=args.host,
                               prefix=args.prefix, args=args)

    exit(0)


if __name__ == "__main__":
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d-%Y %H:%M:%S', )
    __main__()
