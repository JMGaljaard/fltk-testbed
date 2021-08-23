import argparse
import logging
from multiprocessing.pool import ThreadPool
from pathlib import Path

import yaml
from dotenv import load_dotenv

from fltk.launch import run_single

from fltk.util.base_config import BareConfig
from fltk.util.cluster.client import ClusterManager
from fltk.util.generator.arrival_generator import ExperimentGenerator

logging.basicConfig(level=logging.INFO)


def add_default_arguments(parser):
    parser.add_argument('--world_size', type=str, default=None,
                        help='Number of entities in the world. This is the number of clients + 1')


def main():
    # TODO: Clean up the parsers
    parser = argparse.ArgumentParser(description='Experiment launcher for the Federated Learning Testbed')

    subparsers = parser.add_subparsers(dest="mode")

    # Create single experiment parser
    single_parser = subparsers.add_parser('single')
    single_parser.add_argument('config', type=str)
    single_parser.add_argument('--rank', type=int)
    single_parser.add_argument('--nic', type=str, default=None)
    single_parser.add_argument('--host', type=str, default=None)
    add_default_arguments(single_parser)

    # Create spawn parser
    spawn_parser = subparsers.add_parser('spawn')
    spawn_parser.add_argument('config', type=str)
    add_default_arguments(spawn_parser)

    # Create remote parser
    remote_parser = subparsers.add_parser('remote')
    remote_parser.add_argument('--rank', type=int)
    remote_parser.add_argument('--nic', type=str, default=None)
    remote_parser.add_argument('--host', type=str, default=None)
    add_default_arguments(remote_parser)

    # Create poisoned parser
    poison_parser = subparsers.add_parser('poison')
    poison_parser.add_argument('config', type=str)
    poison_parser.add_argument('--rank', type=int)
    poison_parser.add_argument('--nic', type=str, default=None)
    poison_parser.add_argument('--host', type=str, default=None)
    add_default_arguments(poison_parser)

    poison_parser = subparsers.add_parser('cluster')
    poison_parser.add_argument('config', type=str)
    poison_parser.add_argument('--rank', type=int)
    poison_parser.add_argument('--nic', type=str, default=None)
    poison_parser.add_argument('--host', type=str, default=None)
    add_default_arguments(poison_parser)

    args = parser.parse_args()

    if args.mode == 'cluster':
        logging.info("[Fed] Starting in cluster mode.")
        # TODO: Load configuration path
        config_path: Path = None
        cluster_manager = ClusterManager()
        arrival_generator = ExperimentGenerator(config_path)

        pool = ThreadPool(4)
        pool.apply(cluster_manager.start)
        pool.apply(arrival_generator.run)

        pool.join()
    else:
        with open(args.config) as config_file:
            cfg = BareConfig()
            yaml_data = yaml.load(config_file, Loader=yaml.FullLoader)
            cfg.merge_yaml(yaml_data)
            if args.mode == 'poison':
                perform_poison_experiment(args, cfg, parser, yaml_data)


def perform_single_experiment(args, cfg, parser, yaml_data):
    if args.rank is None:
        print('Missing rank argument when in \'single\' mode!')
        parser.print_help()
        exit(1)
    world_size = args.world_size
    master_address = args.host
    nic = args.nic

    if not world_size:
        world_size = yaml_data['system']['clients']['amount'] + 1
    if not master_address:
        master_address = yaml_data['system']['federator']['hostname']
    if not nic:
        nic = yaml_data['system']['federator']['nic']
    print(f'rank={args.rank}, world_size={world_size}, host={master_address}, args=cfg, nic={nic}')
    run_single(rank=args.rank, world_size=world_size, host=master_address, args=cfg, nic=nic)


def perform_poison_experiment(args, cfg, yaml_data):
    """
    Function to start poisoned experiment.
    """
    if args.rank is None:
        print('Missing rank argument when in \'poison\' mode!')
        exit(1)
    if not yaml_data.get('poison'):
        print(f'Missing poison configuration for \'poison\' mode')
        exit(1)

    world_size = args.world_size
    master_address = args.host
    nic = args.nic

    if not world_size:
        world_size = yaml_data['system']['clients']['amount'] + 1
    if not master_address:
        master_address = yaml_data['system']['federator']['hostname']
    if not nic:
        nic = yaml_data['system']['federator']['nic']
    print(f'rank={args.rank}, world_size={world_size}, host={master_address}, args=cfg, nic={nic}')
    run_single(rank=args.rank, world_size=world_size, host=master_address, args=cfg, nic=nic)


if __name__ == "__main__":
    load_dotenv()
    main()
