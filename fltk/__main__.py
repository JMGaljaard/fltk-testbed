import argparse
import logging
from multiprocessing.pool import ThreadPool
from pathlib import Path

from dotenv import load_dotenv

from fltk.launch import run_single

from fltk.util.config.base_config import BareConfig
from fltk.util.cluster.client import ClusterManager
from fltk.util.task.generator.arrival_generator import ExperimentGenerator

logging.basicConfig(level=logging.INFO)


def add_default_arguments(parser):
    parser.add_argument('--world_size', type=str, default=None,
                        help='Number of entities in the world. This is the number of clients + 1')


def main():
    # TODO: Clean up the parsers
    parser = argparse.ArgumentParser(description='Experiment launcher for the Federated Learning Testbed')

    subparsers = parser.add_subparsers(dest="mode")

    # Create single experiment parser
    cluster_parser = subparsers.add_parser('cluster')
    cluster_parser.add_argument('config', type=str)
    cluster_parser.add_argument('--rank', type=int)
    cluster_parser.add_argument('--nic', type=str, default=None)
    cluster_parser.add_argument('--host', type=str, default=None)
    add_default_arguments(cluster_parser)

    arguments = parser.parse_args()


    with open(arguments.config) as config_file:
        try:
            config = BareConfig.from_json(config_file)
        except Exception as e:
            print("Cannot load provided configuration, exiting...")
            exit(-1)

    if arguments.mode == 'orchestrator':
        cluster_start(arguments, config)
    elif arguments.mode == 'client':
        run_single()


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


def cluster_start(args: dict, config: BareConfig):
    """
    Function to start poisoned experiment.
    """
    logging.info("[Fed] Starting in cluster mode.")
    # TODO: Load configuration path
    config_path: Path = None
    cluster_manager = ClusterManager()
    arrival_generator = ExperimentGenerator(config_path)

    pool = ThreadPool(4)
    pool.apply(cluster_manager.start)
    pool.apply(arrival_generator.run)

    pool.join()

    print(f'rank={args.rank}, world_size={world_size}, host={master_address}, args=cfg, nic={nic}')
    run_single(rank=args.rank, args=config, nic=nic)


if __name__ == "__main__":
    load_dotenv()
    main()
