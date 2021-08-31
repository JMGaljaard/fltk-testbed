import json
import logging
from argparse import Namespace, ArgumentParser

from dotenv import load_dotenv

from fltk.launch import launch_client, launch_orchestrator
from fltk.util.config.arguments import create_client_parser, create_cluster_parser, extract_learning_parameters
from fltk.util.config.base_config import BareConfig


def main():
    parser = ArgumentParser(description='Experiment launcher for the Federated Learning Testbed')
    subparsers = parser.add_subparsers(dest="mode")
    create_client_parser(subparsers)
    create_cluster_parser(subparsers)

    """
    To create your own parser mirror the construction in the 'client_parser' object.
    Or refer to the ArgumentParser library documentation.
    """

    arguments = parser.parse_args()

    with open(arguments.config, 'r') as config_file:
        config = BareConfig.from_dict(json.load(config_file))

    if arguments.mode == 'cluster':
        logging.info("Starting in cluster mode.")
        cluster_start(arguments, config)
    elif arguments.mode == 'client':
        logging.info("Starting in client mode")
        client_start(arguments, config)
        logging.info("Stopping client...")
        exit(0)
    else:
        print("Provided mode is not supported...")
        exit(1)


def cluster_start(args: Namespace, configuration: BareConfig):
    """
    Function to to launch Orchestrator for execution with provided configurations. Currently
    this assumes that a single Orchestrator is started that manages all the resources in the cluster.
    """
    launch_orchestrator(args=args, config=configuration)


def client_start(args: Namespace, configuration: BareConfig):
    learning_params = extract_learning_parameters(args)
    task_id = args.task_id
    launch_client(task_id, config=configuration, learning_params=learning_params)


if __name__ == "__main__":
    # Load dotenv with default values. However, the Pytorch-Operator should set the necessary
    # environmental variables to get started.
    load_dotenv()
    main()
