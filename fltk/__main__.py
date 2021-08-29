import json
import logging
from argparse import Namespace, ArgumentParser

from dotenv import load_dotenv

from fltk.launch import launch_client, launch_orchestrator
from fltk.util.config.base_config import BareConfig


def main():
    parser = ArgumentParser(description='Experiment launcher for the Federated Learning Testbed')
    subparsers = parser.add_subparsers(dest="mode")

    cluster_parser = subparsers.add_parser('cluster')
    cluster_parser.add_argument('config', type=str)

    client_parser = subparsers.add_parser('client')
    # Option to override rank, by default provided by PytorchJob in Kubeflow.
    client_parser.add_argument('--rank', type=int)
    # Option to override default nic, by default is 'eth0' in containers.
    client_parser.add_argument('--nic', type=str, default=None)
    # Option to override 'master' host name, by default provided by PytorchJob in Kubeflow.
    client_parser.add_argument('--host', type=str, default=None)

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
    else:
        print("Provided mode is not supported...")
        exit(1)


def cluster_start(args: Namespace, configuration: BareConfig):
    """
    Function to to launch Orchestrator for execution with provided configurations. Currently
    this assumes that a single Orchestrator is started that manages all the resources in the cluster.
    """
    logging.info("Starting in ")
    launch_orchestrator(args=args, config=configuration)


def client_start(args: Namespace, configuration: BareConfig):
    logging.info("Starting in client mode.")
    launch_client(args=args, config=configuration)


if __name__ == "__main__":
    # Load dotenv with default values. However, the Pytorch-Operator should set the necessary
    # environmental variables to get started.
    load_dotenv()
    main()
