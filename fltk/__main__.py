import json
import logging
from argparse import Namespace, ArgumentParser
from pathlib import Path

from fltk.launch import launch_client, launch_orchestrator, launch_extractor
from fltk.util.config.arguments import create_client_parser, create_cluster_parser, extract_learning_parameters, \
    create_extractor_parser
from fltk.util.config.base_config import BareConfig


def __main__():
    parser = ArgumentParser(description='Experiment launcher for the Federated Learning Testbed')
    subparsers = parser.add_subparsers(dest="mode")
    create_client_parser(subparsers)
    create_cluster_parser(subparsers)
    create_extractor_parser(subparsers)
    """
    To create your own parser mirror the construction in the 'client_parser' object.
    Or refer to the ArgumentParser library documentation.
    """

    arguments = parser.parse_args()

    with open(arguments.config, 'r') as config_file:
        config: BareConfig = BareConfig.from_dict(json.load(config_file))
        config.config_path = Path(arguments.config)

    if arguments.mode == 'cluster':
        logging.info("Starting in cluster mode.")
        cluster_start(arguments, config)
    elif arguments.mode == 'client':
        logging.info("Starting in client mode")
        client_start(arguments, config)
        logging.info("Stopping client...")
        exit(0)
    elif arguments.mode == 'extractor':
        launch_extractor(arguments, config)
    else:
        print("Provided mode is not supported...")
        exit(1)


def cluster_start(args: Namespace, configuration: BareConfig):
    """
    Function to to launch Orchestrator for execution with provided configurations. Currently
    this assumes that a single Orchestrator is started that manages all the resources in the cluster.
    """
    logging.basicConfig(level=logging.DEBUG,
                        datefmt='%m-%d %H:%M')
    # Set the seed for arrivals, torch seed is mostly ignored. Set the `arrival_seed` to a different value
    # for each repetition that you want to run an experiment with.
    configuration.set_seed()
    launch_orchestrator(args=args, conf=configuration)


def client_start(args: Namespace, configuration: BareConfig):
    learning_params = extract_learning_parameters(args)
    # Set the seed for PyTorch, numpy seed is mostly ignored. Set the `torch_seed` to a different value
    # for each repetition that you want to run an experiment with.
    configuration.set_seed()
    task_id = args.task_id
    launch_client(task_id, config=configuration, learning_params=learning_params, namespace=args)


if __name__ == "__main__":
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d-%Y %H:%M:%S',)
    __main__()
