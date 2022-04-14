import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Optional, Any

from fltk.launch import launch_extractor, launch_client, launch_single, \
    launch_remote, launch_cluster
from fltk.util.config import DistributedConfig
from fltk.util.config.arguments import create_all_subparsers
from fltk.util.generate_experiments import generate, run


__run_op_dict = {
    'util-generate': generate,
    'util-run': run,
    'remote': launch_remote,
    'single': launch_single,
    'cluster': launch_cluster,
    'client': launch_client,
    'extractor': launch_extractor
}


def _save_get(args, param) -> Optional[Any]:
    save_argument = None
    if args is not None and hasattr(args, param):
        save_argument = args.__dict__[param]
    msg = f"Getting {param}: {save_argument}"
    logging.debug(msg)
    return save_argument


def _get_distributed_config(args) -> Optional[DistributedConfig]:
    config = None
    try:
        with open(args.config, 'r', encoding='utf-8') as config_file:
            config = DistributedConfig.from_dict(json.load(config_file)) # pylint: disable=no-member
            config.config_path = Path(args.config)
    except Exception as excep: # pylint: disable=broad-except
        msg = f"Failed to get distributed config: {excep}"
        logging.info(msg)
    return config


def __main__():
    """
    Main loop to perform learning (either Federated or Distributed). Note that Orchestrator is part of this setup for
    now.
    @return: Nothing.
    @rtype: None
    """
    parser = argparse.ArgumentParser(prog='fltk',
                                     description='Experiment launcher for the Federated Learning Testbed (fltk)')
    subparsers = parser.add_subparsers(dest="action", required=True)
    create_all_subparsers(subparsers)
    # To create your own parser mirror the construction in the 'client_parser' object.
    # Or refer to the ArgumentParser library documentation.
    args = parser.parse_args()
    distributed_config = _get_distributed_config(args)

    arg_path, conf_path = None, None
    try:
        arg_path = Path(args.path)
    except Exception as _: # pylint: disable=broad-except
        print('No argument path is provided.')
    try:
        conf_path = Path(args.config)
    except Exception as _: # pylint: disable=broad-except
        print('No configuration path is provided.')

    __run_op_dict[args.action](arg_path, conf_path,
                               rank=_save_get(args, 'rank'),
                               parser=parser,
                               nic=_save_get(args, 'nic'),
                               host=_save_get(args, 'host'),
                               prefix=_save_get(args, 'prefix'),
                               args=args,
                               config=distributed_config)

    sys.exit(0)


if __name__ == "__main__":
    # Get loger and set format for logging before starting the main loop.
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d-%Y %H:%M:%S', )
    __main__()
