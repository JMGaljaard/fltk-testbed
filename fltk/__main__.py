import argparse
import logging
from pathlib import Path
from typing import Optional, Any, Dict

import sys

from fltk.launch import launch_extractor, launch_client, launch_single, \
    launch_remote, launch_cluster, launch_signature
from fltk.util.config import get_distributed_config
from fltk.util.config.arguments import create_all_subparsers
from fltk.util.generate_experiments import generate, run

__run_op_dict: Dict[str, launch_signature] = {
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
    distributed_config = get_distributed_config(args)

    arg_path, conf_path = None, None
    try:
        arg_path = Path(args.path)
    except Exception as _:  # pylint: disable=broad-except
        print('No argument path is provided.')
    try:
        conf_path = Path(args.config)
    except Exception as _:  # pylint: disable=broad-except
        print('No configuration path is provided.')

    launch_fn: launch_signature = __run_op_dict[args.action]
    launch_fn(arg_path, conf_path,
              _save_get(args, 'rank'),
              _save_get(args, 'nic'),
              _save_get(args, 'host'),
              _save_get(args, 'prefix'),
              args,
              distributed_config)
    try:
        pass
    except Exception as e:
        print(f"Failed with reason: {e}")
        parser.print_help()
        sys.exit(1)
    sys.exit(0)


if __name__ == "__main__":
    # Get loger and set format for logging before starting the main loop.
    root = logging.getLogger()
    if root.handlers:
        for handler in root.handlers:
            root.removeHandler(handler)
    logging.basicConfig(format='%(asctime)s %(name)-12s %(levelname)-8s %(message)s', datefmt='%m-%d-%Y %H:%M:%S', )
    __main__()
