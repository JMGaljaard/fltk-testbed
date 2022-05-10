import json
from pathlib import Path
from typing import Optional

import yaml
import logging

from fltk.util.config.distributed_config import DistributedConfig
from fltk.util.config.config import Config, get_safe_loader
from fltk.util.config.arguments import DistLearningConfig


def retrieve_config_network_params(conf: Config, nic=None, host=None):
    if hasattr(conf, 'system'):
        system_attr = getattr(conf, 'system')
        if 'federator' in system_attr:
            if 'hostname' in system_attr['federator'] and not host:
                host = system_attr['federator']['hostname']
            if 'nic' in system_attr['federator'] and not nic:
                nic = system_attr['federator']['nic']
    return nic, host


def get_distributed_config(args, alt_path: str = None) -> Optional[DistributedConfig]:
    if args:
        config_path = args.config
    else:
        config_path = alt_path
    config = None
    try:
        with open(config_path, 'r') as config_file:
            logging.info(f"Loading file {config_path}")
            config = DistributedConfig.from_dict(json.load(config_file))  # pylint: disable=no-member
            config.config_path = Path(args.config)
    except Exception as e:  # pylint: disable=broad-except
        msg = f"Failed to get distributed config: {e}"
        logging.info(msg)
    return config


def get_learning_param_config(args, alt_path: str = None) -> Optional[DistLearningConfig]:
    if args:
        config_path = args.experiment_config
    else:
        config_path = alt_path
    safe_loader = get_safe_loader()
    try:
        with open(config_path) as f:
            learning_params_dict = yaml.load(f, Loader=safe_loader)
            learning_params = DistLearningConfig.from_dict(learning_params_dict)
    except Exception as e:
        msg = f"Failed to get learning parameter configuration for distributed experiments: {e}"
        logging.info(msg)
        raise e
    return learning_params
