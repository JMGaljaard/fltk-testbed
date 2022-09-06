from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fltk.util.config.definitions import Loss
from fltk.util.config.distributed_config import DistributedConfig
from fltk.util.config.learning_config import FedLearningConfig, get_safe_loader, DistLearningConfig


def retrieve_config_network_params(conf: FedLearningConfig, nic=None, host=None):
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
            config = DistributedConfig.from_json(config_file.read())  # pylint: disable=no-member
            config.config_path = Path(args.config)
    except Exception as e:  # pylint: disable=broad-except
        msg = f"Failed to get distributed config: {e}"
        logging.info(msg)
    return config


def get_learning_param_config(args, alt_path: str = None) -> Optional[DistLearningConfig]:
    """
    Retrieve learning parameter configuration from Disk for distributed learning experiments.
    """
    if args:
        config_path = args.experiment_config
    else:
        config_path = alt_path
    try:
        learning_params: DistLearningConfig = DistLearningConfig.from_yaml(Path(config_path))
    except Exception as e:
        msg = f"Failed to get learning parameter configuration for distributed experiments: {e}"
        logging.info(msg)
        raise e
    return learning_params
