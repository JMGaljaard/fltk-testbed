from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

from fltk.util.config.definitions import Loss
from fltk.util.config.distributed_config import DistributedConfig
from fltk.util.config.learner_config import FedLearnerConfig, get_safe_loader, DistLearnerConfig
from fltk.util.config.experiment_config import ExperimentConfig, ExperimentParser


def retrieve_config_network_params(conf: FedLearnerConfig, nic=None, host=None):
    """

    Args:
      conf: FedLearnerConfig: 
      nic:  (Default value = None)
      host:  (Default value = None)

    Returns:
        str: NIC to use.
        str: host to use.
    """
    if hasattr(conf, 'system'):
        system_attr = getattr(conf, 'system')
        if 'federator' in system_attr:
            if 'hostname' in system_attr['federator'] and not host:
                host = system_attr['federator']['hostname']
            if 'nic' in system_attr['federator'] and not nic:
                nic = system_attr['federator']['nic']
    return nic, host


def get_distributed_config(args, alt_path: str = None) -> Optional[DistributedConfig]:
    """

    Args:
      args: 
      alt_path: str:  (Default value = None)

    Returns:
        Optional[DistributedConfig]: When provided, DistributedConfig from Path specified during startup.
    """
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
        msg = f"Failed to get distributed config: {e}, path: {config_path}"
        logging.info(msg)
    return config


def get_learning_param_config(args, alt_path: str = None) -> Optional[DistLearnerConfig]:
    """Retrieve learning parameter configuration from Disk for distributed learning experiments.

    Args:
      args: 
      alt_path: str:  (Default value = None)

    Returns:

    """
    if args:
        config_path = args.experiment_config
    else:
        config_path = alt_path
    try:
        learning_params: DistLearnerConfig = DistLearnerConfig.from_yaml(Path(config_path))
    except Exception as e:
        msg = f"Failed to get learning parameter configuration for distributed experiments: {e}"
        logging.info(msg)
        raise e
    return learning_params
