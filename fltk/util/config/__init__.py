from fltk.util.config import Config
from fltk.util.config.distributed_config import DistributedConfig
from fltk.util.config.config import Config


def retrieve_config_network_params(conf: Config, nic=None, host=None):
    if hasattr(conf, 'system'):
        system_attr = getattr(conf, 'system')
        if 'federator' in system_attr:
            if 'hostname' in system_attr['federator'] and not host:
                host = system_attr['federator']['hostname']
            if 'nic' in system_attr['federator'] and not nic:
                nic = system_attr['federator']['nic']
    return nic, host
