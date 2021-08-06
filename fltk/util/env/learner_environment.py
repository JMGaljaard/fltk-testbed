import os

def prepare_environment(host: str, nic: str):
    if host:
        os.environ['MASTER_ADDR'] = host
    else:
        os.environ['MASTER_ADDR'] = '10.5.0.2'
        os.environ['MASTER_PORT'] = '5000'
    if nic:
        os.environ['GLOO_SOCKET_IFNAME'] = nic
        os.environ['TP_SOCKET_IFNAME'] = nic
    else:
        os.environ['GLOO_SOCKET_IFNAME'] = 'etho0'
        os.environ['TP_SOCKET_IFNAME'] = 'eth0'
