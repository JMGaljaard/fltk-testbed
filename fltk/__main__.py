import os
import sys
import torch.distributed.rpc as rpc
import logging

import yaml
import argparse

import torch.multiprocessing as mp
from fltk.federator import Federator
from fltk.launch import run_single, run_spawn
from fltk.util.base_config import BareConfig

logging.basicConfig(level=logging.DEBUG)

def main():
    parser = argparse.ArgumentParser(description='Experiment launcher for the Federated Learning Testbed')
    parser.add_argument('mode', choices=['single', 'spawn'])
    parser.add_argument('config', type=str)
    parser.add_argument('--rank', type=int)
    parser.add_argument('--nic', type=str, default=None)
    parser.add_argument('--host', type=str, default=None)
    parser.add_argument('--world_size', type=str, default=None, help='Number of entities in the world. This is the number of clients + 1')

    args = parser.parse_args()
    with open(args.config) as file:
        cfg = BareConfig()
        yaml_data = yaml.load(file, Loader=yaml.FullLoader)
        cfg.merge_yaml(yaml_data)
        if args.mode == 'single':
            if args.rank is None:
                print('Missing rank argument when in \'single\' mode!')
                parser.print_help()
                exit(1)
            world_size = args.world_size
            master_address = args.host
            nic = args.nic

            if not world_size:
                world_size = yaml_data['system']['clients']['amount'] + 1
            if not master_address:
                master_address = yaml_data['system']['federator']['hostname']
            if not nic:
                nic = yaml_data['system']['federator']['nic']
            print(f'rank={args.rank}, world_size={world_size}, host={master_address}, args=cfg, nic={nic}')
            run_single(rank=args.rank, world_size=world_size, host=master_address, args=cfg, nic=nic)
        else:
            run_spawn(cfg)

if __name__ == "__main__":
    main()