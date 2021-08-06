import os
import sys
from pathlib import Path

import torch.distributed.rpc as rpc
import logging

import yaml
import argparse
from dotenv import load_dotenv

import torch.multiprocessing as mp
from fltk.federator import Federator
from fltk.launch import run_single, run_spawn
from fltk.strategy.antidote import create_antidote
from fltk.strategy.attack import create_attack
from fltk.util.base_config import BareConfig
from fltk.util.poison.poisonpill import FlipPill

logging.basicConfig(level=logging.DEBUG)


def add_default_arguments(parser):
    parser.add_argument('--world_size', type=str, default=None,
                        help='Number of entities in the world. This is the number of clients + 1')


def main():
    parser = argparse.ArgumentParser(description='Experiment launcher for the Federated Learning Testbed')

    subparsers = parser.add_subparsers(dest="mode")

    # Create single experiment parser
    single_parser = subparsers.add_parser('single')
    single_parser.add_argument('config', type=str)
    single_parser.add_argument('--rank', type=int)
    single_parser.add_argument('--nic', type=str, default=None)
    single_parser.add_argument('--host', type=str, default=None)
    add_default_arguments(single_parser)

    # Create spawn parser
    spawn_parser = subparsers.add_parser('spawn')
    spawn_parser.add_argument('config', type=str)
    add_default_arguments(spawn_parser)

    # Create remote parser
    remote_parser = subparsers.add_parser('remote')
    remote_parser.add_argument('--rank', type=int)
    remote_parser.add_argument('--nic', type=str, default=None)
    remote_parser.add_argument('--host', type=str, default=None)
    add_default_arguments(remote_parser)

    # Create poisoned parser
    poison_parser = subparsers.add_parser('poison')
    poison_parser.add_argument('config', type=str)
    poison_parser.add_argument('--rank', type=int)
    poison_parser.add_argument('--nic', type=str, default=None)
    poison_parser.add_argument('--host', type=str, default=None)
    add_default_arguments(poison_parser)

    args = parser.parse_args()
    if args.mode == 'remote':
        if args.rank is None or args.host is None or args.world_size is None or args.nic is None:
            print('Missing rank, host, world-size, or nic argument when in \'remote\' mode!')
            parser.print_help()
            exit(1)
        world_size = int(args.world_size)
        master_address = args.host
        nic = args.nic
        rank = int(args.rank)
        if rank == 0:
            print('Remote mode only supports ranks > 0!')
            exit(1)
        print(f'rank={args.rank}, world_size={world_size}, host={master_address}, args=None, nic={nic}')
        run_single(rank=args.rank, world_size=world_size, host=master_address, args=None, nic=nic)
    else:
        with open(args.config) as config_file:
            cfg = BareConfig()
            yaml_data = yaml.load(config_file, Loader=yaml.FullLoader)
            cfg.merge_yaml(yaml_data)
            if args.mode == 'poison':
                for ratio in [0.0, 0.05, 0.1, 0.15, 0.2]:
                    perform_poison_experiment(args, cfg, parser, yaml_data, ratio)
            elif args.mode == 'single':
                perform_single_experiment(args, cfg, parser, yaml_data)
            else:
                run_spawn(cfg)


def perform_single_experiment(args, cfg, parser, yaml_data):
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




def perform_poison_experiment(args, cfg, parser, yaml_data, ratio=None):
    """
    Function to start poisoned experiment.
    """
    if args.rank is None:
        print('Missing rank argument when in \'poison\' mode!')
        exit(1)
    if not yaml_data.get('poison'):
        print(f'Missing poison configuration for \'poison\' mode')
        exit(1)

    world_size = args.world_size
    master_address = args.host
    nic = args.nic

    attack = create_attack(cfg)
    antidote = create_antidote(cfg)
    if not world_size:
        world_size = yaml_data['system']['clients']['amount'] + 1
    if not master_address:
        master_address = yaml_data['system']['federator']['hostname']
    if not nic:
        nic = yaml_data['system']['federator']['nic']
    print(f'rank={args.rank}, world_size={world_size}, host={master_address}, args=cfg, nic={nic}')
    if ratio:
        print(f'Setting ratio to {ratio}')
        attack.ratio = ratio
    run_single(rank=args.rank, world_size=world_size, host=master_address, args=cfg, nic=nic, attack=attack, antidote=antidote)


if __name__ == "__main__":
    load_dotenv()
    main()
