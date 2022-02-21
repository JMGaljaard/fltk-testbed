import os
import sys
import torch
import torch.distributed.rpc as rpc

from fltk.core.client import Client
from fltk.core.federator import Federator
from fltk.core.node import Node
from fltk.util.config import Config

if __name__ == '__main__':
    world_size = 2
    config = Config()
    config.num_clients = world_size - 1
    config.world_size = world_size
    config.clients_per_round = 1
    config.epochs = 2
    config.rounds = 20
    config.cuda = False
    config.single_machine = True

    fed = Federator('fed0', 0, world_size, config)
    fed.run()

    # n1 = Client('c1', 0, world_size, config)
    # n2 = Client('c2', 1, world_size, config)
    # n3 = Client('c3', 2, world_size, config)
    # n1.init_dataloader()
    # n2.init_dataloader()
    # n3.init_dataloader()
    #
    # response = n1.message(n2, Client.ping, 'new_sender')
    # print(response)
    # response = n3.message(n1, Client.ping, 'new_sender', be_weird=True)
    # print(response)
    #
    # _, _, accuracy_n1, _ = n3.message(n1, Client.exec_round, 1)
    # _, _, accuracy_n2, _ = n1.message(n2, Client.exec_round, 1)
    # _, _, accuracy_n3, _ = n1.message(n3, Client.exec_round, 1)
    # print(f'Client n1 has an accuracy of {accuracy_n1}')
    # print(f'Client n2 has an accuracy of {accuracy_n2}')
    # print(f'Client n3 has an accuracy of {accuracy_n3}')
    #
    # print(config)
