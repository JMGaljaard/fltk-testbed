import logging
import time
from pathlib import Path
from typing import List, Callable

from dataclass_csv import DataclassWriter
from torch.distributed import rpc
from torch.utils.tensorboard import SummaryWriter

from fltk.client import Client
from fltk.strategy.attack import Attack
from fltk.strategy.client_selection import random_selection
from fltk.util.base_config import BareConfig
from fltk.util.fed_avg import average_nn_parameters
from fltk.util.log import FLLogger
from fltk.util.results import EpochData

logging.basicConfig(level=logging.DEBUG)


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _remote_method(method: Callable, rref, *args, **kwargs):
    """
    Wrapper function for executing remote code. This will launch an inference job at the federator learning side.
    """
    arguments = [method, rref] + list(args)
    # Send marshalled request to the child process
    return rpc.rpc_sync(rref.owner(), _call_method, args=arguments, kwargs=kwargs)


def _remote_method_async(method: Callable, rref, *args, **kwargs):
    """
    Wrapper function for executing remote code in asynchronous manner.
    This will launch an inference job at the federator learning side, without a blocking request. A a callback must be pro
    """
    arguments = [method, rref] + list(args)
    # Send marshalled request to the child process
    return rpc.rpc_async(rref.owner(), _call_method, args=arguments, kwargs=kwargs)


class ClientRef:
    ref = None
    name = ""
    data_size = 0
    tb_writer = None

    def __init__(self, name, ref, tensorboard_writer):
        self.name = name
        self.ref = ref
        self.tb_writer = tensorboard_writer

    def __repr__(self):
        return self.name


class Federator(object):
    """
    Central component of the Federated Learning System: The Federator

    The Federator is in charge of the following tasks:
    - Have a copy of the global model
    - Client selection
    - Aggregating the client model weights/gradients
    - Saving all the metrics
        - Use tensorboard to report metrics
    - Keep track of timing

    """
    clients: List[ClientRef] = []
    epoch_counter = 0
    client_data = {}
    poisoned_workers = {}

    def __init__(self, client_id_triple, num_epochs=3, config: BareConfig = None, attack: Attack = None):
        log_rref = rpc.RRef(FLLogger())
        self.log_rref = log_rref
        self.num_epoch = num_epochs
        self.config = config
        self.tb_path = config.output_location
        self.ensure_path_exists(self.tb_path)
        self.tb_writer = SummaryWriter(f'{self.tb_path}/{config.experiment_prefix}_federator')
        self.create_clients(client_id_triple)
        self.config.init_logger(logging)

        logging.info("Creating test client")
        copy_sampler = config.data_sampler
        config.data_sampler = "uniform"
        self.test_data = Client("test", None, 1, 2, config)
        self.test_data.init_dataloader()
        config.data_sampler = copy_sampler

        # Poisoning
        self.attack = attack

    def create_clients(self, client_id_triple):
        for id, rank, world_size in client_id_triple:
            client = rpc.remote(id, Client, kwargs=dict(id=id, log_rref=self.log_rref, rank=rank, world_size=world_size,
                                                        config=self.config))
            writer = SummaryWriter(f'{self.tb_path}/{self.config.experiment_prefix}_client_{id}')
            self.clients.append(ClientRef(id, client, tensorboard_writer=writer))
            self.client_data[id] = []

    def select_clients(self, n=2):
        """
        TODO: Make this extensible.
         1. E.g. 'time dependent' function.
         2. E.g. make a progress dependent function.
        """
        return random_selection(self.clients, n)

    def ping_all(self):
        for client in self.clients:
            logging.info(f'Sending ping to {client}')
            t_start = time.time()
            answer = _remote_method(Client.ping, client.ref)
            t_end = time.time()
            duration = (t_end - t_start) * 1000
            logging.info(f'Ping to {client} is {duration:.3}ms')

    def rpc_test_all(self):
        for client in self.clients:
            res = _remote_method_async(Client.rpc_test, client.ref)
            while not res.done():
                pass

    def client_load_data(self, poison_pill):
        for client in self.clients:
            _remote_method_async(Client.init_dataloader, client.ref,
                                 pill=None if poison_pill and client not in self.poisoned_workers else poison_pill)

    def clients_ready(self):
        all_ready = False
        ready_clients = []
        while not all_ready:
            responses = []
            for client in self.clients:
                if client.name not in ready_clients:
                    responses.append((client, _remote_method_async(Client.is_ready, client.ref)))
            all_ready = True
            for res in responses:
                result = res[1].wait()
                if result:
                    logging.info(f'{res[0]} is ready')
                    ready_clients.append(res[0])
                else:
                    logging.info(f'Waiting for {res[0]}')
                    all_ready = False

            time.sleep(2)
        logging.info('All clients are ready')

    def remote_run_epoch(self, epochs):
        responses = []
        client_weights = []
        selected_clients = self.select_clients(self.config.clients_per_round)
        for client in selected_clients:
            """
            TODO: Implement poisioning by providing arguments to the different clients. 
            Either the federator selects n nodes at the start, or a (configurable) function is selected, which 
            determines to send to which nodes and which are poisoned
            """
            pill = None
            if client in self.poisoned_workers:
                pill = self.attack.get_poison_pill()
            responses.append((client, _remote_method_async(Client.run_epochs, client.ref, num_epoch=epochs, pill=pill)))
        self.epoch_counter += epochs
        for res in responses:
            epoch_data, weights = res[1].wait()
            self.client_data[epoch_data.client_id].append(epoch_data)
            logging.info(f'{res[0]} had a loss of {epoch_data.loss}')
            logging.info(f'{res[0]} had a epoch data of {epoch_data}')

            res[0].tb_writer.add_scalar('training loss',
                                        epoch_data.loss_train,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            res[0].tb_writer.add_scalar('accuracy',
                                        epoch_data.accuracy,  # for every 1000 minibatches
                                        self.epoch_counter * res[0].data_size)

            res[0].tb_writer.add_scalar('training loss per epoch',
                                        epoch_data.loss_train,  # for every 1000 minibatches
                                        self.epoch_counter)

            res[0].tb_writer.add_scalar('accuracy per epoch',
                                        epoch_data.accuracy,  # for every 1000 minibatches
                                        self.epoch_counter)

            client_weights.append(weights)
        updated_model = average_nn_parameters(client_weights)

        # test global model
        logging.info("Testing on global test set")
        self.test_data.update_nn_parameters(updated_model)
        accuracy, loss, class_precision, class_recall = self.test_data.test()
        # self.tb_writer.add_scalar('training loss', loss, self.epoch_counter * self.test_data.get_client_datasize()) # does not seem to work :( )
        self.tb_writer.add_scalar('accuracy', accuracy, self.epoch_counter * self.test_data.get_client_datasize())
        self.tb_writer.add_scalar('accuracy per epoch', accuracy, self.epoch_counter)

        responses = []
        for client in self.clients:
            responses.append(
                (client, _remote_method_async(Client.update_nn_parameters, client.ref, new_params=updated_model)))

        for res in responses:
            res[1].wait()
        logging.info('Weights are updated')

    def update_client_data_sizes(self):
        responses = []
        for client in self.clients:
            responses.append((client, _remote_method_async(Client.get_client_datasize, client.ref)))
        for res in responses:
            res[0].data_size = res[1].wait()
            logging.info(f'{res[0]} had a result of datasize={res[0].data_size}')

    def remote_test_sync(self):
        responses = []
        for client in self.clients:
            responses.append((client, _remote_method_async(Client.test, client.ref)))

        for res in responses:
            accuracy, loss, class_precision, class_recall = res[1].wait()
            logging.info(f'{res[0]} had a result of accuracy={accuracy}')

    def save_epoch_data(self):
        file_output = f'./{self.config.output_location}'
        self.ensure_path_exists(file_output)
        for key in self.client_data:
            filename = f'{file_output}/{key}_epochs.csv'
            logging.info(f'Saving data at {filename}')
            with open(filename, "w") as f:
                w = DataclassWriter(f, self.client_data[key], EpochData)
                w.write()

    def ensure_path_exists(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def run(self):
        """
        Main loop of the Federator
        :return:
        """

        # # Select clients which will be poisened
        # TODO: get attack type and ratio from config, temp solution now
        poison_pill = None
        if self.attack:
            self.poisoned_workers = self.attack.select_poisoned_workers(self.clients)
            poison_pill = self.attack.get_poison_pill()

        self.client_load_data(poison_pill)
        self.ping_all()
        self.clients_ready()
        self.update_client_data_sizes()

        epoch_to_run = self.num_epoch
        addition = 0
        epoch_to_run = self.config.epochs
        epoch_size = self.config.epochs_per_cycle
        for epoch in range(epoch_to_run):
            print(f'Running epoch {epoch}')
            self.remote_run_epoch(epoch_size)
            addition += 1
        logging.info('Printing client data')
        print(self.client_data)

        logging.info(f'Saving data')
        self.save_epoch_data()
        logging.info(f'Federator is stopping')
