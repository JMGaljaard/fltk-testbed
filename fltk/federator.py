import time
from typing import List

from dataclass_csv import DataclassWriter
from torch.distributed import rpc

from fltk.client import Client
from fltk.strategy.client_selection import random_selection
from fltk.util.fed_avg import average_nn_parameters
from fltk.util.log import FLLogger
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging

from fltk.util.remote import ClientRef, AsyncCall, timed_remote_async_call, _remote_method, TimingRecord
from fltk.util.results import EpochData

logging.basicConfig(level=logging.DEBUG)

class Federator:
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

    def __init__(self, client_id_triple, num_epochs = 3, config=None):
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


    def create_clients(self, client_id_triple):
        for id, rank, world_size in client_id_triple:
            client = rpc.remote(id, Client, kwargs=dict(id=id, log_rref=self.log_rref, rank=rank, world_size=world_size, config=self.config))
            writer = SummaryWriter(f'{self.tb_path}/{self.config.experiment_prefix}_client_{id}')
            self.clients.append(ClientRef(id, client, tensorboard_writer=writer))
            self.client_data[id] = []

    def select_clients(self, n = 2):
        return random_selection(self.clients, n)

    def ping_all(self):
        for client in self.clients:
            logging.info(f'Sending ping to {client}')
            t_start = time.time()
            answer = _remote_method(Client.ping, client.ref)
            t_end = time.time()
            duration = (t_end - t_start)*1000
            client.timing_data.append(TimingRecord(f'{client.name}', 'ping', duration))
            logging.info(f'Ping to {client} is {duration:.3}ms')

    def rpc_test_all(self):
        for client in self.clients:
            res = timed_remote_async_call(client, Client.rpc_test, client.ref)
            while not res.future.done():
                pass

    def client_load_data(self):
        for client in self.clients:
            timed_remote_async_call(client, Client.init_dataloader, client.ref)

    def clients_ready(self):
        all_ready = False
        ready_clients = []
        while not all_ready:
            responses = []
            for client in self.clients:
                if client.name not in ready_clients:
                    response = timed_remote_async_call(client, Client.is_ready, client.ref)
                    responses.append(response)
            all_ready = True
            for res in responses:
                result = res.future.wait()
                if result:
                    logging.info(f'{res.client} is ready')
                    ready_clients.append(res.client)
                else:
                    logging.info(f'Waiting for {res.client}')
                    all_ready = False

            time.sleep(2)
        logging.info('All clients are ready')

    def remote_run_epoch(self, epochs):

        responses: List[AsyncCall] = []
        client_weights = []
        selected_clients = self.select_clients(self.config.clients_per_round)
        for client in selected_clients:
            response = timed_remote_async_call(client, Client.run_epochs, client.ref, num_epoch=epochs)
            responses.append(response)

        self.epoch_counter += epochs
        durations = []
        for res in responses:
            res.future.wait()
            epoch_data, weights = res.future.wait()
            fed_stop_time = time.time()
            self.client_data[epoch_data.client_id].append(epoch_data)
            logging.info(f'{res.client.name} had a loss of {epoch_data.loss}')
            logging.info(f'{res.client.name} had a epoch data of {epoch_data}')
            res.client.timing_data.append(TimingRecord(f'{res.client.name}', 'epoch_time_round_trip', res.duration()))

            res.client.tb_writer.add_scalar('training loss',
                                        epoch_data.loss_train,  # for every 1000 minibatches
                                        self.epoch_counter * res.client.data_size)

            res.client.tb_writer.add_scalar('accuracy',
                                        epoch_data.accuracy,  # for every 1000 minibatches
                                        self.epoch_counter * res.client.data_size)

            res.client.tb_writer.add_scalar('training loss per epoch',
                                        epoch_data.loss_train,  # for every 1000 minibatches
                                        self.epoch_counter)

            res.client.tb_writer.add_scalar('accuracy per epoch',
                                        epoch_data.accuracy,  # for every 1000 minibatches
                                        self.epoch_counter)

            client_weights.append(weights)
        updated_model = average_nn_parameters(client_weights)

        # test global model
        logging.info("Testing on global test set")
        self.test_data.update_nn_parameters(updated_model)
        accuracy, loss, class_precision, class_recall = self.test_data.test()
        self.tb_writer.add_scalar('accuracy', accuracy, self.epoch_counter * self.test_data.get_client_datasize())
        self.tb_writer.add_scalar('accuracy per epoch', accuracy, self.epoch_counter)

        responses = []
        for client in self.clients:
            response = timed_remote_async_call(client, Client.update_nn_parameters, client.ref, new_params=updated_model)
            responses.append(response)

        for res in responses:
            func_duration = res.future.wait()
            res.client.timing_data.append(TimingRecord(res.client.name, 'update_param_inner', func_duration))
            res.client.timing_data.append(TimingRecord(f'{res.client.name}', 'update_param_round_trip', res.duration()))
        logging.info('Weights are updated')

    def update_client_data_sizes(self):
        responses = []
        for client in self.clients:
            response = timed_remote_async_call(client, Client.get_client_datasize, client.ref)
            responses.append(response)
        for res in responses:
            res.client.data_size = res.future.wait()
            logging.info(f'{res.client.name} had a result of datasize={res.client.data_size}')

    def remote_test_sync(self):
        responses = []
        for client in self.clients:
            response = timed_remote_async_call(client, Client.test, client.ref)
            responses.append(response)

        for res in responses:
            accuracy, loss, class_precision, class_recall = res.future.wait()
            logging.info(f'{res.client.name} had a result of accuracy={accuracy}')

    def save_epoch_data(self):
        file_output = f'./{self.config.output_location}/{self.config.experiment_prefix}_data'
        self.ensure_path_exists(file_output)
        for key in self.client_data:
            filename = f'{file_output}/{key}_epochs.csv'
            logging.info(f'Saving data at {filename}')
            with open(filename, "w") as f:
                w = DataclassWriter(f, self.client_data[key], EpochData)
                w.write()

    def save_profiling_data(self):
        file_output = f'./{self.config.output_location}/{self.config.experiment_prefix}_data'
        filename = f'{file_output}/profiling_data.csv'
        self.ensure_path_exists(file_output)
        with open(filename, "w") as f:
            for client in self.clients:
                for record in client.timing_data:
                    w = DataclassWriter(f, [record], TimingRecord)
                    w.write()



    def ensure_path_exists(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def run(self):
        """
        Main loop of the Federator
        :return:
        """
        # # Make sure the clients have loaded all the data
        self.client_load_data()
        self.ping_all()
        self.clients_ready()
        self.update_client_data_sizes()

        epoch_to_run = self.num_epoch
        addition = 0
        epoch_to_run = self.config.epochs
        epoch_size = self.config.epochs_per_cycle
        for epoch in range(epoch_to_run):
            logging.info(f'Running epoch {epoch}')
            self.remote_run_epoch(epoch_size)
            addition += 1
        logging.info('Available clients with data')
        logging.info(self.client_data.keys())

        logging.info('Saving data')
        self.save_epoch_data()

        logging.info('Printing all clients timing data')
        for client in self.clients:
            logging.info(f"Timing data for client {client}")
            for record in client.timing_data:
                logging.info(f'{record}')

        logging.info('Saving profiling data')
        self.save_profiling_data()

        logging.info(f'Federator is stopping')

