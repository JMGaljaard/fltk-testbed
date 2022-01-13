import datetime
import time
from dataclasses import dataclass
from typing import List

import pandas as pd
import torch
from dataclass_csv import DataclassWriter
from torch.distributed import rpc
from torch.distributed.rpc import RRef, get_worker_info
from torch.utils.data._utils.worker import WorkerInfo

from fltk.client import Client
from fltk.datasets.data_distribution import distribute_batches_equally
from fltk.strategy.aggregation import FedAvg
from fltk.strategy.client_selection import random_selection
from fltk.strategy.offloading import OffloadingStrategy
from fltk.util.arguments import Arguments
from fltk.util.base_config import BareConfig
from fltk.util.data_loader_utils import load_train_data_loader, load_test_data_loader, \
    generate_data_loaders_from_distributed_dataset
from fltk.util.fed_avg import average_nn_parameters
from fltk.util.log import FLLogger
from torchsummary import summary
from torch.utils.tensorboard import SummaryWriter
from pathlib import Path
import logging

# from fltk.util.profile_plots import stability_plot, parse_stability_data
from fltk.util.results import EpochData
from fltk.util.tensor_converter import convert_distributed_data_into_numpy

logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
)


def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)


def _call_method_2(method, rref, *args, **kwargs):
    print(method)
    return method(rref, *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _remote_method_async(method, rref, *args, **kwargs) -> torch.Future:
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _remote_method_async_by_name(method, client_name, *args, **kwargs) -> torch.Future:
    args = [method, client_name] + list(args)
    print(client_name)
    print(_call_method_2)
    return rpc.rpc_sync(client_name, _call_method_2, args=args, kwargs=kwargs)


class ClientRef:
    ref = None
    name = ""
    data_size = 0
    tb_writer = None
    available = False

    def __init__(self, name, ref, tensorboard_writer):
        self.name = name
        self.ref = ref
        self.tb_writer = tensorboard_writer

    def __repr__(self):
        return self.name

@dataclass
class ClientResponse:
    id: int
    client: ClientRef
    future: torch.Future
    start_time: float = time.time()
    end_time: float = 0
    done: bool = False
    dropped = True

    def finish(self):
        self.end_time = time.time()
        self.done = True
        self.dropped = False

    def duration(self):
        return self.end_time - self.start_time


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
    response_list : List[ClientResponse] = []
    response_id = 0

    reference_lookup = {}
    performance_estimate = {}

    # Strategies
    deadline_enabled = False
    swyh_enabled = False
    freeze_layers_enabled = False
    offload_enabled = False
    warmup_active = False

    exp_start_time = 0

    strategy = OffloadingStrategy.VANILLA


    # Keep track of the experiment data
    exp_data_general = []

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
        self.performance_data = {}

        logging.info("Creating test client")
        copy_sampler = config.data_sampler
        config.data_sampler = "uniform"
        self.test_data = Client("test", None, 1, 2, config)
        config.data_sampler = copy_sampler
        self.reference_lookup[get_worker_info().name] = RRef(self)
        self.strategy = OffloadingStrategy.Parse(config.offload_strategy)
        self.configure_strategy(self.strategy)



    def configure_strategy(self, strategy : OffloadingStrategy):
        if strategy == OffloadingStrategy.VANILLA:
            logging.info('Running with offloading strategy: VANILLA')
            self.deadline_enabled = False
            self.swyh_enabled = False
            self.freeze_layers_enabled = False
            self.offload_enabled = False
        if strategy == OffloadingStrategy.DEADLINE:
            logging.info('Running with offloading strategy: DEADLINE')
            self.deadline_enabled = True
            self.swyh_enabled = False
            self.freeze_layers_enabled = False
            self.offload_enabled = False
        if strategy == OffloadingStrategy.SWYH:
            logging.info('Running with offloading strategy: SWYH')
            self.deadline_enabled = True
            self.swyh_enabled = True
            self.freeze_layers_enabled = False
            self.offload_enabled = False
        if strategy == OffloadingStrategy.FREEZE:
            logging.info('Running with offloading strategy: FREEZE')
            self.deadline_enabled = True
            self.swyh_enabled = False
            self.freeze_layers_enabled = True
            self.offload_enabled = False
        if strategy == OffloadingStrategy.MODEL_OFFLOAD:
            logging.info('Running with offloading strategy: MODEL_OFFLOAD')
            self.deadline_enabled = True
            self.swyh_enabled = False
            self.freeze_layers_enabled = True
            self.offload_enabled = True
        logging.info(f'Offload strategy params: deadline={self.deadline_enabled}, swyh={self.swyh_enabled}, freeze={self.freeze_layers_enabled}, offload={self.offload_enabled}')

    def create_clients(self, client_id_triple):
        for id, rank, world_size in client_id_triple:
            client = rpc.remote(id, Client, kwargs=dict(id=id, log_rref=self.log_rref, rank=rank, world_size=world_size, config=self.config))
            writer = SummaryWriter(f'{self.tb_path}/{self.config.experiment_prefix}_client_{id}')
            self.clients.append(ClientRef(id, client, tensorboard_writer=writer))
            self.client_data[id] = []

    def select_clients(self, n = 2):
        available_clients = list(filter(lambda x : x.available, self.clients))
        return random_selection(available_clients, n)

    def ping_all(self):
        for client in self.clients:
            logging.info(f'Sending ping to {client}')
            t_start = time.time()
            answer = _remote_method(Client.ping, client.ref)
            t_end = time.time()
            duration = (t_end - t_start)*1000
            logging.info(f'Ping to {client} is {duration:.3}ms')

    def rpc_test_all(self):
        for client in self.clients:
            res = _remote_method_async(Client.rpc_test, client.ref)
            while not res.done():
                pass

    def client_load_data(self):
        for client in self.clients:
            _remote_method_async(Client.init_dataloader, client.ref)

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
                result, client_ref = res[1].wait()
                if result:
                    self.reference_lookup[res[0].name] = client_ref
                    logging.info(f'{res[0]} is ready')
                    ready_clients.append(res[0])
                    # Set the client to available
                    res[0].available = True
                else:
                    logging.info(f'Waiting for {res[0]}')
                    all_ready = False

            time.sleep(2)

        # WorkerInfo(id=1, name="client1").local_value()
        # rpc.rpc_sync(self.nameclients[0].ref.owner(), Client.ping, args=(self.clients[0].ref))
        logging.info(f'Sending a ping to client {self.clients[0].name}')
        r_ref = rpc.remote(self.clients[0].name, Client.static_ping, args=())
        print(f'Result of rref: {r_ref.to_here()}')
        logging.info('All clients are ready')
        for idx, c in enumerate(self.clients):
            logging.info(f'[{idx}]={c}')


    def perf_metric_endpoint(self, node_id, perf_data):
        if node_id not in self.performance_data.keys():
            self.performance_data[node_id] = []
        self.performance_data[node_id].append(perf_data)

    def perf_est_endpoint(self, node_id, performance_data):
        logging.info(f'Received performance estimate of node {node_id}')
        self.performance_estimate[node_id] = performance_data

    def send_clients_ref(self):

        for c in self.clients:
            # _remote_method_async(Client.send_reference, c.ref, rpc.get_worker_info())
            _remote_method_async(Client.send_reference, c.ref, RRef(self))

    def num_available_clients(self):
        return sum(c.available == True for c in self.clients)

    def process_response_list(self):
        for resp in self.response_list:
            if resp.future.done():
                resp.finish()
                resp.client.available = True
        self.response_list = list(filter(lambda x: not x.done, self.response_list))

    def ask_client_to_offload(self, client1_ref, client2_ref):
        logging.info(f'Offloading call from {client1_ref} to {client2_ref}')
        # args = [method, rref] + list(args)
        # rpc.rpc_sync(client1_ref, Client.call_to_offload_endpoint, args=(client2_ref))
        # print(_remote_method_async_by_name(Client.client_to_offload_to, client1_ref, client2_ref))
        _remote_method(Client.call_to_offload_endpoint, client1_ref, client2_ref)
        logging.info(f'Done with call to offload')

    def remote_run_epoch(self, epochs, warmup=False, first_epoch=False):
        if warmup:
            logging.info('This is a WARMUP round')
        start_epoch_time = time.time()
        deadline = self.config.deadline
        deadline_time = self.config.deadline
        if first_epoch:
            deadline = self.config.first_deadline
            deadline_time = self.config.first_deadline
        """
        1. Client selection
        2. Run local updates
        3. Retrieve data
        4. Aggregate data
        """

        client_weights = []

        client_weights_dict = {}
        client_training_process_dict = {}
        while self.num_available_clients() < self.config.clients_per_round:
            logging.warning(f'Waiting for enough clients to become available. # Available Clients = {self.num_available_clients()}, but need {self.config.clients_per_round}')
            self.process_response_list()
            time.sleep(1)

        #### Client Selection ####
        selected_clients = self.select_clients(self.config.clients_per_round)

        #### Send model to clients ####
        responses = []
        for client in selected_clients:
            logging.info(f'Send updated model to selected client: {client.name}')
            responses.append(
                (client, _remote_method_async(Client.update_nn_parameters, client.ref, new_params=self.test_data.get_nn_parameters())))

        for res in responses:
            res[1].wait()
        logging.info('Weights are updated')

        # Let clients train locally

        if not self.deadline_enabled:
            deadline = 0
        responses: List[ClientResponse] = []
        for client in selected_clients:
            cr = ClientResponse(self.response_id, client, _remote_method_async(Client.run_epochs, client.ref, num_epoch=epochs, deadline=deadline, warmup=warmup))
            self.response_id += 1
            self.response_list.append(cr)
            responses.append(cr)
            client.available = False
            # responses.append((client, time.time(), _remote_method_async(Client.run_epochs, client.ref, num_epoch=epochs)))
        self.epoch_counter += epochs

        # deadline_time = None
        # Wait loop with deadline
        start = time.time()
        def reached_deadline():
            if deadline_time is None:
                return False
            # logging.info(f'{(time.time() - start)} >= {deadline_time}')
            return (time.time() -start) >= deadline_time

        logging.info('Starting waiting period')
        # Wait loop without deadline
        all_finished = False

        # Debug for testing!
        has_not_called = True

        show_perf_data = True
        while not all_finished and not ((self.deadline_enabled and reached_deadline()) or warmup):
            # if self.deadline_enabled and reached_deadline()
            # if has_not_called and (time.time() -start) > 10:
            #     logging.info('Sending call to offload')
            #     has_not_called = False
            #
            #     self.ask_client_to_offload(self.reference_lookup[selected_clients[0].name], selected_clients[1].name)

            # Check if all performance data has come in
            has_all_perf_data = True

            if show_perf_data:
                for sc in selected_clients:
                    if sc.name not in self.performance_estimate.keys():
                        has_all_perf_data = False

                if has_all_perf_data:
                    logging.info('Got all performance data')
                    print(self.performance_estimate)
                    show_perf_data = False

                    # Make offloading call
                    # @NOTE: this will only work for the two node scenario

                    lowest_est_time = 0
                    est_keys = list(self.performance_estimate.keys())

                    # for k, v in self.performance_estimate.items():
                    #     if v[1] > lowest_est_time:
                    #         lowest_est_time = v[1]
                    #         weak_client = k
                    #     else:
                    #         strong_client = k
                    if self.offload_enabled and not warmup:
                        weak_client = est_keys[0]
                        strong_client = est_keys[1]
                        if self.performance_estimate[est_keys[1]][1] > self.performance_estimate[est_keys[0]][1]:
                            weak_client = est_keys[1]
                            strong_client = est_keys[0]

                        logging.info(f'Offloading from {weak_client} -> {strong_client} due to {self.performance_estimate[weak_client]} and {self.performance_estimate[strong_client]}')
                        logging.info('Sending call to offload')
                        self.ask_client_to_offload(self.reference_lookup[selected_clients[0].name], selected_clients[1].name)

                # selected_clients[0]
            # logging.info(f'Status of all_finished={all_finished} and deadline={reached_deadline()}')
            all_finished = True
            for client_response in responses:
                if client_response.future.done():
                    if not client_response.done:
                        client_response.finish()
                else:
                    all_finished = False
            time.sleep(0.1)
        logging.info(f'Stopped waiting due to all_finished={all_finished} and deadline={reached_deadline()}')
        for client_response in responses:
            if warmup:
                break
            client = client_response.client
            logging.info(f'{client} had a exec time of {client_response.duration()} dropped?={client_response.dropped}')
            if client_response.dropped:
                client_response.end_time = time.time()
                logging.info(
                    f'{client} had a exec time of {client_response.duration()} dropped?={client_response.dropped}')

            if not client_response.dropped:
                client.available = True
                epoch_data, weights = client_response.future.wait()
                self.client_data[epoch_data.client_id].append(epoch_data)
                logging.info(f'{client} had a loss of {epoch_data.loss}')
                logging.info(f'{client} had a epoch data of {epoch_data}')
                logging.info(f'{client} has trained on {epoch_data.training_process} samples')
                elapsed_time = client_response.end_time - self.exp_start_time

                client.tb_writer.add_scalar('training loss',
                                            epoch_data.loss_train,  # for every 1000 minibatches
                                            self.epoch_counter * client.data_size)

                client.tb_writer.add_scalar('accuracy',
                                            epoch_data.accuracy,  # for every 1000 minibatches
                                            self.epoch_counter * client.data_size)

                client.tb_writer.add_scalar('accuracy wall time',
                                            epoch_data.accuracy,  # for every 1000 minibatches
                                            elapsed_time)
                client.tb_writer.add_scalar('training loss per epoch',
                                            epoch_data.loss_train,  # for every 1000 minibatches
                                            self.epoch_counter)

                client.tb_writer.add_scalar('accuracy per epoch',
                                            epoch_data.accuracy,  # for every 1000 minibatches
                                            self.epoch_counter)

                client_weights.append(weights)
                client_weights_dict[client.name] = weights
                client_training_process_dict[client.name] = epoch_data.training_process

        self.performance_estimate = {}
        if len(client_weights):
            updated_model = FedAvg(client_weights_dict, client_training_process_dict)
            # updated_model = average_nn_parameters(client_weights)

            # test global model
            logging.info("Testing on global test set")
            self.test_data.update_nn_parameters(updated_model)
        accuracy, loss, class_precision, class_recall = self.test_data.test()
        # self.tb_writer.add_scalar('training loss', loss, self.epoch_counter * self.test_data.get_client_datasize()) # does not seem to work :( )
        self.tb_writer.add_scalar('accuracy', accuracy, self.epoch_counter * self.test_data.get_client_datasize())
        self.tb_writer.add_scalar('accuracy per epoch', accuracy, self.epoch_counter)
        elapsed_time = time.time() - self.exp_start_time
        self.tb_writer.add_scalar('accuracy wall time',
                                    accuracy,  # for every 1000 minibatches
                                    elapsed_time)
        end_epoch_time = time.time()
        duration = end_epoch_time - start_epoch_time

        self.exp_data_general.append([self.epoch_counter, duration, accuracy, loss, class_precision, class_recall])


    def save_experiment_data(self):
        p = Path(f'./{self.config.output_location}')
        # file_output = f'./{self.config.output_location}'
        exp_prefix = self.config.experiment_prefix
        self.ensure_path_exists(p)
        p /= f'{exp_prefix}-general_data.csv'
        # general_filename = f'{file_output}/general_data.csv'
        df = pd.DataFrame(self.exp_data_general, columns=['epoch', 'duration', 'accuracy', 'loss', 'class_precision', 'class_recall'])
        df.to_csv(p)

    def update_client_data_sizes(self):
        responses = []
        for client in self.clients:
            responses.append((client, _remote_method_async(Client.get_client_datasize, client.ref)))
        for res in responses:
            res[0].data_size = res[1].wait()
            logging.info(f'{res[0]} had a result of datasize={res[0].data_size}')
            # @TODO: Use datasize in aggregation method

    def remote_test_sync(self):
        responses = []
        for client in self.clients:
            responses.append((client, _remote_method_async(Client.test, client.ref)))

        for res in responses:
            accuracy, loss, class_precision, class_recall = res[1].wait()
            logging.info(f'{res[0]} had a result of accuracy={accuracy}')

    def save_epoch_data(self):
        file_output = f'./{self.config.output_location}'
        exp_prefix = self.config.experiment_prefix
        self.ensure_path_exists(file_output)
        for key in self.client_data:
            filename = f'{file_output}/{exp_prefix}_{key}_epochs.csv'
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



        Steps in federated learning process

        1. Client selection
        2. Run local updates
        3. Retrieve data
        4. Aggregate data
        """
        # # Make sure the clients have loaded all the data
        self.send_clients_ref()
        self.client_load_data()
        self.test_data.init_dataloader()
        self.ping_all()
        self.clients_ready()
        self.update_client_data_sizes()

        epoch_to_run = self.num_epoch
        addition = 0
        epoch_to_run = self.config.epochs
        epoch_size = self.config.epochs_per_cycle

        if self.config.warmup_round:
            logging.info('Running warmup round')
            self.remote_run_epoch(epoch_size, warmup=True)

        self.exp_start_time = time.time()
        for epoch in range(epoch_to_run):
            self.process_response_list()
            logging.info(f'Running epoch {epoch}')
            self.remote_run_epoch(epoch_size)
            addition += 1


        logging.info(f'Saving data')
        self.save_epoch_data()
        self.save_experiment_data()

        # Ignore profiler for now
        # logging.info(f'Reporting profile data')
        # for key in self.performance_data.keys():
        #     parse_stability_data(self.performance_data[key], save_to_file=True)
        logging.info(f'Federator is stopping')

