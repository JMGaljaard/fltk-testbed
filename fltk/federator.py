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
from fltk.strategy.client_selection import random_selection, tifl_update_probs, tifl_select_tier_and_decrement
from fltk.strategy.offloading import OffloadingStrategy, parse_strategy
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
import numpy as np
import copy

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
    tb_writer_offload = None
    available = False
    rank=None

    def __init__(self, name, ref, tensorboard_writer, tensorboard_writer_offload, rank):
        self.name = name
        self.ref = ref
        self.tb_writer = tensorboard_writer
        self.tb_writer_offload = tensorboard_writer_offload
        self.rank = rank

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
    terminated = False

    def finish(self):
        self.end_time = time.time()
        self.done = True
        self.dropped = False
        print(f'>>>> \t\tClient {self.id} has a duration of {self.duration()}')

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
    dyn_terminate = False
    dyn_terminate_swyh = False
    warmup_active = False
    node_groups = {}
    tifl_tier_data = []
    tifl_tier_names = []
    tifl_selected_tier = ''

    exp_start_time = 0

    strategy = OffloadingStrategy.VANILLA


    # Keep track of the experiment data
    exp_data_general = []

    epoch_events = []

    def __init__(self, client_id_triple, num_epochs = 3, config=None):
        log_rref = rpc.RRef(FLLogger())
        self.log_rref = log_rref
        self.num_epoch = num_epochs
        self.config = config
        self.tb_path = f'{config.output_location}/{config.experiment_prefix}'
        self.ensure_path_exists(self.tb_path)
        self.tb_writer = SummaryWriter(f'{self.tb_path}/{config.experiment_prefix}_federator')
        self.strategy = OffloadingStrategy.Parse(config.offload_strategy)
        self.configure_strategy(self.strategy)
        self.create_clients(client_id_triple)
        self.config.init_logger(logging)
        self.performance_data = {}

        logging.info("Creating test client")
        copy_sampler = config.data_sampler
        config.data_sampler = "uniform"
        self.test_data = Client("test", None, 1, 2, config)
        config.data_sampler = copy_sampler
        self.reference_lookup[get_worker_info().name] = RRef(self)
        
        if self.strategy == OffloadingStrategy.TIFL_BASIC or self.strategy == OffloadingStrategy.TIFL_ADAPTIVE:
            for k, v in self.config.node_groups.items():
                self.node_groups[k] = list(range(v[0], v[1]+1))
                self.tifl_tier_names.append(k)

        if self.strategy == OffloadingStrategy.TIFL_ADAPTIVE:
            num_tiers = len(self.tifl_tier_names) * 1.0
            start_credits = np.ceil(self.config.epochs / num_tiers)
            logging.info(f'Tifl starting credits is {start_credits}')
            for tier_name in self.tifl_tier_names:
                self.tifl_tier_data.append([tier_name, 0, start_credits, 1 / num_tiers])
            residue = 1
            for t in self.tifl_tier_data:
                residue -= t[3]
            self.tifl_tier_data[0][3] += residue

    # def configure_strategy(self, strategy : OffloadingStrategy):
    #     if strategy == OffloadingStrategy.VANILLA:
    #         logging.info('Running with offloading strategy: VANILLA')
    #         self.deadline_enabled = False
    #         self.swyh_enabled = False
    #         self.freeze_layers_enabled = False
    #         self.offload_enabled = False
    #     if strategy == OffloadingStrategy.DEADLINE:
    #         logging.info('Running with offloading strategy: DEADLINE')
    #         self.deadline_enabled = True
    #         self.swyh_enabled = False
    #         self.freeze_layers_enabled = False
    #         self.offload_enabled = False
    #     if strategy == OffloadingStrategy.SWYH:
    #         logging.info('Running with offloading strategy: SWYH')
    #         self.deadline_enabled = True
    #         self.swyh_enabled = True
    #         self.freeze_layers_enabled = False
    #         self.offload_enabled = False
    #     if strategy == OffloadingStrategy.FREEZE:
    #         logging.info('Running with offloading strategy: FREEZE')
    #         self.deadline_enabled = True
    #         self.swyh_enabled = False
    #         self.freeze_layers_enabled = True
    #         self.offload_enabled = False
    #     if strategy == OffloadingStrategy.MODEL_OFFLOAD:
    #         logging.info('Running with offloading strategy: MODEL_OFFLOAD')
    #         self.deadline_enabled = True
    #         self.swyh_enabled = False
    #         self.freeze_layers_enabled = True
    #         self.offload_enabled = True
    #     if strategy == OffloadingStrategy.TIFL_BASIC:
    #         logging.info('Running with offloading strategy: TIFL_BASIC')
    #         self.deadline_enabled = False
    #         self.swyh_enabled = False
    #         self.freeze_layers_enabled = False
    #         self.offload_enabled = False
    #     logging.info(f'Offload strategy params: deadline={self.deadline_enabled}, swyh={self.swyh_enabled}, freeze={self.freeze_layers_enabled}, offload={self.offload_enabled}')
    #
    def configure_strategy(self, strategy : OffloadingStrategy):
        deadline_enabled, swyh_enabled, freeze_layers_enabled, offload_enabled, dyn_terminate, dyn_terminate_swyh = parse_strategy(strategy)
        self.deadline_enabled = deadline_enabled
        self.swyh_enabled = swyh_enabled
        self.freeze_layers_enabled = freeze_layers_enabled
        self.offload_enabled = offload_enabled
        self.dyn_terminate = dyn_terminate
        self.dyn_terminate_swyh = dyn_terminate_swyh
        logging.info(f'Offloading strategy={strategy}')
        logging.info(f'Offload strategy params: deadline={self.deadline_enabled}, '
                     f'swyh={self.swyh_enabled}, freeze={self.freeze_layers_enabled}, '
                     f'offload={self.offload_enabled}, dyn_terminate={self.dyn_terminate}, '
                     f'dyn_terminate_swyh={self.dyn_terminate_swyh}')

    def create_clients(self, client_id_triple):
        for id, rank, world_size in client_id_triple:
            client = rpc.remote(id, Client, kwargs=dict(id=id, log_rref=self.log_rref, rank=rank, world_size=world_size, config=self.config))
            writer = SummaryWriter(f'{self.tb_path}/{self.config.experiment_prefix}_client_{id}')
            writer_offload = None
            if self.offload_enabled:
                writer_offload = SummaryWriter(f'{self.tb_path}/{self.config.experiment_prefix}_client_{id}_offload')
            self.clients.append(ClientRef(id, client, tensorboard_writer=writer, tensorboard_writer_offload=writer_offload, rank=rank))
            self.client_data[id] = []

    def record_epoch_event(self, event: str):
        self.epoch_events.append(f'{time.time()} - [{self.epoch_counter}] - {event}')

    def select_clients(self, n = 2):
        available_clients = list(filter(lambda x : x.available, self.clients))
        if self.strategy == OffloadingStrategy.TIFL_ADAPTIVE:
            tifl_update_probs(self.tifl_tier_data)
            self.tifl_selected_tier = tifl_select_tier_and_decrement(self.tifl_tier_data)
            client_subset = self.node_groups[self.tifl_selected_tier]
            available_clients = list(filter(lambda x: x.rank in client_subset, self.clients))
        if self.strategy == OffloadingStrategy.TIFL_BASIC:
            self.tifl_selected_tier = np.random.choice(list(self.node_groups.keys()), 1, replace=False)[0]
            logging.info(f'TIFL: Sampling from group {self.tifl_selected_tier} out of{list(self.node_groups.keys())}')
            client_subset = self.node_groups[self.tifl_selected_tier]
            available_clients = list(filter(lambda x : x.rank in client_subset, self.clients))
            logging.info(f'TIFL: Sampling subgroup {available_clients}')
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

    def ask_client_to_offload(self, client1_ref, client2_ref, soft_deadline):
        logging.info(f'Offloading call from {client1_ref} to {client2_ref}')
        # args = [method, rref] + list(args)
        # rpc.rpc_sync(client1_ref, Client.call_to_offload_endpoint, args=(client2_ref))
        # print(_remote_method_async_by_name(Client.client_to_offload_to, client1_ref, client2_ref))
        _remote_method(Client.call_to_offload_endpoint, client1_ref, client2_ref, soft_deadline)
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


        self.record_epoch_event('Starting new round')
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

        ### Clients train locally
        # Structure of the async message:
        # - Client will respond with two messages:

        # Let clients train locally

        if not self.deadline_enabled:
            deadline = 0
        responses: List[ClientResponse] = []
        for client in selected_clients:
            cr = ClientResponse(self.response_id, client, _remote_method_async(Client.run_epochs, client.ref, num_epoch=epochs, deadline=deadline, warmup=warmup))
            cr.start_time = time.time()
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
        has_send_terminate = False
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

                    # (time_per_batch, est_total_time, number_of_training_samples)
                    if self.offload_enabled and not warmup:
                        first = True
                        weakest = 0
                        strongest = 0
                        weak_performance = 0
                        strong_performance = 0
                        summed_time = 0
                        perf_estimate_copy = copy.deepcopy(self.performance_estimate)
                        offload_calls = []
                        for i in range(int(np.floor(len(self.performance_estimate)/2))):
                            for k, v in perf_estimate_copy.items():
                                summed_time += v[1]
                                # print(v)
                                if first:
                                    first = False
                                    est_total_time = v[1]
                                    weakest = k
                                    strongest = k
                                    weak_performance = est_total_time
                                    strong_performance = est_total_time
                                else:
                                    est_total_time = v[1]
                                    if est_total_time > weak_performance:
                                        weak_performance = est_total_time
                                        weakest = k
                                    if est_total_time < strong_performance:
                                        strong_performance = est_total_time
                                        strongest = k
                            self.record_epoch_event(f'Offloading from {weakest} -> {strongest} due to {self.performance_estimate[weakest]} and {self.performance_estimate[strongest]}')
                            logging.info(
                                f'Offloading from {weakest} -> {strongest} due to {self.performance_estimate[weakest]} and {self.performance_estimate[strongest]}')
                            offload_calls.append([weakest, strongest])
                            perf_estimate_copy.pop(weakest, None)
                            perf_estimate_copy.pop(strongest, None)
                        mean_time_est_time = (summed_time * 1.0) / len(self.performance_estimate.items())
                        logging.info(f'Mean time for offloading={mean_time_est_time}')
                        logging.info('Sending call to offload')
                        for weak_node, strong_node in offload_calls:
                            self.ask_client_to_offload(self.reference_lookup[weak_node], strong_node, mean_time_est_time)
                        logging.info('Releasing clients')
                        for client in selected_clients:
                            _remote_method_async(Client.release_from_offloading_endpoint, client.ref)

                    # if self.offload_enabled and not warmup:
                    #     logging.info(f'self.performance_estimate={self.performance_estimate}')
                    #     logging.info(f'est_keys={est_keys}')
                    #     weak_client = est_keys[0]
                    #     strong_client = est_keys[1]
                    #     if self.performance_estimate[est_keys[1]][1] > self.performance_estimate[est_keys[0]][1]:
                    #         weak_client = est_keys[1]
                    #         strong_client = est_keys[0]
                    #
                    #     logging.info(f'Offloading from {weak_client} -> {strong_client} due to {self.performance_estimate[weak_client]} and {self.performance_estimate[strong_client]}')
                    #     logging.info('Sending call to offload')
                    #     self.ask_client_to_offload(self.reference_lookup[selected_clients[0].name], selected_clients[1].name)

                # selected_clients[0]
            # logging.info(f'Status of all_finished={all_finished} and deadline={reached_deadline()}')
            all_finished = True

            for client_response in responses:
                if client_response.future.done():
                    if not client_response.done:
                        client_response.finish()
                else:
                    all_finished = False
            if not has_send_terminate and (self.dyn_terminate or self.dyn_terminate_swyh):
                num_finished_responses = sum([1 for x in responses if x.done])
                percentage = num_finished_responses / len(responses)
                if percentage > self.config.termination_percentage:
                    logging.info('Sending termination signal')
                    for cr in responses:
                        if not cr.done:
                            if self.dyn_terminate:
                                cr.terminated = True
                            _remote_method_async(Client.terminate_training_endpoint, cr.client.ref)
                    has_send_terminate = True
                logging.info(f'Percentage of finished responses: {percentage}, do terminate ? {percentage} > {self.config.termination_percentage} = {percentage > self.config.termination_percentage}')
            time.sleep(0.1)
        logging.info(f'Stopped waiting due to all_finished={all_finished} and deadline={reached_deadline()}')
        client_accuracies = []
        for client_response in responses:
            if warmup:
                break
            client = client_response.client
            logging.info(f'{client} had a exec time of {client_response.duration()} dropped?={client_response.dropped}')
            if client_response.dropped:
                client_response.end_time = time.time()
                logging.info(
                    f'{client} had a exec time of {client_response.duration()} dropped?={client_response.dropped}')

            if not client_response.dropped and not client_response.terminated:
                client.available = True
                logging.info(f'Fetching response for client: {client}')
                response_obj = client_response.future.wait()

                epoch_data, weights, scheduler_data, perf_data = response_obj['own']
                self.client_data[epoch_data.client_id].append(epoch_data)

                # logging.info(f'{client} had a loss of {epoch_data.loss}')
                # logging.info(f'{client} had a epoch data of {epoch_data}')
                # logging.info(f'{client} has trained on {epoch_data.training_process} samples')
                self.record_epoch_event(f'{client} had an accuracy of {epoch_data.accuracy}')
                self.record_epoch_event(f'{client} had an duration of {client_response.duration()}')
                client_accuracies.append(epoch_data.accuracy)
                # logging.info(f'{client} has perf data: {perf_data}')
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

                client.tb_writer.add_scalar('Client time per epoch',
                                            client_response.duration(),  # for every 1000 minibatches
                                            self.epoch_counter)

                client.tb_writer.add_scalar('learning rate',
                                            scheduler_data['lr'],
                                            self.epoch_counter)

                client.tb_writer.add_scalar('momentum',
                                            scheduler_data['momentum'],
                                            self.epoch_counter)

                client.tb_writer.add_scalar('weight decay',
                                            scheduler_data['wd'],
                                            self.epoch_counter)
                total_time_t1 = perf_data['total_duration']
                loop_duration = perf_data['loop_duration']
                p_v1_time = perf_data['p_v1_data'].mean() * perf_data['n_batches']
                p_v1_time_sum = perf_data['p_v1_data'].sum()
                p_v1_pre_loop = perf_data['p_v1_pre_loop']
                p_v1_post_loop = perf_data['p_v1_post_loop']
                pre_train_loop_data = perf_data['pre_train_loop_data']
                post_train_loop_data = perf_data['post_train_loop_data']
                p_v2_forwards = (perf_data['p_v2_data'][0].mean() + perf_data['p_v2_data'][1].mean()) * perf_data['n_batches']
                p_v2_backwards = (perf_data['p_v2_data'][2].mean() + perf_data['p_v2_data'][3].mean()) * perf_data['n_batches']
                p_v3_forwards = (perf_data['p_v3_data'][0].mean() + perf_data['p_v3_data'][1].mean()) * perf_data[
                    'n_batches']
                p_v3_backwards = (perf_data['p_v3_data'][2].mean() + perf_data['p_v3_data'][3].mean()) * perf_data[
                    'n_batches']
                p_v2_time = sum([x.mean() for x in perf_data['p_v2_data']]) * perf_data['n_batches']
                p_v1_forwards = perf_data['p_v1_forwards'].mean() * perf_data['n_batches']
                p_v1_backwards = perf_data['p_v1_backwards'].mean() * perf_data['n_batches']

                # logging.info(f'{client} has time estimates: {[total_time_t1, loop_duration, p_v1_time_sum, p_v1_time, p_v2_time, [p_v1_forwards, p_v1_backwards], [p_v2_forwards, p_v2_backwards]]}')
                # logging.info(f'{client} combined times pre post loop stuff: {[p_v1_pre_loop, loop_duration, p_v1_post_loop]} = {sum([p_v1_pre_loop, loop_duration, p_v1_post_loop])} ? {total_time_t1}')
                # logging.info(f'{client} p3 time = {p_v3_forwards} + {p_v3_backwards} = {p_v3_forwards+ p_v3_backwards}')
                # logging.info(f'{client} Pre train loop time = {pre_train_loop_data.mean()}, post train loop time = {post_train_loop_data.mean()}')
                # logging.info(f'{client} p_v1 data: {perf_data["p_v1_data"]}')



                client.tb_writer.add_scalar('train_time_estimate_delta', loop_duration - (p_v3_forwards+ p_v3_backwards), self.epoch_counter)
                client.tb_writer.add_scalar('train_time_estimate_delta_2', loop_duration - (p_v2_forwards+ p_v2_backwards), self.epoch_counter)

                client_weights.append(weights)
                client_weights_dict[client.name] = weights
                client_training_process_dict[client.name] = epoch_data.training_process

                if self.strategy == OffloadingStrategy.TIFL_ADAPTIVE:
                    mean_tier_accuracy = np.mean(client_accuracies)
                    logging.info(f'TIFL:: the mean accuracy is {mean_tier_accuracy}')
                    for t in self.tifl_tier_data:
                        if t[0] == self.tifl_selected_tier:
                            t[1] = mean_tier_accuracy

                if 'offload' in response_obj:
                    epoch_data_offload, weights_offload, scheduler_data_offload, perf_data_offload, sender_id = response_obj['offload']
                    if epoch_data_offload.client_id not in self.client_data:
                        self.client_data[epoch_data_offload.client_id] = []
                    self.client_data[epoch_data_offload.client_id].append(epoch_data_offload)

                    writer = client.tb_writer_offload

                    writer.add_scalar('training loss',
                                                epoch_data_offload.loss_train,  # for every 1000 minibatches
                                                self.epoch_counter * client.data_size)

                    writer.add_scalar('accuracy',
                                                epoch_data_offload.accuracy,  # for every 1000 minibatches
                                                self.epoch_counter * client.data_size)

                    writer.add_scalar('accuracy wall time',
                                                epoch_data_offload.accuracy,  # for every 1000 minibatches
                                                elapsed_time)
                    writer.add_scalar('training loss per epoch',
                                                epoch_data_offload.loss_train,  # for every 1000 minibatches
                                                self.epoch_counter)

                    writer.add_scalar('accuracy per epoch',
                                                epoch_data_offload.accuracy,  # for every 1000 minibatches
                                                self.epoch_counter)

                    writer.add_scalar('Client time per epoch',
                                                client_response.duration(),  # for every 1000 minibatches
                                                self.epoch_counter)

                    writer.add_scalar('learning rate',
                                                scheduler_data_offload['lr'],
                                                self.epoch_counter)

                    writer.add_scalar('momentum',
                                                scheduler_data_offload['momentum'],
                                                self.epoch_counter)

                    writer.add_scalar('weight decay',
                                                scheduler_data_offload['wd'],
                                                self.epoch_counter)
                    client_weights.append(weights_offload)
                    client_weights_dict[epoch_data_offload.client_id] = weights_offload
                    client_training_process_dict[epoch_data_offload.client_id] = epoch_data_offload.training_process

        self.performance_estimate = {}
        if len(client_weights):
            logging.info(f'Aggregating {len(client_weights)} models')
            updated_model = FedAvg(client_weights_dict, client_training_process_dict)
            # updated_model = average_nn_parameters(client_weights)

            # test global model
            logging.info("Testing on global test set")
            self.test_data.update_nn_parameters(updated_model)
        accuracy, loss, class_precision, class_recall, accuracy_per_class = self.test_data.test()
        # logging.info('Class precision')
        # logging.warning(accuracy_per_class)
        # logging.info('Class names')
        # logging.info(self.test_data.dataset.test_dataset.class_to_idx)
        # self.tb_writer.add_scalar('training loss', loss, self.epoch_counter * self.test_data.get_client_datasize()) # does not seem to work :( )
        self.tb_writer.add_scalar('Number of clients dropped', sum([1 for x in responses if x.dropped or x.terminated]), self.epoch_counter)

        self.tb_writer.add_scalar('accuracy', accuracy, self.epoch_counter * self.test_data.get_client_datasize())
        self.record_epoch_event(f'Global accuracy is {accuracy}')
        self.tb_writer.add_scalar('accuracy per epoch', accuracy, self.epoch_counter)
        elapsed_time = time.time() - self.exp_start_time
        self.tb_writer.add_scalar('accuracy wall time',
                                    accuracy,  # for every 1000 minibatches
                                    elapsed_time)

        class_acc_dict = {}
        for idx, acc in enumerate(accuracy_per_class):
            class_acc_dict[f'{idx}'] = acc
        self.tb_writer.add_scalars('accuracy per class', class_acc_dict, self.epoch_counter)
        self.record_epoch_event(f'Accuracy per class is {class_acc_dict}')
        end_epoch_time = time.time()
        duration = end_epoch_time - start_epoch_time


        self.exp_data_general.append([self.epoch_counter, end_epoch_time, duration, accuracy, loss, class_precision, class_recall])


    def set_tau_eff(self):
        total = sum(client.data_size for client in self.clients)
        responses = []
        for client in self.clients:
            responses.append((client, _remote_method_async(Client.set_tau_eff, client.ref, total)))
        torch.futures.wait_all([x[1] for x in responses])
        # for client in self.clients:
        #     client.set_tau_eff(total)

    def save_experiment_data(self):
        p = Path(f'./{self.tb_path}')
        # file_output = f'./{self.config.output_location}'
        exp_prefix = self.config.experiment_prefix
        self.ensure_path_exists(p)
        p /= f'{exp_prefix}-general_data.csv'
        # general_filename = f'{file_output}/general_data.csv'
        df = pd.DataFrame(self.exp_data_general, columns=['epoch', 'wall_time', 'duration', 'accuracy', 'loss', 'class_precision', 'class_recall'])
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

    def flush_epoch_events(self):
        file_output = f'./{self.tb_path}'
        exp_prefix = self.config.experiment_prefix
        file_epoch_events = f'{file_output}/{exp_prefix}_federator_events.txt'
        self.ensure_path_exists(file_output)

        with open(file_epoch_events, 'a') as f:
            for ev in self.epoch_events:
                f.write(f'{ev}\n')
            f.flush()

        self.epoch_events = []

    def save_epoch_data(self):
        file_output = f'./{self.tb_path}'
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
        self.set_tau_eff()

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
            self.flush_epoch_events()
            addition += 1


        logging.info(f'Saving data')
        self.save_epoch_data()
        self.save_experiment_data()

        # Ignore profiler for now
        # logging.info(f'Reporting profile data')
        # for key in self.performance_data.keys():
        #     parse_stability_data(self.performance_data[key], save_to_file=True)
        logging.info(f'Federator is stopping')

