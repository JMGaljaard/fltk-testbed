from __future__ import annotations

import copy
import numpy as np
import sklearn
import time
from dataclasses import dataclass
from pathlib import Path
from typing import List, Union, Tuple

import torch

from fltk.core.client import Client
from fltk.core.node import Node
from fltk.strategy import get_aggregation
from fltk.strategy import random_selection

from fltk.util.data_container import DataContainer, FederatorRecord, ClientRecord

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from fltk.util.config import FedLearningConfig
NodeReference = Union[Node, str]


@dataclass
class LocalClient:
    """
    Dataclass for local execution references to 'virtual' Clients.
    """
    name: str
    ref: NodeReference
    data_size: int
    exp_data: DataContainer


def cb_factory(future: torch.Future, method, *args, **kwargs):  # pylint: disable=no-member
    """
    Callback factory function to attach callbacks to a future.
    @param future: Future promise for remote function.
    @type future: torch.Future.
    @param method: Callable method to call on a remote.
    @type method: Callable
    @param args: Arguments to pass to the callback function.
    @type args: List[Any]
    @param kwargs: Keyword arguments to pass to the callback function.
    @type kwargs: Dict[str, Any]
    @return: None
    @rtype: None
    """
    future.then(lambda x: method(x, *args, **kwargs))


class Federator(Node):
    """
    Federator implementation that governs the (possibly) distributed learning process. Learning is initiated by the
    Federator and performed by the Clients. The Federator also performs centralized logging for easier execution.
    """
    clients: List[LocalClient] = []
    # clients: List[NodeReference] = []
    num_rounds: int
    exp_data: DataContainer

    def __init__(self, identifier: str, rank: int, world_size: int, config: FedLearningConfig):
        super().__init__(identifier, rank, world_size, config)
        self.loss_function = self.config.get_loss_function()()
        self.num_rounds = config.rounds
        self.config = config
        prefix_text = ''
        if config.replication_id:
            prefix_text = f'_r{config.replication_id}'
        config.output_path = Path(config.output_path) / f'{config.experiment_prefix}{prefix_text}'
        self.exp_data = DataContainer('federator', config.output_path, FederatorRecord, config.save_data_append)
        self.aggregation_method = get_aggregation(config.aggregation)

    def create_clients(self):
        """
        Function to create references to all the clients that will perform the learning process.
        @return: None.
        @rtype: None
        """
        self.logger.info('Creating clients')
        if self.config.single_machine:
            # Create direct clients
            world_size = self.config.num_clients + 1
            for client_id in range(1, self.config.world_size):
                client_name = f'client{client_id}'
                client = Client(client_name, client_id, world_size, copy.deepcopy(self.config))
                self.clients.append(
                        LocalClient(client_name, client, 0, DataContainer(client_name, self.config.output_path,
                                                                          ClientRecord, self.config.save_data_append)))
                self.logger.info(f'Client "{client_name}" created')

    def register_client(self, client_name: str, rank: int):
        """
        Function to be called by remote Client to register the learner to the Federator.
        @param client_name: Name of the client.
        @type client_name: str
        @param rank: Rank of the client that registers.
        @type rank: int
        @return: None.
        @rtype: None
        """
        self.logger.info(f'Got new client registration from client {client_name}')
        if self.config.single_machine:
            self.logger.warning('This function should not be called when in single machine mode!')
        self.clients.append(
                LocalClient(client_name, client_name, rank, DataContainer(client_name, self.config.output_path,
                                                                          ClientRecord, self.config.save_data_append)))

    def stop_all_clients(self):
        """
        Method to stop all running clients that were registered by the Federator during initialziation.
        @return: None.
        @rtype: None
        """
        for client in self.clients:
            self.message(client.ref, Client.stop_client)

    def _num_clients_online(self) -> int:
        return len(self.clients)

    def _all_clients_online(self) -> bool:
        return len(self.clients) == self.world_size - 1

    def clients_ready(self):
        """
        Synchronous implementation to wait for all remote Clients to register themselves to the Federator.
        """
        all_ready = False
        ready_clients = []
        while not all_ready:
            responses = []
            all_ready = True
            for client in self.clients:
                resp = self.message(client.ref, Client.is_ready)
                if resp:
                    self.logger.info(f'Client {client} is ready')
                else:
                    self.logger.info(f'Waiting for client {client}')
                    all_ready = False
            time.sleep(2)

    def get_client_data_sizes(self):
        """
        Function to request the dataset sizes of the Clients training DataLoaders.
        @return: None.
        @rtype: None
        """
        for client in self.clients:
            client.data_size = self.message(client.ref, Client.get_client_datasize)

    def run(self):
        """
        Spinner function to perform the experiment that was provided by either the Orhcestrator in Kubernetes or the
        generated docker-compose file in case running in Docker.
        @return: None.
        @rtype: None.
        """
        # Load dataset with world size 2 to load the whole dataset.
        # Caused by the fact that the dataloader subtracts 1 from the world size to exclude the federator by default.
        self.init_dataloader(world_size=2)

        self.create_clients()
        while not self._all_clients_online():
            msg = f'Waiting for all clients to come online. ' \
                  f'Waiting for {self.world_size - 1 - self._num_clients_online()} clients'
            self.logger.info(msg)
            time.sleep(2)
        self.logger.info('All clients are online')
        # self.logger.info('Running')
        # time.sleep(10)
        self.client_load_data()
        self.get_client_data_sizes()
        self.clients_ready()
        # self.logger.info('Sleeping before starting communication')
        # time.sleep(20)
        for communication_round in range(self.config.rounds):
            self.exec_round(communication_round)

        self.save_data()
        self.logger.info('Federator is stopping')

    def save_data(self):
        """
        Function to store all data obtained from the Clients during their training efforts.
        @return: None.
        @rtype: None
        """
        self.exp_data.save()
        for client in self.clients:
            client.exp_data.save()

    def client_load_data(self):
        """
        Function to contact all clients to initialize their dataloaders, to be called to prepare the Clients for the
        learning loop.
        @return: None.
        @rtype: None
        """
        for client in self.clients:
            self.message(client.ref, Client.init_dataloader)

    def set_tau_eff(self):
        total = sum(client.data_size for client in self.clients)
        # responses = []
        for client in self.clients:
            self.message(client.ref, Client.set_tau_eff, client.ref, total)
            # responses.append((client, _remote_method_async(Client.set_tau_eff, client.ref, total)))
        # torch.futures.wait_all([x[1] for x in responses])

    def test(self, net) -> Tuple[float, float, np.array]:
        """
        Function to test the learned global model by the Federator. This does not take the client distributions in
        account but is centralized.
        @param net: Global network to be tested on the Federator centralized testing dataset.
        @type net: torch.nn.Module
        @return: Accuracy and loss of the global network on a (subset) of the testing data, and the confusion matrix
        corresponding to the models' predictions.
        @rtype: Tuple[float, float, np.array]
        """
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = net(images)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total
        confusion_mat = sklearn.metrics.confusion_matrix(targets_, pred_)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        return accuracy, loss, confusion_mat

    def exec_round(self, com_round_id: int):
        """
        Helper method to call a remote Client to perform a training round during the training loop.
        @param com_round_id: Identifier of the communication round.
        @type com_round_id: int
        @return: None
        @rtype: None
        """
        start_time = time.time()
        num_epochs = self.config.epochs

        # Client selection
        selected_clients: List[LocalClient]
        selected_clients = random_selection(self.clients, self.config.clients_per_round)

        last_model = self.get_nn_parameters()
        for client in selected_clients:
            self.message(client.ref, Client.update_nn_parameters, last_model)

        # Actual training calls
        client_weights = {}
        client_sizes = {}
        # pbar = tqdm(selected_clients)
        # for client in pbar:

        # Client training
        training_futures: List[torch.Future] = []  # pylint: disable=no-member

        def training_cb(fut: torch.Future, client_ref: LocalClient, client_weights, client_sizes, num_epochs):  # pylint: disable=no-member
            train_loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration, c_mat = fut.wait()
            self.logger.info(f'Training callback for client {client_ref.name} with accuracy={accuracy}')
            client_weights[client_ref.name] = weights
            client_data_size = self.message(client_ref.ref, Client.get_client_datasize)
            client_sizes[client_ref.name] = client_data_size
            c_record = ClientRecord(com_round_id, train_duration, test_duration, round_duration, num_epochs, 0,
                                    accuracy, train_loss, test_loss, confusion_matrix=c_mat)
            client_ref.exp_data.append(c_record)

        for client in selected_clients:
            future = self.message_async(client.ref, Client.exec_round, num_epochs, com_round_id)
            cb_factory(future, training_cb, client, client_weights, client_sizes, num_epochs)
            self.logger.info(f'Request sent to client {client.name}')
            training_futures.append(future)

        def all_futures_done(futures: List[torch.Future]) -> bool:  # pylint: disable=no-member
            return all(map(lambda x: x.done(), futures))

        while not all_futures_done(training_futures):
            time.sleep(0.1)
            # self.logger.info('')
            # self.logger.info(f'Waiting for other clients')

        self.logger.info('Continue with rest [1]')
        time.sleep(3)

        updated_model = self.aggregation_method(client_weights, client_sizes)
        self.update_nn_parameters(updated_model)

        test_accuracy, test_loss, conf_mat = self.test(self.net)
        self.logger.info(f'[Round {com_round_id:>3}] Federator has a accuracy of {test_accuracy} and loss={test_loss}')

        end_time = time.time()
        duration = end_time - start_time
        record = FederatorRecord(len(selected_clients), com_round_id, duration, test_loss, test_accuracy,
                                 confusion_matrix=conf_mat)
        self.exp_data.append(record)
        self.logger.info(f'[Round {com_round_id:>3}] Round duration is {duration} seconds')
