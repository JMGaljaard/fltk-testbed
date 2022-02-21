import copy
import time
from typing import List, Union

import torch

from fltk.core.client import Client
from fltk.core.node import Node
from fltk.datasets.loader_util import get_dataset
from fltk.strategy import FedAvg, random_selection, average_nn_parameters, average_nn_parameters_simple
from fltk.util.config import Config


NodeReference = Union[Node, str]


class Federator(Node):
    clients: List[NodeReference] = []
    num_rounds: int

    def __init__(self, id: int, rank: int, world_size: int, config: Config):
        super().__init__(id, rank, world_size, config)
        self.loss_function = self.config.get_loss_function()()
        self.num_rounds = config.rounds
        self.config = config



    def create_clients(self):
        if self.config.single_machine:
            # Create direct clients
            world_size = self.config.num_clients + 1
            for client_id in range(1, self.config.num_clients+ 1):
                client_name = f'client{client_id}'
                self.clients.append(Client(client_name, client_id, world_size, copy.deepcopy(self.config)))

    def register_client(self, client_name, rank):
        if self.config.single_machine:
            self.logger.warning('This function should not be called when in single machine mode!')
        self.clients.append(client_name)

    def _num_clients_online(self) -> int:
        return len(self.clients)

    def _all_clients_online(self) -> bool:
        return len(self.clients) == self.world_size - 1

    def clients_ready(self):
        """
        Synchronous implementation
        """
        all_ready = False
        ready_clients = []
        while not all_ready:
            responses = []
            all_ready = True
            for client in self.clients:
                resp = self.message(client, Client.is_ready)
                if resp:
                    self.logger.info(f'Client {client} is ready')
                else:
                    self.logger.info(f'Waiting for client {client}')
                    all_ready = False
            time.sleep(2)

    def run(self):
        self.init_dataloader()
        self.create_clients()
        while not self._all_clients_online():
            self.logger.info(f'Waiting for all clients to come online. Waiting for {self.world_size - 1 -self._num_clients_online()} clients')
            time.sleep(2)
        self.client_load_data()
        self.clients_ready()

        for communications_round in range(self.config.rounds):
            self.exec_round()

        self.logger.info('Federator is stopping')

    def client_load_data(self):
        for client in self.clients:
            self.message(client, Client.init_dataloader)

    def test(self, net):
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

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100 * correct / total
        # confusion_mat = confusion_matrix(targets_, pred_)
        # accuracy_per_class = confusion_mat.diagonal() / confusion_mat.sum(1)
        #
        # class_precision = calculate_class_precision(confusion_mat)
        # class_recall = calculate_class_recall(confusion_mat)
        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        return accuracy, loss

    def exec_round(self):
        start_time = time.time()
        num_epochs = self.config.epochs

        # Client selection
        selected_clients = random_selection(self.clients, self.config.clients_per_round)

        last_model = self.get_nn_parameters()
        for client in selected_clients:
            self.message(client, Client.update_nn_parameters, last_model)

        # Actual training calls
        client_weights = {}
        client_sizes = {}
        for client in selected_clients:
            train_loss, weights, accuracy, test_loss = self.message(client, Client.exec_round, num_epochs)
            client_weights[client] = weights
            client_data_size = self.message(client, Client.get_client_datasize)
            client_sizes[client] = client_data_size
            self.logger.info(f'Client {client} has a accuracy of {accuracy}, train loss={train_loss}, test loss={test_loss},datasize={client_data_size}')

        # updated_model = FedAvg(client_weights, client_sizes)
        updated_model = average_nn_parameters_simple(list(client_weights.values()))
        self.update_nn_parameters(updated_model)

        test_accuracy, test_loss = self.test(self.net)
        self.logger.info(f'Federator has a accuracy of {test_accuracy} and loss={test_loss}')

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Round duration is {duration} seconds')

