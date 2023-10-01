from __future__  import annotations

import gc
import multiprocessing
import queue
from typing import Tuple, Any, Callable

import numpy as np
import sklearn
import time
import torch

from fltk.core.node import Node
from fltk.schedulers import MinCapableStepLR
from fltk.strategy import get_optimizer


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fltk.util.config import FedLearnerConfig


class Client(Node):
    """
    Federated experiment client.
    """
    running = False
    request_queue = queue.Queue()
    result_queue = queue.Queue()

    def __init__(self, identifier: str, rank: int, world_size: int, config: FedLearnerConfig):
        super(Client, self).__init__(identifier, rank, world_size, config)

        self.loss_function = self.config.get_loss_function()()
        self.optimizer = get_optimizer(self.config.optimizer)(self.net.parameters(),
                                                              **self.config.optimizer_args)
        self.scheduler = MinCapableStepLR(self.optimizer,
                                          self.config.scheduler_step_size,
                                          self.config.scheduler_gamma,
                                          self.config.min_lr)

    def remote_registration(self):
        """
        Function to perform registration to the remote. Currently, this will connect to the Federator Client. Future
        version can provide functionality to register to an arbitrary Node, including other Clients.
        @return: None.
        @rtype: None
        """
        self.logger.info('Sending registration')
        self.message('federator', 'ping', 'new_sender')
        self.message('federator', 'register_client', self.id, self.rank)

    def run(self):
        """
        Function to start running the Client after registration. This allows for processing requests by the main thread,
        while the RPC requests can be made asynchronously.

        Returns: None

        """
        self.running = True
        event = multiprocessing.Event()
        while self.running:
            # Hack for running on Kubeflow
            if not self.request_queue.empty():
                request = self.request_queue.get()
                self.logger.info(f"Got request, args: {request} running synchronously.")
                self.result_queue.put(self.exec_round(*request))
            event.wait(1)
        self.logger.info(f"Exiting client {self.id}")

    def stop_client(self):
        """
        Function to stop client after training. This allows remote clients to stop the client within a specific
        timeframe.
        @return: None
        @rtype: None
        """
        self.logger.info('Got call to stop event loop')
        self.running = False

    def train(self, num_epochs: int, round_id: int):
        """
        Function implementing federated learning training loop, allowing to run for a configurable number of epochs
        on a local dataset. Note that only the last statistics of a run are sent to the caller (i.e. Federator).
        @param num_epochs: Number of epochs to run during a communication round's training loop.
        @type num_epochs: int
        @param round_id: Global communication round ID to be used during training.
        @type round_id: int
        @return: Final running loss statistic and acquired parameters of the locally trained network. NOTE that
        intermediate information is only logged to the STD-out.
        @rtype: Tuple[float, Dict[str, torch.Tensor]]
        """
        start_time = time.time()
        running_loss = 0.0
        final_running_loss = 0.0
        self.logger.info(f"[RD-{round_id}] kicking of local training for {num_epochs} local epochs")
        for local_epoch in range(num_epochs):
            effective_epoch = round_id * num_epochs + local_epoch
            progress = f'[RD-{round_id}][LE-{local_epoch}][EE-{effective_epoch}]'
            if self.distributed:
                # In case a client occurs within (num_epochs) communication rounds as this would cause
                # an order or data to re-occur during training.
                self.dataset.train_sampler.set_epoch(effective_epoch)

            training_cardinality = len(self.dataset.get_train_loader())
            self.logger.info(f'{progress}{self.id}: Number of training samples: {training_cardinality}')
            for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
                # zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)

                running_loss += loss.detach().item()
                loss.backward()
                self.optimizer.step()
                # Mark logging update step
                if i % self.config.log_interval == 0:
                    self.logger.info(
                            f'[{self.id}] [{local_epoch}/{num_epochs:d}, {i:5d}] loss: {running_loss / self.config.log_interval:.3f}')
                    final_running_loss = running_loss / self.config.log_interval
                    running_loss = 0.0
                del loss, inputs, labels

            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f'{progress} Train duration is {duration} seconds')
        # Clear gradients before we send.
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        return final_running_loss, self.get_nn_parameters()

    def set_tau_eff(self, total):
        client_weight = self.get_client_datasize() / total
        n = self.get_client_datasize()  # pylint: disable=invalid-name
        E = self.config.epochs  # pylint: disable=invalid-name
        B = 16  # nicely hardcoded :) # pylint: disable=invalid-name
        tau_eff = int(E * n / B) * client_weight
        if hasattr(self.optimizer, 'set_tau_eff'):
            self.optimizer.set_tau_eff(tau_eff)

    def test(self, round_id: int = None) -> Tuple[float, float, np.array]:
        """
        Function implementing federated learning test loop.
        @param round_id: Indicator of round, currently unused.
        @type round_id: integer representing the round provided by the Federator.
        @return: Statistics on test-set given a (partially) trained model; accuracy, loss, and confusion matrix.
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

                outputs = self.net(images)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += int(labels.size(0))
                correct += (predicted == labels).sum().detach().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).detach().item()
        # Calculate learning statistics
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total

        confusion_mat = sklearn.metrics.confusion_matrix(targets_, pred_)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        del targets_, pred_
        return accuracy, loss, confusion_mat

    def get_client_datasize(self):  # pylint: disable=missing-function-docstring
        return len(self.dataset.get_train_sampler())

    def request_round(self, num_epochs: int, round_id:int):
        event = multiprocessing.Event()
        self.request_queue.put([num_epochs, round_id])

        while self.result_queue.empty():
            event.wait(5)
        self.logger.info("Finished request!")
        return self.result_queue.get()

    def exec_round(self, num_epochs: int, round_id: int) -> Tuple[Any, Any, Any, Any, float, float, float, np.array]:
        """
        Function as access point for the Federator Node to kick off a remote learning round on a client.
        @param num_epochs: Number of epochs to run
        @type num_epochs: int
        @return: Tuple containing the statistics of the training round; loss, weights, accuracy, test_loss, make-span,
        training make-span, testing make-span, and confusion matrix.
        @rtype: Tuple[Any, Any, Any, Any, float, float, float, np.array]
        """
        self.logger.info(f"[EXEC] running {num_epochs} locally...")
        start = time.time()
        loss, weights = self.train(num_epochs, round_id)
        time_mark_between = time.time()
        accuracy, test_loss, test_conf_matrix = self.test(round_id)

        end = time.time()
        round_duration = end - start
        train_duration = time_mark_between - start
        test_duration = end - time_mark_between
        # self.logger.info(f'Round duration is {duration} seconds')

        if hasattr(self.optimizer, 'pre_communicate'):
            # aka fednova or fedprox
            self.optimizer.pre_communicate()
        for k, value in weights.items():
            weights[k] = value.cpu()
        gc.collect()
        return loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration, test_conf_matrix

    def __del__(self):
        self.logger.info(f'Client {self.id} is stopping')


class ContinuousClient(Client):
    """
    Federated Continual Learning experiment Client. See also Client implementation for ordinary Federated Learning
    Experiments.
    """

    def __init__(
        self,
        client_id: str,
        rank: int,
        world_size: int,
        fed_config: FedLearnerConfig,
        *args,
        **kwargs
    ):
        """
        model,
        tr_dataloader,
        nepochs=100,
        lr=0.001,
        lr_min=1e-6,
        lr_factor=3,
        lr_patience=5,
        clipgrad=100,
        args=None,
        num_classes=10,
        """
        super(ContinuousClient, self).__init__(client_id, rank, world_size, fed_config)

    def train(self, num_epochs: int, round_id: int):
        """
        Function implementing federated learning training loop, allowing to run for a configurable number of epochs
        on a local dataset. Note that only the last statistics of a run are sent to the caller (i.e. Federator).
        @param num_epochs: Number of epochs to run during a communication round's training loop.
        @type num_epochs: int
        @param round_id: Global communication round ID to be used during training.
        @type round_id: int
        @return: Final running loss statistic and acquired parameters of the locally trained network. NOTE that
        intermediate information is only logged to the STD-out.
        @rtype: Tuple[float, Dict[str, torch.Tensor]]
        """
        start_time = time.time()
        running_loss = 0.0
        final_running_loss = 0.0
        self.logger.info(f"[RD-{round_id}] kicking of local continuous training for {num_epochs} local epochs")
        # FIXME: Add means to compute task_id given round ID.
        task_id = ...
        train_loader = self.dataset.get_train_loader(task_id=task_id)
        for local_epoch in range(num_epochs):
            effective_epoch = round_id * num_epochs + local_epoch
            progress = f'[RD-{round_id}][LE-{local_epoch}][EE-{effective_epoch}]'
            if self.distributed:
                # In case a client occurs within (num_epochs) communication rounds as this would cause
                # an order or data to re-occur during training.
                self.dataset.train_sampler.set_epoch(effective_epoch)

            training_cardinality = len(train_loader)
            self.logger.info(f'{progress}{self.id}: Number of training samples: {training_cardinality}')
            for i, (inputs, labels) in enumerate(train_loader, 0):
                # zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)

                running_loss += loss.detach().item()
                loss.backward()
                self.optimizer.step()
                # Mark logging update step
                if i % self.config.log_interval == 0:
                    self.logger.info(
                            f'[{self.id}] [{local_epoch}/{num_epochs:d}, {i:5d}] loss: {running_loss / self.config.log_interval:.3f}')
                    final_running_loss = running_loss / self.config.log_interval
                    running_loss = 0.0
                del loss, inputs, labels

            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f'{progress} Train duration is {duration} seconds')
        # Clear gradients before we send.
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        return final_running_loss, self.get_nn_parameters()

    def test(self, epoch_id=-1):
        """Function implementing federated continual learning evaluation loop. Provides only (scaled) accuracies and
        losses for tasks seen until now, rather than
        @return: Statistics on test-set given a (partially) trained model; accuracy, loss, and confusion matrix.
        @rtype: Tuple[float, float, np.array]
        """
        start_time = time.time()
        # FIXME: Add means to calculate task_id given current
        task_id = ...
        accuracies = []
        counts = []
        losses = []
        with torch.no_grad():
            # Currently this is evaluating the Average accuracy for client i by looping over the tasks with size T
            for task in range(task_id + 1):
                correct, total = 0, 0
                test_loss = 0
                for (images, labels) in self.dataset.get_test_loader(task_id=task):
                    images, labels = images.to(self.device), labels.to(self.device)
                    # FIXME: Add shared/local weights to the model.
                    outputs = self.net(images)

                    _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                    correct += (predicted == labels).sum().detach().item()
                    total += int(labels.size(0))

                    test_loss += self.loss_function(outputs, labels).detach().item()
                test_accuracy = correct / total
                test_loss = test_loss / total
                self.logger.info(
                    f"[Client {self.id}] Test on task {task:2d} : loss={test_loss:.3f}, acc={100 * test_accuracy:5.1f}%"
                )
                counts.append(int(total))
                accuracies.append(test_accuracy)
                losses.append(test_loss)
            counts = torch.tensor(counts)
            # Scale statistics per run.
            mean_acc = torch.mean(torch.tensor(losses) / counts)
            mean_lss = torch.mean(torch.tensor(accuracies) / counts)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'[Client {self.id}] Test duration is {duration} seconds')
        return mean_lss, mean_acc, total


def _client_constructor(client_name: str, rank: int, config: FedLearnerConfig, *args, **kwargs) -> Client:
    """Constructor helper method for standard Federated Learning Clients.

    @param client_name: Identifier of the client during experiment.
    @rtype client_name: str
    @param rank: Rank (relative to worlds size) of client.
    @rtype rank: int
    @param config: Federated Learning configuration object.
    """
    return Client(client_name, rank, config.world_size, config)


def _continous_client_constructor(client_name: str, rank: int, config: FedLearnerConfig, *args, **kwargs) -> ContinuousClient:
    """Constructor helper method for Continuous Federated Learning Clients.

    @param client_name: Identifier of the client during experiment.
    @rtype client_name: str
    @param rank: Rank (relative to worlds size) of client.
    @rtype rank: int
    @param config: Federated Learning configuration object.
    """
    raise NotImplementedError()
    return ContinuousClient(client_name, rank, config.world_size, config)


def get_constructor(config: FedLearnerConfig) -> Callable[[str, int, FedLearnerConfig, ...], Client]:
    """Helper method to infer required constructor method to instantiate and prepare different Client's during
    experiments.
    @param config: FederatedLearning Configuration for current experiment.
    @type config: FedLearnerConfig
    @return: Callable function which implements the instantiation of the requested Client using the callers' provided
        arguments.
    @rtype: Callable[[str, int, FedLearnerConfig, ...], Client]
    """
    # FIXME: Implement way to discern between continous and ordinary federated learning.
    continous = False

    if continous:
        return _continous_client_constructor
    else:
        return _client_constructor


class FedClientConstructor:
    """Constructor object allowing the caller to defer the inference of the type of required Client ot the Constructor.
    Default behavior is to instantiate a standard Federated Learning client.
    """
    def construct(self, config: FedLearnerConfig, client_name: str, rank: int, *args, **kwargs):
        """
        Constructor method to automatically infer the required type of Client from the provided learner configuration.

        @param config: FederatedLearning configuration object for current experiment.
        @type config: FedLearnerConfig
        @param client_name: Name of client to use during communication.
        @type client_name: str
        @param rank: Rank of the client during the experiment (relative to world size)
        @type rank: int
        @param world_size: Total number of clients + federator to participate during experiment.
        @type world_size: int
        @param args: Additional arguments to pass to constructors as required.
        @type args: Any
        @param kwargs: Additional keyword arguments to pass to consturctors as required.
        @type kwargs: Dict[str, Any]
        @return: Instantiated client with provided arguments.
        @rtype: Client
        """
        constructor = get_constructor(config)
        return constructor(client_name, rank, config, *args, **kwargs)
