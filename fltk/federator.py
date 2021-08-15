import logging
import pathlib
import time
from pathlib import Path
from typing import List, Callable, Dict

import torch
from dataclass_csv import DataclassWriter
from torch.distributed import rpc
from torch.utils.tensorboard import SummaryWriter

from fltk.client import Client
from fltk.nets.util.utils import flatten_params, save_model
from fltk.util.base_config import BareConfig
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
    # Dictionary containing active (e.g. deployed) tasks.
    active_tasks: Dict[str, Dict[str, ClientRef]]
    # List of active clients
    clients: List[ClientRef] = []
    # epoch_counter = 0
    # client_data = {}
    # poisoned_clients = {}
    # healthy_clients = {}

    def __init__(self, client_id_triple, config: BareConfig = None):
        log_rref = rpc.RRef(FLLogger())


        self.log_rref = log_rref
        self.config = config

        # TODO: Change to Kubernetes spawning
        self.create_clients(client_id_triple)

        # TODO: Decide on using a more persitent logging approach
        self.config.init_logger(logging)

    def create_clients(self, client_id_triple):
        """
        Function to spin up worker clients for a task.
        @param client_id_triple:
        @type client_id_triple:
        @return:
        @rtype:
        """
        # TODO: Change to spinning up different clients.
        for id, rank, world_size in client_id_triple:
            client = rpc.remote(id, Client, kwargs=dict(id=id, log_rref=self.log_rref, rank=rank, world_size=world_size,
                                                        config=self.config))
            writer = SummaryWriter(f'{self.tb_path_base}/{self.config.experiment_prefix}_client_{id}')
            self.clients.append(ClientRef(id, client, tensorboard_writer=writer))
            self.client_data[id] = []
        # In additino we store our own data through the process
        self.client_data['federator'] = []

    def update_clients(self, ratio):
        """
        TODO remove functionality, move to new function & clean up
        @param ratio:
        @type ratio:
        @return:
        @rtype:
        """
        # Prevent abrupt ending of the client
        self.tb_writer = SummaryWriter(f'{self.tb_path_base}/{self.config.experiment_prefix}_federator_{ratio}')
        for client in self.clients:
            # Create new writer and close old writer
            writer = SummaryWriter(f'{self.tb_path_base}/{self.config.experiment_prefix}_client_{client.name}_{ratio}')
            client.tb_writer.close()
            client.tb_writer = writer
            # Clear client updates ofteraf
            self.client_data[client.name] = []

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

    def client_reset_model(self):
        """
        Function to reset the model at all learners
        @return:
        @rtype:
        """
        self.epoch_counter = 0
        for client in self.clients:
            _remote_method_async(Client.reset_model, client.ref)

    def client_load_data(self, poison_pill):
        for client in self.clients:
            _remote_method_async(Client.init_dataloader, client.ref,
                                 pill=None if poison_pill and client not in self.poisoned_clients else poison_pill)

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

    def remote_run_epoch(self, epochs, ratio=None, store_grad=False):
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
            if (client in self.poisoned_clients) & self.attack.is_active():
                pill = self.attack.get_poison_pill()
            responses.append((client, _remote_method_async(Client.run_epochs, client.ref, num_epoch=epochs, pill=pill)))
        try:
            # Test the model before waiting for the model.
            # Append to client data to keep better track of progress
            self.client_data.get('federator', []).append(self.test_model())
        except Exception as e:
            print(e)
        self.epoch_counter += epochs
        flat_current = None

        if store_grad:
            flat_current = flatten_params(self.test_data.net.state_dict())
        for res in responses:
            epoch_data, weights = res[1].wait()
            if store_grad:
                # get flatten
                self.store_gradient(flatten_params(weights) - flat_current, epoch_data.client_id, self.epoch_counter,
                                    ratio)
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
        # TODO: Make sure that we keep track of whose gradient we are dealing with
        updated_model = self.antidote.process_gradients(client_weights, epoch=self.epoch_counter,
                                                        clients=selected_clients,
                                                        model=self.test_data.net.state_dict())
        self.test_data.net.load_state_dict(updated_model)
        # test global model
        logging.info("Testing on global test set")
        self.test_data.update_nn_parameters(updated_model)
        self.distribute_new_model(updated_model)
        return updated_model

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

    def save_epoch_data(self, ratio=None):
        file_output = f'./{self.config.output_location}'
        self.ensure_path_exists(file_output)
        for key in self.client_data:
            if ratio:
                filename = f'{file_output}/{key}_epochs_{ratio}.csv'
            else:
                filename = f'{file_output}/{key}_{ratio}.csv'
            logging.info(f'Saving data at {filename}')
            with open(filename, "w") as f:
                w = DataclassWriter(f, self.client_data[key], EpochData)
                w.write()

    def ensure_path_exists(self, path):
        Path(path).mkdir(parents=True, exist_ok=True)

    def run(self, ratios=[0.12, 0.18, 0.6, 0.0]):
        """
        Main loop of the Federator
        :return:
        """

        # # Select clients which will be poisened
        # TODO: get attack type and ratio from config, temp solution now
        poison_pill = None
        save_path = self.config
        for rat in ratios:
            # Update writer to logdir
            self.update_clients(rat)
            if self.attack:
                self.poisoned_clients: List[ClientRef] = self.attack.select_poisoned_clients(self.clients, rat)
                self.healthy_clients = list(set(self.clients).symmetric_difference(set(self.poisoned_clients)))
                print(f"Poisoning workers: {self.poisoned_clients}")
                with open(f"{self.tb_path_base}/config_{rat}_poisoned.txt", 'w') as f:
                    f.writelines(list(map(lambda worker: worker.name, self.poisoned_clients)))
                poison_pill = self.attack.get_poison_pill()
            self.client_load_data(poison_pill)
            self.ping_all()
            self.clients_ready()
            self.update_client_data_sizes()
            addition = 0
            epoch_to_run = self.config.epochs
            epoch_size = self.config.epochs_per_cycle
            print(f"Running a total of {epoch_to_run} epochs...")
            for epoch in range(epoch_to_run):
                print(f'Running epoch {epoch}')
                # Get new model during run, update iteratively. The model is needed to calculate the
                # gradient by the federator.
                self.remote_run_epoch(epoch_size, rat)
                addition += 1
            logging.info('Printing client data')

            # Perform last test on the current model.
            self.client_data.get('federator', []).append(self.test_model())
            logging.info(f'Saving model')
            save_model(self.test_data.net, './output', self.epoch_counter, self.config, rat)
            logging.info(f'Saving data')
            self.save_epoch_data(rat)

            # Reset the model to continue with the next round
            self.client_reset_model()
            # Reset dataloader, etc. for next experiment
            self.set_data()
            self.antidote.save_data_and_reset(rat)

        logging.info(f'Federator is stopping')

    def store_gradient(self, gradient, client_id, epoch, ratio):
        """
        Function to save the gradient of a client in a specific directory.
        @param gradient:
        @type gradient:
        @param client_id:
        @type client_id:
        @param epoch:
        @type epoch:
        @param ratio:
        @type ratio:
        @return:
        @rtype:
        """
        directory: str = f"{self.tb_path_base}/gradient/{ratio}/{epoch}/{client_id}"
        # Ensure path exists (mkdir -p)
        pathlib.Path(directory).mkdir(parents=True, exist_ok=True)
        # Save using pytorch.
        torch.save(gradient, f"{directory}/gradient.pt")

    def distribute_new_model(self, updated_model):
        """
        Function to update the model on the clients
        @return:
        @rtype:
        """
        responses = []
        for client in self.clients:
            responses.append(
                (client, _remote_method_async(Client.update_nn_parameters, client.ref, new_params=updated_model)))

        for res in responses:
            res[1].wait()
        logging.info('Weights are updated')

    def test_model(self, model, writer) -> EpochData:
        """
        TODO: Move this function somewhere else. Maybe even the federator shouldn't be bothered with testing.
        Function to test the model on the test dataset.
        @return:
        @rtype:
        """
        # Test interleaved to speed up execution, i.e. don't keep the clients waiting.
        accuracy, loss, class_precision, class_recall = model.test()
        data = EpochData(epoch_id=self.epoch_counter,
                         duration_train=0,
                         duration_test=0,
                         loss_train=0,
                         accuracy=accuracy,
                         loss=loss,
                         class_precision=class_precision,
                         class_recall=class_recall,
                         client_id='federator')
        writer.add_scalar('accuracy', accuracy, self.epoch_counter * self.test_data.get_client_datasize())
        writer.add_scalar('accuracy per epoch', accuracy, self.epoch_counter)
        return data
