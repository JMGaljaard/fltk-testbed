import logging
import time
from multiprocessing.pool import ThreadPool
from pathlib import Path
from typing import List, Callable, Dict

import torch
from dataclass_csv import DataclassWriter
from torch.distributed import rpc
from torch.utils.tensorboard import SummaryWriter

from fltk.client import Client
from fltk.nets.util.utils import flatten_params, save_model
from fltk.util.cluster.client import ClientRef, ClusterManager
from fltk.util.config.base_config import BareConfig
from fltk.util.log import DistLearningLogger
from fltk.util.results import EpochData
from fltk.util.task.generator.arrival_generator import ArrivalGenerator, ExperimentGenerator


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


class Orchestrator(object):
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
    active_tasks: Dict[str, Dict[str, ClientRef]] = {}
    # List of active clients
    clients: List[ClientRef] = []

    task_generator: ArrivalGenerator

    def __init__(self, client_id_triple, config: BareConfig = None):


        self.log_rref = None
        self.config = config

        # TODO: Change to Kubernetes spawning
        # self.create_clients(client_id_triple)

        # TODO: Decide on using a more persitent logging approach
        # self.config.init_logger(logging)

    def init_generator(self) -> None:
        """
        Function to initialize task generation according to provided config files.

        TODO: Rename function to match description
        TODO: Find way to provide scheduling characteristcs back to the task generator in case of an experiment runner.

        @return: None
        @rtype: None
        """

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
        :@deprecated Function to be removed in future commit.
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

    def client_load_data(self):
        """
        TODO: Make this compatible with job registration...
        @return:
        @rtype:
        """
        for client in self.clients:
            _remote_method_async(Client.init_dataloader, client.ref)

    def clients_ready(self):
        """
        TODO: Make compatible with Job registration
        TODO: Make push based instead of pull based.
        @return:
        @rtype:
        """
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

    def run(self) -> None:
        """
        Main loop of the Orchestrator
        :return:
        """
        save_path = Path(self.config.execution_config.general_net.save_model_path)
        logging_dir = self.config.execution_config.tensorboard.record_dir

        cluster_manager: ClusterManager = ClusterManager()
        arrival_generator = ExperimentGenerator()

        logger = logging.getLogger('Orchestrator')
        while True:
            logger.info(cluster_manager._watchdog._resource_lookup)
            time.sleep(10)

        self.client_load_data()
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
            self.remote_run_epoch(epoch_size)
            addition += 1
        logging.info('Printing client data')

        # Perform last test on the current model.
        self.client_data.get('federator', []).append(self.test_model())
        logging.info(f'Saving model')
        save_model(self.test_data.net, './output', self.epoch_counter, self.config)
        logging.info(f'Saving data')
        self.save_epoch_data()

        # Reset the model to continue with the next round
        self.client_reset_model()
        # Reset dataloader, etc. for next experiment
        self.set_data()
        self.antidote.save_data_and_reset()

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
        Path(directory).mkdir(parents=True, exist_ok=True)
        # Save using pytorch.
        torch.save(gradient, f"{directory}/gradient.pt")

    def distribute_new_model(self, updated_model) -> None:
        """
        Function to update the model on the
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


def run_orchestrator(rpc_ids_triple, configuration: BareConfig):
    """
    Function to run as 'orchestrator', this will
    @param rpc_ids_triple:
    @type rpc_ids_triple:
    @param configuration:
    @type configuration:
    @param config_path:
    @type config_path:
    @return:
    @rtype:
    """
    logging.info("Starting Orchestrator, initializing resources....")
    orchestrator = Orchestrator(rpc_ids_triple, config=configuration)
    cluster_manager = ClusterManager()
    arrival_generator = ExperimentGenerator()

    pool = ThreadPool(3)
    logging.info("Starting cluster manager")
    pool.apply_async(cluster_manager.start)
    logging.info("Starting arrival generator")
    pool.apply_async(arrival_generator.run)
    logging.info("Starting orchestrator")
    pool.apply(orchestrator.run)
    pool.join()