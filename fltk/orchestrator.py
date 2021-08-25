import logging
import queue
import time
from multiprocessing.pool import ThreadPool
from typing import List

import kubernetes.config
from dataclass_csv import DataclassWriter
from kubernetes import client

from fltk.client import Client
from fltk.nets.util.utils import flatten_params, save_model
from fltk.util.cluster.client import ClusterManager, deploy_job
from fltk.util.config.base_config import BareConfig
from fltk.util.results import EpochData
from fltk.util.task.generator.arrival_generator import ExperimentGenerator


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
    pending_tasks: queue.Queue = queue.Queue()
    completed_tasks: List[str] = []


    def __init__(self, config: BareConfig = None):

        self.log_rref = None
        self.config = config
        kubernetes.config.load_incluster_config()
        self._v1 = client.CoreV1Api()
        self._batch_api = client.BatchV1Api()

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


    def save_epoch_data(self):
        file_output = f'./{self.config.output_location}'
        # self.ensure_path_exists(file_output)
        for key in self.client_data:
            filename = f'{file_output}/{key}.csv'
            logging.info(f'Saving data at {filename}')
            with open(filename, "w") as f:
                w = DataclassWriter(f, self.client_data[key], EpochData)
                w.write()

    def run(self) -> None:
        """
        Main loop of the Orchestrator
        :return:
        """
        logger = logging.getLogger('Orchestrator')

        time.sleep(10)

        job =
        deploy_job(self._batch_api, job)
        while True:
            logger.info("Still alive...")
            time.sleep(5)

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


def run_orchestrator(configuration: BareConfig) -> None:
    """
    Function to start the different components of the orchestrator.
    @param configuration: Configuration for components, needed for spinning up components of the Orchestrator.
    @type configuration: BareConfig
    @return: None
    @rtype: None
    """
    logging.info("Starting Orchestrator, initializing resources....")
    orchestrator = Orchestrator(configuration)
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
    logging.info("Stopped execution of Orchestrator...")
