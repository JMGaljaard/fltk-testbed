import logging
import time
from queue import PriorityQueue
from typing import List

import kubernetes.config
from dataclass_csv import DataclassWriter
from kubernetes import client

from fltk.util.config.base_config import BareConfig
from fltk.util.results import EpochData
from fltk.util.task.config.parameter import TrainTask


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
    _alive = False
    # Priority queue, requires an orderable object, otherwise a Tuple[int, Any] can be used to insert.
    pending_tasks: PriorityQueue[TrainTask] = PriorityQueue()
    deployed_tasks: List[TrainTask] = []
    completed_tasks: List[str] = []

    def __init__(self, config: BareConfig = None):
        self._logger = logging.getLogger('Orchestrator')

        self._logger.debug("Loading in-cluster configuration")
        kubernetes.config.load_incluster_config()

        self.config = config
        self._v1 = client.CoreV1Api()
        self._batch_api = client.BatchV1Api()

    def remote_run_epoch(self, ):
        """
        @deprecated
        """

        #     """
        #     TODO: Implement poisioning by providing arguments to the different clients.
        #     Either the federator selects n nodes at the start, or a (configurable) function is selected, which
        #     determines to send to which nodes and which are poisoned
        #     """
        #     responses.append((client, _remote_method_async(Client.run_epochs, client.ref, num_epoch=epochs)))

        # TODO: Decide on how to combine logging in KubeFlow/Tensorboard/otherwise.
        # res[0].tb_writer.add_scalar('training loss',
        #                             epoch_data.loss_train,  # for every 1000 minibatches
        #                             self.epoch_counter * res[0].data_size)
        #
        # res[0].tb_writer.add_scalar('accuracy',
        #                             epoch_data.accuracy,  # for every 1000 minibatches
        #                             self.epoch_counter * res[0].data_size)
        #
        # res[0].tb_writer.add_scalar('training loss per epoch',
        #                             epoch_data.loss_train,  # for every 1000 minibatches
        #                             self.epoch_counter)
        #
        # res[0].tb_writer.add_scalar('accuracy per epoch',
        #                             epoch_data.accuracy,  # for every 1000 minibatches
        #                             self.epoch_counter)

    def save_epoch_data(self):
        file_output = f'./{self.config.output_location}'
        # self.ensure_path_exists(file_output)
        for key in self.client_data:
            filename = f'{file_output}/{key}.csv'
            self._logger.info(f'Saving data at {filename}')
            with open(filename, "w") as f:
                w = DataclassWriter(f, self.client_data[key], EpochData)
                w.write()

    def stop(self) -> None:
        """

        @return:
        @rtype:
        """
        self._logger.info("Received stop signal for the Orchestrator.")
        self._alive = False

    def run(self) -> None:
        """
        Main loop of the Orchestrator
        :return:
        """
        self._alive = True
        while self._alive:
            # 1. Check arrivals
            # If new arrivals, store them in arrival list
            if not self.pending_tasks.empty():
                pass

            self._logger.info("Still alive...")
            time.sleep(5)

        # TODO: Implement run loop:

        # 2. If unscheduled tassk
        # Take first / highest priority job
        # Check for available resources in cluster, break if not
        # Create Job description
        # Spawn job
        # 3. Check for job completion status
        # 4. Record something? idk.

        logging.info(f'Federator is stopping')
