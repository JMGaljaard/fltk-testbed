import collections
import logging
import multiprocessing
import time
from abc import abstractmethod
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from queue import Queue
from random import choices
from typing import Dict, List, Union, OrderedDict, Optional

import numpy as np

from fltk.util.config.definitions.net import Nets
from fltk.datasets.dataset import Dataset
from fltk.util.singleton import Singleton
from fltk.util.task.config.parameter import (TrainTask, JobDescription, ExperimentParser, SystemParameters,
                                             HyperParameters,LearningParameters)


@dataclass
class ArrivalGenerator(metaclass=Singleton): # pylint: disable=too-many-instance-attributes
    """
    Abstract Base Class for generating arrivals in the system. These tasks must be run
    """

    configuration_path: Path
    job_dict: OrderedDict[str, JobDescription] = None
    logger: logging.Logger = None
    arrivals: "Queue[Arrival]" = Queue()

    start_time: float = -1
    stop_time: float = -1
    alive: bool = False

    def load_config(self):
        """
        Load configuration from default path, if alternative path is not provided.
        @return: None
        @rtype: None
        """
        parser = ExperimentParser(config_path=self.configuration_path)
        experiment_descriptions = parser.parse()
        self.job_dict = collections.OrderedDict(
                {f'train_job_{indx}': item for indx, item in enumerate(experiment_descriptions.train_tasks)})

    def start(self, duration: Union[float, int]):
        """
        Function to start arrival generator, requires to
        @return: None
        @rtype: None
        """
        if not self.logger:
            self.set_logger()
        self.logger.info("Starting execution of arrival generator...")
        self.alive = True
        self.run(duration)

    def stop(self) -> None:
        """
        Function to call when the generator needs to stop. By default, the generator will run for 1 hour.
        @return: None
        @rtype: None
        """
        self.logger.info("Received stopping signal")
        self.alive = False

    @abstractmethod
    def run(self, duration: float):
        """
        Abstract function to run experiment generator for a specified time duration.
        @param duration: Time in seconds to run experiment generation.
        @type duration: int
        @return: None
        @rtype: None
        """

    @abstractmethod
    def set_logger(self, name: str = None):
        """
        Function to set logger to keep track of execution.
        @param name: Name to use for the logger.
        @type name: str
        @return: None
        @rtype: None
        """


@dataclass
class Arrival:
    """
    Dataclass containing the information needed to keep track of Arrivals to allow their arrival to be scheduled.
    Uses a single timer to allow for generation of tasks with lower overhead.
    """
    ticks: Optional[int]
    task: TrainTask
    task_id: str

    def get_priority(self): # pylint: disable=missing-function-docstring
        return self.task.priority

    def get_network(self) -> Nets: # pylint: disable=missing-function-docstring
        return self.task.network_configuration.network

    def get_dataset(self) -> Dataset: # pylint: disable=missing-function-docstring
        return self.task.network_configuration.dataset

    def get_system_config(self) -> SystemParameters: # pylint: disable=missing-function-docstring
        return self.task.system_parameters

    def get_parameter_config(self) -> HyperParameters: # pylint: disable=missing-function-docstring
        return self.task.hyper_parameters

    def get_learning_config(self) -> LearningParameters: # pylint: disable=missing-function-docstring
        return self.task.learning_parameters


class SimulatedArrivalGenerator(ArrivalGenerator):
    """
    Experiments (on K8s) generator that simulates the arrival of training tasks according to a pre-defined distribution.
    As such, a set of clients can be simulated that submit various types of training jobs. See also
    BatchArrivalGenerator for an implementation that will directly schedule all arrivals on the cluster.

    N.B. it's intended purpose is to easily execute simulate different users/components requesting training jobs.
    Allowing to schedule different configuration of experiments, to see how a scheduling algorithm behaves. For example
    simulating users/systems deploying training pipelines with regular intervals.
    """
    job_dict: Dict[str, JobDescription] = None

    _tick_list: List[Arrival] = []
    _decrement = 10

    def __init__(self, custom_config: Path = None):
        super(SimulatedArrivalGenerator, self).__init__(custom_config or self.configuration_path)
        self.load_config()

    def set_logger(self, name: str = None):
        """
        Set logging name of the ArrrivalGenerator object to a recognizable name. Needs to be called once, as otherwise
        the logger is Uninitialized, resulting in failed execution.
        @param name: Name to use, by default the name 'ArrivalGenerator' is used.
        @type name: str
        @return: None
        @rtype: None
        """
        logging_name = name or self.__class__.__name__
        self.logger = logging.getLogger(logging_name)

    def generate_arrival(self, task_id: str, inter_arrival_unit: timedelta = timedelta(minutes=1)) -> Arrival:
        """
        Generate a training task for a JobDescription once the inter-arrival time has been 'deleted'.
        @param task_id: identifier for a training task corresponding to the JobDescription.
        @type task_id: str
        @return: generated arrival corresponding to the unique task_id.
        @rtype: Arrival
        """
        msg = f"Creating task for {task_id}"
        self.logger.info(msg)
        job: JobDescription = self.job_dict[task_id]

        # Select job configuration according to the weight of the `classProbability` (limit 1)
        parameters, *_ = choices(job.job_class_parameters,
                                 [job_param.class_probability for job_param in job.job_class_parameters], k=1)
        # Select job configuration according to the weight of the selected `classParameter`'s priorities (limit 1)
        priority, *_ = choices(parameters.priorities, [prio.probability for prio in parameters.priorities], k=1)

        inter_arrival_ticks = np.random.poisson(lam=job.arrival_statistic) * inter_arrival_unit.seconds
        train_task = TrainTask(identity=task_id,
                               job_parameters=parameters,
                               priority=priority,
                               experiment_type=job.experiment_type)

        return Arrival(inter_arrival_ticks, train_task, task_id)

    def run(self, duration: float):
        """
        Run function to generate arrivals during existence of the Orchestrator. Accounts time-drift correction for
        long-term execution duration of the generator (i.e. for time taken by Python interpreter).
        @return: None
        @rtype: None
        """
        self.start_time = time.time()
        self.logger.info("Populating tick lists with initial arrivals")
        for task_id in self.job_dict.keys():
            new_arrival: Arrival = self.generate_arrival(task_id)
            self._tick_list.append(new_arrival)
            msg = f"Arrival {new_arrival} arrives at {new_arrival.ticks} seconds"
            self.logger.info(msg)
        event = multiprocessing.Event()
        while self.alive and time.time() - self.start_time < duration:
            save_time = time.time()
            new_scheduled = []
            for entry in self._tick_list:
                entry.ticks -= self._decrement
                if entry.ticks <= 0:
                    self.arrivals.put(entry)
                    new_arrival = self.generate_arrival(entry.task_id)
                    new_scheduled.append(new_arrival)
                    msg = f"Arrival {new_arrival.task_id} arrives in {new_arrival.ticks} seconds"
                    self.logger.info(msg)
                else:
                    new_scheduled.append(entry)
            self._tick_list = new_scheduled
            # Correct for time drift between execution, otherwise drift adds up, and arrivals don't generate correctly
            correction_time = time.time() - save_time
            event.wait(timeout=self._decrement - correction_time)
        self.stop_time = time.time()
        msg = f"Stopped execution at: {self.stop_time}, duration: {self.stop_time - self.start_time}/{duration}"
        self.logger.info(msg)


class SequentialArrivalGenerator(ArrivalGenerator):
    """
    Experiments (on K8s) generator that directly generates all arrivals to be executed. This will rely on the scheduling
    policy of Kubeflows' Pytorch TrainOperator.

    This allows for running batches of train jobs, e.g. to run a certain experiment configuration with a number of
    replications in a fire-and-forget fashion. SimulatedArrivalGenerator for an implementation that will simulate
    arrivals following a pre-defined distribution.

    N.B. it's intended purpose is to easily execute a range of experiments, with possibly different configurations,
    where reproducability is important. For example, running a batch of experiments of a training algorithm to see
    the effect of hyperparameters on test/validation performance.
    """

    def __init__(self, custom_config: Path):
        super(SequentialArrivalGenerator, self).__init__(custom_config)
        self.load_config()

    def set_logger(self, name: str = None):
        logging_name = name or self.__class__.__name__
        self.logger = logging.getLogger(logging_name)

    def run(self, duration: float):
        """
        Helper method to start experiments. Curent implementations only runs without duration. I.e. this method
        runs in a fire-and-forget fashion without obeying to the duration parameter that may have been set.
        @param duration:
        @type duration:
        @return:
        @rtype:
        """
        self.start_time = time.time()

        description: JobDescription
        for job_name, description in self.job_dict.items():
            # TODO: Ensure seeds are set properly
            raise NotImplementedError("Run is to be re-implemented for BatchedArrivals in an upcomming release")
            for repl, seed in enumerate(description.job_class_parameters.experiment_configuration.random_seed):
                replication_name = f"{job_name}_{repl}_{seed}"
                train_task = TrainTask(identity=replication_name,
                                       job_parameters=description.job_class_parameters,
                                       priority=description.priority,
                                       # experiment_config=description.get_experiment_configuration(),
                                       replication=repl,
                                       experiment_type=description.experiment_type)

                arrival = Arrival(None, train_task, job_name)
                self.arrivals.put(arrival)
