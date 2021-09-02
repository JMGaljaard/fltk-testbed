import logging
import time
from abc import abstractmethod
from asyncio import sleep
from dataclasses import dataclass
from pathlib import Path
from queue import Queue
from random import choices
from typing import Dict, List, Union

import numpy as np

from fltk.util.singleton import Singleton
from fltk.util.task.config.parameter import TrainTask, JobDescription, ExperimentParser


@dataclass
class ArrivalGenerator(metaclass=Singleton):
    """
    Abstract Base Class for generating arrivals in the system. These tasks must be run
    """
    configuration_path: Path
    logger: logging.Logger = None
    arrivals = Queue()

    @abstractmethod
    def load_config(self):
        raise NotImplementedError("Cannot call abstract function")

    @abstractmethod
    def generate_arrival(self, task_id):
        """
        Function to generate arrival based on a Task ID.
        @param task_id:
        @type task_id:
        @return:
        @rtype:
        """
        raise NotImplementedError("Cannot call abstract function")


@dataclass
class Arrival:
    ticks: int
    task: TrainTask
    task_id: str


class ExperimentGenerator(ArrivalGenerator):
    start_time: float = -1
    stop_time: float = -1
    job_dict: Dict[str, JobDescription] = None

    _tick_list: List[Arrival] = []
    _alive: bool = False
    _decrement = 1
    __default_config: Path = Path('configs/example_cloud_experiment.json')

    def __init__(self, custom_config: Path = None):
        super(ExperimentGenerator, self).__init__(custom_config or self.__default_config)

    def set_logger(self, name: str = None):
        """
        Set logging name to make debugging easier.
        @param name:
        @type name:
        @return:
        @rtype:
        """
        logging_name = name or self.__class__.__name__
        self.logger = logging.getLogger(logging_name)

    def load_config(self):
        """
        Generate
        """
        parser = ExperimentParser(config_path=self.configuration_path)
        experiment_descriptions = parser.parse()
        self.job_description = {f'train_job_{indx}': item for indx, item in enumerate(experiment_descriptions)}

    def generate_arrival(self, task_id: str) -> None:
        """
        Generate a training task for a JobDescription once the inter-arrival time has been 'deleted'.
        @param train_id: id for a training task correspnoding to the JobDescription.
        @type train_id: str
        """
        self.logger.info(f"Creating task for {task_id}")
        job = self.job_description[task_id]
        parameters = choices(job.job_class_parameters, [param.probability for param in job.job_class_parameters])[0]
        priority = choices(parameters.priorities, [prio.probabilities for prio in parameters.priorities], k=1)[0]

        inter_arrival_ticks = np.random.poisson(lam=job.arrival_statistic)
        train_task = TrainTask(task_id, parameters, priority)

        self._tick_list.append(Arrival(inter_arrival_ticks, train_task, task_id))

    def start(self, duration: Union[float, int]):
        """
        Function to start arrival generator, requires to
        @return:
        @rtype: int, float
        """
        if not self.logger:
            self.set_logger()
        self.logger.info("Starting execution of arrival generator...")
        self._alive = True
        self.run(duration)

    def stop(self) -> None:
        self.logger.info("Received stopping signal")
        self._alive = False

    def run(self, duration: float):
        """
        Run function to generate arrivals during existence of the Orchestrator. WIP.

        Currently supports for time-drift correction to account for execution duration of the generator.
        @return:
        @rtype:
        """
        self.start_time = time.time()
        self.logger.info("Populating tick lists with initial arrivals")
        for task_id, job_description in self.job_dict.keys():
            self.generate_arrival(task_id)
            self.logger.debug(f"Arrival {task_id} arrives at {self._tick_list[-1].ticks} seconds")

        while self._alive and time.time() - self.start_time < duration:
            save_time = time.time()
            for indx, entry in enumerate(self._tick_list):
                entry.ticks -= self._decrement
                if entry.ticks <= 0:
                    self._tick_list.pop(indx)
                    self.arrivals.put(entry)
                    self.generate_arrival(entry.task_id)
            # Correct for time drift between execution, otherwise drift adds up, and arrivals don't generate correctly
            correction_time = time.time() - save_time
            sleep(self._decrement - correction_time)
        self.stop_time = time.time()
        self.logger.info(f"Stopped execution at: {self.stop_time}, duration: {self.stop_time - self.start_time}/{duration}")
