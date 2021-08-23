import logging
import random
from abc import ABC, abstractmethod
from asyncio import sleep
from dataclasses import dataclass
from pathlib import Path
from random import choices
from time import time
from typing import Dict, List

import numpy as np

from fltk.util.task.config.parameter import TrainTask, JobDescription, ExperimentParser


@dataclass
class ArrivalGenerator(ABC):
    """
    Abstract Base Class for generating arrivals in the system. These tasks must be run
    """
    configuration_path: Path
    logger: logging.Logger = None

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
    job_description: Dict[str, JobDescription] = None

    _tick_list: List[Arrival] = []
    _alive: bool = False
    _decrement = 1

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

    def set_seed(self, seed: int = 42):
        """
        Function to pre-set the seed used by the Experiment generator, this allows for better reproducability of the
        experiments.
        @param seed: Seed to be used by the `random` library for experiment generation
        @type seed: int
        @return:
        @rtype:
        """
        random.seed(seed)

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
        @param train_id: identifier for a training task correspnoding to the JobDescription.
        @type train_id: String
        """
        # TODO: logging
        job = self.job_description[task_id]
        parameters = choices(job.job_class_parameters, [param.probability for param in job.job_class_parameters])[0]
        priority = choices(parameters.priorities, [prio.probabilities for prio in parameters.priorities], k=1)[0]

        inter_arrival_ticks = np.random.poisson(lam=job.arrival_statistic)
        train_task = TrainTask(parameters, priority, task_id)

        self._tick_list.append(Arrival(inter_arrival_ticks, train_task, task_id))

    def start(self):
        """
        Function to start arrival generator, requires to
        @return:
        @rtype:
        """
        if not self.logger:
            self.set_logger()
        self.logger.info("Starting execution of arrival generator...")
        self._alive = True
        self.run()

    def stop(self) -> None:
        self.logger.info("Received stopping signal")
        self._alive = False

    def run(self):
        """
        Run function to generate arrivals during existence of the Orchestrator. WIP.

        Currently supports for time-drift correction to account for execution duration of the generator.
        @return:
        @rtype:
        """
        self.start_time = time()
        while self._alive:
            arrived = []
            save_time = time()
            for indx, entry in enumerate(self._tick_list):
                entry.ticks -= self._decrement
                if entry.ticks <= 0:
                    self._tick_list.pop(indx)
                    arrived.append(entry)
                    self.generate_arrival(entry.task_id)

            # Correct for time drift between execution, otherwise drift adds up, and arrivals don't generate correctly
            correction_time = time() - save_time
            sleep(self._decrement - correction_time)
        self.stop_time = time()
        self.logger.info(f"Stopped execution at: {self.stop_time}, duration: {self.stop_time - self.start_time}")
