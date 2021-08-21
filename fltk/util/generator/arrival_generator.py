from abc import ABC, abstractmethod
from asyncio import sleep
from dataclasses import dataclass
from pathlib import Path
from random import choices
from time import time
from typing import Dict, List

import numpy as np

from fltk.util.config.parameter import ExperimentParser, TrainTask, JobDescription


@dataclass
class ArrivalGenerator(ABC):
    """
    Abstract Base Class for generating arrivals in the system. These tasks must be run
    """
    configuration_path: Path

    @abstractmethod
    def load_config(self):
        pass

    @abstractmethod
    def generate_arrival(self, task_id):
        """
        Function to generate arrival based on a Task ID.
        @param task_id:
        @type task_id:
        @return:
        @rtype:
        """
        pass


@dataclass
class Arrival:
    ticks: int
    task: TrainTask
    task_id: str


class ExperimentGenerator(ArrivalGenerator):
    alive: bool = True
    decrement = 1

    start_time: float
    stop_time: float
    job_description: Dict[str, JobDescription]

    tick_list: List[Arrival] = []

    def populate(self):
        # TODO: logging
        for key in self.job_description.keys():
            self.generate_arrival(key)

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

        self.tick_list.append(Arrival(inter_arrival_ticks, train_task, task_id))

    def run(self):
        """
        Run function to generate arrivals during existence of the Orchestrator. WIP.

        Currently supports for time-drift correctino to account for execution duration of the generator
        @return:
        @rtype:
        """
        # TODO: logging
        self.start_time = time()
        while self.alive:
            arrived = []
            save_time = time()
            for indx, entry in enumerate(self.tick_list):
                entry.ticks -= self.decrement
                if entry.ticks <= 0:
                    self.tick_list.pop(indx)
                    arrived.append(entry)
                    self.generate_arrival(entry.task_id)

            # Correct for time drift between execution, otherwise drift adds up, and arrivals don't line up correctly
            correction_time = time() - save_time
            sleep(self.decrement - correction_time)


@dataclass
class EvaluationGenerator(ArrivalGenerator):
    def load_config(self):
        pass

    def generate_arrivals(self):
        pass
