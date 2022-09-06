import abc
import logging

import torch


class LearningScheduler(abc.ABC):
    """
    Abstract base class for learning rate scheduler objects.
    """

    @abc.abstractmethod
    def step(self): # pylint: disable=missing-function-docstring
        raise NotImplementedError()


class MinCapableStepLR(LearningScheduler):
    """
    Stepping learning rate scheduler with minimum learning rate.
    """

    def __init__(self, optimizer: torch.optim.Optimizer, step_size, gamma, min_lr):
        """
        :param optimizer:
        :type optimizer: torch.optim
        :param step_size: # of epochs between LR updates
        :type step_size: int
        :param gamma: multiplication factor for LR update
        :type gamma: float
        :param min_lr: minimum learning rate
        :type min_lr: float
        """
        self.logger = logging.getLogger('MinCapableStepLR')

        self.optimizer = optimizer
        self.step_size = step_size
        self.gamma = gamma
        self.min_lr = min_lr

        self.epoch_idx = 0

    def step(self):
        """
        Adjust the learning rate as necessary.
        """
        self.increment_epoch_index()

        if self.is_time_to_update_lr():
            self.logger.debug("Updating LR for optimizer")

            self.update_lr()

    def is_time_to_update_lr(self):
        """
        Helper function to caluclate whether the learning rate should be updated.
        @return: Boolean indicating whether it is time to update.
        @rtype: bool.
        """
        return self.epoch_idx % self.step_size == 0

    def update_lr(self):
        """
        Function to update the learning rate of module in the encapsulated torch.optim.Optimizer object.
        @return: None
        @rtype: None
        """
        if self.optimizer.param_groups[0]['lr'] * self.gamma >= self.min_lr:
            self.optimizer.param_groups[0]['lr'] *= self.gamma
        else:
            self.logger.warning("Updating LR would place it below min LR. Skipping LR update.")

        self.logger.debug(f"New LR: {self.optimizer.param_groups[0]['lr']}")

    def increment_epoch_index(self):
        """
        Update the epoch tracker attribute after completing an epoch.
        @return: None.
        @rtype: None
        """
        self.epoch_idx += 1
