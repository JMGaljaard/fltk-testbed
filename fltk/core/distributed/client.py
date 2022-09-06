from __future__ import annotations

import datetime
import logging
from pathlib import Path
from typing import List, Tuple, TYPE_CHECKING

import numpy as np
import torch
from sklearn.metrics import confusion_matrix
from torch.utils.tensorboard import SummaryWriter

from fltk.core.distributed.dist_node import DistNode
from fltk.nets import get_net
from fltk.nets.util import calculate_class_precision, calculate_class_recall, save_model, load_model_from_file
from fltk.schedulers import MinCapableStepLR, LearningScheduler
from fltk.strategy import get_optimizer
from fltk.util.config.definitions.dataset import get_dist_dataset
from fltk.util.results import EpochData

if TYPE_CHECKING:
    from fltk.util.config import DistributedConfig, DistLearningConfig


class DistClient(DistNode):

    def __init__(self, rank: int, task_id: str, world_size: int, config: DistributedConfig = None,
                 learning_params: DistLearningConfig = None):
        """
        @param rank: PyTorch rank provided by KubeFlow setup.
        @type rank: int
        @param task_id: String id representing the UID of the training task
        @type task_id: str
        @param config: Parsed configuration file representation to extract runtime information from.
        @type config: DistributedConfig
        @param learning_params: Hyper-parameter configuration to be used during the training process by the learner.
        @type learning_params: DistLearningConfig
        """
        self._logger = logging.getLogger(f'Client-{rank}-{task_id}')

        self._logger.info("Initializing learning client")
        self._id = rank
        self._world_size = world_size
        self._task_id = task_id

        self.config = config
        self.learning_params = learning_params

        # Create model and dataset
        self.loss_function = self.learning_params.get_loss_function()()
        self.dataset = get_dist_dataset(self.learning_params.dataset)(self.config, self.learning_params, self._id, self._world_size)
        self.model = get_net(self.learning_params.model)()
        self.device = self._init_device()

        self.optimizer: torch.optim.Optimizer
        self.scheduler: LearningScheduler
        self.tb_writer: SummaryWriter

    def prepare_learner(self, distributed: bool = False) -> None:
        """
        Function to prepare the learner, i.e. load all the necessary data into memory.
        @param distributed: Indicates whether the execution must be run in Distributed fashion with DDP.
        @type distributed: bool
        @return: None
        @rtype: None
        """
        self._logger.info(f"Preparing learner model with distributed={distributed}")
        self.model.to(self.device)
        if distributed:
            # Wrap the model to use pytorch DistributedDataParallel wrapper for all reduce.
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        # Currently, it is assumed to use an SGD optimizer, using non-federated optimizer types.
        optim_type = get_optimizer(self.learning_params.optimizer, federated=False)
        self.optimizer = optim_type(self.model.parameters(), **self.learning_params.optimizer_args)
        self.scheduler = MinCapableStepLR(self.optimizer,
                                          self.learning_params.scheduler_step_size,
                                          self.learning_params.scheduler_gamma,
                                          self.learning_params.min_lr)

        if self.config.execution_config.tensorboard.active and self._id == 0:
            self.tb_writer = SummaryWriter(
                str(self.config.get_log_path(self._task_id, self._id, self.learning_params)))

    def stop_learner(self):
        """
        @deprecated Function to stop a learner upon command of another learner.
        @return: None
        @rtype: None
        """
        self._logger.info(f"Tearing down Client {self._id}")
        self.tb_writer.close()

    def _init_device(self, default_device: torch.device = torch.device('cpu')): # pylint: disable=no-member
        """
        Initialize Torch to use available devices. Either prepares CUDA device, or disables CUDA during execution to run
        with CPU only inference/training.
        @param default_device: Torch device to use, refers to the CUDA device to be used in case there are multiple.
        Defaults to the first cuda device when CUDA is enabled at index 0.
        @type default_device: torch.device
        @return: None
        @rtype: None
        """
        if self.config.cuda_enabled() and torch.cuda.is_available():
            return torch.device('cuda') # pylint: disable=no-member
        # Force usage of CPU
        torch.cuda.is_available = lambda: False
        return default_device

    def load_default_model(self):
        """
        @deprecated Load a model from default model file. This function could be used to ensure consistent default model
        behavior. When using PyTorch's DistributedDataParallel, however, the first step will always synchronize the
        model.
        """

        model_file = Path(f'{self.model.__name__}.model')
        default_model_path = Path(self.config.get_default_model_folder_path()).joinpath(model_file)
        load_model_from_file(self.model, default_model_path)

    def train(self, epoch, log_interval: int = 50):
        """
        Function to start training, regardless of DistributedDataParallel (DPP) or local training. DDP will account for
        synchronization of nodes. If extension requires to make use of torch.distributed.send and torch.distributed.recv
        (for example for customized training or Federated Learning), additional torch.distributed.barrier calls might
        be required to launch.

        :param epoch: Current epoch number.
        :type epoch: int
        @param log_interval: Iteration interval at which to log.
        @type log_interval: int
        """
        running_loss = 0.0
        final_running_loss = 0.0
        self.model.train()
        for i, (inputs, labels) in enumerate(self.dataset.get_train_loader()):
            # zero the parameter gradients
            self.optimizer.zero_grad()

            # Forward through the net to train
            outputs = self.model(inputs.to(self.device))

            # Calculate the loss
            loss = self.loss_function(outputs, labels.to(self.device))

            # Update weights, DPP will account for synchronization of the weights.
            loss.backward()
            self.optimizer.step()

            running_loss += float(loss.detach().item())
            if i % log_interval == 0:
                self._logger.info(f'[{epoch:d}, {i:5d}] loss: {running_loss / log_interval:.3f}')
                final_running_loss = running_loss / log_interval
                running_loss = 0.0
        self.scheduler.step()

        # Save model
        if self.config.should_save_model(epoch):
            # Note that currently this is not supported in the Framework. However, the creation of a ReadWriteMany
            # PVC in the deployment charts, and mounting this in the appropriate directory, would resolve this issue.
            # This can be done by copying the setup of the PVC used to record the TensorBoard information (used by
            # logger created by the rank==0 node during the training process (i.e. to keep track of process)).
            self.save_model(epoch)

        return final_running_loss

    def test(self) -> Tuple[float, float, np.array, np.array, np.array]:
        """
        Function to test the trained model using the test dataset. Returns a number of statistics of the training
        process.
        @warning Currently the testing process assumes that the model performs classification, for different types of
        tasks this function would need to be updated.
        @return: (accuracy, loss, class_precision, class_recall, confusion_mat): class_precision, class_recal and
        confusion_mat will be in a `np.array`, which corresponds to the number of classes in a classification task.
        @rtype: Tuple[float, float, np.array, np.array, np.array]:
        """
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0

        # Disable gradient calculation, as we are only interested in predictions
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.model(images)
                # Currently, the FLTK framework assumes that a classification task is performed (hence max).
                # Future work may add support for non-classification training.
                _, predicted = torch.max(outputs.data, 1) # pylint: disable=no-member
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.detach().cpu().view_as(predicted).numpy())
                pred_.extend(predicted.detach().cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        accuracy = 100.0 * correct / total
        confusion_mat: np.array = confusion_matrix(targets_, pred_)

        class_precision: np.array = calculate_class_precision(confusion_mat)
        class_recall: np.array = calculate_class_recall(confusion_mat)

        self._logger.debug(f'Test set: Accuracy: {correct}/{total} ({accuracy:.0f}%)')
        self._logger.debug(f'Test set: Loss: {loss}')
        self._logger.debug(f"Confusion Matrix:\n{confusion_mat}")
        self._logger.debug(f"Class precision: {class_precision}")
        self._logger.debug(f"Class recall: {class_recall}")

        return accuracy, loss, class_precision, class_recall, confusion_mat

    def run_epochs(self) -> List[EpochData]:
        """
        Function to run training epochs using the pre-set Hyper-Parameters.
        @return: A list of data gathered during the execution, containing progress information such as accuracy.
        See also the EpochData dataclass.
        @rtype: List[EpochData]
        """
        max_epoch = self.learning_params.max_epoch + 1
        start_time_train = datetime.datetime.now()
        epoch_results = []
        for epoch in range(1, max_epoch):
            train_loss = self.train(epoch)

            # Let only the 'master node' work on training. Possibly DDP can be used
            # to have a distributed test loader as well to speed up (would require
            # aggregation of data).
            elapsed_time_train = datetime.datetime.now() - start_time_train
            train_time_ms = int(elapsed_time_train.total_seconds() * 1000)

            start_time_test = datetime.datetime.now()
            accuracy, test_loss, class_precision, class_recall, confusion_mat = self.test()

            elapsed_time_test = datetime.datetime.now() - start_time_test
            test_time_ms = int(elapsed_time_test.total_seconds() * 1000)

            data = EpochData(epoch_id=epoch,
                             duration_train=train_time_ms,
                             duration_test=test_time_ms,
                             loss_train=train_loss,
                             accuracy=accuracy,
                             loss=test_loss,
                             class_precision=class_precision,
                             class_recall=class_recall,
                             confusion_mat=confusion_mat,
                             num_epochs=max_epoch)

            epoch_results.append(data)
            if self._id == 0:
                self.log_progress(data, epoch)
        return epoch_results

    def save_model(self, epoch):
        """
        @deprecated Move function to utils directory.
        """
        self._logger.debug(f"Saving model to flat file storage. Saved at epoch #{epoch}")
        save_model(self.model, str(self.config.get_save_model_folder_path()), epoch)

    def log_progress(self, epoch_data: EpochData, epoch: int):
        """
        Function to log the progress of the learner between epochs. Only the MASTER/RANK=0 process should call this
        function. Other learners' SummaryWriters data will be gone after the pod reached 'Completed' status.
        @param epoch_data: data object which needs to be logged with the learners SummaryWriter.
        @type epoch_data: EpochData
        @param epoch: Number of the epoch.
        @type epoch: int
        @return: None
        @rtype: None
        """
        if self.config.execution_config.tensorboard.active:
            self.tb_writer.add_scalar('training loss per epoch',
                                      epoch_data.loss_train,
                                      epoch)

            self.tb_writer.add_scalar('accuracy per epoch',
                                      epoch_data.accuracy,
                                      epoch)
