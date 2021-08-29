import datetime
import logging
from pathlib import Path

import torch
import torch.distributed as dist
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix

from fltk.nets.util.evaluation import calculate_class_precision, calculate_class_recall
from fltk.nets.util.utils import save_model, load_model_from_file
from fltk.util.config.base_config import BareConfig
from fltk.util.results import EpochData

logging.basicConfig(level=logging.DEBUG)


class Client:

    def __init__(self, rank, task_id, config: BareConfig = None):
        """
        @param rank:
        @type rank:
        @param task_id:
        @type task_id:
        @param config:
        @type config:
        """
        self._logger = logging.getLogger(f'Client-{rank}-{task_id}')

        self._logger.info("Initializing learning client")
        self._logger.debug(f"Configuration received: {config}")
        self._id = rank
        self._task_id = task_id
        self.args = config
        self.device = self._init_device()
        self.dataset = self.args.get_dataset()
        self.model = self.args.get_model()
        self.loss_function = self.args.get_loss_function()
        self.optimizer: torch.nn.Module
        self.scheduler = self.args.get_scheduler()

    def prepare_learner(self, distributed: bool = False, backend=None):
        self._logger.info(f"Preparing learner model with distributed={distributed}")
        self.model.to(self.device)
        if distributed:
            dist.init_process_group(backend)
            self.model = torch.nn.parallel.DistributedDataParallel(self.model)

        self.optimizer = self.args.get_optimizer(self.model.parameters(),
                                                 self.args)

    def _init_device(self, cuda_device: torch.device = torch.device('cuda:0')):
        """
        Initialize Torch to use available devices. Either prepares CUDA device, or disables CUDA during execution to run
        with CPU only inference/training.
        @param cuda_device: Torch device to use, refers to the CUDA device to be used in case there are multiple.
        Defaults to the first cuda device when CUDA is enabled at index 0.
        @type cuda_device: torch.device
        @return:
        @rtype:
        """
        if self.args.cuda and torch.cuda.is_available():
            return torch.device(cuda_device)
        else:
            # Force usage of CPU
            torch.cuda.is_available = lambda: False
            return torch.device("cpu")

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """

        model_file = Path(f'{self.model.__name__}.model')
        default_model_path = Path(self.args.get_default_model_folder_path()).joinpath(model_file)
        load_model_from_file(self.model, default_model_path)

    def train(self, epoch):
        """
        :param epoch: Current epoch #
        :type epoch: int
        """

        running_loss = 0.0
        final_running_loss = 0.0

        for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize

            outputs = self.model(inputs)

            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()
            running_loss += float(loss.detach().item())
            del loss, outputs
            if i % self.args.get_log_interval() == 0:
                self.args.get_logger().info(
                    '[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))
                final_running_loss = running_loss / self.args.get_log_interval()
                running_loss = 0.0
        self.scheduler.step()
        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        return final_running_loss

    def test(self):
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
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.detach().cpu().view_as(predicted).numpy())
                pred_.extend(predicted.detach().cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        accuracy = 100.0 * correct / total
        confusion_mat = confusion_matrix(targets_, pred_)

        class_precision = calculate_class_precision(confusion_mat)
        class_recall = calculate_class_recall(confusion_mat)

        self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
        self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))

        return accuracy, loss, class_precision, class_recall

    def run_epochs(self):
        """
        Function to run epochs wit
        """
        num_epochs = self.config.epochs
        start_time_train = datetime.datetime.now()
        # Make epochs 1 index.
        for epoch in range(1, num_epochs + 1):
            loss = self.train(epoch)

            if self._id == 0:
                # Let only the 'master node' work on training. Possibly DDP can be used
                # to have a distributed test loader as well to speed up (would require
                # aggregration of data.
                # Example https://github.com/fabio-deep/Distributed-Pytorch-Boilerplate/blob/0206247150720ca3e287e9531cb20ef68dc9a15f/src/datasets.py#L271-L303.
                accuracy, loss, class_precision, class_recall = self.test()

        elapsed_time_train = datetime.datetime.now() - start_time_train
        train_time_ms = int(elapsed_time_train.total_seconds() * 1000)

        start_time_test = datetime.datetime.now()
        accuracy, test_loss, class_precision, class_recall = self.test()
        elapsed_time_test = datetime.datetime.now() - start_time_test
        test_time_ms = int(elapsed_time_test.total_seconds() * 1000)

        data = EpochData(train_time_ms, test_time_ms, loss, accuracy, test_loss, class_precision,
                         class_recall, client_id=self._id)

    def save_model(self, epoch, suffix):
        """
        Move function to utils directory.
        Saves the model if necessary.
        """
        self.args.get_logger().debug(f"Saving model to flat file storage. Saved at epoch #{epoch}")

        save_model(self.model, epoch, self.args.get_save_model_folder_path(), self.args)


