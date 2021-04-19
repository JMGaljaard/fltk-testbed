import copy
import datetime
import os
import random
import time
from dataclasses import dataclass
from typing import List

import torch
from torch.distributed import rpc
import logging
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.metrics import classification_report

from fltk.datasets.distributed import DistCIFAR10Dataset
from fltk.schedulers import MinCapableStepLR
from fltk.util.arguments import Arguments
from fltk.util.log import FLLogger

import yaml

from fltk.util.results import EpochData

logging.basicConfig(level=logging.DEBUG)



def _call_method(method, rref, *args, **kwargs):
    """helper for _remote_method()"""
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    """
    executes method(*args, **kwargs) on the from the machine that owns rref

    very similar to rref.remote().method(*args, **kwargs), but method() doesn't have to be in the remote scope
    """
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

def _remote_method_async(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)

class Client:
    counter = 0
    finished_init = False
    dataset = None
    epoch_results: List[EpochData] = []
    epoch_counter = 0


    def __init__(self, id, log_rref, rank, world_size, config = None):
        logging.info(f'Welcome to client {id}')
        self.id = id
        self.log_rref = log_rref
        self.rank = rank
        self.world_size = world_size
        # self.args = Arguments(logging)
        self.args = config
        self.args.init_logger(logging)
        self.device = self.init_device()
        self.set_net(self.load_default_model())
        self.loss_function = self.args.get_loss_function()()
        self.optimizer = torch.optim.SGD(self.net.parameters(),
                                         lr=self.args.get_learning_rate(),
                                         momentum=self.args.get_momentum())
        self.scheduler = MinCapableStepLR(self.args.get_logger(), self.optimizer,
                                          self.args.get_scheduler_step_size(),
                                          self.args.get_scheduler_gamma(),
                                          self.args.get_min_lr())

    def init_device(self):
        if self.args.cuda and torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def ping(self):
        return 'pong'

    def rpc_test(self):
        sleep_time = random.randint(1, 5)
        time.sleep(sleep_time)
        self.local_log(f'sleep for {sleep_time} seconds')
        self.counter += 1
        log_line = f'Number of times called: {self.counter}'
        self.local_log(log_line)
        self.remote_log(log_line)

    def remote_log(self, message):
        _remote_method_async(FLLogger.log, self.log_rref, self.id, message, time.time())

    def local_log(self, message):
        logging.info(f'[{self.id}: {time.time()}]: {message}')

    def set_configuration(self, config: str):
        yaml_config = yaml.safe_load(config)

    def init(self):
        pass

    def init_dataloader(self, ):
        self.args.distributed = True
        self.args.rank = self.rank
        self.args.world_size = self.world_size
        self.dataset = DistCIFAR10Dataset(self.args)
        self.finished_init = True
        logging.info('Done with init')

    def is_ready(self):
        return self.finished_init

    def set_net(self, net):
        self.net = net
        self.net.to(self.device)

    def load_model_from_file(self, model_file_path):
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def get_nn_parameters(self):
        """
        Return the NN's parameters.
        """
        return self.net.state_dict()

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def load_model_from_file(self, model_file_path):
        """
        Load a model from a file.

        :param model_file_path: string
        """
        model_class = self.args.get_net()
        model = model_class()

        if os.path.exists(model_file_path):
            try:
                model.load_state_dict(torch.load(model_file_path))
            except:
                self.args.get_logger().warning("Couldn't load model. Attempting to map CUDA tensors to CPU to solve error.")

                model.load_state_dict(torch.load(model_file_path, map_location=torch.device('cpu')))
        else:
            self.args.get_logger().warning("Could not find model: {}".format(model_file_path))

        return model

    def get_client_index(self):
        """
        Returns the client index.
        """
        return self.client_idx

    def update_nn_parameters(self, new_params):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        self.net.load_state_dict(copy.deepcopy(new_params), strict=True)
        self.remote_log(f'Weigths of the model are updated')

    def load_default_model(self):
        """
        Load a model from default model file.

        This is used to ensure consistent default model behavior.
        """
        model_class = self.args.get_net()
        default_model_path = os.path.join(self.args.get_default_model_folder_path(), model_class.__name__ + ".model")

        return self.load_model_from_file(default_model_path)

    def train(self, epoch):
        """
        :param epoch: Current epoch #
        :type epoch: int
        """
        # self.net.train()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())

        running_loss = 0.0
        final_running_loss = 0.0
        if self.args.distributed:
            self.dataset.train_sampler.set_epoch(epoch)

        for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
            inputs, labels = inputs.to(self.device), labels.to(self.device)

            # zero the parameter gradients
            self.optimizer.zero_grad()

            # forward + backward + optimize
            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)
            loss.backward()
            self.optimizer.step()

            # print statistics
            running_loss += loss.item()
            if i % self.args.get_log_interval() == 0:
                self.args.get_logger().info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))
                final_running_loss = running_loss / self.args.get_log_interval()
                running_loss = 0.0

        self.scheduler.step()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        return final_running_loss, self.get_nn_parameters()

    def test(self):
        self.net.eval()

        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        accuracy = 100 * correct / total
        confusion_mat = confusion_matrix(targets_, pred_)

        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)

        self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
        self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
        self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
        self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
        self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
        self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))

        return accuracy, loss, class_precision, class_recall

    def run_epochs(self, num_epoch):
        start_time_train = datetime.datetime.now()
        loss = weights = None
        for e in range(num_epoch):
            loss, weights = self.train(self.epoch_counter)
            self.epoch_counter += 1
        elapsed_time_train = datetime.datetime.now() - start_time_train
        train_time_ms = int(elapsed_time_train.total_seconds()*1000)

        start_time_test = datetime.datetime.now()
        accuracy, test_loss, class_precision, class_recall = self.test()
        elapsed_time_test = datetime.datetime.now() - start_time_test
        test_time_ms = int(elapsed_time_test.total_seconds()*1000)

        data = EpochData(self.epoch_counter, train_time_ms, test_time_ms, loss, accuracy, test_loss, class_precision, class_recall, client_id=self.id)
        self.epoch_results.append(data)

        # Copy GPU tensors to CPU
        for k, v in weights.items():
            weights[k] = v.cpu()
        return data, weights

    def save_model(self, epoch, suffix):
        """
        Saves the model if necessary.
        """
        self.args.get_logger().debug("Saving model to flat file storage. Save #{}", epoch)

        if not os.path.exists(self.args.get_save_model_folder_path()):
            os.mkdir(self.args.get_save_model_folder_path())

        full_save_path = os.path.join(self.args.get_save_model_folder_path(), "model_" + str(self.client_idx) + "_" + str(epoch) + "_" + suffix + ".model")
        torch.save(self.get_nn_parameters(), full_save_path)

    def calculate_class_precision(self, confusion_mat):
        """
        Calculates the precision for each class from a confusion matrix.
        """
        return np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=0)

    def calculate_class_recall(self, confusion_mat):
        """
        Calculates the recall for each class from a confusion matrix.
        """
        return np.diagonal(confusion_mat) / np.sum(confusion_mat, axis=1)

    def get_client_datasize(self):
        return len(self.dataset.get_train_sampler())

    def __del__(self):
        print(f'Client {self.id} is stopping')
