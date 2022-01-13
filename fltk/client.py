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
from torch.distributed.rpc import RRef

from fltk.schedulers import MinCapableStepLR
from fltk.strategy.aggregation import FedAvg
from fltk.strategy.offloading import OffloadingStrategy
from fltk.util.arguments import Arguments
from fltk.util.fed_avg import average_nn_parameters
from fltk.util.log import FLLogger

import yaml

from fltk.util.profiler import Profiler
from fltk.util.results import EpochData

logging.basicConfig(
    level=logging.DEBUG,

    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
)

global_dict = {}
global_model_weights = {}
global_model_data_size = 0
global_offload_received = False


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


def _remote_method_async(method, rref, *args, **kwargs) -> torch.Future:
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _remote_method_async_by_info(method, worker_info, *args, **kwargs):
    args = [method, worker_info] + list(args)
    return rpc.rpc_async(worker_info, _call_method, args=args, kwargs=kwargs)

class Client:
    counter = 0
    finished_init = False
    dataset = None
    epoch_results: List[EpochData] = []
    epoch_counter = 0
    server_ref = None

    # Model offloading
    received_offload_model = False
    offloaded_model_weights = None
    call_to_offload = False
    client_to_offload_to : str = None

    strategy = OffloadingStrategy.VANILLA


    def __init__(self, id, log_rref, rank, world_size, config = None):
        logging.info(f'Welcome to client {id}')
        self.id = id
        global_dict['id'] = id
        global global_model_weights, global_offload_received, global_model_data_size
        global_model_weights = None
        global_offload_received = False
        global_model_data_size = 0
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
        self.strategy = OffloadingStrategy.Parse(config.offload_strategy)
        self.configure_strategy(self.strategy)


    def configure_strategy(self, strategy : OffloadingStrategy):
        if strategy == OffloadingStrategy.VANILLA:
            logging.info('Running with offloading strategy: VANILLA')
            self.deadline_enabled = False
            self.swyh_enabled = False
            self.freeze_layers_enabled = False
            self.offload_enabled = False
        if strategy == OffloadingStrategy.DEADLINE:
            logging.info('Running with offloading strategy: DEADLINE')
            self.deadline_enabled = True
            self.swyh_enabled = False
            self.freeze_layers_enabled = False
            self.offload_enabled = False
        if strategy == OffloadingStrategy.SWYH:
            logging.info('Running with offloading strategy: SWYH')
            self.deadline_enabled = True
            self.swyh_enabled = True
            self.freeze_layers_enabled = False
            self.offload_enabled = False
        if strategy == OffloadingStrategy.FREEZE:
            logging.info('Running with offloading strategy: FREEZE')
            self.deadline_enabled = True
            self.swyh_enabled = False
            self.freeze_layers_enabled = True
            self.offload_enabled = False
        if strategy == OffloadingStrategy.MODEL_OFFLOAD:
            logging.info('Running with offloading strategy: MODEL_OFFLOAD')
            self.deadline_enabled = True
            self.swyh_enabled = False
            self.freeze_layers_enabled = True
            self.offload_enabled = True
        logging.info(f'Offload strategy params: deadline={self.deadline_enabled}, swyh={self.swyh_enabled}, freeze={self.freeze_layers_enabled}, offload={self.offload_enabled}')


    def init_device(self):
        if self.args.cuda and torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def send_reference(self, server_ref):
        self.local_log(f'Got worker_info from server {server_ref}')
        self.server_ref = server_ref

    @staticmethod
    def static_ping():
        print(f'Got static ping with global_dict={global_dict}')

    def ping(self):
        self.local_log(f'Pong!')
        self.local_log(f'Pong2! {self.id}')
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
        # self.dataset = DistCIFAR10Dataset(self.args)
        self.dataset = self.args.DistDatasets[self.args.dataset_name](self.args)
        self.finished_init = True
        logging.info('Done with init')

    def is_ready(self):
        logging.info("Client is ready")
        return self.finished_init, RRef(self)

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
        if self.log_rref:
            self.remote_log(f'Weights of the model are updated')

    def report_performance_async(self, performance_data):
        self.local_log('Reporting performance')
        from fltk.federator import Federator
        return _remote_method_async(Federator.perf_metric_endpoint, self.server_ref, self.id, performance_data)

    def report_performance_estimate(self, performance_data):
        self.local_log('Reporting performance estimate')
        from fltk.federator import Federator
        return _remote_method_async(Federator.perf_est_endpoint, self.server_ref, self.id, performance_data)

    @staticmethod
    def offload_receive_endpoint(model_weights, num_train_samples):
        print(f'Got the offload_receive_endpoint endpoint')
        global global_model_weights, global_offload_received, global_model_data_size
        global_model_weights = copy.deepcopy(model_weights.copy())
        global_model_data_size = num_train_samples
        global_offload_received = True

    @staticmethod
    def offload_receive_endpoint_2(string):
        print(f'Got the offload_receive_endpoint endpoint')
        print(f'Got the offload_receive_endpoint endpoint with arg={string}')
        # global global_model_weights, global_offload_received
        # global_model_weights = model_weights.copy(deep=True)
        # global_offload_received = True


    def call_to_offload_endpoint(self, client_to_offload: RRef):
        self.local_log(f'Got the call to offload endpoint to {client_to_offload}')
        self.client_to_offload_to = client_to_offload
        self.call_to_offload = True

    def freeze_layers(self, until):
        ct = 0
        for child in self.net.children():
            ct += 1
            if ct < until:
                for param in child.parameters():
                    param.requires_grad = False

    def unfreeze_layers(self):
        for param in self.net.parameters():
            param.requires_grad = True

    def train(self, epoch, deadline: int = None, warmup=False):
        """

        Different modes:
        1. Vanilla
        2. Deadline
        3. SWYH
        4. Just Freeze
        5. Model Offload


        :: Vanilla
        Disable deadline
        Disable swyh
        Disable offload

        :: Deadline
        We need to keep track of the incoming deadline
        We don't need to send data before the deadline

        :param epoch: Current epoch #
        :type epoch: int
        """
        start_time = time.time()
        deadline_threshold = 10
        train_stop_time = None
        if self.deadline_enabled and deadline is not None:
            train_stop_time = start_time + deadline - deadline_threshold

        # Ignore profiler for now
        # p = Profiler()
        # p.attach(self.net)

        # self.net.train()
        global global_model_weights, global_offload_received
        # deadline_time = None
        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_start_suffix())

        running_loss = 0.0
        final_running_loss = 0.0
        if self.args.distributed:
            self.dataset.train_sampler.set_epoch(epoch)
        self.args.get_logger().info(f'{self.id}: Number of training samples: {len(list(self.dataset.get_train_loader()))}')
        number_of_training_samples = len(list(self.dataset.get_train_loader()))
        # Ignore profiler for now
        # performance_metric_interval = 20
        # perf_resp = None

        # Profiling parameters
        profiling_size = self.args.profiling_size
        profiling_data = np.zeros(profiling_size)
        active_profiling = True

        control_start_time = time.time()
        training_process = 0
        for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
            start_train_time = time.time()

            if self.offload_enabled and not warmup:
                # Check if there is a call to offload
                if self.call_to_offload:
                    self.args.get_logger().info('Got call to offload model')
                    model_weights = self.get_nn_parameters()

                    ret = rpc.rpc_sync(self.client_to_offload_to, Client.offload_receive_endpoint, args=([model_weights, i]))
                    print(f'Result of rref: {ret}')

                    self.call_to_offload = False
                    self.client_to_offload_to = None
                    # This number only works for cifar10cnn
                    # @TODO: Make this dynamic for other networks
                    self.freeze_layers(15)

                # Check if there is a model to incorporate
                if global_offload_received:
                    self.args.get_logger().info('Merging offloaded model')
                    self.args.get_logger().info('FedAvg locally with offloaded model')
                    updated_weights = FedAvg({'own': self.get_nn_parameters(), 'remote': global_model_weights}, {'own': i, 'remote': global_model_data_size})

                    # updated_weights = average_nn_parameters([self.get_nn_parameters(), global_model_weights])
                    self.args.get_logger().info('Updating local weights due to offloading')
                    self.update_nn_parameters(updated_weights)
                    global_offload_received = False
                    global_model_weights = None

            if self.deadline_enabled and not warmup:
                # Deadline
                if train_stop_time is not None:
                    if time.time() >= train_stop_time:
                        self.args.get_logger().info('Stopping training due to deadline time')
                        break
                    # else:
                    #     self.args.get_logger().info(f'Time to deadline: {train_stop_time - time.time()}')




            inputs, labels = inputs.to(self.device), labels.to(self.device)
            training_process = i
            # zero the parameter gradients
            self.optimizer.zero_grad()

            outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)

            # Ignore profiler for now
            # p.signal_backward_start()
            loss.backward()
            self.optimizer.step()
            # print statistics
            running_loss += loss.item()
            if i % self.args.get_log_interval() == 0:
                self.args.get_logger().info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))
                final_running_loss = running_loss / self.args.get_log_interval()
                running_loss = 0.0

            # Ignore profiler for now
            # p.set_warmup(True)
            # if i % performance_metric_interval == 0:
            #     # perf_metrics = p.calc_metric(15)
            #     perf_metrics = p.export_data()
            #     self.args.get_logger().info(f'Number of events = {len(perf_metrics)}')
            #     perf_resp = self.report_performance_async(perf_metrics)
            #     p.reset()
            if active_profiling:
                # print(i)
                end_train_time = time.time()
                batch_duration = end_train_time - start_train_time
                profiling_data[i] = batch_duration
                if i == profiling_size-1:
                    active_profiling = False
                    time_per_batch = profiling_data.mean()
                    logging.info(f'Average batch duration is {time_per_batch}')

                    # Estimated training time
                    est_total_time = number_of_training_samples * time_per_batch
                    logging.info(f'Estimated training time is {est_total_time}')
                    self.report_performance_estimate((time_per_batch, est_total_time, number_of_training_samples))

                    if self.freeze_layers_enabled  and not warmup:
                        logging.info(f'Checking if need to freeze layers ? {est_total_time} > {deadline}')
                        if est_total_time > deadline:
                            logging.info('Will freeze layers to speed up computation')
                            # This number only works for cifar10cnn
                            # @TODO: Make this dynamic for other networks
                            self.freeze_layers(15)
            # logging.info(f'Batch time is {batch_duration}')

            # Break away from loop for debug purposes
            # if i > 5:
            #     break

        control_end_time = time.time()

        logging.info(f'Measure end time is {(control_end_time - control_start_time)}')
        logging.info(f'Trained on {training_process} samples')

        if not warmup:
            self.scheduler.step()

        # Reset the layers
        self.unfreeze_layers()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())

        return final_running_loss, self.get_nn_parameters(), training_process

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

    def run_epochs(self, num_epoch, deadline: int = None, warmup=False):
        start_time_train = datetime.datetime.now()

        self.dataset.get_train_sampler().set_epoch_size(num_epoch)
        # Train locally
        loss, weights, training_process = self.train(self.epoch_counter, deadline, warmup)
        if not warmup:
            self.epoch_counter += num_epoch
        elapsed_time_train = datetime.datetime.now() - start_time_train
        train_time_ms = int(elapsed_time_train.total_seconds()*1000)

        start_time_test = datetime.datetime.now()
        accuracy, test_loss, class_precision, class_recall = self.test()
        elapsed_time_test = datetime.datetime.now() - start_time_test
        test_time_ms = int(elapsed_time_test.total_seconds()*1000)

        data = EpochData(self.epoch_counter, num_epoch, train_time_ms, test_time_ms, loss, accuracy, test_loss, class_precision, class_recall, training_process, self.id)
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
