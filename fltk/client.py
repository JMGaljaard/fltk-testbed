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
from fltk.strategy.offloading import OffloadingStrategy, parse_strategy
from fltk.util.arguments import Arguments
from fltk.util.fed_avg import average_nn_parameters
from fltk.util.log import FLLogger

import yaml

from fltk.util.profiler import Profiler
from fltk.util.profilerV2 import Profiler as P2
from fltk.util.profilerV3 import Profiler as P3
from fltk.util.results import EpochData
from fltk.util.timer import elapsed_timer

logging.basicConfig(
    level=logging.DEBUG,

    format='%(asctime)s %(levelname)s %(module)s - %(funcName)s: %(message)s',
)

global_dict = {}
global_model_weights = {}
global_model_data_size = 0
global_sender_id = ""
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
    offloaded_net = None

    # Model offloading
    received_offload_model = False
    offloaded_model_weights = None
    call_to_offload = False
    client_to_offload_to : str = None
    offloaded_model_ready = False

    strategy = OffloadingStrategy.VANILLA

    deadline_enabled = False
    swyh_enabled = False
    freeze_layers_enabled = False
    offload_enabled = False
    dyn_terminate = False
    dyn_terminate_swyh = False

    terminate_training = False

    def __init__(self, id, log_rref, rank, world_size, config = None):
        # logging.info(f'Welcome to client {id}')
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

    def load_offloaded_model(self):
        self.offloaded_net = self.load_default_model()
        self.offloaded_net.to(self.device)
        logging.info('Offloaded network loaded')

    def copy_offloaded_model_weights(self):
        self.update_nn_parameters(global_model_weights, True)
        logging.info('Parameters of offloaded model updated')
        self.offloaded_model_ready = True

    def configure_strategy(self, strategy : OffloadingStrategy):
        deadline_enabled, swyh_enabled, freeze_layers_enabled, offload_enabled, dyn_terminate, dyn_terminate_swyh = parse_strategy(strategy)
        self.deadline_enabled = deadline_enabled
        self.swyh_enabled = swyh_enabled
        self.freeze_layers_enabled = freeze_layers_enabled
        self.offload_enabled = offload_enabled
        self.dyn_terminate = dyn_terminate
        self.dyn_terminate_swyh = dyn_terminate_swyh
        logging.info(f'Offloading strategy={strategy}')
        logging.info(f'Offload strategy params: deadline={self.deadline_enabled}, '
                     f'swyh={self.swyh_enabled}, freeze={self.freeze_layers_enabled}, '
                     f'offload={self.offload_enabled}, dyn_terminate={self.dyn_terminate}, '
                     f'dyn_terminate_swyh={self.dyn_terminate_swyh}')


    def init_device(self):
        if self.args.cuda and torch.cuda.is_available():
            return torch.device("cuda:0")
        else:
            return torch.device("cpu")

    def send_reference(self, server_ref):
        self.local_log(f'Got worker_info from server {server_ref}')
        self.server_ref = server_ref


    def terminate_training_endpoint(self):
        self.terminate_training = True


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

    def update_nn_parameters(self, new_params, is_offloaded_model = False):
        """
        Update the NN's parameters.

        :param new_params: New weights for the neural network
        :type new_params: dict
        """
        if is_offloaded_model:
            self.offloaded_net.load_state_dict(copy.deepcopy(new_params), strict=True)
        else:
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
    def offload_receive_endpoint(model_weights, num_train_samples, sender_id):
        print(f'Got the offload_receive_endpoint endpoint')
        global global_model_weights, global_offload_received, global_model_data_size, global_sender_id
        global_model_weights = copy.deepcopy(model_weights.copy())
        global_model_data_size = num_train_samples
        global_sender_id = sender_id
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

    def freeze_layers2(self, until, net):

        def get_children(model: torch.nn.Module):
            children = list(model.children())
            flatt_children = []
            if children == []:
                return model
            else:
                for child in children:
                    try:
                        flatt_children.extend(get_children(child))
                    except TypeError:
                        flatt_children.append(get_children(child))
            return flatt_children

        for idx, layer in enumerate(get_children(net)):
            if idx < until:
                print(f'[{idx}] Freezing layer: {layer}')
                for param in layer.parameters():
                    param.requires_grad = False
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

    def train(self, epoch, deadline: int = None, warmup=False, use_offloaded_model=False):
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


        perf_data = {
            'total_duration': 0,
            'p_v2_data': None,
            'p_v1_data': None,
            'n_batches': 0
        }

        start_time = time.time()

        if use_offloaded_model:
            for param in self.offloaded_net.parameters():
                param.requires_grad = True
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
        # # save model
        # if self.args.should_save_model(epoch):
        #     self.save_model(epoch, self.args.get_epoch_save_start_suffix())

        running_loss = 0.0
        final_running_loss = 0.0
        if self.args.distributed:
            self.dataset.train_sampler.set_epoch(epoch)
        number_of_training_samples = len(self.dataset.get_train_loader())
        self.args.get_logger().info(f'{self.id}: Number of training samples: {number_of_training_samples}')
        # self.args.get_logger().info(f'{self.id}: Number of training samples: {len(self.dataset.get_train_loader())}')
        # Ignore profiler for now
        # performance_metric_interval = 20
        # perf_resp = None

        # Profiling parameters
        profiling_size = self.args.profiling_size
        if profiling_size == -1:
            profiling_size = number_of_training_samples
        profiling_data = np.zeros(profiling_size)
        profiling_forwards_data = np.zeros(profiling_size)
        profiling_backwards_data = np.zeros(profiling_size)
        pre_train_loop_data = np.zeros(profiling_size)
        post_train_loop_data = np.zeros(profiling_size)
        active_profiling = True
        p = P2(profiling_size, 7)
        p3 = P3(profiling_size, 7)
        if use_offloaded_model:
            p.attach(self.offloaded_net)
            p3.attach(self.offloaded_net)
        else:
            p.attach(self.net)
            p3.attach(self.net)
        profiler_active = True

        control_start_time = time.time()
        training_process = 0

        def calc_optimal_offloading_point(profiler_data, time_till_deadline, iterations_left):
            logging.info(f'Calc optimal point: profiler_data={profiler_data}, time_till_deadline={time_till_deadline}, iterations_left={iterations_left}')
            ff, cf, cb, fb = profiler_data
            full_network = ff + cf + cb + fb
            frozen_network = ff + cf + cb
            split_point = 0
            for z in range(iterations_left, -1, -1):
                x = z
                y = iterations_left - x
                # print(z)
                new_est_split = (x * full_network) + (y * frozen_network)
                split_point = x
                if new_est_split < time_till_deadline:
                    break
            logging.info(f'The offloading point is a iteration: {split_point}')
            logging.info(f'Estimated default runtime={full_network* iterations_left}')
            logging.info(f'new_est_split={new_est_split}, deadline={deadline}')

        start_loop_time = time.time()
        for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
            loop_pre_train_start = time.time()
            start_train_time = time.time()

            if self.dyn_terminate_swyh or self.dyn_terminate:
                if self.terminate_training:
                    logging.info('Got a call to terminate training')
                    break

            if self.offload_enabled and not warmup:
                # Check if there is a call to offload
                if self.call_to_offload:
                    self.args.get_logger().info('Got call to offload model')
                    model_weights = self.get_nn_parameters()

                    ret = rpc.rpc_async(self.client_to_offload_to, Client.offload_receive_endpoint, args=([model_weights, i, self.id]))
                    print(f'Result of rref: {ret}')
                    #
                    self.call_to_offload = False
                    self.client_to_offload_to = None
                    # This number only works for cifar10cnn
                    # @TODO: Make this dynamic for other networks
                    # self.freeze_layers(5)
                    self.freeze_layers2(8, self.net)

                # Check if there is a model to incorporate
                # Disable for now to offloading testing
                # if global_offload_received:
                #     self.args.get_logger().info('Merging offloaded model')
                #     self.args.get_logger().info('FedAvg locally with offloaded model')
                #     updated_weights = FedAvg({'own': self.get_nn_parameters(), 'remote': global_model_weights}, {'own': i, 'remote': global_model_data_size})
                #
                #     # updated_weights = average_nn_parameters([self.get_nn_parameters(), global_model_weights])
                #     self.args.get_logger().info('Updating local weights due to offloading')
                #     self.update_nn_parameters(updated_weights)
                #     global_offload_received = False
                #     global_model_weights = None

            if self.swyh_enabled and not warmup:
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
            loop_pre_train_end = time.time()
            if profiler_active:
                p.signal_forward_start()
                p3.signal_forward_start()
            outputs = None
            if use_offloaded_model:
                outputs = self.offloaded_net(inputs)
            else:
                outputs = self.net(inputs)
            loss = self.loss_function(outputs, labels)
            post_train_time = time.time()
            if active_profiling:
                profiling_forwards_data[i] = post_train_time - start_train_time

            # Ignore profiler for now
            # p.signal_backward_start()
            if profiler_active:
                p.signal_backward_start()
                p3.signal_forward_end()
                p3.signal_backwards_start()
            loss.backward()
            self.optimizer.step()
            if profiler_active:
                p3.signal_backwards_end()
                p.step()
                p3.step()
            loop_post_train_start = time.time()
            # print statistics
            running_loss += loss.item()
            if i % self.args.get_log_interval() == 0:
                self.args.get_logger().info('[%d, %5d] loss: %.3f' % (epoch, i, running_loss / self.args.get_log_interval()))
                final_running_loss = running_loss / self.args.get_log_interval()
                running_loss = 0.0
            if active_profiling:
                profiling_backwards_data[i] = time.time() - post_train_time

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
                    profiler_active = False
                    active_profiling = False
                    p.remove_all_handles()
                    p3.remove_all_handles()
                    time_per_batch = profiling_data.mean()
                    logging.info(f'Average batch duration is {time_per_batch}')
                    profiler_data = p.aggregate_values()
                    p3_data = p3.aggregate_values()
                    logging.info(f'Profiler data: {profiler_data}')
                    logging.info(f'P3 Profiler data: {p3_data}')
                    calc_optimal_offloading_point(profiler_data, deadline, number_of_training_samples - i)

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
                            # self.freeze_layers(5)
                            self.freeze_layers2(8, self.net)
            # logging.info(f'Batch time is {batch_duration}')

            # Break away from loop for debug purposes
            # if i > 5:
            #     break
            loop_post_train_end = time.time()
            if active_profiling:
                pre_train_loop_data[i] = loop_pre_train_end - loop_pre_train_start
                post_train_loop_data[i] = loop_post_train_end - loop_post_train_start

        control_end_time = time.time()
        end_loop_time = time.time()
        logging.info(f'Measure end time is {(control_end_time - control_start_time)}')
        logging.info(f'Trained on {training_process} samples')
        # logging.info(f'Profiler data: {p.get_values()}')

        perf_data['total_duration'] = control_end_time - control_start_time
        perf_data['n_batches'] = len(self.dataset.get_train_loader())
        perf_data['p_v2_data'] = p.get_values()
        perf_data['p_v3_data'] = p3.get_values()
        perf_data['p_v1_data'] = profiling_data
        perf_data['pre_train_loop_data'] = pre_train_loop_data
        perf_data['post_train_loop_data'] = post_train_loop_data
        perf_data['p_v1_pre_loop'] = start_loop_time - start_time
        perf_data['p_v1_forwards'] = profiling_forwards_data
        perf_data['p_v1_backwards'] = profiling_backwards_data
        perf_data['loop_duration'] = end_loop_time - start_loop_time
        if not warmup:
            self.scheduler.step()
        # logging.info(self.optimizer.param_groups)
        scheduler_data = {
            'lr': self.scheduler.optimizer.param_groups[0]['lr'],
            'momentum': self.scheduler.optimizer.param_groups[0]['momentum'],
            'wd': self.scheduler.optimizer.param_groups[0]['weight_decay'],
        }

        # Reset the layers
        self.unfreeze_layers()

        # save model
        if self.args.should_save_model(epoch):
            self.save_model(epoch, self.args.get_epoch_save_end_suffix())
        perf_data['p_v1_post_loop'] = time.time() - control_end_time
        return final_running_loss, self.get_nn_parameters(), training_process, scheduler_data, perf_data

    def test(self, use_offloaded_model = False):
        if use_offloaded_model:
            self.offloaded_net.eval()
        else:
            self.net.eval()

        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                if use_offloaded_model:
                    outputs = self.offloaded_net(images)
                else:
                    outputs = self.net(images)

                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).item()

        accuracy = 100 * correct / total
        confusion_mat = confusion_matrix(targets_, pred_)
        accuracy_per_class = confusion_mat.diagonal() / confusion_mat.sum(1)

        class_precision = self.calculate_class_precision(confusion_mat)
        class_recall = self.calculate_class_recall(confusion_mat)
        if False:
            self.args.get_logger().debug('Test set: Accuracy: {}/{} ({:.0f}%)'.format(correct, total, accuracy))
            self.args.get_logger().debug('Test set: Loss: {}'.format(loss))
            self.args.get_logger().debug("Classification Report:\n" + classification_report(targets_, pred_))
            self.args.get_logger().debug("Confusion Matrix:\n" + str(confusion_mat))
            self.args.get_logger().debug("Class precision: {}".format(str(class_precision)))
            self.args.get_logger().debug("Class recall: {}".format(str(class_recall)))

        return accuracy, loss, class_precision, class_recall, accuracy_per_class


    def run_epochs(self, num_epoch, deadline: int = None, warmup=False):
        """
        Timing data to measure:
        Total execution tim:
        """
        start = time.time()

        start_time_train = datetime.datetime.now()

        self.dataset.get_train_sampler().set_epoch_size(num_epoch)
        # Train locally
        loss, weights, training_process, scheduler_data, perf_data = self.train(self.epoch_counter, deadline, warmup)
        if self.dyn_terminate:
            logging.info('Not testing data due to termination call')
            self.dyn_terminate = False
            return {'own': []}
        elif self.dyn_terminate_swyh:
            self.dyn_terminate_swyh = False
            logging.info('Sending back weights due to terminate with swyh')
        if not warmup:
            self.epoch_counter += num_epoch
        elapsed_time_train = datetime.datetime.now() - start_time_train
        train_time_ms = int(elapsed_time_train.total_seconds()*1000)
        post_training_time = time.time()

        start_time_test = datetime.datetime.now()
        accuracy, test_loss, class_precision, class_recall, _accuracy_per_class = self.test()
        elapsed_time_test = datetime.datetime.now() - start_time_test
        test_time_ms = int(elapsed_time_test.total_seconds()*1000)
        post_test_time = time.time()

        # Timing data that needs to be send back
        duration_train = post_training_time - start
        duration_test = post_test_time - post_training_time
        logging.info(
            f'Time for training={duration_train}, time for testing={duration_test}, total time={duration_train + duration_test}')
        data = EpochData(self.epoch_counter, num_epoch, train_time_ms, test_time_ms, loss, accuracy, test_loss,
                         class_precision, class_recall, training_process, self.id)
        self.epoch_results.append(data)
        for k, v in weights.items():
            weights[k] = v.cpu()
        response_obj = {'own': [data, weights, scheduler_data, perf_data]}

        global global_offload_received
        if self.offload_enabled and global_offload_received:
            self.configure_strategy(OffloadingStrategy.SWYH)
            logging.info('Processing offloaded model')
            self.load_offloaded_model()
            self.copy_offloaded_model_weights()
            loss_offload, weights_offload, training_process_offload, scheduler_data_offload, perf_data_offload = self.train(self.epoch_counter, deadline, warmup, use_offloaded_model=True)
            accuracy, test_loss, class_precision, class_recall, _accuracy_per_class = self.test(use_offloaded_model=True)
            global global_sender_id
            data_offload = EpochData(self.epoch_counter, num_epoch, train_time_ms, test_time_ms, loss_offload, accuracy, test_loss,
                                     class_precision, class_recall, training_process, f'{global_sender_id}-offload')
            # Copy GPU tensors to CPU
            for k, v in weights_offload.items():
                weights_offload[k] = v.cpu()
            response_obj['offload'] = [ data_offload, weights_offload, scheduler_data_offload, perf_data_offload, global_sender_id]
            self.configure_strategy(OffloadingStrategy.MODEL_OFFLOAD)
        else:
            logging.info(f'Not doing offloading due to offload_enabled={self.offload_enabled} and global_offload_received={global_offload_received}')
        return response_obj

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
