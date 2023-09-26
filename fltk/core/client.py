from __future__  import annotations

import gc
import multiprocessing
import queue
from collections import OrderedDict
from typing import Tuple, Any, Type, Callable

import numpy as np
import sklearn
import time
import torch

from fltk.core.node import Node
from fltk.schedulers import MinCapableStepLR
from fltk.strategy import get_optimizer


from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from fltk.util.config import FedLearnerConfig


class Client(Node):
    """
    Federated experiment client.
    """
    running = False
    request_queue = queue.Queue()
    result_queue = queue.Queue()

    def __init__(self, identifier: str, rank: int, world_size: int, config: FedLearnerConfig):
        super(Client, self).__init__(identifier, rank, world_size, config)

        self.loss_function = self.config.get_loss_function()()
        self.optimizer = get_optimizer(self.config.optimizer)(self.net.parameters(),
                                                              **self.config.optimizer_args)
        self.scheduler = MinCapableStepLR(self.optimizer,
                                          self.config.scheduler_step_size,
                                          self.config.scheduler_gamma,
                                          self.config.min_lr)

    def remote_registration(self):
        """
        Function to perform registration to the remote. Currently, this will connect to the Federator Client. Future
        version can provide functionality to register to an arbitrary Node, including other Clients.
        @return: None.
        @rtype: None
        """
        self.logger.info('Sending registration')
        self.message('federator', 'ping', 'new_sender')
        self.message('federator', 'register_client', self.id, self.rank)

    def run(self):
        """
        Function to start running the Client after registration. This allows for processing requests by the main thread,
        while the RPC requests can be made asynchronously.

        Returns: None

        """
        self.running = True
        event = multiprocessing.Event()
        while self.running:
            # Hack for running on Kubeflow
            if not self.request_queue.empty():
                request = self.request_queue.get()
                self.logger.info(f"Got request, args: {request} running synchronously.")
                self.result_queue.put(self.exec_round(*request))
            event.wait(1)
        self.logger.info(f"Exiting client {self.id}")

    def stop_client(self):
        """
        Function to stop client after training. This allows remote clients to stop the client within a specific
        timeframe.
        @return: None
        @rtype: None
        """
        self.logger.info('Got call to stop event loop')
        self.running = False

    def train(self, num_epochs: int, round_id: int):
        """
        Function implementing federated learning training loop, allowing to run for a configurable number of epochs
        on a local dataset. Note that only the last statistics of a run are sent to the caller (i.e. Federator).
        @param num_epochs: Number of epochs to run during a communication round's training loop.
        @type num_epochs: int
        @param round_id: Global communication round ID to be used during training.
        @type round_id: int
        @return: Final running loss statistic and acquired parameters of the locally trained network. NOTE that
        intermediate information is only logged to the STD-out.
        @rtype: Tuple[float, Dict[str, torch.Tensor]]
        """
        start_time = time.time()
        running_loss = 0.0
        final_running_loss = 0.0
        self.logger.info(f"[RD-{round_id}] kicking of local training for {num_epochs} local epochs")
        for local_epoch in range(num_epochs):
            effective_epoch = round_id * num_epochs + local_epoch
            progress = f'[RD-{round_id}][LE-{local_epoch}][EE-{effective_epoch}]'
            if self.distributed:
                # In case a client occurs within (num_epochs) communication rounds as this would cause
                # an order or data to re-occur during training.
                self.dataset.train_sampler.set_epoch(effective_epoch)

            training_cardinality = len(self.dataset.get_train_loader())
            self.logger.info(f'{progress}{self.id}: Number of training samples: {training_cardinality}')
            for i, (inputs, labels) in enumerate(self.dataset.get_train_loader(), 0):
                # zero the parameter gradients
                self.optimizer.zero_grad()

                outputs = self.net(inputs)
                loss = self.loss_function(outputs, labels)

                running_loss += loss.detach().item()
                loss.backward()
                self.optimizer.step()
                # Mark logging update step
                if i % self.config.log_interval == 0:
                    self.logger.info(
                            f'[{self.id}] [{local_epoch}/{num_epochs:d}, {i:5d}] loss: {running_loss / self.config.log_interval:.3f}')
                    final_running_loss = running_loss / self.config.log_interval
                    running_loss = 0.0
                del loss, inputs, labels

            end_time = time.time()
            duration = end_time - start_time
            self.logger.info(f'{progress} Train duration is {duration} seconds')
        # Clear gradients before we send.
        self.optimizer.zero_grad(set_to_none=True)
        gc.collect()
        return final_running_loss, self.get_nn_parameters()

    def set_tau_eff(self, total):
        client_weight = self.get_client_datasize() / total
        n = self.get_client_datasize()  # pylint: disable=invalid-name
        E = self.config.epochs  # pylint: disable=invalid-name
        B = 16  # nicely hardcoded :) # pylint: disable=invalid-name
        tau_eff = int(E * n / B) * client_weight
        if hasattr(self.optimizer, 'set_tau_eff'):
            self.optimizer.set_tau_eff(tau_eff)

    def test(self) -> Tuple[float, float, np.array]:
        """
        Function implementing federated learning test loop.
        @return: Statistics on test-set given a (partially) trained model; accuracy, loss, and confusion matrix.
        @rtype: Tuple[float, float, np.array]
        """
        start_time = time.time()
        correct = 0
        total = 0
        targets_ = []
        pred_ = []
        loss = 0.0
        with torch.no_grad():
            for (images, labels) in self.dataset.get_test_loader():
                images, labels = images.to(self.device), labels.to(self.device)

                outputs = self.net(images)

                _, predicted = torch.max(outputs.data, 1)  # pylint: disable=no-member
                total += int(labels.size(0))
                correct += (predicted == labels).sum().detach().item()

                targets_.extend(labels.cpu().view_as(predicted).numpy())
                pred_.extend(predicted.cpu().numpy())

                loss += self.loss_function(outputs, labels).detach().item()
        # Calculate learning statistics
        loss /= len(self.dataset.get_test_loader().dataset)
        accuracy = 100.0 * correct / total

        confusion_mat = sklearn.metrics.confusion_matrix(targets_, pred_)

        end_time = time.time()
        duration = end_time - start_time
        self.logger.info(f'Test duration is {duration} seconds')
        del targets_, pred_
        return accuracy, loss, confusion_mat

    def get_client_datasize(self):  # pylint: disable=missing-function-docstring
        return len(self.dataset.get_train_sampler())

    def request_round(self, num_epochs: int, round_id:int):
        event = multiprocessing.Event()
        self.request_queue.put([num_epochs, round_id])

        while self.result_queue.empty():
            event.wait(5)
        self.logger.info("Finished request!")
        return self.result_queue.get()

    def exec_round(self, num_epochs: int, round_id: int) -> Tuple[Any, Any, Any, Any, float, float, float, np.array]:
        """
        Function as access point for the Federator Node to kick off a remote learning round on a client.
        @param num_epochs: Number of epochs to run
        @type num_epochs: int
        @return: Tuple containing the statistics of the training round; loss, weights, accuracy, test_loss, make-span,
        training make-span, testing make-span, and confusion matrix.
        @rtype: Tuple[Any, Any, Any, Any, float, float, float, np.array]
        """
        self.logger.info(f"[EXEC] running {num_epochs} locally...")
        start = time.time()
        loss, weights = self.train(num_epochs, round_id)
        time_mark_between = time.time()
        accuracy, test_loss, test_conf_matrix = self.test()

        end = time.time()
        round_duration = end - start
        train_duration = time_mark_between - start
        test_duration = end - time_mark_between
        # self.logger.info(f'Round duration is {duration} seconds')

        if hasattr(self.optimizer, 'pre_communicate'):  # aka fednova or fedprox
            self.optimizer.pre_communicate()
        for k, value in weights.items():
            weights[k] = value.cpu()
        gc.collect()
        return loss, weights, accuracy, test_loss, round_duration, train_duration, test_duration, test_conf_matrix

    def __del__(self):
        self.logger.info(f'Client {self.id} is stopping')

class ContinuousClient(Client):
    """
    Federated Continual Learning experiment Client. See also Client implementation for ordinary Federated Learning
    Experiments.
    """

    def __init__(
        self,
        client_id: str,
        rank: int,
        world_size: int,
        fed_config: FedLearnerConfig,
        *args,
        **kwargs
    ):
        """
        model,
        tr_dataloader,
        nepochs=100,
        lr=0.001,
        lr_min=1e-6,
        lr_factor=3,
        lr_patience=5,
        clipgrad=100,
        args=None,
        num_classes=10,
        """
        super(ContinuousClient, self).__init__(client_id, rank, world_size, fed_config)


        # self.tb_writer = tb_writer
        #
        # self.accuracy_hist = []
        # self.num_classes = num_classes
        # self.model_old = model
        # self.fisher = None
        # self.nepochs = nepochs
        # self.tr_dataloader = tr_dataloader
        # self.lr = lr
        # self.lr_min = lr_min * 1 / 3
        # self.lr_factor = lr_factor
        # self.lr_patience = lr_patience
        # self.lr_decay = args.lr_decay
        # self.optim_type = args.optim
        # self.clipgrad = clipgrad
        # self.args = args
        # self.ce = torch.nn.CrossEntropyLoss()
        # self.optimizer = self._get_optimizer()
        # self.lamb = args.lamb
        # self.e_rep = args.local_local_ep
        # self.old_task = -1
        # self.grad_dims = []
        # self.pre_weight = {"weight": [], "aw": [], "mask": []}
        #
        # return

    def set_sw(self, glob_weights):
        """
        Function to set (partial) weights according to the orderd dict received by the
        @param glob_weights:
        @type glob_weights:
        @return:
        @rtype:
        """
        i = 0
        keys = [k for k, _ in self.net.named_parameters()]
        if len(glob_weights) > 0:
            all_weights = []
            for name, para in self.net.named_parameters():
                if "sw" in name:
                    all_weights.append(glob_weights[i])
                    i = i + 1
                else:
                    all_weights.append(para)
            model_dict = self.net.state_dict()
            feature_dict = zip(keys, all_weights)
            # last_dict = OrderedDict({k: torch.Tensor(v) for k, v in zip(last_keys,last_para)})
            save_model = OrderedDict({k: v for k, v in feature_dict})
            state_dict = {k: v for k, v in save_model.items() if k in model_dict.keys()}
            model_dict.update(state_dict)
            self.net.load_state_dict(model_dict)
        # logging.info('')

    def get_sw(self):
        sws = []
        for name, para in self.net.named_parameters():
            if "sw" in name:
                sws.append(para)
        return sws


    def _get_optimizer(self, lr=None):
        if lr is None:
            lr = self.lr
        if "SGD" in self.optim_type:
            optimizer = torch.optim.SGD(
                self.net.parameters(), lr=lr, weight_decay=self.lr_decay
            )
        else:
            optimizer = torch.optim.Adam(
                self.net.parameters(), lr=lr, weight_decay=self.lr_decay
            )
        return optimizer

    def train(self, t, from_kbs, know, writer: SummaryWriter = None, aggNum: int = 0):
        if t != self.old_task:
            self.old_task = t
        lr = self.lr
        for name, para in self.net.named_parameters():
            para.requires_grad = True
        self.net.set_knowledge(t, from_kbs)
        self.optimizer = self._get_optimizer()
        if torch.cuda.is_available():
            self.net.cuda()
        # Loop epochs
        for e in range(self.nepochs):
            # Train
            self.train_epoch(t)
            train_loss, train_acc = self.eval(t)
            if e % self.e_rep == self.e_rep - 1:
                # logging.info('| Epoch {:3d} | Train: loss={:.3f}, acc={:5.1f}% | \n'.format(e + 1,  train_loss, 100 * train_acc), end='')
                logging.debug(
                    f"| Epoch {e + 1:3d} | Train: loss={train_loss:.3f}, acc={100 * train_acc:5.1f}%"
                )

                writer.add_scalar("train_loss", train_loss, aggNum)
                writer.add_scalar("train_acc", train_acc, aggNum)
        if len(self.pre_weight["aw"]) <= t:
            self.pre_weight["aw"].append([])
            self.pre_weight["mask"].append([])
            self.pre_weight["weight"].append([])
            for name, para in self.net.named_parameters():
                if "aw" in name:
                    aw = para.detach()
                    aw.requires_grad = False
                    self.pre_weight["aw"][-1].append(aw)
                elif "mask" in name:
                    mask = para.detach()
                    mask.requires_grad = False
                    self.pre_weight["mask"][-1].append(mask)
            self.pre_weight["weight"][-1] = self.net.get_weights()
        else:
            self.pre_weight["aw"].pop()
            self.pre_weight["mask"].pop()
            self.pre_weight["weight"].pop()
            self.pre_weight["aw"].append([])
            self.pre_weight["mask"].append([])
            self.pre_weight["weight"].append([])
            for name, para in self.net.named_parameters():
                if "aw" in name:
                    self.pre_weight["aw"][-1].append(para)
                elif "mask" in name:
                    self.pre_weight["mask"][-1].append(para)
            self.pre_weight["weight"][-1] = self.net.get_weights()

        return self.get_sw(), train_loss, train_acc

    def train_epoch(self, t):
        self.net.train()
        for images, targets in self.tr_dataloader:
            if torch.cuda.is_available():
                images = images.cuda()
                # targets = (targets - self.num_classes * t).cuda()
                targets.apply_(lambda x: int(labelMapper(x) + 5 * t))
                targets = targets.cuda()
            # Forward current model
            offset1, offset2 = compute_offsets(t, 5)
            self.optimizer.zero_grad()
            self.net.zero_grad()
            # store_grad(self.model.parameters,grads, self.grad_dims,0)
            outputs = self.net.forward(images, t)
            # outputs = self.model.forward(images, t)
            _, pred = outputs.max(1)
            loss = self.get_loss(outputs, targets, t)
            ## 根据这个损失计算梯度，变换此梯度

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return

    def l2_loss(self, para):
        return torch.sum(torch.pow(para, 2)) / 2

    def get_loss(self, outputs, targets, t):
        loss = self.ce(outputs, targets)
        i = 0
        weight_decay = 0
        sparseness = 0
        approx_loss = 0
        sw = None
        aw = None
        mask = None
        for name, para in self.net.named_parameters():
            if "sw" in name:
                sw = para
            elif "aw" in name:
                aw = para
            elif "mask" in name:
                mask = para
            elif "atten" in name:
                weight_decay += self.args.wd * self.l2_loss(aw)
                weight_decay += self.args.wd * self.l2_loss(mask)
                sparseness += self.args.lambda_l1 * torch.sum(torch.abs(aw))
                sparseness += self.args.lambda_mask * torch.sum(torch.abs(mask))
                if torch.isnan(weight_decay).sum() > 0:
                    logging.warning("weight_decay nan")
                if torch.isnan(sparseness).sum() > 0:
                    logging.warning("sparseness nan")
                if t == 0:
                    weight_decay += self.args.wd * self.l2_loss(sw)
                else:
                    for tid in range(t):
                        prev_aw = self.pre_weight["aw"][tid][i]
                        prev_mask = self.pre_weight["mask"][tid][i]
                        m = torch.nn.Sigmoid()
                        g_prev_mask = m(prev_mask)
                        #################################################
                        sw2 = sw.transpose(0, -1)
                        restored = (sw2 * g_prev_mask).transpose(0, -1) + prev_aw
                        a_l2 = self.l2_loss(
                            restored - self.pre_weight["weight"][tid][i]
                        )
                        approx_loss += self.args.lambda_l2 * a_l2
                        #################################################
                    i += 1
        loss += weight_decay + sparseness + approx_loss
        return loss

    def eval(self, t, train=True):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.net.eval()
        dataloaders = self.tr_dataloader

        # Loop batches
        with torch.no_grad():
            for images, targets in dataloaders:
                if torch.cuda.is_available():
                    images = images.cuda()
                    targets.apply_(lambda x: int(labelMapper(x) + 5 * t))
                    # targets = (targets - self.num_classes*t).cuda()
                    targets = targets.cuda()
                # Forward
                offset1, offset2 = compute_offsets(t, 5)
                output = self.net.forward(images, t)
                # output = self.model.forward(images,t)
                loss = self.ce(output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num

    def evalCustom(self, t):
        total_loss = 0
        total_acc = 0
        total_num = 0
        self.net.eval()
        dataloaders = self.tr_dataloader
        # Loop batches
        with torch.no_grad():
            for images, targets in dataloaders:
                if torch.cuda.is_available():
                    # logging.warning(f'Mapping images for testing to cuda! -> {t}')
                    images = images.cuda()
                    # targets = (targets - self.num_classes*t).cuda()
                    targets.apply_(lambda x: int(labelMapper(x) + 5 * t))
                    targets = targets.cuda()
                    # logging.warning(f'Image type: {images.device}, target type: {targets.device}')
                # Forward
                offset1, offset2 = compute_offsets(t, 5)
                # m : nn.Module= self.model
                # for name, p in m.named_parameters():
                # logging.warning(f'{name}: {p.device}')

                output = self.net.forward(images, t)
                # output = self.model.forward(images,t)
                loss = self.ce(output, targets)
                _, pred = output.max(1)
                hits = (pred == targets).float()

                # Log
                total_loss += loss.data.cpu().numpy() * len(images)
                total_acc += hits.sum().data.cpu().numpy()
                total_num += len(images)

        return total_loss / total_num, total_acc / total_num

    def criterion(self, t, output, targets):
        # Regularization for all previous tasks
        loss_reg = 0
        if t > 0:
            for (name, param), (_, param_old) in zip(
                self.net.feature_net.named_parameters(),
                self.model_old.feature_net.named_parameters(),
            ):
                loss_reg += (
                    torch.sum(self.fisher[name] * (param_old - param).pow(2)) / 2
                )
        return self.ce(output, targets) + self.lamb * loss_reg

    def lifeLongTrain(self, args, iter, kb):
        logging.debug(f"cur round :{iter}  cur client: {self.client_id}")
        taskcla = []
        # acc = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        # lss = np.zeros((len(taskcla), len(taskcla)), dtype=np.float32)
        t = iter // args.round
        logging.debug(f"cur task:{t}")
        r = iter % args.round
        # for t, ncla in taskcla:
        know = False
        if r == args.round - 1:
            know = True
        # logging.info('*' * 100)
        # logging.info('Task {:2d} ({:s})'.format(t, data[t]['name']))
        # logging.info('*' * 100)

        # Get data
        task = t

        # Train
        sws, loss, _ = self.train(task, kb, know, self.tb_writer, iter)
        logging.debug("-" * 100)
        if know:
            return sws, self.pre_weight["aw"][-1], loss, 0
        else:
            return sws, None, loss, 0

    def lifeLongTest(self, args, task_id: int, testdatas):
        """To evaluate the (Federated) Continual Learning metrics
        Args:
            args (_type_): scenario arguments
            appr (_type_): _description_
            t (_type_): Task Id
            testdatas (_type_): test-datasets
            writer (_type_): Tensorboard writer
            client_id (int, optional): Client Id. Defaults to 0.
            accuracy_hist (list, optional): _description_. Defaults to [].

        """
        # this returns the test accuracy for a client on all the tasks learned until now
        acc = np.zeros((1, task_id + 1), dtype=np.float32)
        lss = np.zeros((1, task_id + 1), dtype=np.float32)
        # Currently this is evaluating the Average accuracy for client i by looping over the tasks with size T
        for u in range(task_id + 1):
            taskdata = testdatas[u]
            DatasetObj = DatasetConverter(taskdata, "cifar100")
            tr_dataloader = DataLoader(
                DatasetObj, batch_size=args.local_bs, shuffle=True
            )
            self.tr_dataloader = tr_dataloader
            model_device = next(self.net.parameters()).device
            # logging.warning(f'appr.model.device type = {model_device} ')
            test_loss, test_acc = self.evalCustom(u)
            logging.debug(
                f">>> [Client {self.client_id}] Test on task {u:2d} : loss={test_loss:.3f}, acc={100 * test_acc:5.1f}% <<<"
            )
            acc[0, u] = test_acc
            lss[0, u] = test_loss
            if u == task_id:
                # logging.warning(f'This is the current task! (T_{u},{t})')
                self.accuracy_hist.append(test_acc)
            # else:
            #     logging.warning(f'Not there yet! (T_{u},{t})')
        # logging.info(f'Forgetting debug: acc[0, :t] - accuracy_hist[:t] = {acc[0, :t]} - {accuracy_hist[:t]}')
        forgetting = acc[0, :task_id] - self.accuracy_hist[:task_id]
        # logging.info(f'Forgetting for client {client_id}= {forgetting}')
        if task_id:
            backward_transfer = 1.0 / float(task_id) * np.sum(forgetting)
        else:
            backward_transfer = 0.0
        logging.debug(f"Avg forgetting: {backward_transfer}")
        # Save
        mean_acc = np.mean(acc[0, : task_id + 1])
        mean_lss = np.mean(lss[0, : task_id + 1])
        logging.debug(f"Average accuracy={100 * mean_acc:5.1f}")
        logging.debug(f"Average loss={mean_lss:5.1f}")
        # logging.debug('Save at ' + args.output)
        # if r == args.round - 1:
        self.tb_writer.add_scalar("task_finish_and_agg", mean_acc, task_id + 1)
        # np.savetxt(args.agg_output + 'aggNum_already_'+str(aggNum)+args.log_name, acc, '%.4f')
        return mean_lss, mean_acc, backward_transfer, self.accuracy_hist


def _client_constructor(client_name: str, rank: int, config: FedLearnerConfig) -> Client:
    """Constructor helper method for standard Federated Learning Clients.

    @param client_name: Identifier of the client during experiment.
    @rtype client_name: str
    @param rank: Rank (relative to worlds size) of client.
    @rtype rank: int
    @param config: Federated Learning configuration object.
    """
    return Client(client_name, rank, config.world_size, config)

def _continous_client_constructor(client_name: str, rank: int, config) -> ContinuousClient:
    """Constructor helper method for Continuous Federated Learning Clients.

    @param client_name: Identifier of the client during experiment.
    @rtype client_name: str
    @param rank: Rank (relative to worlds size) of client.
    @rtype rank: int
    @param config: Federated Learning configuration object.
    """
    raise NotImplementedError()
    return ContinuousClient(client_name, rank, config.world_size, config)


def get_constructor(config: FedLearnerConfig) -> Callable[[str, int, FedLearnerConfig, ...], Client]:
    """Helper method to infer required consturctor method to properly instantiate and prepare different Client's during
    experiments.
    @param config: FederatedLearning Configuration for current experiment.
    @type config: FedLearnerConfig
    @return: Callable function which implements the instantiation of the requested Client using the callers' provided
        arguments.
    @rtype: Callable[[str, int, FedLearnerConfig, ...], Client]
    """
    raise not NotImplementedError("Point to ")
    continous = ...

    if continous:
        return _continous_client_constructor
    else:
        return _client_constructor


class FedClientConstructor:
    """Constructor object allowing the caller to defer the inference of the type of required Client ot the Constructor.
    Default behavior is to instantiate a standard Federated Learning client.
    """
    def construct(self, config: FedLearnerConfig, client_name: str, rank: int, world_size: int, *args, **kwargs):
        """
        Constructor method to automatically infer the required type of Client from the provided learner configuration.

        @param config: FederatedLearning configuration object for current experiment.
        @type config: FedLearnerConfig
        @param client_name: Name of client to use during communication.
        @type client_name: str
        @param rank: Rank of the client during the experiment (relative to world size)
        @type rank: int
        @param world_size: Total number of clients + federator to participate during experiment.
        @type world_size: int
        @param args: Additional arguments to pass to constructors as required.
        @type args: Any
        @param kwargs: Additional keyword arguments to pass to consturctors as required.
        @type kwargs: Dict[str, Any]
        @return: Instantiated client with provided arguments.
        @rtype: Client
        """
        constructor = get_constructor(config)
        return constructor(client_name, rank, world_size, config, *args, **kwargs)
