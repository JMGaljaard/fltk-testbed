# pylint: disable-all
import torch
from torch.nn import Module
import time

import numpy as np


class Profiler:
    current_layer = 0
    last_time = 0
    execution_id = 0
    last_forward_time = None
    warmup = False
    hook_handles = []

    feature_layers_ends: int = 0
    ff: np.ndarray
    fb: np.ndarray
    cf: np.ndarray
    cb: np.ndarray

    batch_idx = 0

    def __init__(self, rounds: int, feature_layers_ends: int):
        self.round = rounds
        self.ff = np.zeros(self.round)
        self.fb = np.zeros(self.round)
        self.cf = np.zeros(self.round)
        self.cb = np.zeros(self.round)
        self.feature_layers_ends = feature_layers_ends

    def attach(self, module: Module):
        def get_children(model: torch.nn.Module):
            # get children form model!
            children = list(model.children())
            flatt_children = []
            if children == []:
                # if model has no children; model is last child! :O
                return model
            else:
                # look for children from children... to the last child!
                for child in children:
                    try:
                        flatt_children.extend(get_children(child))
                    except TypeError:
                        flatt_children.append(get_children(child))
            return flatt_children

        kids = get_children(module)

        print(module)
        for idx, k in enumerate(kids):
            # print(f'[{idx}] Registrating hooks for layer {k}')
            h1 = k.register_forward_hook(self.forward)
            self.hook_handles.append(h1)
            h2 = k.register_forward_pre_hook(self.pre_forward)
            self.hook_handles.append(h2)
            h3 = k.register_backward_hook(self.backward)
            self.hook_handles.append(h3)

    def remove_all_handles(self):
        for handle in self.hook_handles:
            handle.remove()

    def set_warmup(self, value):
        self.warmup = value

    def add(self, layer_id, duration, backprogation: bool = False):
        is_cls = layer_id > self.feature_layers_ends
        if is_cls:
            if backprogation:
                # use cb
                self.cb[self.batch_idx] += duration
            else:
                # use cf
                self.cf[self.batch_idx] += duration
        else:
            if backprogation:
                # use fb
                self.fb[self.batch_idx] += duration
            else:
                # use ff
                self.ff[self.batch_idx] += duration


    def pre_forward(self, other, input):
        if self.warmup:
            return None
        self.last_forward_time = time.time()

    def forward(self, other, input, output):
        if self.warmup:
            return None
        # print(f'Forward: {other.__class__.__name__}')
        self.last_forward_time = time.time() - self.last_forward_time
        # self.event_list.append(self.last_forward_event)
        # self.add(self.last_forward_event)
        self.add(self.current_layer, self.last_forward_time, False)
        self.current_layer += 1
        self.execution_id += 1

    def backward(self, module, grad_input, grad_output):
        if self.warmup:
            return None
        # print(f'Backward: {module.__class__.__name__}')
        # self.event_list.append(Event(time.time() - self.last_time, self.current_layer, module.__class__.__name__, "backward", self.execution_id))
        self.add(self.current_layer, time.time() - self.last_time, True)
        # self.add(Event(time.time() - self.last_time, self.current_layer, module.__class__.__name__, "backward", self.execution_id))
        self.current_layer -= 1
        self.execution_id += 1
        self.last_time = time.time()
        return None

    def signal_backward_start(self):
        self.current_layer -= 1
        self.last_time = time.time()

    def signal_forward_start(self):
        self.current_layer = 0
        self.execution_id = 0
        self.last_time = None
        self.last_time = 0

    def step(self):
        self.batch_idx += 1

    def get_values(self):
        """
        Returns the measured values in the following order: ff, cf, cb, fb
        ff = feature layers forward propagation
        cf = classifier layers forward propagation
        cb = feature layers backwards propagation
        fb = feature layers backwards propagation
        The order is the execution order of forward and then backward propagation of a network
        """
        return self.ff, self.cf, self.cb, self.fb

    def aggregate_values(self, from_layer: int = 0):
        """
        Returns the measured values in the following order: ff, cf, cb, fb
        ff = feature layers forward propagation
        cf = classifier layers forward propagation
        cb = feature layers backwards propagation
        fb = feature layers backwards propagation
        The order is the execution order of forward and then backward propagation of a network
        """
        return self.ff[from_layer:].mean(), self.cf[from_layer:].mean(), self.fb[from_layer:].mean(), self.cb[
                                                                                                      from_layer:].mean()

    def profile_run(self, module, input, iterations, warmup_time = 0):
        output = module(input)
        g0 = torch.rand_like(output) # pylint: disable=no-member

        self.attach(module)
        module.train()
        self.set_warmup(True)
        for i in range(warmup_time):  # warmup
            print('warmup cycle')
            self.signal_forward_start()
            output = module(input)
            self.signal_backward_start()
            output.backward(g0)
        self.set_warmup(False)
        for i in range(iterations):
            print(i, end='')
            self.signal_forward_start()
            output = module(input)
            self.signal_backward_start()
            output.backward(g0)
            self.step()
        print('')