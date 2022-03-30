from dataclasses import dataclass

import torch
from torch.nn import Module
import time
import pandas as pd

@dataclass
class Event:
    time: int
    layer_id: int
    name: str
    event: str
    execution_id: int

    def to_list(self):
        return [self.time, self.layer_id, self.name, self.event, f'{self.layer_id}-{self.name}', self.execution_id]

class Profiler:
    current_layer = 0
    event_list = []
    last_time = 0
    execution_id = 0
    last_forward_event = None
    warmup = False
    hook_handles = []

    def add(self, event: Event):
        if event.layer_id >= 100:
            print('Error')
            print(event)
            for e in self.event_list[-150:]:
                print(e)
        assert(event.layer_id < 100)
        self.event_list.append(event)

    def pre_forward(self, other, input):
        if self.warmup:
            return None
        # print(f'Pre forward: {other.__class__.__name__}')
        # self.event_list.append(Event(time.time_ns(), self.current_layer, other.__class__.__name__, "pre_forward"))
        self.last_forward_event = Event(time.time_ns(), self.current_layer, other.__class__.__name__, "forward", self.execution_id)

    def forward(self, other, input, output):
        if self.warmup:
            return None
        # print(f'Forward: {other.__class__.__name__}')
        self.last_forward_event.time = time.time_ns() - self.last_forward_event.time
        # self.event_list.append(self.last_forward_event)
        self.add(self.last_forward_event)
        self.current_layer += 1
        self.execution_id += 1

    def backward(self, module, grad_input, grad_output):
        # pass
        if self.warmup:
            return None
        # print(f'Backward: {module.__class__.__name__}')
        # self.event_list.append(Event(time.time_ns() - self.last_time, self.current_layer, module.__class__.__name__, "backward", self.execution_id))
        self.add(Event(time.time_ns() - self.last_time, self.current_layer, module.__class__.__name__, "backward", self.execution_id))
        self.current_layer -= 1
        self.execution_id += 1
        self.last_time = time.time_ns()
        return None

    def signal_backward_start(self):
        self.current_layer -= 1
        self.last_time = time.time_ns()

    def signal_forward_start(self):
        self.current_layer = 0
        self.execution_id = 0
        self.last_time = None
        self.last_forward_event = None

    def print_events(self):
        for e in self.event_list:
            print(e)

    def to_dataframe(self) -> pd.DataFrame:
        data = [x.to_list() for x in self.event_list]
        return pd.DataFrame(data, columns = ['time', 'layer_id', 'layer_type', 'event', 'id_type_combined', 'execution_id'])

    def export_data(self):
        return self.to_dataframe().groupby(['event', 'layer_id']).mean().reset_index()[['event', 'layer_id', 'time']]

    def reset(self):
        self.event_list = []

    def calc_metric(self, start_cls_layer):
        df = self.to_dataframe()
        df['type'] = 'feature'
        mask = df['layer_id'] >= start_cls_layer
        df.loc[mask, 'type'] = 'classifier'
        mask = df['layer_id'] < start_cls_layer
        df.loc[mask, 'type'] = 'feature'
        combined = df.groupby(['event', 'type']).sum().reset_index()

        features_f = combined[(combined['type'] == 'feature') & (combined['event'] == 'forward')]['time'].values[0]
        classifier_f = combined[(combined['type'] == 'classifier') & (combined['event'] == 'forward')]['time'].values[0]
        features_b = combined[(combined['type'] == 'feature') & (combined['event'] == 'backward')]['time'].values[0]
        classifier_b = combined[(combined['type'] == 'classifier') & (combined['event'] == 'backward')]['time'].values[0]
        return features_f, features_b, classifier_f, classifier_b


    def set_warmup(self, value):
        self.warmup = value

    def printnorm(self, other, input, output):
        # input is a tuple of packed inputs
        # output is a Tensor. output.data is the Tensor we are interested
        print('Inside ' + other.__class__.__name__ + ' forward')
        # print('')
        # print('input: ', type(input))
        # print('input[0]: ', type(input[0]))
        # print('output: ', type(output))
        # print('')
        # print('input size:', input[0].size())
        # print('output size:', output.data.size())
        # print('output norm:', output.data.norm())

    def remove_all_handles(self):
        for handle in self.hook_handles:
            handle.remove()

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
        for k in kids:
            # print(f'Registrating hooks for layer {k}')
            h1 = k.register_forward_hook(self.forward)
            self.hook_handles.append(h1)
            h2 = k.register_forward_pre_hook(self.pre_forward)
            self.hook_handles.append(h2)
            h3 = k.register_backward_hook(self.backward)
            self.hook_handles.append(h3)
        # module.register_forward_hook(self.printnorm)
        # for name, m in module.named_children():
        #     print(f'>> Name: {name}')
        #     print(f'>> Content: {m.parameters()}')
        # for child in module.children():
        #     print(f'Registrating hooks for layer {child}')
        #     child.register_forward_hook(self.forward)
        #     child.register_forward_pre_hook(self.pre_forward)
        #     child.register_backward_hook(self.backward)
            # child.register_full_backward_hook(self.backward)

    def profile_run(self, module, input, iterations, warmup_time = 0) -> pd.DataFrame:
        output = module(input)
        g0 = torch.rand_like(output)

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
        print('')
        self.print_events()

        return self.to_dataframe()