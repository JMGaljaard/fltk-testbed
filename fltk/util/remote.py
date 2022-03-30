import time
from typing import Any, List

from torch.distributed import rpc
from dataclasses import dataclass, field
from torch.futures import Future

def _call_method(method, rref, *args, **kwargs):
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)

def _remote_method_async(method, rref, *args, **kwargs):
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)

@dataclass
class TimingRecord:
    client_id: str
    metric: str
    value: Any
    epoch: int = None
    timestamp: float = field(default_factory=time.time)


class ClientRef:
    ref = None
    name = ""
    data_size = 0
    tb_writer = None
    timing_data: List[TimingRecord] = []

    def __init__(self, name, ref, tensorboard_writer):
        self.name = name
        self.ref = ref
        self.tb_writer = tensorboard_writer
        self.timing_data = []

    def __repr__(self):
        return self.name

@dataclass
class AsyncCall:
    future: Future
    client: ClientRef
    start_time: float = 0
    end_time: float = 0

    def duration(self):
        return self.end_time - self.start_time


def bind_timing_cb(response_obj: AsyncCall):
    def callback(fut):
        stop_time = time.time()
        response_obj.end_time = stop_time
    response_obj.future.then(callback)

def timed_remote_async_call(client, method, rref, *args, **kwargs):
    start_time = time.time()
    fut = _remote_method_async(method, rref, *args, **kwargs)
    response = AsyncCall(fut, client, start_time=start_time)
    bind_timing_cb(response)
    return response