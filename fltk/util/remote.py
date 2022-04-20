import time
from dataclasses import dataclass, field
from typing import Any, List

from torch.distributed import rpc
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
    """
    Dataclass containing makespan statistics of epochs for profiling.
    """
    client_id: str
    metric: str
    value: Any
    epoch: int = None
    timestamp: float = field(default_factory=time.time)


class ClientRef:
    """
    Class containing information regarding clients references that work on a learning task. In addition it can keep
    track of TimingRecords describing the statistics of epoch makespans of clients.
    """
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
    """
    Dataclass for asynchronous calls to clients.
    """
    future: Future
    client: ClientRef
    start_time: float = 0
    end_time: float = 0

    def duration(self):
        """
        Function to calculate makespan, or duration, of an AsyncCall.
        @return: Duration of makespan.
        @rtype: float
        """
        return self.end_time - self.start_time


def bind_timing_cb(response_obj: AsyncCall):
    """
    Function to add callbacks for timing information send by clients.
    @param response_obj: Object to attach callback to.
    @type response_obj: AsyncCall
    @return: None
    @rtype: None
    """
    def callback(fut):
        stop_time = time.time()
        response_obj.end_time = stop_time
    response_obj.future.then(callback)

def timed_remote_async_call(client, method, rref, *args, **kwargs):
    """
    Function to add remote asynchronous calls for remote working (i.e. either in Docker or K8s, not on a single
    machine).
    @param client: Client reference
    @type client: ClientRef
    @param method: Method to execute remotely in async fashion.
    @type method: Callable
    @param rref: Remote reference.
    @type rref: ...
    @param args: Arguments to pass to function.
    @type args: List[Any]
    @param kwargs: Keyword arguments to pass to function.
    @type kwargs: Dict[str, Any]
    @return: Asynchrous call function object with attached callback for statistics.
    @rtype: AsyncCall
    """
    start_time = time.time()
    fut = _remote_method_async(method, rref, *args, **kwargs)
    response = AsyncCall(fut, client, start_time=start_time)
    bind_timing_cb(response)
    return response
