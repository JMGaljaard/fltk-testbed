import torch
from torch.distributed import rpc

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

def _remote_method_direct(method, other_node: str, *args, **kwargs):
    args = [method, other_node] + list(args)
    # return rpc.rpc_sync(other_node, _call_method, args=args, kwargs=kwargs)
    return rpc.rpc_sync(other_node, method, args=args, kwargs=kwargs)