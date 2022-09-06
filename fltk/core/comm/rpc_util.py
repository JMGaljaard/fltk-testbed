import torch
from torch.distributed import rpc

def _call_method(method, rref, *args, **kwargs):
    """helper for _remote_method()"""
    return method(rref.local_value(), *args, **kwargs)

def _remote_method(method, rref, *args, **kwargs):
    """
    executes method(*args, **kwargs) on the 'from' the machine that owns rref

    very similar to rref.remote().method(*args, **kwargs), but method() doesn't have to be in the remote scope
    """
    args = [method, rref] + list(args)
    return rpc.rpc_sync(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _remote_method_async(method, rref, *args, **kwargs) -> torch.Future: # pylint: disable=no-member
    args = [method, rref] + list(args)
    return rpc.rpc_async(rref.owner(), _call_method, args=args, kwargs=kwargs)


def _remote_method_async_by_info(method, worker_info, *args, **kwargs):
    args = [method, worker_info] + list(args)
    return rpc.rpc_async(worker_info, _call_method, args=args, kwargs=kwargs)


def _remote_method_direct(method, other_node: str, *args, **kwargs):
    """
    Utility function for RPC communication between nodes.
    :param method: A callable function.
    :param other_node: reference to other node
    :return: any
    """

    args = [method, other_node] + list(args)
    return rpc.rpc_sync(other_node, method, args=args, kwargs=kwargs)
