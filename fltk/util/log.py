import logging

from torch.distributed import rpc

class FLLogger:
    @staticmethod
    @rpc.functions.async_execution
    def log(arg1, node_id, log_line, report_time):
        logging.info(f'[{node_id}: {report_time}]: {log_line}')