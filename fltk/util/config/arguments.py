from argparse import ArgumentParser

import torch.distributed as dist


def _create_extractor_parser(subparsers):
    """
    Helper function to add extractor arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    extractor_parser = subparsers.add_parser('extractor')
    extractor_parser.add_argument('config', type=str)


def _create_client_parser(subparsers) -> None:
    """
    Helper function to add client arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    client_parser = subparsers.add_parser('client')
    client_parser.add_argument('experiment_config', type=str, help="Experiment specific config (yaml).")
    client_parser.add_argument('task_id', type=str, help="Unique identifier for task.")
    client_parser.add_argument('config', type=str, help="General cluster/orchestrator config (json).")
    # Add parameter parser for backend
    client_parser.add_argument('--backend', type=str, help='Distributed backend',
                               choices=[dist.Backend.GLOO, dist.Backend.NCCL, dist.Backend.MPI],
                               default=dist.Backend.GLOO)


def _create_cluster_parser(subparsers) -> None:
    """
    Helper function to add cluster execution arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    cluster_parser = subparsers.add_parser('cluster')
    cluster_parser.add_argument('config', type=str)
    cluster_parser.add_argument('experiment', type=str)
    cluster_parser.add_argument('-l', '--local', type=bool, default=False)


def _create_container_util_parser(subparsers) -> None:
    """
    Helper function to add container util execution arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    util_docker_parser = subparsers.add_parser('util-docker')
    util_docker_parser.add_argument('name', type=str)
    util_docker_parser.add_argument('--clients', type=int)


def _create_util_parser(subparsers):
    """
    Helper function to add util generation execution arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    util_generate_parser = subparsers.add_parser('util-generate')
    util_generate_parser.add_argument('path', type=str)


def _create_util_run_parser(subparsers) -> None:
    """
    Helper function to add util run execution arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    util_run_parser = subparsers.add_parser('util-run')
    util_run_parser.add_argument('path', type=str)


def _create_remote_parser(subparsers) -> None:
    """
    Helper function to add remote Federated Learning execution arguments. Supports both Docker and K8s execution
    using optional (positional) arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    remote_parser = subparsers.add_parser('remote')
    add_default_arguments(remote_parser)

    remote_parser.add_argument('rank', nargs='?', type=int, default=None)
    remote_parser.add_argument('--nic', type=str, default=None)
    remote_parser.add_argument('--host', type=str, default=None)


def _create_single_parser(subparsers) -> None:
    """
    Helper function to add Local single machine execution arguments.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    single_machine_parser = subparsers.add_parser('single')
    add_default_arguments(single_machine_parser)


def add_default_arguments(*parsers):
    """
    Helper function to add default arguments shared between executions.
    @param parsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: None
    """
    for parser in parsers:
        parser.add_argument('config', type=str, help='')
        parser.add_argument('--prefix', type=str, default=None)


def create_all_subparsers(subparsers):
    """
    Helper function to add all subparsers to an argparse object.
    @param subparsers: Subparser to add arguments to.
    @type subparsers: Any
    @return: None
    @rtype: ArgumentParser
    """
    _create_extractor_parser(subparsers)
    _create_client_parser(subparsers)
    _create_cluster_parser(subparsers)
    _create_container_util_parser(subparsers)
    _create_util_parser(subparsers)
    _create_util_run_parser(subparsers)
    _create_remote_parser(subparsers)
    _create_single_parser(subparsers)
