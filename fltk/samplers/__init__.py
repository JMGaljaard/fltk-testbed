from .distributed_sampler import DistributedSamplerWrapper
from .uniform import UniformSampler
from .n_label import N_Labels
from .q_sampler import Probability_q_Sampler
from .dirichlet import DirichletSampler
from .limit_labels import LimitLabelsSampler
from .limit_labels_flex import LimitLabelsSamplerFlex
from ..util.definitions import DataSampler
from ..util.log import getLogger


def get_sampler(dataset, args):
    logger = getLogger(__name__)
    sampler = None
    if args.get_distributed():
        method = args.get_sampler()
        logger.info(
            "Using {} sampler method, with args: {}".format(method, args.get_sampler_args()))

        if method == DataSampler.uniform:
            sampler = UniformSampler(dataset, num_replicas=args.get_world_size(), rank=args.get_rank())
        elif method == DataSampler.q_sampler:
            sampler = Probability_q_Sampler(dataset, num_replicas=args.get_world_size(), rank=args.get_rank(),
                                            args=args.get_sampler_args())
        elif method == DataSampler.limit_labels:
            sampler = LimitLabelsSampler(dataset, num_replicas=args.get_world_size(), rank=args.get_rank(),
                                         args=args.get_sampler_args())
        elif method == DataSampler.limit_labels_flex:
            sampler = LimitLabelsSamplerFlex(dataset, num_replicas=args.get_world_size(), rank=args.get_rank(),
                                             args=args.get_sampler_args())
        elif method == DataSampler.n_labels:
            sampler = N_Labels(dataset, num_replicas=args.get_world_size(), rank=args.get_rank(),
                               args=args.get_sampler_args())
        elif method == DataSampler.dirichlet:
            sampler = DirichletSampler(dataset, num_replicas=args.get_world_size(), rank=args.get_rank(),
                                       args=args.get_sampler_args())
        else:  # default
            logger.warning("Unknown sampler " + method + ", using uniform instead")
            sampler = UniformSampler(dataset, num_replicas=args.get_world_size(), rank=args.get_rank())

    return sampler
