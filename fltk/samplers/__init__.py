from .distributed_sampler import DistributedSamplerWrapper
from .uniform import UniformSampler
from .n_label import N_Labels
from .q_sampler import Probability_q_Sampler
from .dirichlet import DirichletSampler
from .limit_labels import LimitLabelsSampler
from .limit_labels_flex import LimitLabelsSamplerFlex
from fltk.util.log import getLogger

import fltk.util.config.definitions as defs

def get_sampler(dataset, args):
    """
    Helper function to get DataSampler configured with corresponding arguments. Returns None when invalid sampler
    configuration was provided.
    @param dataset: Dataset to pass to DataSampler during instantiation.
    @type dataset: Dataset
    @param args: Configuration object containing arguments to DataSampler instantiation.
    @type args: Any
    @return: Data sampler setup with arguments provided by args and dataset.
    @rtype: Optional[DataSampler]
    """
    logger = getLogger(__name__)
    sampler = None
    if args.get_distributed():
        method = args.get_sampler()
        msg = f"Using {method} sampler method, with args: {args.get_sampler_args()}"
        logger.debug(msg)

        if method == defs.DataSampler.uniform:
            sampler = UniformSampler(dataset, num_replicas=args.get_world_size(), rank=args.get_rank())
        elif method == defs.DataSampler.q_sampler:
            sampler = Probability_q_Sampler(dataset, num_replicas=args.get_world_size(), rank=args.get_rank(),
                                            args=args.get_sampler_args())
        elif method == defs.DataSampler.limit_labels:
            sampler = LimitLabelsSampler(dataset, num_replicas=args.get_world_size(), rank=args.get_rank(),
                                         args=args.get_sampler_args())
        elif method == defs.DataSampler.limit_labels_flex:
            sampler = LimitLabelsSamplerFlex(dataset, num_replicas=args.get_world_size(), rank=args.get_rank(),
                                             args=args.get_sampler_args())
        elif method == defs.DataSampler.n_labels:
            sampler = N_Labels(dataset, num_replicas=args.get_world_size(), rank=args.get_rank(),
                               args=args.get_sampler_args())
        elif method == defs.DataSampler.dirichlet:
            sampler = DirichletSampler(dataset, num_replicas=args.get_world_size(), rank=args.get_rank(),
                                       args=args.get_sampler_args())
        else:  # default
            msg = f"Unknown sampler {method}, using uniform instead"
            logger.warning(msg)
            sampler = UniformSampler(dataset, num_replicas=args.get_world_size(), rank=args.get_rank())

    return sampler
