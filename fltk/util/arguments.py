import torch

SEED = 1
torch.manual_seed(SEED)

class Arguments:

    def __init__(self):
        self.batch_size = 10
        self.test_batch_size = 1000
        self.epochs = 1
        self.lr = 0.001
        self.momentum = 0.9
        self.cuda = False
        self.shuffle = False
        self.log_interval = 10



        self.round_worker_selection_strategy = None
        self.round_worker_selection_strategy_kwargs = None

        self.save_model = False
        self.save_temp_model = False
        self.save_epoch_interval = 1
        self.save_model_path = "models"
        self.epoch_save_start_suffix = "start"
        self.epoch_save_end_suffix = "end"


        self.data_sampler = None
        self.distributed = False

        self.net = None
        self.set_net_by_name('Cifar10CNN')

        self.dataset_name = 'cifar10'
        self.train_data_loader_pickle_path = {
            'cifar10': 'data_loaders/cifar10/train_data_loader.pickle',
            'fashion-mnist': 'data_loaders/fashion-mnist/train_data_loader.pickle',
            'cifar100': 'data_loaders/cifar100/train_data_loader.pickle',
        }

        self.test_data_loader_pickle_path = {
            'cifar10': 'data_loaders/cifar10/test_data_loader.pickle',
            'fashion-mnist': 'data_loaders/fashion-mnist/test_data_loader.pickle',
            'cifar100': 'data_loaders/cifar100/test_data_loader.pickle',
        }


        self.loss_function = torch.nn.CrossEntropyLoss

        self.default_model_folder_path = "default_models"

        self.data_path = "data"


    def get_data_path(self):
        return self.data_path

    def get_epoch_save_start_suffix(self):
        return self.epoch_save_start_suffix

    def get_epoch_save_end_suffix(self):
        return self.epoch_save_end_suffix

    def get_dataloader_list(self):
        return list(self.train_data_loader_pickle_path.keys())

    def get_nets_list(self):
        return list(self.available_nets.keys())


    def set_train_data_loader_pickle_path(self, path, name='cifar10'):
        self.train_data_loader_pickle_path[name] = path

    def get_train_data_loader_pickle_path(self):
        return self.train_data_loader_pickle_path[self.dataset_name]

    def set_test_data_loader_pickle_path(self, path, name='cifar10'):
        self.test_data_loader_pickle_path[name] = path

    def get_test_data_loader_pickle_path(self):
        return self.test_data_loader_pickle_path[self.dataset_name]

    def set_net_by_name(self, name: str):
        self.net = self.available_nets[name]
        # net_dict = {
        #     'cifar10-cnn': Cifar10CNN,
        #     'fashion-mnist-cnn': FashionMNISTCNN,
        #     'cifar100-resnet': Cifar100ResNet,
        #     'fashion-mnist-resnet': FashionMNISTResNet,
        #     'cifar10-resnet': Cifar10ResNet,
        #     'cifar100-vgg': Cifar100VGG,
        # }
        # self.net = net_dict[name]

    def get_cuda(self):
        return self.cuda

    def get_scheduler_step_size(self):
        return self.scheduler_step_size

    def get_scheduler_gamma(self):
        return self.scheduler_gamma

    def get_min_lr(self):
        return self.min_lr

    def get_default_model_folder_path(self):
        return self.default_model_folder_path

    def get_num_epochs(self):
        return self.epochs

    def set_num_poisoned_workers(self, num_poisoned_workers):
        self.num_poisoned_workers = num_poisoned_workers

    def set_num_workers(self, num_workers):
        self.num_workers = num_workers

    def set_model_save_path(self, save_model_path):
        self.save_model_path = save_model_path


    def get_momentum(self):
        return self.momentum

    def get_shuffle(self):
        return self.shuffle

    def get_batch_size(self):
        return self.batch_size

    def get_test_batch_size(self):
        return self.test_batch_size

    def get_log_interval(self):
        return self.log_interval


