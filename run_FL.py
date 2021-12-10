import time

import torch
from utils.data_utils import load_dataset, make_dataloader, make_transforms, create_imbalance, split_dataset
from utils.logger_utils import Logger
from utils.test_utils import make_evaluate_fn, make_monitor_fn
from utils.model_utils import make_model
from utils.loss_utils import focal_loss
from utils.loss_utils import alpha_loss
from core.fed_avg import FEDAVG
from core.fed_pd import FEDPD
from core.scaffold import SCAFFOLD
from config import make_parser
from torch.utils.tensorboard import SummaryWriter
import torch.nn.functional as F
import os, json
import warnings

warnings.filterwarnings("ignore")
FEDERATED_LEARNERS = {
    'fed-avg': FEDAVG,
    'fed-pd' : FEDPD,
    'scaffold': SCAFFOLD,
}

LOSS_FNS = {
    'cross-entropy-loss': F.cross_entropy,
    'focal-loss': focal_loss,
    'alpha-loss': alpha_loss
}

def main():
    # 1. load the configurations
    args = make_parser().parse_args()
    print(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loss_fn = LOSS_FNS[args.loss_fn]

    level = args.homo_ratio if args.heterogeneity == "mix" else args.dir_level
    experiment_setup = f"FL_{args.heterogeneity}_{level}_{args.n_workers}_{args.n_workers_per_round}_{args.dataset}_{args.n_minority}_{args.reduce_to_ratio}_{args.model}_{args.weighted}_{args.loss_fn}"
    hyperparameter_setup = f"{args.learner}_{args.global_lr}_{args.local_lr}_{args.client_step_per_epoch}_{args.local_epoch}"
    if args.learner == "fed-pd":
        hyperparameter_setup += f"_{args.eta}_{args.fed_pd_dual_lr}"

    args.save_dir = 'output/%s/%s' % (experiment_setup, hyperparameter_setup)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(args.save_dir + '/config.json', 'w') as f:
        json.dump(vars(args), f)
    # 2. prepare the data set


    dataset_train, dataset_test, n_classes, n_channels, img_size = load_dataset(args)
    if args.imbalance:
        assert(args.n_minority < n_classes)
        if args.n_minority == 1:
            reduce_classes = (0,)
        elif args.n_minority == 3:
            reduce_classes = (0, 2, 4)
        elif args.n_minority == 5:
            reduce_classes = (0, 2, 4, 6, 8)
        else:
            raise RuntimeError
        dataset_train = create_imbalance(dataset_train, reduce_classes=reduce_classes, reduce_to_ratio=args.reduce_to_ratio)

    transforms = make_transforms(args, train=True) # transforms for data augmentation and normalization
    local_datasets = split_dataset(args, dataset_train, transforms)
    local_dataloaders = [make_dataloader(args, local_dataset) for local_dataset in local_datasets]

    transforms_test = make_transforms(args, train=False)
    dataset_test.transform = transforms_test
    test_dataloader = make_dataloader(args, dataset_test)

    model = make_model(args, n_classes, n_channels, device, img_size)
    if args.load_model:
        model.load_state_dict(torch.load(f"model_{args.dataset}.pth"))

    test_fn_accuracy = make_evaluate_fn(test_dataloader, device, eval_type='accuracy', n_classes=n_classes, loss_fn=loss_fn)
    test_fn_class_wise_accuracy = make_evaluate_fn(test_dataloader, device, eval_type='class_wise_accuracy',
                                                   n_classes=n_classes)
    statistics_monitor_fn = make_monitor_fn()

    # 3. prepare logger
    tb_file = args.save_dir + f'/{time.time()}'
    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)
    logger_accuracy = Logger(writer, test_fn_accuracy, test_metric='accuracy')
    logger_class_wise_accuracy = Logger(writer, test_fn_class_wise_accuracy, test_metric='class_wise_accuracy')
    logger_monitor = Logger(writer, statistics_monitor_fn, test_metric='model_monitor')
    loggers = [logger_accuracy, logger_class_wise_accuracy, logger_monitor]

    # 4. run weighted FL
    if args.weighted:
        weights = [1.] * args.n_workers
        weights[0] = 2.
    else:
        weights = [1.] * args.n_workers
    weights_sum = sum(weights)
    weights = [weight/weights_sum*args.n_workers for weight in weights]


    make_fed_learner = FEDERATED_LEARNERS[args.learner]

    fed_learner = make_fed_learner(model, local_dataloaders, loss_fn, loggers, args, device)



    fed_learner.fit(weights)

    # # 4. save the model
    # save_model(args, fed_learner)


if __name__ == '__main__':
    main()



