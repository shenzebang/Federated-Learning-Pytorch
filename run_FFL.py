import torch.utils.data
from torch.utils.tensorboard import SummaryWriter
from utils.loss_utils import Dx_cross_entropy
from config import make_parser
from utils.data_utils import load_dataset, make_transforms, make_dataloader, split_dataset
from utils.model_utils import make_model as _make_model
from core.ffgb_distill import FFGB_D
from core.fedavg_distill import FEDAVG_D
from utils.logger_utils import make_evaluate_fn, make_monitor_fn, Logger
import json
import time
import os



FEDERATED_LEARNERS = {
    'ffgb-d': FFGB_D,
    'fedavg-d': FEDAVG_D
}


if __name__ == '__main__':
    # 1. load the configurations
    args = make_parser().parse_args()
    print(args)

    # device_ids = [int(a) for a in args.device_ids.split(",")]
    # if device_ids[0] != -1:
    #     os.environ["CUDA_VISIBLE_DEVICES"] = f"{args.device_ids}"

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")


    Dx_loss = Dx_cross_entropy
    loss = torch.nn.functional.cross_entropy

    level = args.homo_ratio if args.heterogeneity == "mix" else args.dir_level
    experiment_setup = f"FFL_{args.heterogeneity}_{level}_{args.n_workers}_{args.n_workers_per_round}_{args.dataset}_{args.model}"
    hyperparameter_setup = f"{args.learner}_{args.local_dataloader_batch_size}_{args.distill_dataloader_batch_size}"
    if args.learner == "ffgb-d":
        hyperparameter_setup += f"_{args.local_steps}_{args.functional_lr}_{args.f_l2_reg}_{args.weak_learner_epoch}_{args.weak_learner_lr}_{args.weak_learner_weight_decay}"
    elif args.learner == 'fedavg-d':
        hyperparameter_setup += f"_{args.fedavg_d_local_lr}_{args.fedavg_d_local_epoch}_{args.fedavg_d_weight_decay}"
    else:
        raise NotImplementedError

    args.save_dir = 'output/%s/%s' % (experiment_setup, hyperparameter_setup)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(args.save_dir + '/config.json', 'w') as f:
        json.dump(vars(args), f)

    # 2. prepare the data set
    dataset_trn, dataset_tst, n_classes, n_channels, img_size = load_dataset(args.dataset)
    dataset_distill, _, _, _, _ = load_dataset(args.dataset_distill)

    transforms = make_transforms(args, args.dataset, train=True)  # transforms for data augmentation and normalization
    local_datasets = split_dataset(args, dataset_trn, transforms)
    client_dataloaders = [make_dataloader(args, "train", local_dataset) for local_dataset in local_datasets]

    transforms_test = make_transforms(args, args.dataset, train=False)
    dataset_tst.transform = transforms_test
    test_dataloader = make_dataloader(args, "test", dataset_tst)

    transforms_distill = make_transforms(args, args.dataset_distill, train=True)
    dataset_distill.transform = transforms_distill
    if args.dataset_distill == 'emnist-digit' or args.dataset_distill == 'emnist-letter':
        distill_dataloader = make_dataloader(args, "distill", dataset_distill, shuffle=False)
    else:
        distill_dataloader = make_dataloader(args, "distill", dataset_distill, shuffle=True)

    test_fn_accuracy = make_evaluate_fn(test_dataloader, device, eval_type='accuracy', n_classes=n_classes,
                                        loss_fn=loss)
    statistics_monitor_fn = make_monitor_fn()

    # 3. prepare logger
    tb_file = args.save_dir + f'/{time.time()}'
    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)
    logger_accuracy = Logger(writer, test_fn_accuracy, test_metric='accuracy')
    logger_monitor = Logger(writer, statistics_monitor_fn, test_metric='model_monitor')
    loggers = [logger_accuracy, logger_monitor]

    # 4. run Functional FL
    weights = [1.] * args.n_workers
    weights_sum = sum(weights)
    weights = [weight / weights_sum * args.n_workers for weight in weights]

    make_model = lambda: _make_model(args, n_classes, n_channels, device, img_size)
    model_init = make_model()

    make_fed_learner = FEDERATED_LEARNERS[args.learner]

    fed_learner = make_fed_learner(model_init, make_model, client_dataloaders, distill_dataloader, Dx_loss, loggers, args, device)


    fed_learner.fit(weights)

    # # 5. save model
    # if args.save_model:
    #     model_file = f"./model_{args.dataset}.pth"
    #     torch.save(fed_learner.server_state.model.state_dict(), model_file)