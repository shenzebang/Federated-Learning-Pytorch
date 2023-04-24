import time
import torch
from utils.general_utils import save_model, seed_everything
from utils.data_utils import load_dataset, make_dataloader, split_dataset, create_imbalance, get_auxiliary_data, make_transforms
from utils.model_utils import make_model
from utils.logger_utils import wandbLogger
from utils.test_utils import make_evaluate_fn, make_monitor_fn
from core.fed_avg import FEDAVG
from core.fed_pd import FEDPD
from core.imbalance_fl import ImbalanceFL, ImbalanceFLRes
from core.ratio_loss_fl import RatioLossFL
from torch.utils.tensorboard import SummaryWriter
from config import make_parser
import os
import json
import wandb
from copy import deepcopy

FEDERATED_LEARNERS = {
    'fed-avg': FEDAVG,
    'fed-pd': FEDPD
}

PD_FEDERATED_LEARNERS = {
    'imbalance-fl': ImbalanceFL,
    'imbalance-fl-res':ImbalanceFLRes,
    'ratioloss-fl': RatioLossFL
}


def main():
    # 1. load the configurations
    args = make_parser().parse_args()
    print(args)
    seed_everything(args.seed)
    wandb.init(project=args.project, name=args.run)
    wandb.config.update(args)
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loss = torch.nn.functional.cross_entropy

    level = args.homo_ratio if args.heterogeneity == "mix" else args.dir_level
    experiment_setup = f"Class_Imbalance_FL_{args.formulation}_{args.heterogeneity}_{level}_{args.n_workers}_{args.n_workers_per_round}_{args.dataset}_{args.n_minority}_{args.reduce_to_ratio}_{args.model}"
    hyperparameter_setup = f"{args.learner}_{args.global_lr}_{args.local_lr}_{args.lambda_lr}_{args.n_p_steps}_{args.tolerance_epsilon}_{args.client_step_per_epoch}_{args.local_epoch}"
    if args.learner == "fed-pd":
        hyperparameter_setup += f"_{args.eta}_{args.fed_pd_dual_lr}"

    args.save_dir = 'output/%s/%s' %(experiment_setup, hyperparameter_setup)
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
    with open(args.save_dir+'/config.json', 'w') as f:
        json.dump(vars(args), f)

    # 2. prepare the data set
    dataset_train, dataset_test, n_classes, n_channels, img_size = load_dataset(args)

    if args.imbalance:
        assert (args.n_minority < n_classes)
        if args.n_minority == 1:
            reduce_classes = (0,)
        elif args.n_minority == 3:
            reduce_classes = (0, 2, 4)
        elif args.n_minority == 5:
            reduce_classes = (0, 2, 4, 6, 8)
        else:
            raise RuntimeError
        dataset_train = create_imbalance(dataset_train, reduce_classes=reduce_classes,
                                         reduce_to_ratio=args.reduce_to_ratio)

    transforms = make_transforms(args, train=True)  # transforms for data augmentation and normalization
    local_datasets, class_distribution = split_dataset(args, dataset_train, transforms, reduce_classes=reduce_classes)
    local_dataloaders = [make_dataloader(args, local_dataset) for local_dataset in local_datasets]

    transforms_test = make_transforms(args, train=False)
    dataset_test.transform = transforms_test
    test_dataloader = make_dataloader(args, dataset_test)

    local_datasets_test, _ = split_dataset(args, dataset_test, transforms_test, ratio_per_client=class_distribution, reduce_classes=reduce_classes)
    local_dataloaders_test = [make_dataloader(args, local_dataset_t) for local_dataset_t in local_datasets_test]
    

    model = make_model(args, n_classes, n_channels, device, img_size)

    test_fn_accuracy = make_evaluate_fn(test_dataloader, device, eval_type='accuracy', n_classes=n_classes, loss_fn=loss)
    test_fn_class_wise_accuracy = make_evaluate_fn(test_dataloader, device, eval_type='class_wise_accuracy', n_classes=n_classes)
    #statistics_monitor_fn = make_monitor_fn()

    # 3. prepare logger
    tb_file = args.save_dir+f'/{time.time()}'
    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)
    logger_accuracy = wandbLogger(writer, test_fn_accuracy, test_metric='accuracy')
    logger_class_wise_accuracy = wandbLogger(writer, test_fn_class_wise_accuracy, test_metric='class_wise_accuracy')
    #logger_monitor = wandbLogger(writer, statistics_monitor_fn, test_metric='model_monitor')
    loggers = [logger_accuracy, logger_class_wise_accuracy]
    # 4. run PD FL

    make_pd_fed_learner = PD_FEDERATED_LEARNERS[args.formulation]
    make_fed_learner = FEDERATED_LEARNERS[args.learner]

    fed_learner = make_fed_learner(init_model=model,
                                   client_dataloaders=local_dataloaders,
                                   client_dataloaders_test = local_dataloaders_test,
                                   loss=loss,
                                   loggers=None,
                                   config=args,
                                   device=device
                                   )
    n_aux = 5
    auxiliary_data = get_auxiliary_data(args, transforms_test, dataset_train, n_classes,
                                        n_aux) if args.formulation == "ratioloss-fl" else None

    pd_fed_learner = make_pd_fed_learner(fed_learner, args, loggers, auxiliary_data)

    pd_fed_learner.fit()

    # # 4. save the model
    save_model(args, pd_fed_learner)


if __name__ == '__main__':
    main()
