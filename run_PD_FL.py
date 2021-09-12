import time
import torch
from utils import load_dataset, make_model, make_dataloader, split_dataset, make_evaluate_fn, save_model, \
    make_transforms, Logger, create_imbalance, make_monitor_fn
from core.fed_avg import FEDAVG
from core.fed_pd import FEDPD
from core.imbalance_fl import ImbalanceFL
from torch.utils.tensorboard import SummaryWriter
from config import make_parser
import os
import json

FEDERATED_LEARNERS = {
    'fed-avg': FEDAVG,
    'fed-pd': FEDPD
}

PD_FEDERATED_LEARNERS = {
    'imbalance-fl': ImbalanceFL
}





def main():
    # 1. load the configurations
    args = make_parser().parse_args()
    print(args)

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
    with open(args.save_dir+'config.json', 'w') as f:
        json.dump(vars(args), f)

    # 2. prepare the data set
    dataset_train, dataset_test, n_classes, n_channels = load_dataset(args)
    if args.imbalance:
        dataset_train = create_imbalance(dataset_train, reduce_to_ratio=args.reduce_to_ratio)

    transforms = make_transforms(args, train=True)  # transforms for data augmentation and normalization
    local_datasets = split_dataset(args.n_workers, args.homo_ratio, dataset_train, transforms)
    local_dataloaders = [make_dataloader(args, local_dataset) for local_dataset in local_datasets]

    transforms_test = make_transforms(args, train=False)
    dataset_test.transform = transforms_test
    test_dataloader = make_dataloader(args, dataset_test)

    model = make_model(args, n_classes, n_channels, device)

    test_fn_accuracy = make_evaluate_fn(test_dataloader, device, eval_type='accuracy', n_classes=n_classes, loss_fn=loss)
    test_fn_class_wise_accuracy = make_evaluate_fn(test_dataloader, device, eval_type='class_wise_accuracy', n_classes=n_classes)
    statistics_monitor_fn = make_monitor_fn()

    # 3. prepare logger
    tb_file = args.save_dir+f'{time.time()}'
    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)
    logger_accuracy = Logger(writer, test_fn_accuracy, test_metric='accuracy')
    logger_class_wise_accuracy = Logger(writer, test_fn_class_wise_accuracy, test_metric='class_wise_accuracy')
    logger_monitor = Logger(writer, statistics_monitor_fn, test_metric='model_monitor')
    loggers = [logger_accuracy, logger_class_wise_accuracy, logger_monitor]
    # 4. run PD FL

    make_pd_fed_learner = PD_FEDERATED_LEARNERS[args.formulation]
    make_fed_learner = FEDERATED_LEARNERS[args.learner]

    fed_learner = make_fed_learner(init_model=model,
                                   client_dataloaders=local_dataloaders,
                                   loss=loss,
                                   loggers=None,
                                   config=args,
                                   device=device
                                   )
    pd_fed_learner = make_pd_fed_learner(fed_learner, args, loggers)

    pd_fed_learner.fit()

    # # 4. save the model
    # save_model(args, fed_learner)


if __name__ == '__main__':
    main()
