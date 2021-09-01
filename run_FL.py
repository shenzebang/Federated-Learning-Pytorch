import time
import argparse
import torch
from utils import load_dataset, make_model, make_dataloader, split_dataset, make_evaluate_fn, save_model,\
    make_transforms, Logger, create_imbalance
from core.fed_avg import FEDAVG
from core.fed_dyn import FEDDYN
from core.fed_pd import FEDPD
from core.scaffold import SCAFFOLD
from torch.utils.tensorboard import SummaryWriter
FEDERATED_LEARNERS = {
    'fed-avg': FEDAVG,
    'fed-dyn': FEDDYN,
    'fed-pd' : FEDPD,
    'scaffold': SCAFFOLD,
}


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cifar10'], default='cifar10')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dense_hid_dims', type=str, default='384-192')
    parser.add_argument('--conv_hid_dims', type=str, default='64-64')
    parser.add_argument('--model', type=str, choices=['mlp', 'convnet', 'resnet'], default='convnet')
    parser.add_argument('--learner', type=str, choices=['fed-avg', 'fed-dyn', 'fed-pd', 'scaffold'], default='fed-avg')
    parser.add_argument('--local_lr', type=float, default=0.1)
    parser.add_argument('--alpha', type=float, default=.1)
    parser.add_argument('--eta', type=float, default=10)
    parser.add_argument('--global_lr', type=float, default=1.)
    parser.add_argument('--homo_ratio', type=float, default=1.)
    parser.add_argument('--n_workers', type=int, default=50)
    parser.add_argument('--n_workers_per_round', type=int, default=5)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--client_step_per_epoch', type=int, default=5)
    parser.add_argument('--test_batch_size', type=int, default=200)
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--n_global_rounds', type=int, default=5000)
    parser.add_argument('--eval_freq', type=int, default=1)
    ################################################################
    # what to report in tensorboard
    parser.add_argument('--test_metric', type=str, choices=['accuracy', 'class_wise_accuracy'], default='class_wise_accuracy')
    ################################################################
    # allow the clients to have different weights
    parser.add_argument('--weighted', action='store_true')
    ################################################################
    # create imbalance among classes
    parser.add_argument('--imbalance', action='store_true')
    parser.add_argument('--reduce_to_ratio', type=float, default=1.)
    # disable the data augmentation
    parser.add_argument('--no_data_augmentation', action='store_true')
    # gradient clip for scaffold may not be correct. removed temporarily
    parser.add_argument('--use_gradient_clip', action='store_true')
    parser.add_argument('--gradient_clip_constant', type=float, default=5.)
    return parser


def main():
    # 1. load the configurations
    args = make_parser().parse_args()
    print(args)

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    loss = torch.nn.functional.cross_entropy
    # 2. prepare the data set
    dataset_train, dataset_test, n_classes, n_channels = load_dataset(args)
    if args.imbalance:
        dataset_train = create_imbalance(dataset_train, reduce_to_ratio=args.reduce_to_ratio)

    transforms = make_transforms(args, train=True) # transforms for data augmentation and normalization
    local_datasets = split_dataset(args.n_workers, args.homo_ratio, dataset_train, transforms)
    local_dataloaders = [make_dataloader(args, local_dataset) for local_dataset in local_datasets]

    transforms_test = make_transforms(args, train=False)
    dataset_test.transform = transforms_test
    test_dataloader = make_dataloader(args, dataset_test)

    model = make_model(args, n_classes, n_channels, device)

    test_fn_accuracy = make_evaluate_fn(test_dataloader, device, eval_type='accuracy', n_classes=n_classes, loss_fn=loss)
    test_fn_class_wise_accuracy = make_evaluate_fn(test_dataloader, device, eval_type='class_wise_accuracy',
                                                   n_classes=n_classes)

    # 3. prepare logger
    ts = time.time()
    if args.model == 'resnet':
        tb_file = f'out/p_fl/{args.dataset}/resnet20/s{args.homo_ratio}' \
                  f'/N{args.n_workers}/rhog{args.local_lr}_{args.learner}_{ts}'
    else:
        tb_file = f'out/p_fl/{args.dataset}/convnet/{args.conv_hid_dims}_{args.dense_hid_dims}/s{args.homo_ratio}' \
              f'/N{args.n_workers}/rhog{args.local_lr}_{args.learner}_{ts}'

    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)
    logger_accuracy = Logger(writer, test_fn_accuracy, test_metric='accuracy')
    logger_class_wise_accuracy = Logger(writer, test_fn_class_wise_accuracy, test_metric='class_wise_accuracy')
    loggers = [logger_accuracy, logger_class_wise_accuracy]
    # 4. run weighted FL
    # todo: assign weights to different clients

    if args.weighted:
        weights = [1.] * args.n_workers
        weights[0] = 2.
    else:
        weights = [1.] * args.n_workers
    weights_sum = sum(weights)
    weights = [weight/weights_sum*args.n_workers for weight in weights]


    make_fed_learner = FEDERATED_LEARNERS[args.learner]

    fed_learner = make_fed_learner(model, local_dataloaders, loss, loggers, args, device)



    fed_learner.fit(weights)

    # # 4. save the model
    # save_model(args, fed_learner)


if __name__ == '__main__':
    main()



