import time
import argparse
import torch
from utils import load_dataset, make_model, make_dataloader, split_dataset, make_evaluate_fn, save_model,\
    make_transforms, Logger
from core.fed_avg import FEDAVG
from torch.utils.tensorboard import SummaryWriter
FEDERATED_LEARNERS = {
    'fed-avg': FEDAVG
}


def make_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', type=str, choices=['cifar10'], default='cifar10')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--dense_hid_dims', type=str, default='120-84')
    parser.add_argument('--conv_hid_dims', type=str, default='6-16')
    parser.add_argument('--model', type=str, choices=['mlp', 'convnet'], default='convnet')
    parser.add_argument('--learner', type=str, choices=['fed-avg'], default='fed-avg')
    parser.add_argument('--local_lr', type=float, default=0.05)
    parser.add_argument('--homo_ratio', type=float, default=0.1)
    parser.add_argument('--n_workers', type=int, default=50)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--client_step_per_epoch', type=int, default=5)
    parser.add_argument('--test_batch_size', type=int, default=200)
    parser.add_argument('--use_ray', type=bool, default=True)
    parser.add_argument('--n_global_rounds', type=int, default=5000)
    parser.add_argument('--eval_freq', type=int, default=1)

    return parser


def main():
    # 1. load the configurations
    args = make_parser().parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    # 2. prepare the data set
    dataset_train, dataset_test, n_classes, n_channels = load_dataset(args)
    transforms = make_transforms(args, train=True) # transforms for data augmentation and normalization
    local_datasets = split_dataset(args, dataset_train, transforms)
    local_dataloaders = [make_dataloader(args, local_dataset) for local_dataset in local_datasets]

    transforms_test = make_transforms(args, train=False)
    dataset_test.transform = transforms_test
    test_dataloader = make_dataloader(args, dataset_test)

    model = make_model(args, n_classes, n_channels, device)

    test_fn = make_evaluate_fn(test_dataloader, device)

    # 3. prepare logger, loss

    loss = torch.nn.functional.cross_entropy
    ts = time.time()
    tb_file = f'out/{args.dataset}/{args.conv_hid_dims}_{args.dense_hid_dims}/s{args.homo_ratio}' \
              f'/N{args.n_workers}/rhog{args.local_lr}_{args.learner}_{ts}'

    print(f"writing to {tb_file}")
    writer = SummaryWriter(tb_file)
    logger = Logger(writer)
    # 4. run weighted FL
    # todo: assign weights to different clients

    make_fed_learner = FEDERATED_LEARNERS[args.learner]

    fed_learner = make_fed_learner(model, local_dataloaders, loss, test_fn, logger, args, device)

    fed_learner.fit()

    # # 4. save the model
    # save_model(args, fed_learner)


if __name__ == '__main__':
    main()



