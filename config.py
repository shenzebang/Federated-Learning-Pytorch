import argparse

def make_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument('--seed', type=int, default=0, help='Random seed For Reproducibility.')

    # general configurations
    parser.add_argument('--n_pd_rounds', type=int, default=5000, help='total dual rounds for PDFL')
    parser.add_argument('--n_global_rounds', type=int, default=5000, help='total communication rounds for FL')
    parser.add_argument('--test_batch_size', type=int, default=200)
    parser.add_argument('--use_ray', action='store_true')
    parser.add_argument('--load_model', action='store_true')
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--test_metric', type=str, choices=['accuracy', 'class_wise_accuracy'],
                        default='class_wise_accuracy', help='what to report in tensorboard')
    parser.add_argument('--eval_freq', type=int, default=1, help='how often the test loss should be checked')
    parser.add_argument('--weighted', action='store_true', help='allow clients to have different weights initially')
    parser.add_argument('--loss_fn', type=str, choices=['focal-loss', 'cross-entropy-loss'],
                        default='cross-entropy-loss', help='loss functional')


    # tricks for NN training
    parser.add_argument('--no_data_augmentation', action='store_true', help='disable the data augmentation')
    parser.add_argument('--use_gradient_clip', action='store_true')
    parser.add_argument('--gradient_clip_constant', type=float, default=5.)
    parser.add_argument('--weight_decay', type=float, default=1e-3)

    # Experiment setup
    parser.add_argument('--heterogeneity', type=str, choices=['mix', 'dir'], default='mix',
                        help='Type of heterogeneity, mix or dir(dirichlet)')
    parser.add_argument('--homo_ratio', type=float, default=1.)
    parser.add_argument('--dir_level', type=float, default=.3, help='hyperparameter of the Dirichlet distribution')
    parser.add_argument('--dataset', type=str, choices=['cifar10', 'cifar100', 'mnist', 'fashion-mnist', 'emnist', 'shakespeare'],
                        default='cifar10', help='dataset (and the corresponding task), now only support cifar10')
    parser.add_argument('--dense_hid_dims', type=str, default='384-192')
    parser.add_argument('--conv_hid_dims', type=str, default='64-64')
    parser.add_argument('--model', type=str, choices=['mlp', 'convnet', 'resnet'], default='convnet')
    parser.add_argument('--learner', type=str, choices=['fed-avg', 'fed-pd', 'scaffold'], default='fed-pd')
    parser.add_argument('--formulation', type=str, choices=['imbalance-fl','imbalance-fl-res', 'ratioloss-fl', 'GHMC_loss'],
                        default='imbalance-fl', help='formulation for handling class imbalance problem')
    parser.add_argument('--n_workers', type=int, default=50)
    parser.add_argument('--n_workers_per_round', type=int, default=5)
    parser.add_argument('--l2_reg', type=float, default=-1.)


    parser.add_argument('--imbalance', action='store_true', help='create imbalance among classes')
    parser.add_argument('--n_minority', type=int, default=1, help='number of minority classes')
    parser.add_argument('--reduce_to_ratio', type=float, default=1.)

    # General hyperparameters
    parser.add_argument('--local_lr', type=float, default=0.1)
    parser.add_argument('--global_lr', type=float, default=1.)
    parser.add_argument('--local_epoch', type=int, default=5)
    parser.add_argument('--client_step_per_epoch', type=int, default=5)

    # Hyperparameters for the formulation "imbalance-fl"
    parser.add_argument('--lambda_lr', type=float, default=1)
    parser.add_argument('--tolerance_epsilon', type=float, default=1.)
    parser.add_argument('--n_p_steps', type=int, default=5, help="primal steps per dual step in PDFL")
    # Resilient added this
    parser.add_argument('--perturbation_lr', type=float, default=0.5)
    parser.add_argument('--perturbation_penalty', type=float, default=4)
    # Hyperparameters for fed-pd
    parser.add_argument('--eta', type=float, default=10)
    parser.add_argument('--fed_pd_dual_lr', type=float, default=1)

    # wandb logging
    parser.add_argument('--run', type=str, default='FL-PD')
    parser.add_argument('--project', type=str, default='Fed')
    return parser