# Federated-Learning-Pytorch



This repo contains code accompaning the following papers (and more to come), 
1. Federated Functional Gradient Boosting (Shen et al., AISTATS 2022).
2. An Agnostic Approach to Federated Learning with Class Imbalance (Shen et al., ICLR 2022).

It includes code for running the multiclass image classification experiments in the Federated Learning paradigm.
A few different settings are considered, including standard Federated Learning, Functional Federated Learning, and Constrained Federated Learning.
The implementation is based on pytorch. To support acceleration with multiple GPUs, we use ray.

To replicate the experiments presented in (Shen et al., AISTATS 2022), please check ``scripts/AISTATS2022-EMNIST.sh``.

## Usage
To run experiments in the standard setting of Federated Learning, please use run_FL.py. For example,
```
python run_FL.py --dataset emnist-digit --homo_ratio 0.1 --n_workers_per_round 100\
  --reduce_to_ratio .1 --use_ray\
  --learner fed-pd --local_lr 5e-2\
  --n_global_rounds 100 --loss_fn cross-entropy-loss --model convnet\
  --n_workers 100 --eval_freq 1 --eta 10\
  --client_step_per_epoch 10 --use_gradient_clip --local_epoch 20\
  --dense_hid_dims 120-84 --conv_hid_dims 64-64
```
To run experiments in the setting of Functional Federated Learning, please use run_FFL.py. For example,
```
python run_FFL.py --device cuda --use_ray --n_global_rounds 20\
  --test_batch_size 1000 --learner ffgb-d --dataset emnist-letter\
  --dataset_distill emnist-digit --homo_ratio .1 --n_workers 100 --n_workers_per_round 100\
  --functional_lr 10 --f_l2_reg 5e-3 --local_steps 1\
  --weak_learner_epoch 120 --weak_learner_lr 1e-3 --weak_learner_weight_decay 0\
  --distill_oracle l2 --distill_oracle_epoch 10 --distill_oracle_lr 1e-3\
   --distill_oracle_weight_decay 0 --dense_hid_dims 120-84 --conv_hid_dims 64-64\
```
To run experiments in the setting of Constrained Federated Learning, please use run_PD_FL.py
```
python run_PD_FL.py
```

## Structure of the Code
Another goal of this repo is to provide a template of the optimization methods for Federated Learning.

Structure of the code:
1. load configuration
2. prepare the local datasets
3. prepare logger, local objective function
4. run FL
5. save model

To adapt the current template to your algorithm, simply implement the following five functions:
1. server_init
2. client_init
3. clients_step
4. server_step
5. clients_update
