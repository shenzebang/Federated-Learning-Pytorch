#P_FL

#python run_FL.py  --dataset cifar10 --n_workers_per_round 100\
#                  --reduce_to_ratio 0.1 --use_ray --imbalance\
#                  --learner fed-avg --local_lr 5e-2\
#                  --n_global_rounds 5000 --loss_fn alpha-loss --model convnet\
#                  --n_workers 100 --eval_freq 5 --n_minority 1 --use_gradient_clip\
#                  --heterogeneity dir --dir_level 0.3
python run_FL.py  --dataset cifar10 --n_workers_per_round 100\
                  --reduce_to_ratio 0.1 --imbalance\
                  --learner fed-avg --local_lr 5e-2\
                  --n_global_rounds 5000 --loss_fn alpha-loss --model convnet\
                  --n_workers 100 --eval_freq 5 --n_minority 1 --use_gradient_clip\
                  --heterogeneity dir --dir_level 0.3
#
#python run_FL.py  --dataset cifar10 --n_workers_per_round 100\
#                  --reduce_to_ratio 0.1 --use_ray --imbalance\
#                  --learner fed-avg --local_lr 5e-2\
#                  --n_global_rounds 5000 --loss_fn alpha-loss --model convnet\
#                  --n_workers 100 --eval_freq 5 --n_minority 1 --use_gradient_clip\
#                  --heterogeneity mix --homo_ratio 0.1


#python run_FL.py  --homo_ratio 0.1 --n_workers_per_round 50 --reduce_to_ratio .2\
#                  --learner fed-avg --local_lr 1e-1 --local_epoch 5 --client_step_per_epoch 5\
#                  --use_ray --imbalance --use_gradient_clip