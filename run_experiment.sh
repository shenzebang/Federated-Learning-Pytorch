#P_FL

#python run_FL.py  --homo_ratio 0.1 --n_workers_per_round 50 --reduce_to_ratio .2\
#                  --learner fed-pd --local_lr 1e-1 --local_epoch 5 --client_step_per_epoch 20\
#                  --eta 100 --use_ray --fed_pd_dual_lr 1 --imbalance
python run_FL.py  --homo_ratio 0.1 --n_workers_per_round 50 --reduce_to_ratio .2\
                  --learner fed-avg --local_lr 1e-1 --local_epoch 5 --client_step_per_epoch 5\
                  --use_ray --imbalance --use_gradient_clip