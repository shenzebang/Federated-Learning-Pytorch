python run_FL.py --homo_ratio 0 --n_workers_per_round 50 --reduce_to_ratio .2 --imbalance --learner fed-avg\
      --local_lr 5e-2 --local_epoch 10 --use_ray --loss_fn focal-loss