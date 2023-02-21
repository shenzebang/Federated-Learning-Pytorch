#P_FL

python run_PD_FL.py --dataset cifar10 --n_workers_per_round 100\
          --reduce_to_ratio .2 --use_ray --imbalance --n_minority 1\
            --formulation imbalance-fl --learner fed-avg --local_lr 5e-2\
              --n_pd_rounds 1000 --loss_fn cross-entropy-loss \
                --n_workers 100 --n_p_steps 5  --no_data_augmentation\
                  --lambda_lr 2 --tolerance_epsilon .01 --use_gradient_clip --n_minority 1\
                  --heterogeneity dir --dir_level 0.6