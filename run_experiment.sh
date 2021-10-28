#P_FL

#python run_FL.py  --homo_ratio 0.1 --n_workers_per_round 50 --reduce_to_ratio .2\
#                  --learner fed-pd --local_lr 1e-1 --local_epoch 5 --client_step_per_epoch 20\
#                  --eta 100 --use_ray --fed_pd_dual_lr 1 --imbalance
#python run_FL.py  --homo_ratio 0.1 --n_workers_per_round 50 --reduce_to_ratio .2\
#                  --learner fed-avg --local_lr 1e-1 --local_epoch 5 --client_step_per_epoch 5\
#                  --use_ray --imbalance --use_gradient_clip





#python run_PD_FL.py --dataset mnist --homo_ratio 0.1 --n_workers_per_round 100\
#          --reduce_to_ratio .05 --use_ray --imbalance\
#            --formulation imbalance-fl --learner fed-avg --local_lr 5e-2\
#              --n_pd_rounds 200 --loss_fn cross-entropy-loss \
#                --n_workers 100 --n_p_steps 5  --no_data_augmentation\
#                  --lambda_lr 2 --tolerance_epsilon .01 --use_gradient_clip --n_minority 3\
#                  --heterogeneity dir --dir_level 0.3 --model mlp --dense_hid_dims 128-128

#python run_FL.py --dataset mnist --homo_ratio 0.1 --n_workers_per_round 100\
#          --reduce_to_ratio .05 --use_ray --imbalance\
#            --formulation imbalance-fl --learner fed-avg --local_lr 5e-2\
#              --n_pd_rounds 200 --loss_fn cross-entropy-loss \
#                --n_workers 100 --n_p_steps 5  --no_data_augmentation\
#                  --lambda_lr 2 --tolerance_epsilon .01 --use_gradient_clip --n_minority 3\
#                  --heterogeneity mix --dir_level 0.3 --model mlp --dense_hid_dims 128-128


python run_PD_FL.py --dataset cifar10 --homo_ratio 0.1 --n_workers_per_round 100\
          --reduce_to_ratio .1 --use_ray --imbalance --n_minority 1\
            --formulation imbalance-fl --learner fed-avg --local_lr 5e-2\
              --n_pd_rounds 200 --loss_fn cross-entropy-loss \
                --n_workers 100 --n_p_steps 5  --no_data_augmentation\
                  --lambda_lr 2 --tolerance_epsilon .01 --use_gradient_clip --n_minority 3\
                  --heterogeneity dir --dir_level 0.3

python run_FL.py --dataset cifar10 --homo_ratio 0.1 --n_workers_per_round 100\
          --reduce_to_ratio .1 --use_ray --imbalance --n_minority 1\
            --formulation imbalance-fl --learner fed-avg --local_lr 5e-2\
              --n_pd_rounds 200 --loss_fn cross-entropy-loss \
                --n_workers 100 --n_p_steps 5  --no_data_augmentation\
                  --lambda_lr 2 --tolerance_epsilon .01 --use_gradient_clip --n_minority 3\
                  --heterogeneity mix --dir_level 0.3

python run_PD_FL.py --dataset mnist --homo_ratio 0.1 --n_workers_per_round 100\
          --reduce_to_ratio .1 --use_ray --imbalance --n_minority 1\
            --formulation imbalance-fl --learner fed-avg --local_lr 5e-2\
              --n_pd_rounds 200 --loss_fn cross-entropy-loss \
                --n_workers 100 --n_p_steps 5  --no_data_augmentation\
                  --lambda_lr 2 --tolerance_epsilon .01 --use_gradient_clip --n_minority 3\
                  --heterogeneity dir --dir_level 0.3 --model mlp --dense_hid_dims 128-128

python run_FL.py --dataset mnist --homo_ratio 0.1 --n_workers_per_round 100\
          --reduce_to_ratio .1 --use_ray --imbalance --n_minority 1\
            --formulation imbalance-fl --learner fed-avg --local_lr 5e-2\
              --n_pd_rounds 200 --loss_fn cross-entropy-loss \
                --n_workers 100 --n_p_steps 5  --no_data_augmentation\
                  --lambda_lr 2 --tolerance_epsilon .01 --use_gradient_clip --n_minority 3\
                  --heterogeneity mix --dir_level 0.3 --model mlp --dense_hid_dims 128-128