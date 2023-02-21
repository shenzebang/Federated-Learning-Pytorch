#P_FL

python run_PD_FL.py --dataset cifar10  --n_workers_per_round 100 --reduce_to_ratio 1\
	          --use_ray --imbalance --formulation imbalance-fl --learner fed-avg\
		  --local_lr 5e-2\
		  --n_pd_rounds 1000 --loss_fn cross-entropy-loss --model convnet\
		  --n_workers 100 --lambda_lr 0.1 --tolerance_epsilon 0.1\
		  --n_p_steps 5 --n_minority 1 --heterogeneity dir --dir_level 0.3\
		  --use_gradient_clip

python run_PD_FL.py --dataset cifar10  --n_workers_per_round 100 --reduce_to_ratio 0.1\
	          --use_ray --imbalance --formulation imbalance-fl --learner fed-avg\
		  --local_lr 5e-2\
		  --n_pd_rounds 1000 --loss_fn cross-entropy-loss --model convnet\
		  --n_workers 100 --lambda_lr 0.1 --tolerance_epsilon 0.1\
		  --n_p_steps 5 --n_minority 1 --heterogeneity dir --dir_level 0.3\
		  --use_gradient_clip
#python run_FL.py  --homo_ratio 0.1 --n_workers_per_round 50 --reduce_to_ratio .2\
#                  --learner fed-avg --local_lr 1e-1 --local_epoch 5 --client_step_per_epoch 5\
#                  --use_ray --imbalance --use_gradient_clip
