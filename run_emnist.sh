python run_FL.py --dataset emnist --homo_ratio 0.1 --n_workers_per_round 100\
	         --reduce_to_ratio .1 --use_ray --heterogeneity dir --dir_level 10\
		           --learner fed-pd --local_lr 5e-2\
			              --n_global_rounds 500 --loss_fn cross-entropy-loss --model convnet\
				                  --n_workers 100 --eval_freq 1 --eta 10\
						               --client_step_per_epoch 10 --use_gradient_clip --local_epoch 20

