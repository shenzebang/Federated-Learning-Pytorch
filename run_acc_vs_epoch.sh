python run_PD_FL.py --dataset cifar10 --homo_ratio 0.2 --n_workers_per_round 100\
	  --reduce_to_ratio .1 --use_ray --imbalance\
	    --formulation imbalance-fl --learner fed-avg --local_lr 5e-2\
	      --n_pd_rounds 1000 --loss_fn cross-entropy-loss --model convnet\
	        --n_workers 100 --lambda_lr .1 --tolerance_epsilon .1\
		  --n_p_steps 5 --use_gradient_clip
