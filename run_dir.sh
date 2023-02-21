## Ratio-loss, juan4
python run_PD_FL.py --dataset cifar10 --n_workers_per_round 100\
	  --reduce_to_ratio .1 --use_ray --imbalance\
	    --formulation ratioloss-fl --learner fed-avg --local_lr 1e-1\
	      --n_pd_rounds 1000 --loss_fn cross-entropy-loss --model convnet\
	        --n_workers 100 \
		  --n_p_steps 5 --use_gradient_clip --n_minority 1\
		    --heterogeneity dir --dir_level 2

