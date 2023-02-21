## baseline, juan4,
python run_FL.py --dataset fashion-mnist --n_workers_per_round 100\
	  --reduce_to_ratio .1 --use_ray --imbalance\
	    --learner fed-avg --local_lr 5e-2\
	      --n_global_rounds 5000 --loss_fn cross-entropy-loss --model convnet\
	        --dense_hid_dims 384-192 --conv_hid_dims 64-64\
		  --n_workers 100 --eval_freq 5 --n_minority 1 --use_gradient_clip\
		    --heterogeneity dir --dir_level 10
