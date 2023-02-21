## baseline,
python run_FL.py --dataset cifar10 --n_workers_per_round 100\
	  --reduce_to_ratio .1 --use_ray --imbalance\
	    --learner fed-avg --local_lr 5e-2\
	      --n_global_rounds 5000 --loss_fn cross-entropy-loss --model convnet\
	        --n_workers 100 --eval_freq 5 --n_minority 1 --use_gradient_clip\
		  --heterogeneity dir --dir_level 20

## CLIMB,
python run_PD_FL.py --dataset cifar10 --n_workers_per_round 100\
	  --reduce_to_ratio .1 --use_ray --imbalance\
	    --formulation imbalance-fl --learner fed-avg --local_lr 5e-2\
	      --n_pd_rounds 1000 --loss_fn cross-entropy-loss --model convnet\
	        --n_workers 100 --lambda_lr .1 --tolerance_epsilon .1\
		  --n_p_steps 5 --n_minority 1 --heterogeneity dir --dir_level 20\
		    --use_gradient_clip

## Focal-loss,
python run_FL.py --dataset cifar10 --n_workers_per_round 100\
	  --reduce_to_ratio .1 --use_ray --imbalance\
	    --learner fed-avg --local_lr 5e-2\
	      --n_global_rounds 5000 --loss_fn focal-loss --model convnet\
	        --n_workers 100 --eval_freq 5 --use_gradient_clip --n_minority 1\
		  --heterogeneity dir --dir_level 20

## Ratio-loss,
python run_PD_FL.py --dataset cifar10 --n_workers_per_round 100\
	  --reduce_to_ratio .1 --use_ray --imbalance\
	    --formulation ratioloss-fl --learner fed-avg --local_lr 5e-2\
	      --n_pd_rounds 1000 --loss_fn cross-entropy-loss --model convnet\
	        --n_workers 100 \
		  --n_p_steps 5 --use_gradient_clip --n_minority 1\
		    --heterogeneity dir --dir_level 20
