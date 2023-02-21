########################################################################################
# massive, alpha=.1, IR=10
## CLIMB
python run_PD_FL.py --dataset mnist --homo_ratio 0.1 --n_workers_per_round 100\
	  --reduce_to_ratio .1 --use_ray --imbalance\
	    --formulation imbalance-fl --learner fed-avg --local_lr 5e-2\
	      --n_pd_rounds 1000 --loss_fn cross-entropy-loss --model mlp\
	        --n_workers 100 --n_p_steps 5 --dense_hid_dims 128-128 --no_data_augmentation\
		  --lambda_lr 2 --tolerance_epsilon .01 --use_gradient_clip

## Baseline
python run_FL.py --dataset mnist --homo_ratio 0.1 --n_workers_per_round 100\
	  --reduce_to_ratio .1 --use_ray --imbalance\
	    --learner fed-avg --local_lr 5e-2\
	      --n_global_rounds 5000 --loss_fn cross-entropy-loss --model mlp\
	        --n_workers 100 --eval_freq 5 --dense_hid_dims 128-128 --no_data_augmentation\
		  --use_gradient_clip

## Ratio-loss
python run_PD_FL.py --dataset mnist --homo_ratio 0.1 --n_workers_per_round 100\
	  --reduce_to_ratio .1 --use_ray --imbalance\
	    --formulation ratioloss-fl --learner fed-avg --local_lr 5e-2\
	      --n_pd_rounds 200 --loss_fn cross-entropy-loss --model mlp\
	        --n_workers 500 --n_p_steps 5 --dense_hid_dims 128-128 --no_data_augmentation\
		  --use_gradient_clip

## Focal-loss
python run_FL.py --dataset mnist --homo_ratio 0.1 --n_workers_per_round 100\
	  --reduce_to_ratio .1 --use_ray --imbalance\
	    --learner fed-avg --local_lr 5e-2\
	      --n_global_rounds 1000 --loss_fn focal-loss --model mlp\
	        --n_workers 500 --eval_freq 5 --dense_hid_dims 128-128 --no_data_augmentation\
		  --use_gradient_clip
########################################################################################
