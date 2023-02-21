for i in 1 2 3 4 5
do

	python run_FL.py --dataset cifar100 --homo_ratio 0.1 --n_workers_per_round 100\
		          --client_step_per_epoch 10 --use_ray\
			              --learner fed-pd --local_lr 1e-1\
				                    --n_global_rounds 100 --loss_fn cross-entropy-loss --model convnet\
						                    --n_workers 100 --eval_freq 2 --eta 100 --local_epoch 10 --load_model

	python run_FL.py --dataset cifar100 --homo_ratio 0.1 --n_workers_per_round 100\
		          --client_step_per_epoch 10 --use_ray\
			              --learner fed-pd --local_lr 1e-1\
				                    --n_global_rounds 100 --loss_fn cross-entropy-loss --model convnet\
						                    --n_workers 100 --eval_freq 2 --eta 100 --local_epoch 10

							    done


