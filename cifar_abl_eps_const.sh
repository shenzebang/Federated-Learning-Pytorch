DEVICE=1
########################################################################################
for minority in 3
do
    for d in 0.3
    do
        for eps in 0.01 0.001 0.05 0.1
        do
            for formulation in "imbalance-fl"
            do
                CUDA_VISIBLE_DEVICES=$DEVICE python run_PD_FL.py --perturbation_penalty 1 --project FedResABl --imbalance --dataset cifar10 --n_workers_per_round 100 --reduce_to_ratio .1 --use_ray --formulation $formulation --learner fed-avg --local_lr 1e-1 --n_pd_rounds 1000 --loss_fn cross-entropy-loss --model convnet --n_workers 100 --n_p_steps 5 --lambda_lr 2 --tolerance_epsilon ${eps} --use_gradient_clip --n_minority 3 --run eps_abl_${eps} --heterogeneity dir --dir_level 0.3
            done
        done
    done
done