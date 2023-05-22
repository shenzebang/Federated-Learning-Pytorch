DEVICE=1
minority=3
########################################################################################
for imbalance in 0.05 0.2 0.01 0.3
do
    for d in 0.3
    do
        for eps in 0.02
        do
            for formulation in "imbalance-fl-res" "imbalance-fl"
            do
                CUDA_VISIBLE_DEVICES=$DEVICE python run_PD_FL.py --perturbation_penalty 1 --project FedResFinal --imbalance --dataset cifar10 --n_workers_per_round 100 --reduce_to_ratio $imbalance --use_ray --formulation $formulation --learner fed-avg --local_lr 1e-1 --n_pd_rounds 1000 --loss_fn cross-entropy-loss --model convnet --n_workers 100 --n_p_steps 5 --lambda_lr 2 --tolerance_epsilon ${eps} --use_gradient_clip --n_minority 3 --run abl_imbalance_$imbalance --heterogeneity dir --dir_level $d
            done
        done
    done
done