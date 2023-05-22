DEVICE=0
eps=0.02
########################################################################################
for minority in 3
do
    for d in 0.3
    do
        for penalty in 2 4 10 40 100 1000
        do
            for formulation in "imbalance-fl-res"
            do
                CUDA_VISIBLE_DEVICES=$DEVICE python run_PD_FL.py --perturbation_penalty $penalty --project FedResFinal --imbalance --dataset cifar10 --n_workers_per_round 100 --reduce_to_ratio .1 --use_ray --formulation $formulation --learner fed-avg --local_lr 1e-1 --n_pd_rounds 1000 --loss_fn cross-entropy-loss --model convnet --n_workers 100 --n_p_steps 5 --lambda_lr 2 --tolerance_epsilon ${eps} --use_gradient_clip --n_minority 3 --run penalty_abl_${penalty} --heterogeneity dir --dir_level 0.3
            done
        done
    done
done