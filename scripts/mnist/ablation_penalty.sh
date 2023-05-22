DEVICE=0
lru=0.1
########################################################################################
for minority in 3 5
do
    for d in 0.3
    do
        for eps in 0.02
        do
        for penalty in 0.1 1.0 2.0 4.0 10.0 0.5 
            do
                for formulation in "imbalance-fl-res"
                do
                    CUDA_VISIBLE_DEVICES=$DEVICE python run_PD_FL.py --project FedResFinal --formulation ${formulation} --run abl_penalty_${penalty} --reduce_to_ratio 0.01 --perturbation_lr 0.1 --perturbation_penalty $penalty --dataset mnist --imbalance --n_workers_per_round 100 --use_ray --learner fed-avg --local_lr 5e-2 --n_pd_rounds 500 --loss_fn cross-entropy-loss --model mlp --n_workers 100 --n_p_steps 5 --dense_hid_dims 128-128 --no_data_augmentation --lambda_lr .1 --tolerance_epsilon $eps --use_gradient_clip --n_minority $minority --heterogeneity dir --dir_level $d
                done
            done
        done
    done
done
########################################################################################

