for minority in 3 5 1
do
    for d in 0.3 10.0
    do
        for formulation in "imbalance-fl-res" "imbalance-fl"
        do
            CUDA_VISIBLE_DEVICES=1 python run_PD_FL.py --project FedResFinal --imbalance --run fmnist_dir_${d}_min_${minority} --dir_level ${d} --perturbation_lr 0.1 --perturbation_penalty 2.0 --dataset fashion-mnist --n_workers_per_round 100 --reduce_to_ratio .1 --use_ray --formulation $formulation --learner fed-avg --local_lr 5e-2 --n_pd_rounds 1000 --loss_fn cross-entropy-loss --model convnet --dense_hid_dims 384-192 --conv_hid_dims 64-64 --n_workers 100 --n_p_steps 5 --lambda_lr 2 --tolerance_epsilon .01 --use_gradient_clip --n_minority $minority --heterogeneity dir
        done
    done
done