########################################################################################
for alpha in 0.0 0.1 0.2
do
for formulation in "imbalance-fl-res"
    do
        CUDA_VISIBLE_DEVICES=0 python run_PD_FL.py --project FedRes --homo_ratio ${alpha} --formulation ${formulation} --run alpha_${imbalance}_${formulation} --reduce_to_ratio 0.05 --perturbation_lr 0.1 --perturbation_penalty 2.0 --dataset mnist --homo_ratio 0.1 --n_workers_per_round 100 --use_ray --imbalance --learner fed-avg --local_lr 5e-2 --n_pd_rounds 500 --loss_fn cross-entropy-loss --model mlp --n_workers 100 --n_p_steps 5 --dense_hid_dims 128-128 --no_data_augmentation --lambda_lr .1 --tolerance_epsilon .1 --use_gradient_clip --n_minority 3
    done
done
########################################################################################

