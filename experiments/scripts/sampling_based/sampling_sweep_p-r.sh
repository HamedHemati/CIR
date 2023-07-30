# ====================> NAIVE
python -m experiments.train_cir --multirun  \
      strategy=er_rs buffer_size=2000 strategy_params.train_epochs=40 \
      dataset="cifar-100" generator=sampling_based \
      dist_first_occurrence.dist_type="geometric" +dist_first_occurrence.p=0.01\
      seed=0,1,3 dist_recurrence.dist_type="fixed" +dist_recurrence.p=0.1,0.2,0.4,0.6,0.8,1.0 \
      n_e=500 s_e=500 p_a=0.0 \
      num_workers=3 wandb_proj="CIR-Sampling" save_results=True 

# ====================> ER-RS
python -m experiments.train_cir --multirun  \
      strategy=er_rs buffer_size=2000 strategy_params.train_epochs=40 \
      dataset="cifar-100" generator=sampling_based \
      dist_first_occurrence.dist_type="geometric" +dist_first_occurrence.p=0.01\
      seed=0,1,3 dist_recurrence.dist_type="fixed" +dist_recurrence.p=0.1,0.2,0.4,0.6,0.8,1.0 \
      n_e=500 s_e=500 p_a=0.0 \
      num_workers=3 wandb_proj="CIR-Sampling" save_results=True
