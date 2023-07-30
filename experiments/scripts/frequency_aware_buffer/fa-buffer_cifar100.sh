# ====================> ER-ACE-CB
python -m experiments.train_cir --multirun  \
      strategy=er_ace buffer_size=2000 strategy_params.train_epochs=20 \
      dataset="cifar-100" generator=sampling_based \
      dist_first_occurrence.dist_type="geometric" +dist_first_occurrence.p=0.2\
      seed=0,1,3 dist_recurrence.dist_type="fixed-dual" +dist_recurrence.p_l=0.1 \
      +dist_recurrence.p_h=1.0  +dist_recurrence.frac=0.1,0.3,0.5 \
      n_e=100 s_e=2000 p_a=0.0 \
      num_workers=3 wandb_proj="FA-Buffer" save_results=True  

# ====================> ER-ACE-Adaptive
python -m experiments.train_cir --multirun  \
      strategy=er_ace_adaptive buffer_size=2000 strategy_params.train_epochs=20 \
      dataset="cifar-100" generator=sampling_based \
      dist_first_occurrence.dist_type="geometric" +dist_first_occurrence.p=0.2\
      seed=0,1,3 dist_recurrence.dist_type="fixed-dual" +dist_recurrence.p_l=0.1 \
      +dist_recurrence.p_h=1.0  +dist_recurrence.frac=0.1,0.3,0.5 \
      n_e=100 s_e=2000 p_a=0.0 \
      num_workers=3 wandb_proj="FA-Buffer" save_results=True  

# ====================> ER-ACE-RS
python -m experiments.train_cir --multirun  \
      strategy=er_ace_rs buffer_size=2000 strategy_params.train_epochs=20 \
      dataset="cifar-100" generator=sampling_based \
      dist_first_occurrence.dist_type="geometric" +dist_first_occurrence.p=0.2 \
      seed=0,1,3 dist_recurrence.dist_type="fixed-dual" +dist_recurrence.p_l=0.1 \
      +dist_recurrence.p_h=1.0  +dist_recurrence.frac=0.1,0.3,0.5 \
      n_e=100 s_e=2000 p_a=0.0 \
      num_workers=3 wandb_proj="FA-Buffer" save_results=True  

# ====================> ER-DER-CB
python -m experiments.train_cir --multirun  \
      strategy=er_der buffer_size=2000 strategy_params.train_epochs=20 \
      dataset="cifar-100" generator=sampling_based \
      dist_first_occurrence.dist_type="geometric" +dist_first_occurrence.p=0.2 \
      seed=0,1,3 dist_recurrence.dist_type="fixed-dual" +dist_recurrence.p_l=0.1 \
      +dist_recurrence.p_h=1.0  +dist_recurrence.frac=0.1,0.3,0.5 \
      n_e=100 s_e=2000 p_a=0.0 \
      num_workers=3 wandb_proj="FA-Buffer" save_results=True  

# ====================> ER-DER-Adaptive
python -m experiments.train_cir --multirun  \
      strategy=er_der_adaptive buffer_size=2000 strategy_params.train_epochs=20 \
      dataset="cifar-100" generator=sampling_based \
      dist_first_occurrence.dist_type="geometric" +dist_first_occurrence.p=0.2 \
      seed=0,1,3 dist_recurrence.dist_type="fixed-dual" +dist_recurrence.p_l=0.1 \
      +dist_recurrence.p_h=1.0  +dist_recurrence.frac=0.1,0.3,0.5 \
      n_e=100 s_e=2000 p_a=0.0 \
      num_workers=3 wandb_proj="FA-Buffer" save_results=True  

# ====================> ER-DER-RS
python -m experiments.train_cir --multirun  \
      strategy=er_der_rs buffer_size=2000 strategy_params.train_epochs=20 \
      dataset="cifar-100" generator=sampling_based \
      dist_first_occurrence.dist_type="geometric" +dist_first_occurrence.p=0.2 \
      seed=0,1,3 dist_recurrence.dist_type="fixed-dual" +dist_recurrence.p_l=0.1 \
      +dist_recurrence.p_h=1.0  +dist_recurrence.frac=0.1,0.3,0.5 \
      n_e=100 s_e=2000 p_a=0.0 \
      num_workers=3 wandb_proj="FA-Buffer" save_results=True  
