# ====================> NAIVE
python -m experiments.train_cir --multirun \
 strategy=naive strategy_params.train_mb_size=128 \
 strategy_params.train_epochs=5 lr=0.1 \
 dataset=mnist model=simple_mlp \
 generator=slot_based \
 N=5 \
 K=2,3,5,8,10 

# ====================> ER-RS
python -m experiments.train_cir --multirun \
  strategy=er_rs strategy_params.train_mb_size=128 \
   strategy_params.train_epochs=5 lr=0.1 \
  dataset=mnist model=simple_mlp \
  generator=slot_based \
  N=5 \
  K=2,3,5,8,10 \
  buffer_size=200 

# ====================> LWF
python -m experiments.train_cir --multirun \
 strategy=lwf strategy_params.train_mb_size=128 \
 strategy_params.train_epochs=5 lr=0.1 \
 dataset=mnist model=simple_mlp \
 generator=slot_based \
 N=5 \
 K=2,3,5,8,10 \
 strategy_params.temperature=1,2 \
 strategy_params.alpha=1,2 

# ====================> EWC
python -m experiments.train_cir --multirun \
 strategy=ewc strategy_params.train_mb_size=128 \
 strategy_params.train_epochs=5 lr=0.1 \
 dataset=mnist model=simple_mlp \
 generator=slot_based \
 strategy_params.ewc_lambda=0.1,10 \
 N=5 \
 K=2,3,5,8,10 

# ====================> AGEM
python -m experiments.train_cir --multirun \
 strategy=agem strategy_params.train_mb_size=128 \
 strategy_params.train_epochs=5 lr=0.1 \
 dataset=mnist model=simple_mlp \
 generator=slot_based \
 N=5 \
 K=2,3,5,8,10
