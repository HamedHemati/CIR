# ====================> NAIVE
python -m experiments.train_cir --multirun \
 strategy=naive \
 strategy_params.train_epochs=30 lr=0.03 \
 dataset=tinyimagenet \
 generator=slot_based \
 N=10 \
 K=20,60,100,160,200

# ====================> ER-RS
python -m experiments.train_cir --multirun \
  strategy=er_rs \
  strategy_params.train_epochs=30 lr=0.03 \
  dataset=tinyimagenet \
  generator=slot_based \
  N=10 \
  K=20,60,100,160,200 \
  buffer_size=4000

# ====================> LWF
python -m experiments.train_cir --multirun \
 strategy=lwf \
 strategy_params.train_epochs=30 lr=0.03 \
 dataset=tinyimagenet \
 generator=slot_based \
 N=10 \
 K=20,60,100,160,200 \
 strategy_params.temperature=1,2 \
 strategy_params.alpha=1,2

# ====================> EWC
python -m experiments.train_cir --multirun \
 strategy=ewc \
 strategy_params.train_epochs=30 lr=0.03 \
 dataset=tinyimagenet \
 generator=slot_based \
 strategy_params.ewc_lambda=0.1,10 \
 N=10 \
 K=20,60,100,160,200 

# ====================> AGEM
python -m experiments.train_cir --multirun \
 strategy=agem \
 strategy_params.train_epochs=30 lr=0.03 \
 dataset=tinyimagenet \
 generator=slot_based \
 N=10 \
 K=20,60,100,160,200
