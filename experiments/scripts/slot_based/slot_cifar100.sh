# ====================> NAIVE
python -m experiments.train_cir --multirun \
 strategy=naive \
 strategy_params.train_epochs=30 lr=0.03 \
 dataset=cifar-100 \
 generator=slot_based \
 N=10 \
 K=10,30,50,80,100 

# ====================> ER-RS
python -m experiments.train_cir --multirun \
  strategy=er_rs \
  strategy_params.train_epochs=30 lr=0.03 \
  dataset=cifar-100 \
  generator=slot_based \
  N=10 \
  K=10,30,50,80,100 \
  buffer_size=2000

# ====================> LWF
python -m experiments.train_cir --multirun \
 strategy=lwf \
 strategy_params.train_epochs=30 lr=0.03 \
 dataset=cifar-100 \
 generator=slot_based \
 N=10 \
 K=10,30,50,80,100 \
 strategy_params.temperature=1,2 \
 strategy_params.alpha=1,2

# ====================> EWC
python -m experiments.train_cir --multirun \
 strategy=ewc \
 strategy_params.train_epochs=30 lr=0.03 \
 dataset=cifar-100 \
 generator=slot_based \
 strategy_params.ewc_lambda=0.1,10 \
 N=10 \
 K=10,30,50,80,100

# ====================> AGEM
python -m experiments.train_cir --multirun \
 strategy=agem \
 strategy_params.train_epochs=30 lr=0.03 \
 dataset=cifar-100 \
 generator=slot_based \
 N=10 \
 K=10,30,50,80,100 
