# ====================> NAIVE
python -m experiments.train_cir --multirun \
 strategy=naive \
 dataset=tinyimagenet \
 generator=slot_based \
 N=10 \
 K=20,60,100,160,200

# ====================> ER-RS
python -m experiments.train_cir --multirun \
  strategy=er_rs \
  dataset=tinyimagenet \
  generator=slot_based \
  N=10 \
  K=20,60,100,160,200 \
  memory_size=4000

# ====================> LWF
python -m experiments.train_cir --multirun \
 strategy=lwf \
 dataset=tinyimagenet \
 generator=slot_based \
 N=10 \
 K=20,60,100,160,200 \
 temperature=1,2 \
 alpha=1,2

# ====================> EWC
python -m experiments.train_cir --multirun \
 strategy=ewc \
 dataset=tinyimagenet \
 generator=slot_based \
 N=10 \
 ewc_lambda=0.1,10 \
 K=20,60,100,160,200

# ====================> AGEM
python -m experiments.train_cir --multirun \
 strategy=agem \
 dataset=tinyimagenet \
 generator=slot_based \
 N=10 \
 K=20,60,100,160,200
