# ====================> NAIVE
python -m experiments.train_cir --multirun \
 strategy=naive \
 dataset=mnist \
 args.generator=slot_based \
 N=5 \
 K=2,3,5,8,10 \

# ====================> ER-RS
python -m experiments.train_cir --multirun \
  strategy=er_rs \
  dataset=mnist \
  args.generator=slot_based \
  N=5 \
  K=2,3,5,8,10 \
  memory_size=200

# ====================> LWF
python -m experiments.train_cir --multirun \
 strategy=lwf \
 dataset=mnist \
 args.generator=slot_based \
 N=5 \
 K=2,3,5,8,10 \
 temperature=1,2 \
 alpha=1,2 \

# ====================> EWC
python -m experiments.train_cir --multirun \
 strategy=ewc \
 dataset=mnist \
 args.generator=slot_based \
 N=5 \
 ewc_lambda=0.1,10 \
 K=2,3,5,8,10 \

# ====================> AGEM
python -m experiments.train_cir --multirun \
 strategy=agem \
 dataset=mnist \
 args.generator=slot_based \
 N=5 \
 K=2,3,5,8,10 \
