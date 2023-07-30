# ====================> NAIVE
python -m experiments.train_cir --multirun \
 strategy=naive \
 dataset=cifar-100 \
 args.generator=slot_based \
 N=10 \
 K=10,30,50,80,100 

# ====================> ER-RS
python -m experiments.train_cir --multirun \
  strategy=er_rs \
  dataset=cifar-100 \
  args.generator=slot_based \
  N=10 \
  K=10,30,50,80,100 \
  memory_size=2000

# ====================> LWF
python -m experiments.train_cir --multirun \
 strategy=lwf \
 dataset=cifar-100 \
 args.generator=slot_based \
 N=10 \
 K=10,30,50,80,100 \
 temperature=1,2 \
 alpha=1,2

# ====================> EWC
python -m experiments.train_cir --multirun \
 strategy=ewc \
 dataset=cifar-100 \
 args.generator=slot_based \
 N=10 \
 ewc_lambda=0.1,10 \
 K=10,30,50,80,100

# ====================> AGEM
python -m experiments.train_cir --multirun \
 strategy=agem \
 dataset=cifar-100 \
 args.generator=slot_based \
 N=10 \
 K=10,30,50,80,100
