import os

os.environ["OMP_NUM_THREADS"] = "8"  # export OMP_NUM_THREADS=4
os.environ["OPENBLAS_NUM_THREADS"] = "8"  # export OPENBLAS_NUM_THREADS=4
os.environ["MKL_NUM_THREADS"] = "8"  # export MKL_NUM_THREADS=6
os.environ["VECLIB_MAXIMUM_THREADS"] = "8"  # export VECLIB_MAXIMUM_THREADS=4
os.environ["NUMEXPR_NUM_THREADS"] = "8"  # export NUMEXPR_NUM_THREADS=6

print("NUMBER OF THREADS IS LIMITED NOW ...")
