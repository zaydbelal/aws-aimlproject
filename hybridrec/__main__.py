import os

# ALS solves thousands of small (factors x factors) systems; multi-threaded
# BLAS only adds overhead there, so pin it to one thread (before numpy loads).
for _var in ("OPENBLAS_NUM_THREADS", "OMP_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

from .cli import main  # noqa: E402

main()
