import os

# Machine rule: cap BLAS/OpenMP pools before any test module imports numpy.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "10")

# Qt tests run without a display.
os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
