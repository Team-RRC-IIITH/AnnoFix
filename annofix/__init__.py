"""AnnoFix: fix object/part/material segmentation annotations with SAM 2.1."""
import os

# Machine rule: cap BLAS/OpenMP pools before numpy is imported anywhere. On this 24-core
# laptop the default (all cores) is several times slower and much hotter for our array sizes.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_var, "10")

__version__ = "0.1.0"
