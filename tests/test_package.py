import os
import subprocess
import sys


def test_import_caps_blas_threads():
    env = {k: v for k, v in os.environ.items()
           if k not in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")}
    code = ("import os, annofix; "
            "print(*(os.environ[v] for v in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', "
            "'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS')))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, check=True)
    assert out.stdout.split() == ["10", "10", "10", "10"]
