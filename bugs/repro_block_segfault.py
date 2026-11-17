"""Minimal reproducer: svb.block._block_update segfaults on a numerically singular 4x4 cluster Gram matrix.
Usage: SVB_PATH=/path/to/2027-research python bugs/repro_block_segfault.py   (exit code 139 = segfault)"""
import os, sys
import numpy as np
sys.path.insert(0, os.environ.get("SVB_PATH", "/home/user/2027-research"))
from svb.block import _block_update

z = np.load(os.path.join(os.path.dirname(__file__), "block_segfault_input.npz"))
print("eigenvalues of Gcc:", np.linalg.eigvalsh(z["Gcc"]), flush=True)
out = _block_update(z["Gcc"], z["rc"], float(z["lam"]), float(z["logw"]), float(z["log1mw"]))
print("returned normally:", out[0])
