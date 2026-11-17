"""Study grid. Everything that defines the experiment lives here."""
import os

N = 1000
N_TEST = 2000
SIGMA = 0.5
DS = [1, 2]
TRUTHS = ["sparse", "matern52"]
PS = [500, 2000, 5000]          # number of random features (cos/sin pairs -> P/2 frequencies)
REPS = 10
N_DRAWS = 2000                  # draws from variational posteriors for predictive quantiles
MATERN_ELL = {1: 0.2, 2: 0.3}   # lengthscale of the Matern-5/2 truth on [0, 1]^d
SPARSE_K = (3, 5)               # number of sinusoids, inclusive range
SPARSE_CYCLES = (1.0, 8.0)      # |omega| / (2 pi) range of the sinusoids on [0, 1]^d
SPARSE_AMP = (0.5, 1.5)

# spike-and-slab VB hyperparameters: the 2027-research defaults (DEVIATIONS.md): a0 = 1, b0 = p, lam = 1,
# ridge init, prioritized order, entropy stopping rule eps = 1e-5, at most 500 sweeps.
VB_LAM, VB_A0 = 1.0, 1.0
VB_EPS, VB_MAX_SWEEPS = 1e-5, 500

SVB_PATH = os.environ.get("SVB_PATH", "/home/user/2027-research")   # read-only dependency
