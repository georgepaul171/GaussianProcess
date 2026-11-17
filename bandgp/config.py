"""Experiment grid and global constants. Everything that defines the study lives here."""
import math

N0 = 10                 # density (points per unit volume) at beta=1; all betas coincide at n=N0
SIGMA = 0.5             # noise sd
ALPHA = 0.75            # truth smoothness (most favourable to banding under infill)
BETAS = [0.0, 0.5, 1.0]
KERNELS = ["se", "m32"]
NS = {1: [10_000, 100_000, 1_000_000], 2: [10_000, 100_000]}

CS = [1.5, 2.0, 2.5, 3.0, 4.0]          # banding radius = c * lengthscale (full sweep on rep 0 only)
PCG_RANKS = [200, 1000]
PCG_MAXIT = 200
SGPR_MS = [100, 1000, 5000]
VECCHIA_MS = [10, 30]

N_TEST = 2000           # RMSE on all test points
N_VAR = 200             # pointwise variances on the first N_VAR test points (all methods)
N_VAL = 2000            # separate validation points, used only for selecting configurations
EXACT_MAX_N = 10_000
TRUNC_TOL = 1e-10       # "exact" kernel entries below this are dropped (matvecs, cross-covariances)
CG_TOL = 1e-6
def time_cap(n):
    """Seconds (single thread) per method configuration; beyond -> "timeout"."""
    return 300 if n <= 10_000 else 120

# feasibility caps (predicted before running; recorded as "infeasible")
NNZ_CAP = 1.2e8         # stored nonzeros of any sparse/banded matrix
PIVCHOL_CAP = 2.5e8     # n * rank doubles
SGPR_FLOP_CAP = 3e12    # n * M^2
DENSE_MATVEC_CAP = 1.0e9  # n^2 per matrix-free dense matvec


def reps(n):
    return 3 if n <= 10_000 else 2


def domain_L(n, d, beta):
    """Side length with L^d = (n/N0)^beta: local density rho = N0^beta n^(1-beta); beta=1 is fixed density N0."""
    return (n / N0) ** (beta / d)


def density(n, d, beta):
    return n / domain_L(n, d, beta) ** d


def lengthscale(n, d, beta, alpha=ALPHA):
    """beta=0: paper's oracle tau_n = n^{-1/(2 alpha + d)}; beta>0: fixed at 1."""
    return n ** (-1.0 / (2 * alpha + d)) if beta == 0 else 1.0


def ball_volume(d, r):
    return 2 * r if d == 1 else math.pi * r * r
