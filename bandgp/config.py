"""Experiment grid and global constants. Everything that defines the study lives here."""
import math

N0 = 10                 # density (points per unit volume) at beta=1; all betas coincide at n=N0
SIGMA = 0.5             # noise sd
NS = [10_000, 30_000, 100_000, 300_000, 1_000_000]
BETAS = [0.0, 0.25, 0.5, 0.75, 1.0]
DS = [1, 2]
KERNELS = ["se", "m32"]
LSMODES = ["fixed", "oracle"]
ALPHAS = [0.75, 1.5]

CS = [1.5, 2.0, 2.5, 3.0, 4.0]          # banding radius = c * lengthscale
PCG_RANKS = [50, 200, 1000]
SGPR_MS = [100, 300, 1000, 3000]
VECCHIA_MS = [10, 20, 40]

N_TEST = 2000
N_VAL = 2000            # separate points, used only for configuration selection
N_SUB = 50              # test subset for CG-based variances / reference variances beyond dense range
EXACT_MAX_N = 10_000    # dense exact GP reference
TRUNC_TOL = 1e-10       # "exact" kernel entries are truncated below this value (matvecs, cross-covariances)
CG_TOL = 1e-6           # relative residual
CG_MAXIT = 2000
CG_VAR_REPS = 3         # CG-based variance blocks (N_SUB columns, time extrapolated) only in reps < 3
TIME_CAP = 600          # seconds (single thread) per method configuration; beyond -> "timeout"

# Feasibility caps (single-threaded, per job). Configurations predicted to exceed them are recorded as skipped.
NNZ_CAP = 1.2e8         # total stored nonzeros for any sparse kernel matrix
PIVCHOL_CAP = 2.5e8     # n * rank doubles for the pivoted-Cholesky factor
SGPR_FLOP_CAP = 1.2e12  # n * M^2
DENSE_MATVEC_CAP = 1.0e9  # n^2 kernel evaluations per matvec for matrix-free dense PCG
EIG_CAP = 6e7           # n * nnz(exact) for the exact lambda_min(E) Lanczos computation

def reps(n):
    return 10 if n <= 100_000 else 3

def domain_L(n, d, beta):
    """Side length: L^d = (n/N0)^beta, so local density rho = N0^beta n^(1-beta)."""
    return (n / N0) ** (beta / d)

def density(n, d, beta):
    return n / domain_L(n, d, beta) ** d

def lengthscale(mode, n, d, beta, alpha):
    if mode == "fixed":
        return 1.0
    # paper's oracle tau_n = n^{-1/(2alpha+d)} on [0,1]; on [0,L]^d with local density rho the
    # same bias-variance balance gives rho^{-1/(2alpha+d)} (equal to tau_n at beta=0)
    return density(n, d, beta) ** (-1.0 / (2 * alpha + d))

def ball_volume(d, r):
    return 2 * r if d == 1 else math.pi * r * r
