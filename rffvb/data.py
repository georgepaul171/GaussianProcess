"""Designs, truths and random Fourier features. Seeds depend only on (d, truth, rep)."""
import numpy as np
from scipy.linalg import cholesky

from . import config as C

TRUTH_ID = {"sparse": 0, "matern52": 1}


def rng_for(d, truth, rep, stream):
    return np.random.default_rng(np.random.SeedSequence([d, TRUTH_ID[truth], rep, stream]))


def matern52(A, B, ell):
    r = np.sqrt(np.maximum(((A[:, None, :] - B[None, :, :]) ** 2).sum(-1), 0.0)) / ell
    s5 = np.sqrt(5.0) * r
    return (1 + s5 + 5.0 * r * r / 3.0) * np.exp(-s5)


def make_dataset(d, truth, rep):
    rng = rng_for(d, truth, rep, 0)
    X = rng.uniform(0, 1, (C.N, d))
    Xt = rng.uniform(0, 1, (C.N_TEST, d))
    if truth == "sparse":
        K = int(rng.integers(C.SPARSE_K[0], C.SPARSE_K[1] + 1))
        dirs = rng.standard_normal((K, d)); dirs /= np.linalg.norm(dirs, axis=1, keepdims=True)
        om = dirs * (2 * np.pi * rng.uniform(*C.SPARSE_CYCLES, K))[:, None]
        ph = rng.uniform(0, 2 * np.pi, K); amp = rng.uniform(*C.SPARSE_AMP, K)
        f = lambda Z: (amp * np.cos(Z @ om.T + ph)).sum(1)
        f_tr, f_te = f(X), f(Xt)
        info = dict(K=K, freqs=np.linalg.norm(om, axis=1).tolist())
    else:
        Z = np.vstack([X, Xt])
        Kz = matern52(Z, Z, C.MATERN_ELL[d]) + 1e-8 * np.eye(len(Z))
        g = cholesky(Kz, lower=True) @ rng.standard_normal(len(Z))
        f_tr, f_te = g[:C.N], g[C.N:]
        info = {}
    y = f_tr + C.SIGMA * rng.standard_normal(C.N)
    return dict(X=X, y=y, Xt=Xt, f_tr=f_tr, f_te=f_te, info=info)


def rff_frequencies(d, truth, rep, ell, p_max):
    """p_max/2 frequencies from the SE spectral density N(0, ell^-2 I), folded into the half-space w_1 >= 0;
    smaller p use the first p/2 (nested). Folding is exact for cos/sin pairs (sum cos(w(x-x')) and the span of each
    pair are invariant to the sign of w) and removes the exactly collinear +-w column pairs, which make the block
    Gram matrices singular (svb.block._block_update segfaults on them)."""
    rng = rng_for(d, truth, rep, 1)
    W = rng.standard_normal((p_max // 2, d)) / ell
    return W * np.where(W[:, :1] < 0, -1.0, 1.0)


def rff_frequencies_lattice(d, truth, rep, ell, p):
    """Separated design: a randomly shifted lattice in [0,1)^d mapped through the SE spectral density (inverse
    normal CDF), restricted to the half-space w_1 >= 0. d = 1: evenly spaced quantiles u_i = (i + U)/P of the
    half-normal; d = 2: golden-ratio (Fibonacci-type) rank-1 lattice (i/P + U1, frac(i g + U2)). Not nested in p.
    A minimum spacing of order 2 pi / (domain width) is infeasible under the SE spectral measure at these p (the
    whole half-line holds ~3 / ell of spectral mass), so this spreads frequencies as evenly as the density allows."""
    from scipy.stats import norm
    P = p // 2
    rng = np.random.default_rng(np.random.SeedSequence([d, TRUTH_ID[truth], rep, 2, p]))
    i = np.arange(P)
    if d == 1:
        U = ((i + rng.random()) / P)[:, None]
    else:
        g = (np.sqrt(5) - 1) / 2
        U = np.stack([(i + rng.random()) / P, np.mod(i * g + rng.random(), 1.0)], 1)
    U = np.clip(U, 1e-12, 1 - 1e-12)
    W = np.empty_like(U)
    W[:, 0] = norm.ppf(0.5 + 0.5 * U[:, 0])               # half-normal: w_1 >= 0
    if d == 2:
        W[:, 1] = norm.ppf(U[:, 1])
    return W / ell


def design_stats(Phi, W):
    """Max |Pearson correlation| between distinct feature columns and the minimum distance between frequencies."""
    Z = Phi - Phi.mean(0)
    Z /= np.maximum(np.linalg.norm(Z, axis=0), 1e-300)
    Cm = np.abs(Z.T @ Z); np.fill_diagonal(Cm, 0.0)
    from scipy.spatial import cKDTree
    dmin = cKDTree(W).query(W, k=2)[0][:, 1].min()
    cmax = Cm.max(1)
    return dict(max_abs_corr=float(cmax.max()), median_col_max_corr=float(np.median(cmax)),
                frac_cols_corr_gt_0p99=float(np.mean(cmax > 0.99)), min_freq_gap=float(dmin))


def rff(X, W):
    """Unit-amplitude cos/sin pairs: columns [cos(X w_1), sin(X w_1), cos(X w_2), ...] (pairs adjacent)."""
    Z = X @ W.T
    out = np.empty((X.shape[0], 2 * W.shape[0]))
    out[:, 0::2] = np.cos(Z); out[:, 1::2] = np.sin(Z)
    return out
