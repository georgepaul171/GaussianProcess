"""Scale diagnostics: pivoted-Cholesky rank needed, analytic effective dimension, analytic far-field row sums."""
import math
import numpy as np
from . import config as C
from .kernels import d_eff_continuum, tail_integral
from .methods import pivoted_cholesky


class _Win:
    pass


def rank_needed(ctx, eta=0.1, target=400, kmax=3000, nmax=20000):
    """Smallest k with max_i (K - L_k L_k^T)_ii <= eta * s2 (greedy pivoted Cholesky). Measured on a sub-window of
    the design whose continuum effective dimension is ~target (and <= nmax points), then scaled by the point-count
    ratio n / n_window (exact when the window is the whole domain)."""
    vol = ctx.L ** ctx.d
    deff_full = d_eff_continuum(ctx.kernel, ctx.ell, ctx.d, ctx.rho, vol, ctx.s2)
    frac = min(1.0, target / max(deff_full, 1e-9), nmax / ctx.n)
    side = ctx.L * frac ** (1.0 / ctx.d)
    side = min(ctx.L, max(side, 6 * ctx.ell))
    lo = (ctx.L - side) / 2
    sel = np.all((ctx.X >= lo) & (ctx.X <= lo + side), axis=1)
    w = _Win(); w.X = ctx.X[sel]; w.n = len(w.X); w.kernel, w.ell = ctx.kernel, ctx.ell
    _, k, md = pivoted_cholesky(w, min(kmax, w.n - 1), stop_maxdiag=eta * ctx.s2)
    censored = md > eta * ctx.s2
    return dict(rank_window=k, n_window=w.n, rank_needed=k * ctx.n / w.n, rank_censored=bool(censored),
                rank_extrapolated=bool(w.n < ctx.n), d_eff=deff_full)


def analytic(ctx):
    out = {}
    for c in C.CS:
        r = c * ctx.ell
        out[f"nnz_pred_c{c}"] = ctx.nnz_row(r)
        out[f"farsum_expected_c{c}"] = ctx.rho * tail_integral(ctx.kernel, ctx.ell, r, ctx.d)
    return out
