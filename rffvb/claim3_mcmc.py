"""Claim 3 (2027-research results/claims.md) against the exact posterior of the SAME model.

Subset: p = 500, d in {1, 2}, both truths, both frequency designs (iid, lattice), reps 0-2. Model: the individual
spike-and-Laplace-slab prior on the sigma-scaled RFF design (unit noise), a0 = 1, b0 = p, lam = 1 -- exactly what
run_cavi fits. Exact posterior: svb.mcmc (read-only), E1f conventions: 4 chains started at the empty model, the
VB selection, the ridge top-s model and a random size-s model; 5000 burn-in, 10000 kept, thin 2. A replicate is
kept iff split R-hat of the model size <= 1.1 and the largest between-chain PIP difference <= 0.1.
Fixed-support exact posterior on the VB-selected set S: svb.mcmc.run_fixed_support.

Group VB is not covered: its multivariate-Laplace group prior is a different model and svb.mcmc does not sample it.

Per (d, truth, design, rep), at coefficient level (j in S) and at function level (f(x) = phi(x)' theta, 2000
test points), with q = VB, E = exact spike-and-slab, F = exact fixed-support (support S), band = VB 95% band:
  cov_obs      = P_E(target in band)                         (observed coverage vs the exact posterior)
  cov_fixed_c  = P_F(target in band re-centred at the F mean)   -> fixed-support (VIF) part = 0.95 - cov_fixed_c
  cov_full_c   = P_E(target in band re-centred at the E mean)   -> inclusion part = cov_fixed_c - cov_full_c
                                                                -> mean-offset part = cov_full_c - cov_obs
  pred_vif     = 2 Phi(1.96 / sqrt(VIF)) - 1 (coefficients) / with R(x) (functions)
  ratio_fixed  = Var_F / Var_q per selected coefficient, compared with VIF (Claim 3: ratio_fixed ~ VIF)
Cancellation test on S: signs of off-diagonal exact (F) posterior correlations, overall and for column pairs with
|corr(X_i, X_j)| > 0.9; and Var_F(f(x)) / sum_j phi_j(x)^2 Var_F(theta_j) (< 1 means cross-covariances cancel).
Model mismatch: exact spike-and-slab vs exact GP (coverage of GP function values by the E band, mean difference).
"""
import json
import math
import os
import time

import numpy as np
from scipy.stats import norm

from . import config as C
from .baselines import exact_gp
from .data import make_dataset, rff, rff_frequencies, rff_frequencies_lattice
from .svbdep import run_cavi, svb_commit  # noqa: F401  (svbdep puts 2027-research on sys.path)
from svb.core import ridge_init                    # noqa: E402
from svb.mcmc import run_chain, run_fixed_support, split_rhat   # noqa: E402

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUT = os.path.join(ROOT, "results", "claim3_mcmc")
P = 500
REPS = 3
MC = dict(n_burn=5000, n_keep=10000, thin=2)
N_FDRAWS = 4000          # posterior draws used for function-level quantities


def _cov_in(draws, lo, hi):
    """Mean over columns of the fraction of draws inside [lo, hi] (per column)."""
    return float(np.mean(((draws >= lo) & (draws <= hi)).mean(0)))


def _vb_draws(rng, r, nd):
    z = rng.random((nd, len(r.mu))) < r.gamma
    return z * (r.mu + r.sigma * rng.standard_normal((nd, len(r.mu))))


def run_one(d, truth, design, rep):
    os.makedirs(OUT, exist_ok=True)
    fn = os.path.join(OUT, f"d{d}_{truth}_{design}_r{rep}.json")
    if os.path.exists(fn):
        return
    t0 = time.perf_counter()
    D = make_dataset(d, truth, rep)
    X, y, Xt = D["X"], D["y"], D["Xt"]
    m_gp, s_gp, gi = exact_gp(X, y, Xt, seed=rep)
    ell = gi["ell"]
    W = rff_frequencies(d, truth, rep, ell, max(C.PS))[:P // 2] if design == "iid" else \
        rff_frequencies_lattice(d, truth, rep, ell, P)
    Phi, Phit = rff(X, W), rff(Xt, W)
    Xs, Ys = Phi / C.SIGMA, y / C.SIGMA
    rng = np.random.default_rng(np.random.SeedSequence([d, rep, 7, 0 if design == "iid" else 1]))

    # ---------------- VB (same call as the grid)
    r = run_cavi(Xs, Ys, order="prioritized", init="ridge", lam=C.VB_LAM, a0=C.VB_A0, b0=None,
                 eps=C.VB_EPS, max_sweeps=C.VB_MAX_SWEEPS, record_elbo=False)
    S = np.flatnonzero(r.gamma > 0.5)
    s = len(S)
    th_q = _vb_draws(rng, r, C.N_DRAWS)
    lo_c, hi_c = np.quantile(th_q, 0.025, 0), np.quantile(th_q, 0.975, 0)
    fq = th_q @ Phit.T
    lo_f, hi_f = np.quantile(fq, 0.025, 0), np.quantile(fq, 0.975, 0)
    mq_c, mq_f = r.gamma * r.mu, Phit @ (r.gamma * r.mu)

    # ---------------- exact spike-and-slab posterior by Gibbs (4 chains, E1f conventions)
    ridge = ridge_init(Xs, Ys)
    init_vb = np.zeros(P); init_vb[S] = r.mu[S]
    top = np.argsort(-np.abs(ridge))[:max(s, 1)]
    init_top = np.zeros(P); init_top[top] = ridge[top]
    rnd = rng.choice(P, max(s, 1), replace=False)
    init_rnd = np.zeros(P); init_rnd[rnd] = ridge[rnd]
    chains = []
    for c, init in enumerate([None, init_vb, init_top, init_rnd]):
        chains.append(run_chain(Xs, Ys, lam=C.VB_LAM, a0=C.VB_A0, b0=None, seed=int(rng.integers(2**31)),
                                init=init, **MC))
    rhat = split_rhat(np.array([ch.n_selected for ch in chains]))
    pips = np.array([ch.pip for ch in chains])
    pip_gap = float((pips.max(0) - pips.min(0)).max())
    mixed = bool(rhat <= 1.1 and pip_gap <= 0.1)
    E = np.vstack([ch.samples for ch in chains])
    pipE = pips.mean(0)
    row = dict(d=d, truth=truth, design=design, rep=rep, p=P, svb_commit=svb_commit(), vb_selected=int(s),
               vb_converged=bool(r.converged), rhat_model_size=float(rhat), max_pip_gap=pip_gap, mixed=mixed,
               exact_model_size_mean=float(np.mean([ch.n_selected.mean() for ch in chains])),
               n_pip_gt_half=int((pipE > 0.5).sum()), mean_abs_gamma_minus_pip=float(np.abs(r.gamma - pipE).mean()))
    if s == 0:
        row["note"] = "VB selected nothing"
    else:
        # ---------------- fixed-support exact posterior on S
        F = run_fixed_support(Xs[:, S], Ys, lam=C.VB_LAM, n_burn=2000, n_keep=10000, thin=1,
                              seed=int(rng.integers(2**31)))
        GS = Xs[:, S].T @ Xs[:, S]
        Ginv = np.linalg.pinv(GS)
        vif = np.diag(Ginv) * np.diag(GS)
        var_q = r.gamma[S] * (r.mu[S] ** 2 + r.sigma[S] ** 2) - (r.gamma[S] * r.mu[S]) ** 2
        var_F = F.var(0)
        ratio_fixed = var_F / var_q
        # ---- coefficient level (j in S)
        lo, hi = lo_c[S], hi_c[S]
        ES = E[:, S]
        cov_obs_c = _cov_in(ES, lo, hi)
        sh_F = F.mean(0) - mq_c[S]; sh_E = ES.mean(0) - mq_c[S]
        cov_fixed_cc = _cov_in(F, lo + sh_F, hi + sh_F)
        cov_full_cc = _cov_in(ES, lo + sh_E, hi + sh_E)
        pred_c = 2 * norm.cdf(1.959964 / np.sqrt(np.maximum(vif, 1.0))) - 1
        # ---- function level
        idx = rng.choice(len(E), min(N_FDRAWS, len(E)), replace=False)
        fE = E[idx] @ Phit.T
        fF = F[rng.choice(len(F), min(N_FDRAWS, len(F)), replace=False)] @ Phit[:, S].T
        mE_f, mF_f = fE.mean(0), fF.mean(0)
        cov_obs_f = _cov_in(fE, lo_f, hi_f)
        cov_fixed_cf = _cov_in(fF, lo_f + (mF_f - mq_f), hi_f + (mF_f - mq_f))
        cov_full_cf = _cov_in(fE, lo_f + (mE_f - mq_f), hi_f + (mE_f - mq_f))
        PhS = Phit[:, S] / C.SIGMA
        v_fix = np.einsum("ij,jk,ik->i", PhS, Ginv, PhS); v_mf = (PhS ** 2 / np.diag(GS)).sum(1)
        R = np.maximum(v_fix / v_mf, 1e-300)
        pred_f = 2 * norm.cdf(1.959964 / np.sqrt(R)) - 1
        # ---- cancellation test (exact fixed-support posterior on S)
        if s >= 2:
            CF = np.corrcoef(F.T); CX = np.corrcoef(Xs[:, S].T)
            off = ~np.eye(s, dtype=bool)
            hi_corr = off & (np.abs(CX) > 0.9)
            # product sign: for positively correlated columns cancellation needs a negative posterior correlation;
            # in general cancellation in f needs sign(corr_post) = -sign(corr_X)
            opp = np.sign(CF) == -np.sign(CX)
            canc = dict(frac_post_corr_negative=float((CF[off] < 0).mean()),
                        frac_opposite_sign_all=float(opp[off].mean()),
                        n_pairs_xcorr_gt_0p9=int(hi_corr.sum() // 2),
                        frac_opposite_sign_xcorr_gt_0p9=float(opp[hi_corr].mean()) if hi_corr.any() else np.nan,
                        median_post_corr_xcorr_gt_0p9=float(np.median((CF * np.sign(CX))[hi_corr])) if hi_corr.any() else np.nan)
        else:
            canc = {}
        Phit_S = Phit[:, S]
        var_f_F = fF.var(0)
        var_f_diag = (Phit_S ** 2 * var_F).sum(1)
        row.update(
            vif_median=float(np.median(vif)), vif_max=float(vif.max()),
            ratio_fixed_over_vif_median=float(np.median(ratio_fixed / vif)),
            spearman_ratio_vif=float(__import__("scipy.stats").stats.spearmanr(ratio_fixed, vif)[0]) if s > 2 else np.nan,
            ratio_full_over_q_median=float(np.median(ES.var(0) / var_q)),
            coef_cov_obs=cov_obs_c, coef_cov_fixed_centred=cov_fixed_cc, coef_cov_full_centred=cov_full_cc,
            coef_pred_vif=float(pred_c.mean()),
            fun_cov_obs=cov_obs_f, fun_cov_fixed_centred=cov_fixed_cf, fun_cov_full_centred=cov_full_cf,
            fun_pred_vif=float(pred_f.mean()), fun_R_median=float(np.median(R)),
            fun_var_ratio_exact_fixed_over_diag=float(np.median(var_f_F / var_f_diag)),
            # model mismatch: exact spike-and-slab vs exact GP
            mm_cov_gp_by_exact_ss=float(np.mean(norm.cdf((np.quantile(fE, 0.975, 0) - m_gp) / s_gp)
                                                - norm.cdf((np.quantile(fE, 0.025, 0) - m_gp) / s_gp))),
            mm_cov_f0_exact_ss=float(np.mean((D["f_te"] >= np.quantile(fE, 0.025, 0)) & (D["f_te"] <= np.quantile(fE, 0.975, 0)))),
            mm_cov_f0_gp=float(np.mean(np.abs(D["f_te"] - m_gp) <= 1.959964 * s_gp)),
            mm_rmse_exact_ss_vs_gp_mean=float(np.sqrt(np.mean((mE_f - m_gp) ** 2))),
            mm_rmse_f0_exact_ss=float(np.sqrt(np.mean((mE_f - D["f_te"]) ** 2))),
            mm_rmse_f0_gp=float(np.sqrt(np.mean((m_gp - D["f_te"]) ** 2))),
            **canc)
    row["runtime"] = time.perf_counter() - t0
    with open(fn, "w") as fh:
        json.dump(row, fh, default=float)
    print(fn, "mixed" if mixed else "NOT MIXED", f"rhat={rhat:.3f} pipgap={pip_gap:.3f} t={row['runtime']:.0f}s",
          flush=True)


def jobs():
    return [(d, t, des, r) for r in range(REPS) for d in C.DS for t in C.TRUTHS for des in ("iid", "lattice")]
