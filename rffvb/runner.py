"""One job = (d, truth, rep): exact GP, then for each p and each frequency design (iid draws, shifted lattice):
RFF-BLR, spike-and-slab VB (individual, group); and SSGP with as many features as individual VB selected (iid). Writes one JSON line per (method, p) to results/raw/<job>.jsonl."""
import json
import math
import os
import time

import numpy as np

from . import config as C
from .baselines import exact_gp, rff_blr, ssgp
from .data import make_dataset, rff, rff_frequencies, rff_frequencies_lattice, design_stats
from .gsvb import run_gsvb
from .svbdep import run_cavi, svb_commit

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")


def metrics(f_te, mean, lo, hi):
    return dict(rmse=float(np.sqrt(np.mean((mean - f_te) ** 2))),
                coverage=float(np.mean((f_te >= lo) & (f_te <= hi))),
                mean_width=float(np.mean(hi - lo)))


def gauss_band(mu, sd):
    return mu - 1.959964 * sd, mu + 1.959964 * sd


def draw_band(theta_draws, Phit):
    F = theta_draws @ Phit.T
    return np.quantile(F, 0.025, axis=0), np.quantile(F, 0.975, axis=0)


def claim3_metrics(Xs, sel, Phit, lo, hi, mean_vb, m_gp, s_gp):
    """Claim 3 (2027-research results/claims.md) on the VB-selected support S (columns of the sigma-scaled design):
    coefficient VIF_i = [(X_S'X_S)^-1]_ii (X_S'X_S)_ii and its predicted coverage 2 Phi(1.96/sqrt(VIF)) - 1;
    the function-level analogue R(x) = phi_S' G_S^-1 phi_S / sum_j phi_j^2 / G_jj (fixed-support Gaussian over
    mean-field variance of f(x)) and its predicted coverage; observed coverage of exact-GP posterior function values
    P_{f ~ GP posterior}(f(x) in VB band), averaged over test points; and the width-only coverage (VB band
    re-centred at the GP mean)."""
    from scipy.stats import norm
    out = dict(support_size=int(len(sel)))
    pz = lambda r: 2 * norm.cdf(1.959964 / np.sqrt(r)) - 1
    obs = norm.cdf((hi - m_gp) / s_gp) - norm.cdf((lo - m_gp) / s_gp)
    half = 0.5 * (hi - lo)
    out.update(cov_gp_observed=float(obs.mean()), cov_gp_width_only=float((2 * norm.cdf(half / s_gp) - 1).mean()),
               mean_shift_over_gp_sd=float(np.median(np.abs(mean_vb - m_gp) / s_gp)))
    if len(sel) == 0:
        return out
    XS = Xs[:, sel]
    GS = XS.T @ XS
    lam, U = np.linalg.eigh(GS)
    sing = lam.min() <= 1e-12 * lam.max()
    lam_c = np.maximum(lam, 1e-12 * lam.max())          # singular support: VIF effectively infinite (capped)
    Gd = np.diag(GS)
    vif = (U ** 2 / lam_c).sum(1) * Gd
    PhS = Phit[:, sel] / C.SIGMA                       # same scaling as the design (f = Phi theta = (Phi/s)(s theta))
    v_fix = ((PhS @ U) ** 2 / lam_c).sum(1)
    v_mf = (PhS ** 2 / Gd).sum(1)
    R = np.maximum(v_fix / v_mf, 1e-300)
    out.update(support_singular=bool(sing), support_cond=float(lam.max() / max(lam.min(), 1e-300)),
               vif_median=float(np.median(vif)), vif_max=float(vif.max()),
               pred_cov_coef=float(pz(np.maximum(vif, 1.0)).mean()),
               fun_vif_median=float(np.median(R)), pred_cov_fun=float(pz(R).mean()), frac_fun_vif_below_1=float(np.mean(R < 1)))
    return out


def draws_mf(rng, mu, sd, incl, sizes=None):
    """Mean-field spike-and-slab draws; with sizes, inclusion is shared within consecutive groups."""
    nd, p = C.N_DRAWS, len(mu)
    if sizes is None:
        z = rng.random((nd, p)) < incl
    else:
        z = np.repeat(rng.random((nd, len(incl))) < incl, sizes, axis=1)
    return z * (mu + sd * rng.standard_normal((nd, p)))


def run_job(j):
    d, truth, rep = j["d"], j["truth"], j["rep"]
    jid = f"d{d}_{truth}_r{rep}"
    os.makedirs(RAW, exist_ok=True)
    out = os.path.join(RAW, jid + ".jsonl")
    if os.path.exists(out):
        return
    D = make_dataset(d, truth, rep)
    X, y, Xt, f_te = D["X"], D["y"], D["Xt"], D["f_te"]
    base = dict(d=d, truth=truth, rep=rep, n=C.N, sigma=C.SIGMA, svb_commit=svb_commit(), **{
        f"truth_{k}": v for k, v in D["info"].items() if k == "K"})
    rows = []
    mu, sd, info = exact_gp(X, y, Xt, seed=rep)
    m_gp, s_gp = mu, sd
    rows.append(dict(base, design="-", method="exact_gp", p=0, n_selected=np.nan, runtime=info["t"],
                     gp_ell=info["ell"], gp_s2=info["s2"], gp_noise=info["noise"], **metrics(f_te, mu, *gauss_band(mu, sd))))
    ell, s2, nz = info["ell"], info["s2"], info["noise"]
    Wall = rff_frequencies(d, truth, rep, ell, max(C.PS))
    rng = np.random.default_rng(np.random.SeedSequence([d, rep, 99]))
    for p in C.PS:
        W = Wall[:p // 2]
        bp = dict(base, p=p, rff_ell=ell)
        m_, s_, inf = rff_blr(X, y, Xt, W, s2, nz)
        rows.append(dict(bp, design="iid", **design_stats(rff(X, W), W), method="rff_blr", n_selected=p,
                         runtime=inf["t"], **metrics(f_te, m_, *gauss_band(m_, s_))))

        k_ind = None
        for design, Wd in (("iid", W), ("lattice", rff_frequencies_lattice(d, truth, rep, ell, p))):
            Phi, Phit = rff(X, Wd), rff(Xt, Wd)
            ds = design_stats(Phi, Wd)
            bd = dict(bp, design=design, **ds)
            if design == "lattice":
                m_, s_, inf = rff_blr(X, y, Xt, Wd, s2, nz)
                rows.append(dict(bd, method="rff_blr", n_selected=p, runtime=inf["t"],
                                 **metrics(f_te, m_, *gauss_band(m_, s_))))
            t0 = time.perf_counter()
            Phi, Phit = rff(X, Wd), rff(Xt, Wd)
            Xs, Ys = Phi / C.SIGMA, y / C.SIGMA                  # run_cavi assumes unit noise variance
            t_feat = time.perf_counter() - t0
            # individual spike-and-slab VB (RS22 CAVI from 2027-research, its default settings)
            t0 = time.perf_counter()
            r = run_cavi(Xs, Ys, order="prioritized", init="ridge", lam=C.VB_LAM, a0=C.VB_A0, b0=None,
                         eps=C.VB_EPS, max_sweeps=C.VB_MAX_SWEEPS, record_elbo=False)
            mean = Phit @ (r.gamma * r.mu)
            band = draw_band(draws_mf(rng, r.mu, r.sigma, r.gamma), Phit)
            k = int((r.gamma > 0.5).sum())
            if design == "iid":
                k_ind = k
            rt = t_feat + time.perf_counter() - t0
            c3 = claim3_metrics(Xs, np.flatnonzero(r.gamma > 0.5), Phit, *band, mean, m_gp, s_gp)
            rows.append(dict(bd, method="svb_individual", n_selected=k, runtime=rt,
                             n_sweeps=r.n_sweeps, converged=r.converged, **metrics(f_te, mean, *band), **c3))
            # group spike-and-slab VB, each cos/sin pair a group (GSVB-style, multivariate Laplace slab)
            t0 = time.perf_counter()
            sizes = np.full(p // 2, 2)
            g = run_gsvb(Xs, Ys, sizes, lam=C.VB_LAM, a0=C.VB_A0, b0=None, eps=C.VB_EPS, max_sweeps=C.VB_MAX_SWEEPS)
            mean = Phit @ g.post_mean
            band = draw_band(draws_mf(rng, g.mu, g.s, g.gamma, sizes), Phit)
            rt = t_feat + time.perf_counter() - t0
            c3 = claim3_metrics(Xs, np.flatnonzero(np.repeat(g.gamma > 0.5, sizes)), Phit, *band, mean, m_gp, s_gp)
            rows.append(dict(bd, method="svb_group", n_selected=int(2 * (g.gamma > 0.5).sum()), runtime=rt,
                             n_sweeps=g.n_sweeps, converged=g.converged, **metrics(f_te, mean, *band), **c3))

        # SSGP with as many features as individual VB selected (m = ceil(k/2) frequencies, >= 1)
        m_freq = max(1, math.ceil(k_ind / 2))
        try:
            m_, s_, inf = ssgp(X, y, Xt, m_freq, ell, s2, nz, seed=rep)
        except Exception as e:
            rows.append(dict(bp, design="optimised", method="ssgp", n_selected=2 * m_freq,
                             status=f"{type(e).__name__}: {e}"[:200]))
            print(f"{jid} p={p} ssgp failed", flush=True)
            continue
        rows.append(dict(bp, design="optimised", method="ssgp", n_selected=2 * m_freq, runtime=inf["t"], ssgp_success=inf["success"], ssgp_failed_restarts=inf["failed_restarts"],
                         **metrics(f_te, m_, *gauss_band(m_, s_))))
        print(f"{jid} p={p} done", flush=True)
    with open(out + ".tmp", "w") as fh:
        for row in rows:
            row.setdefault("status", "ok")
            fh.write(json.dumps(row, default=float) + "\n")
    os.replace(out + ".tmp", out)
