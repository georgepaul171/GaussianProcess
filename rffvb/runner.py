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
            rows.append(dict(bd, method="svb_individual", n_selected=k, runtime=t_feat + time.perf_counter() - t0,
                             n_sweeps=r.n_sweeps, converged=r.converged, **metrics(f_te, mean, *band)))
            # group spike-and-slab VB, each cos/sin pair a group (GSVB-style, multivariate Laplace slab)
            t0 = time.perf_counter()
            sizes = np.full(p // 2, 2)
            g = run_gsvb(Xs, Ys, sizes, lam=C.VB_LAM, a0=C.VB_A0, b0=None, eps=C.VB_EPS, max_sweeps=C.VB_MAX_SWEEPS)
            mean = Phit @ g.post_mean
            band = draw_band(draws_mf(rng, g.mu, g.s, g.gamma, sizes), Phit)
            rows.append(dict(bd, method="svb_group", n_selected=int(2 * (g.gamma > 0.5).sum()),
                             runtime=t_feat + time.perf_counter() - t0, n_sweeps=g.n_sweeps, converged=g.converged,
                             **metrics(f_te, mean, *band)))

        # SSGP with as many features as individual VB selected (m = ceil(k/2) frequencies, >= 1)
        m_freq = max(1, math.ceil(k_ind / 2))
        m_, s_, inf = ssgp(X, y, Xt, m_freq, ell, s2, nz, seed=rep)
        rows.append(dict(bp, design="optimised", method="ssgp", n_selected=2 * m_freq, runtime=inf["t"], ssgp_success=inf["success"],
                         **metrics(f_te, m_, *gauss_band(m_, s_))))
        print(f"{jid} p={p} done", flush=True)
    with open(out + ".tmp", "w") as fh:
        for row in rows:
            row.setdefault("status", "ok")
            fh.write(json.dumps(row, default=float) + "\n")
    os.replace(out + ".tmp", out)
