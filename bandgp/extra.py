"""Supplementary runs and diagnostics (run after the main grid; single-threaded, FTZ, seeded like the main jobs).

  refs     offline reference posteriors (hard banding c=5, no time cap) for d=2, beta=1, n=1e5
  metis    d=2, beta=1, n=1e5 banding re-run with CHOLMOD + forced METIS ordering (default = CHOLMOD's AMD/METIS choice)
  nnpred   NN-Vecchia prediction at m in {50, 100} for SE, beta=1 (settles the 5% borderline)
  plateau  latent-Vecchia plateau check: SE, d in {1,2}, beta=1, n=1e4, rep 0; m x latent jitter sweep with
           condition numbers of the local conditioning matrices; response (NN) Vecchia at the same m.
Supplement rows are merged into results/raw/<job>.jsonl and predictions into results/preds/<job>.npz.
"""
import json
import os
import time

import numpy as np

from . import config as C
from .data import make_data
from .methods import Ctx, band_chol, vecchia_latent, vecchia_pred, _STATE, Result
from .runner import RAW, PREDS, ROOT, job_id, run_methods, add_metrics


def _ctx(j):
    d, n, beta = j["d"], j["n"], j["beta"]
    t0 = time.perf_counter()
    data = make_data(d, C.ALPHA, beta, n, j["rep"])
    t_data = time.perf_counter() - t0
    rho = C.density(n, d, beta); ell = C.lengthscale(n, d, beta)
    ctx = Ctx(data, j["kernel"], ell, d, n, rho)
    base = dict(j, alpha=C.ALPHA, L=data["L"], rho=rho, ell=ell, pts_per_ell=rho * ell ** d, R_ex=ctx.R_ex,
                t_data=t_data, supplement=True)
    return data, ctx, base


def supplement(j, mlist, cap):
    jid = job_id(j)
    raw, prd = os.path.join(RAW, jid + ".jsonl"), os.path.join(PREDS, jid + ".npz")
    if not (os.path.exists(raw) and os.path.exists(prd)):
        print("missing main results for", jid); return
    data, ctx, base = _ctx(j)
    base["time_cap"] = cap
    rows, preds = run_methods(ctx, mlist, base, cap)
    add_metrics(rows, preds, data["ft"], data["fv"], ctx.sub)
    new = {r["method"] for r in rows}
    old = [json.loads(l) for l in open(raw)]
    with open(raw + ".tmp", "w") as fh:
        for r in [r for r in old if r["method"] not in new] + rows:
            fh.write(json.dumps(r, default=float) + "\n")
    os.replace(raw + ".tmp", raw)
    z = dict(np.load(prd)); names = list(z["names"])
    keep = [i for i, nm in enumerate(names) if nm not in preds]
    add = sorted(preds)
    out = dict(z)
    out["names"] = np.array([names[i] for i in keep] + add)
    for k in ("mt", "mv", "vsub"):
        out[k] = np.concatenate([z[k][keep], np.array([getattr(preds[a], k) for a in add]).reshape(len(add), -1)]) \
            if add else z[k][keep]
    np.savez_compressed(prd.replace(".npz", ".tmp.npz"), **out)
    os.replace(prd.replace(".npz", ".tmp.npz"), prd)
    for r in rows:
        print(f"  {jid} {r['method']:24s} {r['status']:10s} t={r.get('t_total', float('nan')):8.1f}s "
              f"order={r.get('info_order', '-')} nnzL={r.get('info_nnz_used', float('nan')):.3g} {r.get('reason', '')[:60]}",
              flush=True)


def jobs(d, kernels, betas, n, reps):
    return [dict(d=d, kernel=k, beta=b, n=n, rep=r) for k in kernels for b in betas for r in reps]


def do_refs():
    C.NNZ_CAP = 1e9                       # offline: memory permitting
    for j in jobs(2, ["se", "m32"], [1.0], 100_000, range(C.reps(100_000))):
        supplement(j, [("ref", "ref_band_c5", {"c": 5.0}, band_chol, {"c": 5.0})], cap=None)


def do_metis():
    for j in jobs(2, ["se", "m32"], [1.0], 100_000, range(C.reps(100_000))):
        cs = [2.0, 2.5, 3.0] if j["kernel"] == "se" else [2.0, 2.5, 3.0, 4.0]
        supplement(j, [("band", f"band_c{c}_metis", {"c": c, "order": "metis"}, band_chol, {"c": c, "order": "metis"})
                       for c in cs], cap=600)


def do_nnpred():
    for d in (1, 2):
        for n in C.NS[d]:
            for j in jobs(d, ["se"], [1.0], n, range(C.reps(n))):
                supplement(j, [("vecchia_pred", f"vecchia_pred_m{m}", {"m": m}, vecchia_pred, {"m": m})
                               for m in (50, 100)], cap=C.time_cap(n))


def do_latent(part):
    """Recompute latent Vecchia (m = 10, 30) for every job after the exact-predecessor fix.
    part: 'd1small' (d=1, n<=1e5), 'd1big' (d=1, n=1e6), 'd2' (d=2 except beta=1, n=1e5), 'd2b1' (the rest)."""
    from .methods import vecchia_latent as vl
    for d in (1, 2):
        for n in C.NS[d]:
            for k in C.KERNELS:
                for b in C.BETAS:
                    key = ("d1small" if n <= 100_000 else "d1big") if d == 1 else \
                          ("d2b1" if (b == 1.0 and n == 100_000) else "d2")
                    if key != part:
                        continue
                    for r in range(C.reps(n)):
                        supplement(dict(d=d, kernel=k, beta=b, n=n, rep=r),
                                   [("vecchia_latent", f"vecchia_latent_m{m}", {"m": m}, vl, {"m": m})
                                    for m in C.VECCHIA_MS], cap=C.time_cap(n))


def do_latent_big():
    """Is latent Vecchia m=30 at n=1e6 (d=1, beta=1) method- or implementation-limited? Run it with the predicted
    nnz cap disabled and a 12 GB address-space limit; log actual nnz(L), peak memory, flops, time."""
    from .methods import vecchia_latent as vl
    C.NNZ_CAP = 1e12
    for k in C.KERNELS:
        for r in range(C.reps(1_000_000)):
            supplement(dict(d=1, kernel=k, beta=1.0, n=1_000_000, rep=r),
                       [("vecchia_latent", "vecchia_latent_m30", {"m": 30}, vl, {"m": 30})], cap=600)


def do_plateau():
    C.NNZ_CAP = 1e9
    out = []
    for d in (1, 2):
        j = dict(d=d, kernel="se", beta=1.0, n=10_000, rep=0)
        z = np.load(os.path.join(PREDS, job_id(j) + ".npz"))
        names = list(z["names"]); ie = names.index("exact")
        mt0, v0, ft = z["mt"][ie], z["vsub"][ie], z["ft"]
        stat = float(np.sqrt(np.mean((mt0 - ft) ** 2)))
        data, ctx, base = _ctx(j)
        runs = [("latent", m, nug) for m in (10, 30, 50, 100) for nug in (1e-10, 1e-8, 1e-6, 1e-4)] + \
               [("response_nn", m, C.SIGMA ** 2) for m in (10, 30, 50, 100)]
        for kind, m, nug in runs:
            res = Result(); _STATE["conds"] = []; _STATE["deadline"] = None
            t0 = time.perf_counter()
            try:
                (vecchia_latent(ctx, res, m, nugget=nug) if kind == "latent" else vecchia_pred(ctx, res, m))
                st = "ok"
            except Exception as e:
                st = f"{type(e).__name__}: {str(e)[:80]}"
            conds = np.array(_STATE["conds"]); _STATE["conds"] = None
            row = dict(d=d, kind=kind, m=m, jitter=nug, status=st, t=time.perf_counter() - t0,
                       cond_median=float(np.median(conds)) if conds.size else np.nan,
                       cond_max=float(np.max(conds)) if conds.size else np.nan)
            if st == "ok":
                row.update(flops=float(res.flops), nnz_used=res.info.get("nnz_used"),
                           err_rel=float(np.sqrt(np.mean((res.mt - mt0) ** 2))) / stat,
                           var_ratio=float(np.mean(res.vsub / v0)),
                           var_ratio_min=float(np.min(res.vsub / v0)), var_ratio_max=float(np.max(res.vsub / v0)))
            out.append(row)
            print(row, flush=True)
    import pandas as pd
    pd.DataFrame(out).to_csv(os.path.join(ROOT, "results", "diag_vecchia_plateau.csv"), index=False)


def main(what, part=None):
    if what == "latent":
        return do_latent(part)
    if what == "latent_big":
        return do_latent_big()
    {"refs": do_refs, "metis": do_metis, "nnpred": do_nnpred, "plateau": do_plateau}[what]()
