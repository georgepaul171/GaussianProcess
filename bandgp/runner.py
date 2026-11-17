"""Executes one job = (d, kernel, beta, n, rep): all methods, metrics, JSONL output (results/raw)."""
import json
import os
import threading
import time
import traceback

import numpy as np
import psutil

from . import config as C
from .data import make_data
from .methods import Ctx, Result, Skip, Invalid, Timeout, method_list, _STATE, band_banded, band_chol

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "results", "raw")
PREDS = os.path.join(ROOT, "results", "preds")
REF_C = 5.0   # reference posterior where exact is unavailable: hard banding at c = 5 (validated vs exact at n = 1e4)
USES_XCOV = {"exact", "band", "pcg"}


def job_id(j):
    return "d{d}_{kernel}_b{beta}_n{n}_r{rep}".format(**j)


class PeakRSS:
    """Peak resident memory above the pre-method baseline, sampled every 5 ms."""
    def __enter__(self):
        self.p = psutil.Process(); self.base = self.p.memory_info().rss; self.peak = self.base; self._stop = False
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start(); return self

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self.p.memory_info().rss); time.sleep(0.005)

    def __exit__(self, *a):
        self._stop = True; self.th.join()

    @property
    def mb(self):
        return (self.peak - self.base) / 2 ** 20


def band_radii(j):
    """Rep 0: full sweep. Later reps: c* (smallest c valid on rep 0) and the next value up."""
    if j["rep"] == 0:
        return list(C.CS)
    p = os.path.join(RAW, job_id(dict(j, rep=0)) + ".jsonl")
    ok = sorted(r["c"] for r in map(json.loads, open(p)) if r["family"] == "band" and r["status"] == "ok")
    if not ok:
        return []
    i = C.CS.index(ok[0])
    return C.CS[i:i + 2]


def add_metrics(rows, preds, ft, fv, sub, ref=None):
    """Per-row accuracy metrics vs f0 (and vs exact when present in preds or given as ref)."""
    ref = ref if ref is not None else preds.get("exact")
    for row in rows:
        r = preds.get(row["method"])
        if r is None:
            continue
        row["rmse_f0"] = float(np.sqrt(np.mean((r.mt - ft) ** 2)))
        row["rmse_val"] = float(np.sqrt(np.mean((r.mv - fv) ** 2)))
        v = r.vsub; fs = ft[sub]; ms = r.mt[sub]
        row["frac_negvar"] = float(np.mean(v < 0))
        row["cover"] = float(np.mean(np.abs(fs - ms) <= 1.96 * np.sqrt(np.clip(v, 0, None))))
        if ref is not None:
            row["rmse_to_exact"] = float(np.sqrt(np.mean((r.mt - ref.mt) ** 2)))
            row["var_ratio"] = float(np.mean(v / ref.vsub)); row["var_ratio_min"] = float(np.min(v / ref.vsub))
        if row["frac_negvar"] > 0:
            row["status"] = "invalid"; row["reason"] = "negative predictive variances"


def run_methods(ctx, mlist, base, cap, crashed=(), pf=None):
    """Run each (family, name, params, fn, kwargs) with the time cap, peak-memory sampling and status capture."""
    rows, preds = [], {}
    for family, name, params, fn, kw in mlist:
        row = dict(base, family=family, method=name, **params)
        if name in crashed:
            row.update(status="oom", reason="process killed (memory) during this method"); rows.append(row); continue
        if pf:
            pf.write(f"start {name}\n"); pf.flush()
        res = Result()
        _STATE.update(deadline=(time.perf_counter() + cap) if cap else None, cap=cap)
        pm = None
        try:
            with PeakRSS() as pm:
                fn(ctx, res, **kw)
            status, reason = "ok", ""
        except Skip as e:
            status, reason = "infeasible", str(e)
        except Invalid as e:
            status, reason = "invalid", str(e)
        except Timeout as e:
            status, reason = "timeout", str(e)
        except MemoryError:
            status, reason = "oom", "MemoryError"
        except Exception as e:
            nm = type(e).__name__
            status = "oom" if "memory" in (nm + str(e)).lower() else "error"
            reason = f"{nm}: {e}"[:300]
            if status == "error":
                traceback.print_exc()
        _STATE.update(deadline=None)
        if pf:
            pf.write(f"done {name}\n"); pf.flush()
        row.update(status=status, reason=reason, **{f"info_{k}": v for k, v in res.info.items()})
        if status == "ok":
            t_fit = res.t["fit"] + (ctx.t_tree if family != "exact" else 0.0)
            t_mean = res.t["mean"] + (ctx.t_xcov if family in USES_XCOV else 0.0)
            row.update(t_xcov=ctx.t_xcov if family in USES_XCOV else 0.0, t_fit=t_fit, t_mean=t_mean,
                       t_var=res.t["var"], t_total=t_fit + t_mean + res.t["var"],
                       flops=res.flops, nnz_used=res.info.get("nnz_used"), peak_mem_mb=pm.mb)
            preds[name] = res
        rows.append(row)
    return rows, preds


def run_job(j):
    os.makedirs(RAW, exist_ok=True)
    jid = job_id(j)
    out_path = os.path.join(RAW, jid + ".jsonl"); prog = os.path.join(RAW, jid + ".progress")
    if os.path.exists(out_path):
        return
    crashed = set()
    if os.path.exists(prog):                      # previous attempt killed: blame the unfinished method
        started, done = [], set()
        for line in open(prog):
            k, nm = line.split(); (started.append(nm) if k == "start" else done.add(nm))
        crashed = {nm for nm in started if nm not in done}
    pf = open(prog, "a")

    d, n, beta = j["d"], j["n"], j["beta"]
    t0 = time.perf_counter()
    data = make_data(d, C.ALPHA, beta, n, j["rep"])
    t_data = time.perf_counter() - t0
    rho = C.density(n, d, beta); ell = C.lengthscale(n, d, beta)
    ctx = Ctx(data, j["kernel"], ell, d, n, rho)
    cap = C.time_cap(n)
    base = dict(j, alpha=C.ALPHA, L=data["L"], rho=rho, ell=ell, pts_per_ell=rho * ell ** d, R_ex=ctx.R_ex,
                t_data=t_data, time_cap=cap)
    radii = band_radii(j)

    ref_fn = band_banded if d == 1 else band_chol
    mlist = method_list(d, radii) + [("ref", f"ref_band_c{REF_C:g}", {"c": REF_C}, ref_fn, {"c": REF_C})]
    rows, preds = run_methods(ctx, mlist, base, cap, crashed, pf)

    ft, fv = data["ft"], data["fv"]
    add_metrics(rows, preds, ft, fv, ctx.sub)
    os.makedirs(PREDS, exist_ok=True)
    names = sorted(preds)
    np.savez_compressed(os.path.join(PREDS, jid + ".npz"), names=np.array(names),
                        mt=np.array([preds[k].mt for k in names]), mv=np.array([preds[k].mv for k in names]), vsub=np.array([preds[k].vsub for k in names]),
                        ft=ft, fv=fv, sub=ctx.sub)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=float) + "\n")
    os.replace(tmp, out_path)
    pf.close(); os.remove(prog)
