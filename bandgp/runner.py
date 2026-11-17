"""Executes one job = (d, kernel, lsmode, alpha, beta, n, rep): all methods, metrics, JSONL output."""
import json
import os
import threading
import time
import traceback

import numpy as np
import psutil

from . import config as C
from .data import make_data
from .methods import Ctx, Result, Skip, Invalid, Timeout, method_list, _STATE
from . import diagnostics

RAW = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results", "raw")


def job_id(j):
    return "d{d}_{kernel}_{lsmode}_a{alpha}_b{beta}_n{n}_r{rep}".format(**j)


class PeakRSS:
    def __init__(self):
        self.p = psutil.Process(); self.peak = 0; self._stop = False

    def __enter__(self):
        self.base = self.p.memory_info().rss; self.peak = self.base
        self.th = threading.Thread(target=self._run, daemon=True); self.th.start(); return self

    def _run(self):
        while not self._stop:
            self.peak = max(self.peak, self.p.memory_info().rss); time.sleep(0.005)

    def __exit__(self, *a):
        self._stop = True; self.th.join(); self.peak = max(self.peak, self.p.memory_info().rss)

    @property
    def mb(self):
        return (self.peak - self.base) / 2 ** 20


def run_job(j):
    os.makedirs(RAW, exist_ok=True)
    jid = job_id(j)
    out_path = os.path.join(RAW, jid + ".jsonl"); prog = os.path.join(RAW, jid + ".progress")
    if os.path.exists(out_path):
        return
    crashed = set()
    if os.path.exists(prog):                       # previous attempt died (e.g. killed): blame the unfinished method
        started, done = [], set()
        for line in open(prog):
            k, nm = line.split(); (started.append(nm) if k == "start" else done.add(nm))
        crashed = {nm for nm in started if nm not in done}
    pf = open(prog, "a")

    d, n = j["d"], j["n"]
    t0 = time.perf_counter()
    data = make_data(d, j["alpha"], j["beta"], n, j["rep"])
    t_data = time.perf_counter() - t0
    rho = C.density(n, d, j["beta"])
    ell = C.lengthscale(j["lsmode"], n, d, j["beta"], j["alpha"])
    ctx = Ctx(data, j["kernel"], ell, d, n, rho, j["rep"])
    base = dict(j, L=data["L"], rho=rho, ell=ell, rho_ell_d=rho * ell ** d, R_ex=ctx.R_ex, t_data=t_data)

    rows, preds = [], {}
    for family, name, params, fn, kw in method_list(d):
        row = dict(base, family=family, method=name, **params)
        if name in crashed:
            row.update(status="oom_killed", valid=False, reason="process killed during this method")
            rows.append(row); continue
        pf.write(f"start {name}\n"); pf.flush()
        res = Result()
        _STATE["deadline"] = time.perf_counter() + C.TIME_CAP
        try:
            with PeakRSS() as pm:
                fn(ctx, res, **kw)
            status, reason = "ok", ""
        except Skip as e:
            status, reason = "skip", str(e)
        except Invalid as e:
            status, reason = "invalid", str(e)
        except Timeout as e:
            status, reason = "timeout", str(e)
        except MemoryError as e:
            status, reason = "oom", "MemoryError"
        except Exception as e:
            nm = type(e).__name__
            status = "oom" if "Memory" in nm or "memory" in str(e).lower() else "error"
            reason = f"{nm}: {e}"[:300]
            if status == "error":
                traceback.print_exc()
        pf.write(f"done {name}\n"); pf.flush()
        row.update(status=status, reason=reason, **{f"info_{k}": v for k, v in res.info.items()})
        if status != "skip":
            row["peak_mem_mb"] = pm.mb if status == "ok" else np.nan
        if status == "ok":
            uses_xc = family not in ("sgpr", "vecchia_pred")
            t_xc = ctx.t_xcov if (uses_xc and hasattr(ctx, "t_xcov")) else 0.0
            t_fit = res.t["fit"] + (ctx.t_tree if family not in ("exact",) else 0.0)
            t_mean = res.t["mean"] + t_xc
            row.update(t_fit=t_fit, t_mean=t_mean, t_var=res.t["var"], var_extrapolated=res.var_extrapolated,
                       t_meanonly=t_fit + t_mean)
            preds[name] = res
        rows.append(row)
        if family == "infl_exact":
            ctx.band_cache = {}   # free memory after last use of this radius

    # ---------------------------------------------------------------- reference and metrics
    ft, fv = data["ft"], data["fv"]
    ref_name = next((nm for nm in ["exact", "pcg_r1000", "pcg_r200", "pcg_r50"] if nm in preds), None)
    ref = preds.get(ref_name)
    ref_mt = ref.mt if ref else None
    ref_vsub = (ref.vt[ctx.sub] if ref.vt is not None else ref.vsub) if ref else None
    ref_vt = ref.vt if (ref is not None and ref.vt is not None) else None
    for row in rows:
        row["ref"] = ref_name
        r = preds.get(row["method"])
        if r is None:
            row["valid"] = False; continue
        row["rmse_f0"] = float(np.sqrt(np.mean((r.mt - ft) ** 2)))
        row["rmse_val"] = float(np.sqrt(np.mean((r.mv - fv) ** 2)))
        if ref_mt is not None:
            row["rmse_ref"] = float(np.sqrt(np.mean((r.mt - ref_mt) ** 2)))
        v = r.vt if r.vt is not None else r.vsub
        if v is None:
            row["valid"] = True; continue
        fsel = ft if r.vt is not None else ft[ctx.sub]
        msel = r.mt if r.vt is not None else r.mt[ctx.sub]
        row["frac_negvar"] = float(np.mean(v < 0))
        row["cover"] = float(np.mean(np.abs(fsel - msel) <= 1.96 * np.sqrt(np.clip(v, 0, None))))
        row["cover_npts"] = int(len(v))
        if ref_vt is not None and r.vt is not None:
            row["var_ratio"] = float(np.mean(r.vt / ref_vt)); row["var_ratio_min"] = float(np.min(r.vt / ref_vt))
        elif ref_vsub is not None:
            vs = r.vt[ctx.sub] if r.vt is not None else r.vsub
            row["var_ratio"] = float(np.mean(vs / ref_vsub)); row["var_ratio_min"] = float(np.min(vs / ref_vsub))
        row["valid"] = bool(row["frac_negvar"] == 0)
    # ---------------------------------------------------------------- scale diagnostics (rep 0 only)
    _STATE["deadline"] = None
    if j["rep"] == 0:
        drow = dict(base, family="diag", method="diag", status="ok")
        try:
            drow.update(diagnostics.analytic(ctx)); drow.update(diagnostics.rank_needed(ctx))
        except Exception as e:
            drow.update(status="error", reason=str(e)[:300])
        rows.append(drow)
    tmp = out_path + ".tmp"
    with open(tmp, "w") as fh:
        for row in rows:
            fh.write(json.dumps(row, default=float) + "\n")
    os.replace(tmp, out_path)
    pf.close(); os.remove(prog)
