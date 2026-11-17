"""Single entry point.
  python -m rffvb run [--workers 4] [--only d=1,truth=sparse,rep=0]   # all (d, truth, rep) jobs, resumable
  python -m rffvb claim3 [--workers 4]                               # Claim 3 vs exact spike-and-slab (MCMC)
  python -m rffvb analyse                                            # results.csv, figures, summary tables
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMBA_NUM_THREADS"):
    os.environ[v] = "1"                      # single-threaded fits: per-fit runtimes are comparable
import argparse
import itertools
from concurrent.futures import ProcessPoolExecutor


def _job(j):
    from .runner import run_job
    run_job(j)
    return j


def _c3(args):
    from .claim3_mcmc import run_one
    run_one(*args)


def main():
    ap = argparse.ArgumentParser(prog="rffvb")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--workers", type=int, default=4); r.add_argument("--only", default="")
    sp.add_parser("analyse")
    c3 = sp.add_parser("claim3"); c3.add_argument("--workers", type=int, default=4)
    a = ap.parse_args()
    if a.cmd == "claim3":
        from .claim3_mcmc import jobs
        with ProcessPoolExecutor(a.workers) as ex:
            list(ex.map(_c3, jobs()))
        return
    if a.cmd == "analyse":
        from .analysis import main as am
        return am()
    from . import config as C
    jobs = [dict(d=d, truth=t, rep=k) for k, d, t in itertools.product(range(C.REPS), C.DS, C.TRUTHS)]
    for kv in filter(None, a.only.split(",")):
        k, v = kv.split("=")
        jobs = [j for j in jobs if str(j[k]) == v]
    with ProcessPoolExecutor(a.workers) as ex:
        for j in ex.map(_job, jobs):
            print("done", j, flush=True)


if __name__ == "__main__":
    main()
