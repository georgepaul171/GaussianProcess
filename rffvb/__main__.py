"""Single entry point.
  python -m rffvb run [--workers 4] [--only d=1,truth=sparse,rep=0]   # all (d, truth, rep) jobs, resumable
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


def main():
    ap = argparse.ArgumentParser(prog="rffvb")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--workers", type=int, default=4); r.add_argument("--only", default="")
    sp.add_parser("analyse")
    a = ap.parse_args()
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
