"""Single entry point.

  python -m bandgp run      [--slots 4] [--mem-per-slot-gb 3.5] [--filter key=val ...]
  python -m bandgp analyse  # results.csv, figures/, summary tables
  python -m bandgp extra refs|metis|nnpred|plateau   # supplementary runs and diagnostics (after `run`)
  python -m bandgp job '<json>'   (internal: one job in this process)
"""
import os
for v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ[v] = "1"                     # single-threaded timings everywhere
import argparse
import itertools
import json
import resource
import subprocess
import sys
import time


def all_jobs():
    """Rep-0 jobs first (they fix c* for later reps), small n first."""
    from . import config as C
    jobs = []
    for rep_phase in (0, 1):
        for d in (1, 2):
            for n in C.NS[d]:
                for kernel in C.KERNELS:
                    for beta in C.BETAS:
                        reps = [0] if rep_phase == 0 else list(range(1, C.reps(n)))
                        jobs += [dict(d=d, kernel=kernel, beta=beta, n=n, rep=r) for r in reps]
    return jobs


def slots_for(n):
    return 1 if n <= 100_000 else 2


def run(args):
    from .runner import job_id, RAW
    jobs = all_jobs()
    for f in args.filter or []:
        k, v = f.split("=")
        jobs = [j for j in jobs if str(j[k]) == v]
    jobs = [j for j in jobs if not os.path.exists(os.path.join(RAW, job_id(j) + ".jsonl"))]
    print(f"{len(jobs)} jobs to run", flush=True)
    os.makedirs(os.path.join(RAW, "..", "logs"), exist_ok=True)
    running = []; free = args.slots; t0 = time.time(); ndone = 0
    while jobs or running:
        while jobs and slots_for(jobs[0]["n"]) <= free:
            if jobs[0]["rep"] > 0 and any(r["rep"] == 0 for _, r, _, _ in running):
                break                                   # later reps need rep-0 c*
            j = jobs.pop(0); s = slots_for(j["n"])
            log = open(os.path.join(RAW, "..", "logs", job_id(j) + ".log"), "w")
            mem = int(args.mem_per_slot_gb * s * 2 ** 30)
            p = subprocess.Popen([sys.executable, "-m", "bandgp", "job", json.dumps(j), "--mem", str(mem)],
                                 stdout=log, stderr=subprocess.STDOUT, cwd=os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            running.append((p, j, s, time.time())); free -= s
        time.sleep(0.5)
        for item in list(running):
            p, j, s, ts = item
            if p.poll() is not None:
                running.remove(item); free += s; ndone += 1
                ok = os.path.exists(os.path.join(RAW, job_id(j) + ".jsonl"))
                if not ok and p.returncode != 0:      # killed: resubmit once; runner blames the unfinished method
                    j.setdefault("_retries", 0)
                    if j["_retries"] < 3:
                        j["_retries"] += 1; jobs.insert(0, j)
                print(f"[{time.time() - t0:7.0f}s] {job_id(j)} rc={p.returncode} {time.time() - ts:.0f}s "
                      f"done={ndone} left={len(jobs)}", flush=True)


def set_ftz_daz():
    """Flush subnormals to zero (MXCSR FTZ|DAZ). Far-apart kernel entries otherwise create subnormal floats that
    slow dense LAPACK/BLAS by ~35x on this CPU; values below 1e-308 are numerically irrelevant here."""
    import ctypes, subprocess
    d = os.path.join(os.path.dirname(os.path.abspath(__file__)), "_ftz")
    so = os.path.join(d, "libftz.so")
    if not os.path.exists(so):
        subprocess.run(["gcc", "-O2", "-shared", "-fPIC", "-o", so, os.path.join(d, "ftz.c")], check=True)
    ctypes.CDLL(so).set_ftz_daz()


def job(args):
    set_ftz_daz()
    if args.mem:
        resource.setrlimit(resource.RLIMIT_AS, (args.mem, args.mem))
    from .runner import run_job
    j = json.loads(args.spec); j.pop("_retries", None)
    run_job(j)


def main():
    ap = argparse.ArgumentParser(prog="bandgp")
    sp = ap.add_subparsers(dest="cmd", required=True)
    r = sp.add_parser("run"); r.add_argument("--slots", type=int, default=4)
    r.add_argument("--mem-per-slot-gb", type=float, default=3.5); r.add_argument("--filter", nargs="*")
    jb = sp.add_parser("job"); jb.add_argument("spec"); jb.add_argument("--mem", type=int, default=0)
    sp.add_parser("analyse")
    ex = sp.add_parser("extra"); ex.add_argument("what", choices=["refs", "metis", "nnpred", "plateau"])
    ex.add_argument("--mem", type=int, default=0)
    a = ap.parse_args()
    if a.cmd == "run":
        run(a)
    elif a.cmd == "job":
        job(a)
    elif a.cmd == "extra":
        set_ftz_daz()
        if a.mem:
            resource.setrlimit(resource.RLIMIT_AS, (a.mem, a.mem))
        from .extra import main as emain
        emain(a.what)
    else:
        from .analysis import main as amain
        amain()


if __name__ == "__main__":
    main()
