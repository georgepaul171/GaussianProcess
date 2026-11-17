# Bug report for georgepaul171/2027-research (found while using it read-only; not fixed here)

## `svb/block.py::_block_update` segfaults on a numerically singular cluster Gram matrix

- **Repo state:** 2027-research @ `e275fa0`; numba from `pip install numba` (Python 3.13, Linux x86_64).
- **Symptom:** the process dies with `Fatal Python error: Segmentation fault`, exit 139. No Python exception is raised, so callers cannot catch it.
  - Traceback via `-X faulthandler`: `run_block` (block.py line 212), at the call to `_block_update(GCC, rc, lam, logw, log1mw)`.
- **Failing input:** a cluster of size **4** (kmax = 4) whose Gram matrix G_CC is singular to machine precision.
  - eigenvalues = [−2.2e-13, 7.9e-06, 0.70, 8.0e3]
  - The cluster's columns are `sin(ω x)` random Fourier features with ω = 15.474, 15.430, 15.445, 15.443 on x ∈ [0, 1], n = 1000, scaled by 1/σ = 2. Near-equal frequencies give nearly collinear columns.
  - The same happens at kmax = 6. It never happened in the 2027-research tests, which use Toeplitz designs.
- **How the input arose:** `make_clusters(X, ridge_init(X, Y), threshold=0.5, kmax=4)` on a 1000 × 2000 random-Fourier-feature design. The failing cluster is cluster 81 at the start of sweep 1.
- **Minimal reproducer:** in this branch, `bugs/block_segfault_input.npz` holds `Gcc` (4 × 4), `rc` (4,), `lam = 1`, `logw = log(1/2001)` and `log1mw`.

  ```bash
  SVB_PATH=/path/to/2027-research python bugs/repro_block_segfault.py; echo $?   # prints eigenvalues, then 139
  ```

- **Not checked, just suspicions:** inside `_solve_subset`, `Sig = np.linalg.inv(Gs + np.diag(D))` can be singular or indefinite when D_j ≈ 0 (large |μ_j/s_j|). That makes `s_new = sqrt(diag(Sig))` NaN or negative, and LAPACK calls later in numba (`slogdet`, `solve`) then receive NaN or indefinite input.
- **Possible guards:**
  - add a relative ridge to `Gs` before `inv`/`solve`;
  - check `np.isfinite` and positive definiteness and fall back to a pattern-level skip (F_S = −inf);
  - or cap the cluster condition number in `make_clusters`.
- **Workaround that avoids it (tested, not used in the study):** append √ε·I pseudo-observation rows to X, which is equivalent to G + εI, with ε = 1e-8 × mean diag(G). With that, `run_block` runs at p = 2000 (1.3–1.4 s per sweep at kmax = 4).
- **Related performance note:** on random-Fourier-feature designs with kmax = 6, a single sweep at p = 500 did not finish in several minutes. Every chunk is full (64 patterns) and highly ill-conditioned, so `_solve_subset` likely runs to its 300 × 20 iteration limits.
