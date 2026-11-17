# Where does hard banding beat low-rank and Vecchia GP approximations?

*Minimal decisive study, 2026-11-17. Code: `python -m bandgp run|extra|analyse`. Single-threaded timings, Intel Xeon @ 2.1 GHz, 3.5–7 GB per job.*

## Answer

1. **Banding vs low-rank: banding wins once the domain grows enough that a lengthscale ball holds only ~10–100 points. It loses badly before that.**
   - **β ≤ 0.5 (all n ≤ 1e6).** Hard banding is indefinite below c = 2.5 (SE) or c = 4 (Matérn-3/2). At the validity threshold it needs 870–7,500 nnz/row at n = 1e4, and it exceeds memory from n = 1e5 on. SGPR with M = 100–1000 reproduces the exact posterior in about 1 s.
   - **β = 1 (fixed density, 10 points per ℓ^d).** Banding is valid at c* = 2 (SE) or 2.5–3 (Matérn) with 41–260 nnz/row, flat in n. SGPR misses the reference by more than 90% of the statistical error even at M = 5000 for n ≥ 1e5.
   - **β\*.** With n0 = 10 and ℓ = 1, the β\* at which banding beats low-rank lies in **(0.5, 1]** for every n tested. The governing quantity is points per ℓ^d, ρℓ^d = n0^β n^(1−β), not β itself; I did not run β = 0.75.
     - Banding's band is small enough to win at ρℓ^d ≈ 10 and too large at ρℓ^d ≥ 50.
     - So β\*(n) ≈ 1 − log(k/n0)/log(n/n0) with k ≈ 20–50. That gives about 0.8 at n = 1e4 and about 0.9 at n = 1e6, which rises toward 1 as n grows. This is inferred, not measured.
2. **Banding vs Vecchia: it depends on the kernel, so no blanket claim.**
   - **SE, d = 2:** banding dominates latent Vecchia at every cost measured.
   - **SE, d = 1:** banding ties latent Vecchia by flops at a 5% tolerance and wins at tighter tolerances.
   - **Matérn-3/2:** Vecchia wins. In d = 1 this is by construction (Markov).
   - **Pointwise prediction only:** the cheap NN-Vecchia predictor beats everything in SE d = 1, but not in d = 2.
3. **Against the feeler.** Its increasing-domain win for banding holds against low-rank and against Vecchia only for SE. Two caveats: the boundary is a points-per-lengthscale boundary, and in d = 2 sparse-Cholesky fill makes banding cost grow as about n^1.5.

## Setup (what differs from the original spec)

- **Domain.** [0, L]^d with **L^d = (n/n0)^β**, n0 = 10. So β = 1 is fixed density in both d; the literal L = (n/n0)^β only does that for d = 1.
- **Grid.**
  - β ∈ {0, 0.5, 1}; α = 0.75; σ = 0.5.
  - d = 1: n ∈ {1e4, 1e5, 1e6}. d = 2: n ∈ {1e4, 1e5}.
  - SE kernel exp(−r²/ℓ²) and Matérn-3/2.
  - ℓ = n^(−1/(2α+d)) (the paper's oracle) at β = 0, and ℓ = 1 at β > 0.
  - Reps: 3 at n = 1e4, 2 above.
- **f0.** A cosine series on [0, L]^d at frequencies πj/L with random signs and coefficients max(|j|/L, 1)^−(α+d/2), scaled to unit variance. At L = 1 this is exactly the feeler's j^−(α+1/2).
  - For L ≥ 10 its spectrum is fixed in absolute units: the variance share above any frequency moves by less than 1% for L from 10 to 1e5. So **feature scale does not drift with β**.
  - However, 81% (d = 1) and 95% (d = 2) of the variance lies above ω = 1/ℓ. At β > 0 test RMSE is therefore bias-dominated (≈ 0.47) and coverage is 0.5–0.7 *for every method including exact*. That makes "test RMSE within 5%" nearly vacuous at β > 0.
  - Proposed fix, not run: put the flat-spectrum corner at about 0.1 oscillations per unit length (ω_c ≈ 0.6), so only about 20% of the variance lies above 1/ℓ.
- **Primary metric: fidelity to a reference posterior.**
  - Reference: the exact GP at n = 1e4. Above that, **hard banding at c = 5**.
  - Checked against exact on all ten n = 1e4 cells (`results/ref_check.csv`):
    - SE: mean error about 1e-10 of the statistical error; variance ratio 1.000.
    - Matérn, β = 1: 0.7–1.7%.
    - Matérn, β ≤ 0.5: 5–9%, too loose. But c = 5 is infeasible there at n ≥ 1e5 anyway.
  - The d = 2, β = 1, n = 1e5 references were computed offline: CHOLMOD with METIS, about 11 min each, nnz(L) ≈ 3.2e8.
  - Cells with β ≤ 0.5 and n ≥ 1e5 have **no reference posterior**. There banding is simply infeasible, and the secondary f0-rule winners are SGPR / latent Vecchia / NN-Vecchia (`results/winners_f0rule.csv`).
- **Eligibility.** A configuration qualifies if its mean error on validation points is at most **5% of the reference's statistical error** and its |mean variance ratio − 1| ≤ 10% on 200 test points. Configurations are selected on validation points and reported on test points.
- **Cost.** Primary cost is **approximate flops**, with nnz of the used or factorised matrix and peak memory alongside. Wall-clock is secondary.
  - **Implementations differ.** d = 1 banding is LAPACK banded Cholesky (`cholesky_banded` / `dtbtrs`). d = 2 banding is CHOLMOD supernodal; its default ordering chose METIS, and forced METIS gives the same times (SE, β = 1, n = 1e5: c = 2.5 takes 70 s vs 73 s).
  - **Latent Vecchia is my Python/numpy code.** It does batched small solves, a KD-tree neighbour search over the maxmin ordering, then a CHOLMOD factorisation of the posterior precision. Its wall-clock is pessimistic relative to its flops.

## Evidence (primary: `figures/frontiers_d1.png`, `figures/frontiers_d2.png`)

The frontiers plot error relative to the reference against flops for every configuration, with 3%, 5% and 10% lines. Figures `winners_d*.png` and the tables below summarise them.

**Banding vs latent Vecchia at matched flops: the informative cells** (`results/band_vs_vecchia.csv`). Matérn-3/2 in d = 1 is excluded from this comparison, because it is Markov in (f, f′), so its precision is almost exactly sparse and Vecchia is near-exact by construction.

| cell (β = 1) | band c=2.5: err / flops / mem / s | latent Vecchia: err / flops / mem / s | verdict |
|---|---|---|---|
| SE d1 n=1e4 | 0.55% / 2.0e8 / 1 MB / 0.10 | m10: 4.0% / 1.4e8 / 70 MB / 0.23; m30: 0.17% / 7.7e8 / 120 MB / 1.0 | **tie at 5%** (m10 fewer flops); **band wins at ≤ 3%** |
| SE d1 n=1e5 | 0.53% / 2.0e9 / 220 MB / 1.1 | m10: 4.1% / 1.0e9 / 440 MB / 2.4; m30: 0.15% / 5.5e9 / 490 MB / 10 | same |
| SE d1 n=1e6 | 0.54% / 2.3e10 / 2.3 GB / 22 | m30: 0.15% / 5.3e10 / 3.9 GB / 120 (needs a 600 s cap; see note) | **band** (2.3× fewer flops, half the memory) |
| SE d2 n=1e4 | 1.1% / 5.5e9 / 86 MB / 2.9 | m30: 9.1% / 8.5e9; m50: 5.5% / 2.0e10; m100: 2.5% / 5.5e10 / – / 33 | **band dominates** (≥ 10× fewer flops at equal error) |
| SE d2 n=1e5 | 1.1% / 1.9e11 / 1.1 GB / 69 | m30: 9.6% / 2.1e11 / 1.2 GB / 83 | **band dominates** |
| Matérn d2 n=1e4 | c=4: 7.2% / 1.7e10 / 196 MB / 10 | m30: 2.2% / 8.5e9 / 172 MB / 5.0 | **Vecchia dominates** |
| Matérn d2 n=1e5 | c=4: 8.4% / 7.4e11 / 2.8 GB / 284 | m30: 2.9% / 2.1e11 / 1.2 GB / 86 | **Vecchia dominates** |

**Notes on the table:**
- The latent Vecchia m = 50 and 100 entries for SE d = 2 come from the plateau diagnostic (n = 1e4, one rep, `results/diag_vecchia_plateau.csv`).
- **m = 30 at n = 1e6 is implementation-limited, not method-limited.** The main grid skipped it because my predicted nnz(AᵀA) bound of N(m+1)², about 9e8, exceeded the cap. Run without the prediction, it needs nnz(L) = 8.8e7 and 3.9 GB, and takes 120 s. That is right at the n ≥ 1e5 time cap, so its runs used a 600 s cap.
- **Plateau check.** The earlier "≈20% floor" for latent Vecchia was **a bug in my predecessor search**, not (a) jitter, (b) the latent formulation, or (c) a genuine limit. The code searched only the prefix before each block, so it missed within-block predecessors. After the fix:
  - error falls with m: d = 1 gives 4.2% (m = 10), 0.17% (m = 30), 0.015% (m = 50);
  - jitter from 1e-10 to 1e-4 changes the error by at most about 10% of itself;
  - median condition numbers stay below 1e6.
  - In SE d = 2 convergence in m is genuinely slow. That fits Stein's screening conditions, which require algebraic spectral decay and which SE fails.

**Winner tables (cheapest eligible configuration; full tables in `results/winners.md`).**

*Full posterior* (exact, banding, latent Vecchia, SGPR, PCG):

| | β = 0 (infill) | β = 0.5 | β = 1, SE | β = 1, Matérn-3/2 |
|---|---|---|---|---|
| d=1 | SGPR M=100 (SE); latent m10 (Mat.) at 1e4. Banding infeasible from 1e5 | same | **band c=2.5** by wall-clock at all n. By flops: latent m10 at 1e4–1e5, band at 1e6 | latent Vecchia m10–m30. Banding needs c=4 (3.3%, rep 0 only) and is dominated |
| d=2 | SGPR M=1000 (SE); latent m30 (Mat.) at 1e4. Banding infeasible from 1e5 | SGPR M=1000 at 1e4 | **band c=2.5** at 1e4 and 1e5 (wall-clock and flops) | latent Vecchia m30 at 1e4 and 1e5 |

*Pointwise prediction* (NN-Vecchia prediction allowed):
- **Changes vs the full-posterior table:**
  - SE, d = 1, β = 1: NN-Vecchia at m = 50 wins at every n. It has 2.0% error and 1.7e8 flops in 0.26–0.56 s, against 2.0e8–2.3e10 flops and 0.1–22 s for banding.
  - Matérn, d = 1, β = 1: NN-Vecchia at m = 30 wins, with 0.6–0.9% error and 3.6e7 flops.
- **In d = 2 the NN predictor converges slowly** (6.3% at m = 100), so banding still wins SE β = 1 and latent Vecchia wins Matérn.

**Threshold sensitivity** (`results/threshold_sensitivity.csv`, winners by wall-clock):

| tolerance | full-posterior verdicts | pointwise verdicts |
|---|---|---|
| **3%** | none flip; by flops, SE d1 β=1 moves from latent m10 to band c=2.5 | none flip |
| **10%** | SE d1 β=1, n=1e5: band c=2.5 → c=2.0 (same family); no family flips (Matérn d2 band c=4 becomes eligible at 8.4% but stays dominated by Vecchia) | SE d1 β=1, n=1e4: band → NN-Vecchia m30; SE d2 β=1, n=1e5: band → NN-Vecchia m100; Matérn d2 β=1 (both n): latent → NN-Vecchia m30 |

So **banding's wins against the pointwise NN predictor in SE d = 2 hold at 3–5% and are lost at 10%**. Its wins against latent Vecchia in SE d = 2 hold at all three tolerances.

## Validity threshold and other observations

- **c\* and nnz/row are flat in n at β = 1.** In d = 1 that is SE 2.0 (41) and Matérn 2.5 (51). In d = 2 it is SE 2.0 (120–125) and Matérn 3.0 (261). Matérn d = 2 at n = 1e5 is only valid at c = 3 in 130 s, just over the 120 s cap.
- **In infill, c\* holds but nnz/row grows like ρℓ^d:** 870–7,500 at n = 1e4.
- **Matérn tails force a larger c\*.** At c = 2.5–3 the banded Matérn posterior is valid but inaccurate: 14–40% error and variance ratio about 0.8. Validity and accuracy thresholds diverge for heavy-tailed kernels, as the paper's neighbour counts suggest.
- **Exact-system PCG with matrix-free matvecs was infeasible on CPU** without KeOps or a GPU wherever the kernel is not compactly truncatable (n ≥ 1e5 at β ≤ 0.5). It never won.
- **Not done from the original spec:** β ∈ {0.25, 0.75}; α = 1.5; the (neighbours) × (rank) ≈ n heuristic plot; 10 reps.
- **Issues found and fixed during the study, and documented in commits:**
  - subnormal floats slowing LAPACK about 35×, fixed by setting the CPU's flush-to-zero flags;
  - the latent-Vecchia neighbour search;
  - the response-model Vecchia plug-in predictor, which is unstable for smooth kernels and was replaced.

**Reproduce:** `python -m bandgp run && python -m bandgp extra refs && python -m bandgp extra nnpred && python -m bandgp extra metis && for p in d1small d1big d2 d2b1; do python -m bandgp extra latent --part $p; done && python -m bandgp extra latent_big && python -m bandgp extra plateau && python -m bandgp analyse`. Seeds depend only on (d, α, β, n, rep).
