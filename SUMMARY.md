# Spike-and-slab VB on random Fourier features as a GP approximation

*n = 1000, σ = 0.5, x ~ U[0,1]^d, 2000 test points, 10 reps; p ∈ {500, 2000, 5000}; single-thread timings.*
*Code: `python -m rffvb run | claim3 | analyse`. Individual VB is `svb.core.run_cavi` from 2027-research @ e275fa0 (read-only), with DEVIATIONS.md defaults and fitted on X/σ, Y/σ.*

**Truths.**
- **Spectrally sparse:** 3–5 sinusoids at 1–8 cycles per unit.
- **Generic smooth:** a Matérn-5/2 draw, ℓ = 0.2 (d=1) or 0.3 (d=2).

**Methods.**
- **Exact GP:** SE kernel, ML-II hyperparameters.
- **RFF-BLR:** all p features, GP-equivalent prior.
- **SSGP:** Lázaro-Gredilla 2010, analytic gradients, 10 restarts, as many features as individual VB selects.
- **Individual SnS-VB.**
- **Group SnS-VB:** each cos/sin pair is a group (GSVB-style multivariate-Laplace slab; my numba implementation, tested).

**Frequency designs.**
- **iid:** draws from the SE spectrum.
- **Min-separation:** a randomly shifted lattice mapped through the SE spectrum, folded so ω₁ ≥ 0.
- A spacing of about 2π / domain width is infeasible under the SE spectrum at these p.

`run_block` is dropped: it segfaults on near-collinear RFF clusters. See `BUGS_2027-research.md`, which includes a reproducer.

## Results (`results/summary.csv`, `figures/rmse_vs_selected.png`, `figures/coverage_by_method.png`)

**Generic smooth truth: spike-and-slab VB is a worse GP approximation than plain RFF.**

| d=1 & d=2, Matérn truth | test RMSE / exact-GP RMSE | coverage of f0 (exact GP: 0.86–0.89) |
|---|---|---|
| individual VB | 1.2–1.45 | 0.57–0.87 |
| group VB | 1.1–1.3 | 0.78–0.97 |
| RFF-BLR | 1.00–1.08 | 0.81–0.89 |

- Individual VB selects 12–25 features. Group VB selects 25–66 features (13–33 pairs).

**Spectrally sparse truth: selection pays off.**
- **d = 2, p ≥ 2000:** group and individual VB reach **0.57–0.78 × the exact-GP RMSE** with 15–33 features.
  - Coverage of f0 is still only 0.53–0.65, against 0.96 for the exact GP.
- **d = 1:** group VB matches or beats the exact GP (0.84–1.09×) and covers 0.92–0.97.
  - Individual VB is 1.1–1.2× and covers 0.80–0.89.
- **SSGP** is the best method on 1D sparse truths (0.74–0.80× at p ≥ 2000). It fails in 2D (1.15–2.1×) because its frequency optimisation lands in poor local optima despite restarts.

**Group vs individual.**
- Treating each cos/sin pair as a group is better for coverage on every cell. For RMSE it is better on every cell except one tie (1D Matérn, p = 5000, lattice).
- It selects about 2× more features, the same number of frequencies.

**Design effect is small.** The lattice cuts 2D collinearity (columns with max |corr| > 0.99 fall from 25% to 0.8% at p = 5000). Yet RMSE and coverage change by at most about 0.1 in the RMSE ratio and 0.05 in coverage, with no consistent sign. In 1D, collinearity can't be reduced (max |corr| ≈ 1 for both designs).

**Convergence and cost.**
- Individual CAVI hits the 500-sweep cap in 17% of fits; group in 7%.
- VB takes 0.3–17 s per fit, and the exact GP about 20 s (with ML-II). So at n = 1000 there is no speed case.

## Claim 3 (2027-research) on RFF designs (`results/claim3_*.csv`, `figures/claim3_*.png`)

**Reference: the exact posterior of the same spike-and-slab model.** This uses `svb.mcmc` with E1f conventions on the subset p = 500 × d × truth × design × 3 reps (24 replicates).

**Full posterior: 0 of 24 replicates mixed.**
- Model-size split R-hat was 1.19–3.46, and the maximum between-chain PIP gap ≈ 1.0.
- Single-site Gibbs cannot move between near-duplicate features. So **the inclusion part, the mean-offset part, and the exact-SnS-vs-GP mismatch cannot be estimated with this sampler.**
- Blocked or swap moves within correlated feature clusters, or tempering, are needed (claims.md already lists this as missing for ρ = 0.95).

**Fixed-support part is valid on all 24 replicates.** The blocked fixed-support sampler mixes: two independent chains give R-hat 1.000 and variances within 8% (`results/fixed_support_mixing_check.txt`).

**Coefficient level: Claim 3 holds on RFF where the selected support is well-conditioned (d = 2) and breaks down where it is near-singular (d = 1).**
- **d = 2:** fixed-support exact variance / VB variance = VIF.
  - Median ratio/VIF is 0.95–1.00 and Spearman 0.95–1.00.
  - 2Φ(1.96/√VIF) − 1 matches the fixed-support coverage: 0.62 vs 0.62, 0.59 vs 0.60, 0.67 vs 0.67, 0.86 vs 0.86.
- **d = 1:** VIF reaches 10⁷–10¹⁰ on the selected supports.
  - The Laplace prior bounds the exact variance in near-null directions, so the ratio is only 0.34–0.53 × VIF (Spearman 0.43–0.95).
  - The closed form under-predicts coverage (0.24–0.42 predicted vs 0.33–0.46 actual). The fixed-support shortfall is still large (0.49–0.62).

**Function level: the fixed-support (VIF) part disappears.**
- Centred fixed-support coverage of f(x) is 0.94–0.98, and the function-level R(x) prediction is 0.94–0.96.
- **The cancellation explanation holds up when tested.**
  - On the selected support, Var(f(x)) under the exact fixed-support posterior is only 0.002–0.006 (1D sparse), 0.2–0.3 (1D Matérn) and 0.05–0.5 (2D) of Σ_j φ_j² Var(θ_j).
  - For column pairs with |corr| > 0.9, the exact posterior correlation has the opposite sign in **90–100%** of pairs (median sign-adjusted correlation −0.31 to −0.98).
  - Across all pairs the split is close to 50/50 (52–57% negative). So the cancellation is concentrated in the collinear pairs.

**Against the exact GP (model mismatch plus approximation, full grid).**
- At p = 2000, VB function bands cover GP function values with probability 0.25–0.98.
- The function-level VIF share of that shortfall is ≈ 0.
- The rest splits into band width (3–85%) and VB-mean offset. Without a mixed exact spike-and-slab posterior I can't tell how much of the width share is inclusion uncertainty and how much is prior/model mismatch.

## Literature (searched 2026-11-17; nothing found that already does this)

- **Closest:** Hadj-Amar, Krishnan & Vannucci (arXiv 2607.15157). A bivariate spike-and-slab on paired sin/cos coefficients over a frequency grid with minimum separation. It targets time-series spectra (MCMC), not RFF approximations of a GP kernel.
- **Group SnS-VB:** Komodromos, Evangelou, Filippi & Ray (arXiv 2309.10378; *Bayesian Analysis*). CAVI with variational contraction rates for grouped linear regression. These rates have not been transferred to RFF designs.
- **Sparse random features (frequentist):** Hashemi et al. (2023, ACHA), ℓ1 generalisation bounds. Variational sparse-spectrum GP: Gal & Turner (2015); Hoang, Hoang & Low (AAAI 2017). Neither uses a spike-and-slab.
- **Contraction theory for SnS on random features:** none found. Sparse-linear-regression SnS rates (Castillo, Schmidt-Hieber & van der Vaart 2015; Ray & Szabó 2022) need compatibility conditions that near-collinear RFF designs violate.
