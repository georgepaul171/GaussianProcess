# Annotated bibliography: banded GP approximations beyond infill

*Searched 2026-11-17 (web search; arxiv.org full texts were not reachable from this session, so annotations rest on abstracts and search-indexed text plus prior knowledge. Items marked ‡ need a check of the full text).*

**Bottom line.** I found **no work that analyses banded or radius-thresholded GP posteriors under increasing-domain or mixed asymptotics**. Three strands come close:
- tapering theory under increasing domain, which is frequentist, about the tapered (PSD) kernel rather than hard banding, and has no contraction rates;
- Bayesian contraction under increasing domain, but only for point-process intensities;
- spatially partitioned (local) GPs, on a fixed domain.

The direction looks open. The nearest threat is the increasing-domain program of Giordano and coauthors (see (a)).

## (a) Contraction / minimax rates on expanding or mixed domains
- **Giordano, Kirichenko & Rousseau**, *Nonparametric Bayesian intensity estimation for covariate-driven inhomogeneous point processes* (arXiv 2312.14073). Optimal posterior contraction for GP, mixture and Pólya-tree priors as the observation window grows, under ergodic covariates. **Closest existing "increasing-domain contraction" machinery**, but it is for intensities, not regression, and has no approximations. ‡
- **Dolmeta & Giordano** (arXiv 2605.10225, May 2026). Minimax contraction for rescaled Gaussian and Besov–Laplace priors in increasing-domain asymptotics. It mentions a companion "general program" for such rates ‡. **Watch this group**: extending their program to GP regression plus banding is the natural next step for them too.
- **Kurisu** (arXiv 2005.06371), **Kurisu & Matsuda** (2211.13467, 2402.02773). Frequentist kernel, local-polynomial and series-ridge regression under Lahiri's stochastic sampling, covering *pure and mixed* increasing domain (n/A_n^d → ∞). Rates depend on the volume of the sampling region, not on n. Useful template for a mixed regime parameterised by β.
- **Bachoc** (JMVA 2014; Bernoulli 2018, arXiv 1412.1926; tutorial 2009.07002). Increasing-domain asymptotics for covariance-parameter ML/CV and plug-in prediction. Background on why increasing domain is the "easy" regime for estimating parameters.
- *Contrast (fixed domain / bounded design):* Li, Sun & Zhu (arXiv 2207.10239, JASA 2024) cover fixed-domain contraction with a nugget. Nieman, Szabó & van Zanten (JMLR 2022, arXiv 2109.10755) show SVGP reaches minimax rates when M ≳ n^{d/(2α+d)}; this is the low-rank benchmark that banding competes against under infill.

## (b) Local / compactly supported / tapered / banded approximations
- **Furrer, Genton & Nychka** (JCGS 2006). Tapered kriging, fixed domain. **Kaufman, Schervish & Nychka** (JASA 2008) and **Du, Zhang & Mandrekar** (AoS 2009) cover tapered likelihood, the latter on a fixed domain. **Shaby & Ruppert** (JCGS 2012) cover tapered likelihood under increasing domain.
- **Furrer, Bachoc & Du**, *Asymptotic properties of multivariate tapering* (arXiv 1506.01833). **Increasing domain**: the tapered predictor's MSPE matches the untapered predictor's, and naive kriging variances keep the right rate. **This is the most relevant prior result.** It is frequentist, needs PSD tapers and fixed parameters, and gives no posterior contraction and no hard banding.
- **Stein** (JCGS 2013), *Statistical properties of covariance tapering*. Critical of tapering's efficiency; useful as a counterpoint.
- **Bevilacqua, Faouzi, Furrer & Porcu** (AoS 2019). Generalised Wendland compactly supported kernels, fixed domain.
- **Szabó, Hadji & van der Vaart** (JASA 2025, arXiv 2312.14130). Spatially partitioned GPs reach (adaptive) optimal contraction, unlike random splits. This is "locality" theory, but on a fixed domain.
- **Vecchia / screening.** Schäfer, Katzfuss & Owhadi (SISC 2021, arXiv 2004.14455) give the first rigorous Vecchia error bounds via exponential decay, but note that screening degrades with additive noise. Kang, Schäfer, Guinness & Katzfuss (arXiv 2401.15813) show Vecchia matches exact prediction and estimation on a fixed domain. **Szabó & Zhu** (arXiv 2410.10649, *AoS* submission) show minimax contraction for Vecchia GPs with oracle or hierarchical tuning. These are the right templates for the Vecchia competitor in our experiments.

## (c) Follow-ups to the two anchor papers
- **Kassi, Stankewitz & Szabó**, *Posterior contraction rates for banded approximations of GPs* (arXiv 2611.16482; NeurIPS 2026 poster "Theoretical guarantees for Banded Approximations of GPs"). Gives necessary and sufficient neighbour counts for SE and Matérn under the infill / oracle-rescaling regime. **It was posted about 5 days ago, and I found no follow-ups.**
- **Ehrhardt & Tobar**, *Accelerated training of GPs using banded SE covariances* (arXiv 2601.19007, ICASSP 2026). 1-D, SE kernel, bandwidth bound guaranteeing a PD Gram matrix. The only citing work found is **Cazelles & Tobar**, *Efficient GP learning via subspace projections* (arXiv 2601.16332), which cites it as related work and is not an extension.

**Flag for the project:** (1) Giordano et al.'s increasing-domain program and (2) Furrer–Bachoc–Du's increasing-domain tapering are the two places where someone could "already be doing this". Neither treats hard banding, validity thresholds, or the β-mixed regime.
