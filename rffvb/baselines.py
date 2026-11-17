"""Gaussian baselines: exact GP (SE, ML-II hyperparameters), RFF Bayesian linear regression, and the
sparse spectrum GP of Lazaro-Gredilla et al. (2010) with optimised frequencies.
Each returns (mean of f at test points, sd of f at test points, info)."""
import time

import numpy as np
from scipy.linalg import cho_factor, cho_solve, solve_triangular
from scipy.optimize import minimize

from .data import rff


def exact_gp(X, y, Xt, seed=0):
    """SE kernel s^2 exp(-r^2 / (2 ell^2)) + noise, hyperparameters (s^2, ell, sigma^2) by marginal likelihood
    (sklearn, 3 optimiser restarts). Predictive mean/sd of the latent f (noise excluded)."""
    from sklearn.gaussian_process import GaussianProcessRegressor
    from sklearn.gaussian_process.kernels import RBF, ConstantKernel, WhiteKernel
    t0 = time.perf_counter()
    k = ConstantKernel(1.0, (1e-3, 1e3)) * RBF(0.2, (1e-3, 1e2)) + WhiteKernel(0.25, (1e-4, 1e1))
    gp = GaussianProcessRegressor(k, n_restarts_optimizer=3, random_state=seed, normalize_y=False).fit(X, y)
    s2 = gp.kernel_.k1.k1.constant_value; ell = gp.kernel_.k1.k2.length_scale; nz = gp.kernel_.k2.noise_level
    se = lambda A, B: s2 * np.exp(-0.5 * ((A[:, None, :] - B[None, :, :]) ** 2).sum(-1) / ell ** 2)
    L = cho_factor(se(X, X) + nz * np.eye(len(X)), lower=True)
    Ks = se(Xt, X)
    mu = Ks @ cho_solve(L, y)
    V = solve_triangular(L[0], Ks.T, lower=True)
    var = np.maximum(s2 - (V ** 2).sum(0), 1e-12)
    return mu, np.sqrt(var), dict(s2=float(s2), ell=float(ell), noise=float(nz), t=time.perf_counter() - t0)


def blr_posterior(Phi, y, prior_var, noise, Phit):
    """Bayesian linear regression with N(0, prior_var I) weights; returns predictive mean/sd of f."""
    m = Phi.shape[1]
    A = Phi.T @ Phi / noise + np.eye(m) / prior_var
    L = cho_factor(A, lower=True)
    w = cho_solve(L, Phi.T @ y / noise)
    V = solve_triangular(L[0], Phit.T, lower=True)
    return Phit @ w, np.sqrt((V ** 2).sum(0))


def rff_blr(X, y, Xt, W, s2, noise):
    """All p features, GP-equivalent prior N(0, 2 s^2 / p) on unit-amplitude features, ML-II s^2 and noise from the
    exact GP (so this targets the exact GP)."""
    t0 = time.perf_counter()
    Phi, Phit = rff(X, W), rff(Xt, W)
    p = Phi.shape[1]
    mu, sd = blr_posterior(Phi, y, 2.0 * s2 / p, noise, Phit)
    return mu, sd, dict(t=time.perf_counter() - t0)


def _ssgp_nlml(theta, X, y, m):
    """Negative log marginal likelihood of SSGP and its gradient. theta = [W (m x d) ravel, log s2, log noise].
    Model: f = sum_r a_r cos(w_r x) + b_r sin(w_r x), a, b ~ N(0, s2 / m)."""
    n, d = X.shape
    W = theta[:m * d].reshape(m, d); ls2, lnz = theta[m * d:]
    s2, nz = np.exp(ls2), np.exp(lnz)
    a = s2 / m
    Z = X @ W.T
    C, S = np.cos(Z), np.sin(Z)
    Phi = np.empty((n, 2 * m)); Phi[:, 0::2] = C; Phi[:, 1::2] = S
    A = Phi.T @ Phi + (nz / a) * np.eye(2 * m)
    L = cho_factor(A, lower=True)
    Pty = Phi.T @ y
    w = cho_solve(L, Pty)                      # posterior mean of weights
    r = y - Phi @ w
    alpha = r / nz                             # (K + nz I)^{-1} y with K = a Phi Phi'
    logdetA = 2 * np.log(np.diag(L[0])).sum()
    nlml = 0.5 * (y @ alpha) + 0.5 * (logdetA + (n - 2 * m) * lnz + 2 * m * np.log(a)) + 0.5 * n * np.log(2 * np.pi)
    # gradient: dnlml/dtheta = -0.5 tr((alpha alpha' - Kinv) dK/dtheta), K = a Phi Phi' + nz I
    Ainv = cho_solve(L, np.eye(2 * m))
    # Kinv Phi = (Phi - Phi Ainv Phi'Phi) / nz = Phi Ainv (nz/a) / nz = Phi Ainv / a
    KiPhi = Phi @ Ainv / a
    # d/dPhi of nlml: -(alpha alpha' Phi - Kinv Phi) a
    dPhi = -(np.outer(alpha, alpha @ Phi) - KiPhi) * a
    dC, dS = dPhi[:, 0::2], dPhi[:, 1::2]
    gW = ((-dC * S + dS * C).T @ X)            # d cos(z)/dw = -sin(z) x, d sin(z)/dw = cos(z) x
    # d/d log a (a = s2/m): dK = a Phi Phi'
    t_aPP = a * (np.sum(KiPhi * Phi) - np.sum((alpha @ Phi) ** 2))
    g_ls2 = 0.5 * t_aPP
    trKi = (n - np.sum(Phi * KiPhi) * a) / nz  # tr(Kinv) = (n - a tr(Phi' Kinv Phi)) / nz
    g_lnz = 0.5 * nz * (trKi - alpha @ alpha)
    return nlml, np.concatenate([gW.ravel(), [g_ls2, g_lnz]])


def ssgp(X, y, Xt, m, ell0, s2_0, noise0, seed=0, maxiter=500, restarts=10):
    """Lazaro-Gredilla et al. (2010): m frequencies (2m features) optimised jointly with s2 and noise by marginal
    likelihood (L-BFGS, analytic gradients). The objective is highly multimodal in the frequencies, so as in the
    original paper we use several random initialisations (frequencies from the SE spectral density at the exact-GP
    lengthscale; s2 and noise at their exact-GP values) and keep the best marginal likelihood."""
    t0 = time.perf_counter()
    n, d = X.shape
    rng = np.random.default_rng(seed)
    best = None
    bounds = [(None, None)] * (m * d) + [(np.log(1e-3), np.log(1e4)), (np.log(1e-4), np.log(10.0))]
    n_fail = 0
    for _ in range(restarts):
        W0 = rng.standard_normal((m, d)) / ell0
        th0 = np.concatenate([W0.ravel(), [np.log(s2_0), np.log(noise0)]])
        try:
            r = minimize(_ssgp_nlml, th0, args=(X, y, m), jac=True, method="L-BFGS-B", bounds=bounds,
                         options=dict(maxiter=maxiter))
        except np.linalg.LinAlgError:          # numerically non-PD system along the path: count as a failed restart
            n_fail += 1
            continue
        if best is None or r.fun < best.fun:
            best = r
    if best is None:
        raise RuntimeError("all SSGP restarts failed")
    res = best
    th = res.x
    W = th[:m * d].reshape(m, d); s2, nz = np.exp(th[m * d:])
    mu, sd = blr_posterior(rff(X, W), y, s2 / m, nz, rff(Xt, W))
    return mu, sd, dict(t=time.perf_counter() - t0, nit=res.nit, success=bool(res.success), noise=float(nz),
                        failed_restarts=n_fail)
