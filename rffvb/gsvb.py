"""Group spike-and-slab variational Bayes (Komodromos, Evangelou, Filippi & Ray, arXiv 2309.10378), written to
match the conventions of svb.core.run_cavi (unit noise variance, a0/b0 Beta-Bernoulli weight, lambda).

Prior per group g of size m_g: theta_g = z_g * slab + (1 - z_g) * delta_0, z_g ~ Bernoulli(w), w = a0/(a0+b0),
slab density C_g exp(-lam ||theta_g||_2) with C_g = lam^m Gamma(m/2) / (2 pi^(m/2) Gamma(m))  (multivariate Laplace).
Variational family: q(theta_g) = gamma_g N(mu_g, diag(s_g^2)) + (1 - gamma_g) delta_0, independent over groups.
E||theta_g|| has no closed form; as in GSVB we use Jensen's bound E||theta_g|| <= sqrt(||mu_g||^2 + sum s_g^2),
so the objective maximised is a lower bound on the ELBO.

CAVI update of group g with r_g = b_g - sum_{h != g} G_gh E theta_h:
  mu_g    = argmin  0.5 mu' G_gg mu - r_g' mu + lam sqrt(||mu||^2 + S)                 (damped Newton, convex)
  s_g     = argmin  0.5 sum_j G_jj s_j^2 + lam sqrt(||mu||^2 + sum s_j^2) - sum log s_j (coordinatewise, convex)
  logit gamma_g = logit w + log C_g + m/2 log(2 pi e) + sum log s_j
                  - [0.5 (mu' G_gg mu + sum_j G_jj s_j^2) - r_g' mu + lam sqrt(||mu||^2 + sum s^2)]
"""
import math
import time
from dataclasses import dataclass

import numpy as np
from numba import njit
from scipy.special import gammaln


@njit(cache=True)
def _hb(g):
    out = 0.0
    if g > 0.0:
        out -= g * math.log(g)
    if g < 1.0:
        out -= (1.0 - g) * math.log(1.0 - g)
    return out


@njit(cache=True)
def _solve_s(Gd, mu2, s, lam):
    """Coordinatewise minimisation of 0.5 sum G_jj s_j^2 + lam sqrt(mu2 + sum s^2) - sum log s_j (convex)."""
    m = len(s)
    for _outer in range(50):
        smax = 0.0
        for j in range(m):
            A = mu2
            for k in range(m):
                if k != j:
                    A += s[k] * s[k]
            lo, hi = 1e-12, 1.0 / math.sqrt(Gd[j])          # h(s) increasing; h(hi) >= 0
            x = s[j] if lo < s[j] < hi else 0.5 * hi
            for _ in range(200):
                Nn = math.sqrt(A + x * x)
                h = Gd[j] * x + lam * x / Nn - 1.0 / x
                if h > 0:
                    hi = x
                else:
                    lo = x
                hp = Gd[j] + lam * A / (Nn * Nn * Nn) + 1.0 / (x * x)
                xn = x - h / hp
                if not (lo < xn < hi):
                    xn = 0.5 * (lo + hi)
                if abs(xn - x) <= 1e-14 * x:
                    x = xn
                    break
                x = xn
            smax = max(smax, abs(x - s[j]) / s[j])
            s[j] = x
        if smax < 1e-12:
            break
    return s


@njit(cache=True)
def _solve_mu(Ggg, r, mu, S, lam):
    """Damped Newton for 0.5 mu' G mu - r' mu + lam sqrt(||mu||^2 + S) (strictly convex for S > 0)."""
    m = len(r)
    for _ in range(100):
        Nn = math.sqrt(mu @ mu + S)
        g = Ggg @ mu - r + lam * mu / Nn
        H = Ggg + (lam / Nn) * np.eye(m) - (lam / (Nn ** 3)) * np.outer(mu, mu)
        step = np.linalg.solve(H, g)
        f0 = 0.5 * mu @ (Ggg @ mu) - r @ mu + lam * Nn
        t = 1.0
        while t > 1e-10:
            c = mu - t * step
            f1 = 0.5 * c @ (Ggg @ c) - r @ c + lam * math.sqrt(c @ c + S)
            if f1 <= f0 + 1e-15 * (1.0 + abs(f0)):
                break
            t *= 0.5
        mu = mu - t * step
        if np.max(np.abs(t * step)) < 1e-13 * (1.0 + np.max(np.abs(mu))):
            break
    return mu


@njit(cache=True)
def _sweep(G, b, starts, sizes, order, mu, s, gamma, u, lam, logit_w, logC):
    max_dh = 0.0
    for k in range(len(order)):
        g = order[k]
        a, m = starts[g], sizes[g]
        idx = np.arange(a, a + m)
        Ggg = np.empty((m, m))
        for i in range(m):
            for j in range(m):
                Ggg[i, j] = G[idx[i], idx[j]]
        mold = gamma[g] * mu[a:a + m]
        r = np.empty(m)
        for i in range(m):
            r[i] = b[idx[i]] - (u[idx[i]] - Ggg[i] @ mold)
        S = 0.0
        for i in range(m):
            S += s[a + i] * s[a + i]
        mug = _solve_mu(Ggg, r, mu[a:a + m].copy(), S, lam)
        sg = _solve_s(np.diag(Ggg).copy(), mug @ mug, s[a:a + m].copy(), lam)
        S = sg @ sg
        Nn = math.sqrt(mug @ mug + S)
        quad = 0.5 * (mug @ (Ggg @ mug))
        for i in range(m):
            quad += 0.5 * Ggg[i, i] * sg[i] * sg[i]
        L = logit_w + logC[g] + 0.5 * m * (math.log(2 * math.pi) + 1.0) - (quad - r @ mug + lam * Nn)
        for i in range(m):
            L += math.log(sg[i])
        gi = 1.0 / (1.0 + math.exp(-L)) if L >= 0 else math.exp(L) / (1.0 + math.exp(L))
        dh = abs(_hb(gi) - _hb(gamma[g]))
        if dh > max_dh:
            max_dh = dh
        mu[a:a + m] = mug
        s[a:a + m] = sg
        gamma[g] = gi
        dm = gi * mug - mold
        for i in range(m):
            if dm[i] != 0.0:
                for j in range(G.shape[0]):
                    u[j] += dm[i] * G[idx[i], j]
    return max_dh


def elbo_lb(Y, G, b, starts, sizes, mu, s, gamma, lam, logit_w, logC, a0, b0):
    """Jensen lower bound on the ELBO (up to -n/2 log 2 pi), for monotonicity checks."""
    gm = np.repeat(gamma, sizes)
    m = gm * mu
    Ed2 = gm * (mu ** 2 + s ** 2)
    quad = m @ G @ m - np.sum(np.diag(G) * m ** 2) + np.sum(np.diag(G) * Ed2)
    # within-group cross terms: E[theta_i theta_j] = gamma mu_i mu_j (not gamma^2 mu_i mu_j) for i != j in a group
    for g, (a, k) in enumerate(zip(starts, sizes)):
        Gg = G[a:a + k, a:a + k]; mg = mu[a:a + k]
        off = mg @ Gg @ mg - np.sum(np.diag(Gg) * mg ** 2)
        quad += (gamma[g] - gamma[g] ** 2) * off
    lik = -0.5 * Y @ Y + b @ m - 0.5 * quad
    w = a0 / (a0 + b0)
    val = lik
    for g, (a, k) in enumerate(zip(starts, sizes)):
        sg, mg = s[a:a + k], mu[a:a + k]
        Nn = math.sqrt(mg @ mg + sg @ sg)
        val += gamma[g] * (math.log(w) + logC[g] - lam * Nn + 0.5 * k * (math.log(2 * math.pi) + 1) + np.log(sg).sum())
        val += (1 - gamma[g]) * math.log1p(-w) + _hb(gamma[g])
    return float(val)


@dataclass
class GSVBResult:
    mu: np.ndarray
    s: np.ndarray
    gamma: np.ndarray        # per group
    sizes: np.ndarray
    n_sweeps: int
    converged: bool
    runtime: float
    elbo_trace: list

    @property
    def post_mean(self):
        return np.repeat(self.gamma, self.sizes) * self.mu


def run_gsvb(X, Y, sizes, lam=1.0, a0=1.0, b0=None, eps=1e-5, max_sweeps=500, param_tol=None, record_elbo=False):
    """Groups are consecutive column blocks of the given sizes. b0 defaults to the number of groups.
    Init (as run_cavi): ridge mu, s = 1, gamma = 0.5; groups updated in decreasing ||ridge_g|| order."""
    t0 = time.perf_counter()
    X = np.ascontiguousarray(X, float); Y = np.ascontiguousarray(Y, float)
    n, p = X.shape
    sizes = np.asarray(sizes, np.int64)
    starts = np.concatenate([[0], np.cumsum(sizes)[:-1]]).astype(np.int64)
    ng = len(sizes)
    if b0 is None:
        b0 = float(ng)
    logit_w = math.log(a0) - math.log(b0)
    logC = np.array([k * math.log(lam) + gammaln(k / 2) - math.log(2) - (k / 2) * math.log(math.pi) - gammaln(k)
                     for k in sizes])
    G = X.T @ X; b = X.T @ Y
    mu = X.T @ np.linalg.solve(X @ X.T + np.eye(n), Y) if p > n else np.linalg.solve(G + np.eye(p), b)
    s = np.ones(p); gamma = np.full(ng, 0.5)
    order = np.argsort([-np.linalg.norm(mu[a:a + k]) for a, k in zip(starts, sizes)], kind="stable").astype(np.int64)
    u = G @ (np.repeat(gamma, sizes) * mu)
    trace = []; conv = False; k = 0
    for k in range(1, max_sweeps + 1):
        mprev = np.repeat(gamma, sizes) * mu
        dh = _sweep(G, b, starts, sizes, order, mu, s, gamma, u, float(lam), logit_w, logC)
        u = G @ (np.repeat(gamma, sizes) * mu)
        if record_elbo:
            trace.append(elbo_lb(Y, G, b, starts, sizes, mu, s, gamma, lam, logit_w, logC, a0, b0))
        ok = param_tol is None or np.max(np.abs(np.repeat(gamma, sizes) * mu - mprev)) <= param_tol
        if dh <= eps and ok:
            conv = True
            break
    return GSVBResult(mu=mu, s=s, gamma=gamma, sizes=sizes, n_sweeps=k, converged=conv,
                      runtime=time.perf_counter() - t0, elbo_trace=trace)
