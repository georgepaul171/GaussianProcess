import math

import numpy as np
from scipy.optimize import minimize

from rffvb.gsvb import run_gsvb, _solve_mu, _solve_s


def _data(seed, n=80, ng=30, k=2, active=3, amp=3.0, rho=0.0):
    rng = np.random.default_rng(seed)
    X = rng.standard_normal((n, ng * k))
    if rho:
        X[:, 1::2] = rho * X[:, 0::2] + math.sqrt(1 - rho ** 2) * X[:, 1::2]
    th = np.zeros(ng * k)
    for g in rng.choice(ng, active, replace=False):
        th[g * k:(g + 1) * k] = amp * rng.choice([-1, 1], k)
    return X, X @ th + rng.standard_normal(n), th


def test_mu_and_s_solvers_match_scipy():
    rng = np.random.default_rng(1)
    for _ in range(20):
        A = rng.standard_normal((5, 2)); G = A.T @ A + 0.1 * np.eye(2); r = rng.standard_normal(2) * 3
        S, lam = rng.uniform(0.01, 1), rng.uniform(0.2, 3)
        f = lambda m: 0.5 * m @ G @ m - r @ m + lam * math.sqrt(m @ m + S)
        ref = minimize(f, np.zeros(2), method="BFGS", options=dict(gtol=1e-12)).x
        got = _solve_mu(G, r, np.zeros(2), S, lam)
        assert np.max(np.abs(got - ref)) < 1e-6
        mu2 = rng.uniform(0, 4)
        h = lambda ls: 0.5 * np.diag(G) @ np.exp(2 * ls) + lam * math.sqrt(mu2 + np.exp(2 * ls).sum()) - ls.sum()
        sref = np.exp(minimize(h, np.zeros(2), method="BFGS", options=dict(gtol=1e-12)).x)
        sgot = _solve_s(np.diag(G).copy(), mu2, np.ones(2), lam)
        assert np.max(np.abs(sgot - sref) / sref) < 1e-5


def test_elbo_lower_bound_monotone():
    for seed in range(4):
        X, Y, th = _data(seed, rho=0.7)
        res = run_gsvb(X, Y, np.full(30, 2), record_elbo=True, max_sweeps=200, eps=0)
        tr = np.array(res.elbo_trace)
        assert np.all(np.diff(tr) >= -1e-8 * np.maximum(1, np.abs(tr[1:]))), np.diff(tr).min()


def test_recovers_active_groups():
    hits = 0
    for seed in range(5):
        X, Y, th = _data(seed)
        res = run_gsvb(X, Y, np.full(30, 2))
        truth = np.abs(th).reshape(30, 2).sum(1) > 0
        hits += np.array_equal(res.gamma > 0.5, truth)
        assert np.max(np.abs(res.post_mean - th)) < 1.0
    assert hits >= 4
