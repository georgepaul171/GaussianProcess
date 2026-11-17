"""Methods for the minimal decisive study. Each fills a Result: test/val means, variances at the first N_VAR test
points, phase timings, and implementation-independent cost (nnz used, approximate flops).

Conventions: unit-variance kernel, noise s2 = SIGMA^2. Cross-covariances k(X, x*) are exact (entries < TRUNC_TOL
dropped) for every method that uses them, as in Kassi et al. Approximate flop counts (order of magnitude,
consistent across methods):
  exact: n^3/3 + 2 n^2 (1 + N_VAR)        banded Cholesky: n u^2 + 2 n u (1 + N_VAR)   (u = half-bandwidth)
  CHOLMOD: sum_j colcount_j^2 + 2 nnz(L)(1 + N_VAR)
  PCG: n k^2 (pivoted Cholesky) + sum_cols iters * (2 nnz(K) + 4 n k)
  SGPR: n M^2 (or nnz(Kuf) M) + M^3 + (N_TEST+N_VAL+N_VAR) M^2      NN-Vecchia pred: (N_TEST+N_VAL) m^3/3
  latent Vecchia: N m^3/3 + sum_j colcount_j^2 + 2 nnz(L)(1 + N_VAR)
"""
import math
import time
from contextlib import contextmanager

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, solve_triangular, cholesky, lapack, cholesky_banded, cho_solve_banded
from scipy.spatial import cKDTree
import sksparse.cholmod as cholmod

from . import config as C
from .kernels import kfun, kfun_sq, dense_block, DenseKernelOp, r_trunc


class Skip(Exception):
    """Predicted infeasible under the memory/flop caps (recorded as 'infeasible')."""


class Invalid(Exception):
    """Invalid approximation (failed factorisation, non-PD system)."""


class Timeout(Exception):
    """Exceeded the per-method time cap."""


_STATE = {"deadline": None, "cap": None}


def check_deadline():
    if _STATE["deadline"] is not None and time.perf_counter() > _STATE["deadline"]:
        raise Timeout(f"> {_STATE['cap']}s")


class Result:
    def __init__(self):
        self.t = {"fit": 0.0, "mean": 0.0, "var": 0.0}
        self.mt = self.mv = self.vsub = None
        self.flops = 0.0
        self.info = {}

    @contextmanager
    def timer(self, phase):
        t0 = time.perf_counter()
        yield
        self.t[phase] += time.perf_counter() - t0


# ----------------------------------------------------------------------------------------------- shared context
class CrossCov:
    """Exact cross-covariance K(X, Xs): sparse (truncated at TRUNC_TOL) if affordable, else computed blockwise."""
    def __init__(self, ctx, Xs):
        self.ctx, self.Xs = ctx, Xs
        exp_nnz = len(Xs) * ctx.nnz_row(ctx.R_ex)
        self.sparse = exp_nnz <= C.NNZ_CAP / 4
        if self.sparse:
            a = ctx.tree.sparse_distance_matrix(cKDTree(Xs), ctx.R_ex, output_type="ndarray")
            self.M = sparse.csc_matrix((kfun(ctx.kernel, a["v"], ctx.ell), (a["i"], a["j"])), shape=(ctx.n, len(Xs)))
            del a

    def rmatvec(self, a):
        if self.sparse:
            return self.M.T @ a
        out = np.empty(len(self.Xs))
        for s in range(0, len(self.Xs), 64):
            check_deadline()
            out[s:s + 64] = dense_block(self.ctx.kernel, self.Xs[s:s + 64], self.ctx.X, self.ctx.ell) @ a
        return out

    def cols(self, idx):
        if self.sparse:
            return self.M[:, idx].toarray()
        return dense_block(self.ctx.kernel, self.ctx.X, self.Xs[idx], self.ctx.ell)


class Ctx:
    def __init__(self, data, kernel, ell, d, n, rho):
        self.X, self.y = data["X"], data["y"]
        self.Xt, self.Xv, self.L = data["Xt"], data["Xv"], data["L"]
        self.kernel, self.ell, self.d, self.n, self.rho = kernel, ell, d, n, rho
        self.s2 = C.SIGMA ** 2
        self.R_ex = r_trunc(kernel, ell, C.TRUNC_TOL)
        self.sub = np.arange(C.N_VAR)
        t0 = time.perf_counter()
        self.tree = cKDTree(self.X)
        self.t_tree = time.perf_counter() - t0
        self._xc = None
        self.t_xcov = 0.0

    def xcov(self):
        if self._xc is None:
            t0 = time.perf_counter()
            self._xc = (CrossCov(self, self.Xt), CrossCov(self, self.Xv))
            self.t_xcov = time.perf_counter() - t0
        return self._xc

    def nnz_row(self, r):
        return min(self.n, self.rho * C.ball_volume(self.d, r))


def sparse_kernel_chunked(ctx, r):
    """K restricted to pairs with distance <= r (incl. diagonal), CSR, built row-chunk-wise via
    cKDTree.sparse_distance_matrix."""
    n = ctx.n
    chunk = int(max(1000, min(n, 2e7 / max(ctx.nnz_row(r), 1))))
    indptr = [np.zeros(1, np.int64)]; indices = []; data = []; off = 0
    for s in range(0, n, chunk):
        check_deadline()
        a = cKDTree(ctx.X[s:s + chunk]).sparse_distance_matrix(ctx.tree, r, output_type="ndarray")
        m = sparse.csr_matrix((kfun(ctx.kernel, a["v"], ctx.ell), (a["i"], a["j"])), shape=(min(chunk, n - s), n))
        del a
        indices.append(m.indices.astype(np.int32)); data.append(m.data)
        indptr.append(m.indptr[1:].astype(np.int64) + off); off += m.nnz
    K = sparse.csr_matrix((np.concatenate(data), np.concatenate(indices), np.concatenate(indptr)), shape=(n, n))
    K.sort_indices()
    return K


def block_cg(matvec, B, minv=None, tol=C.CG_TOL, maxit=10 ** 6):
    """Independent (preconditioned) CG on each column of B, vectorised.
    Returns X, iterations, per-column iteration counts, converged flag, negative-curvature flag."""
    B = B.reshape(len(B), -1)
    X = np.zeros_like(B); R = B.copy()
    Z = minv(R) if minv else R.copy()
    P = Z.copy(); rz = np.einsum("ij,ij->j", R, Z)
    nb = np.linalg.norm(B, axis=0); nb[nb == 0] = 1
    active = np.ones(B.shape[1], bool); its = np.zeros(B.shape[1], int)
    for it in range(1, maxit + 1):
        check_deadline()
        Pa = P[:, active]
        AP = matvec(Pa)
        pap = np.einsum("ij,ij->j", Pa, AP)
        if np.any(pap <= 0):
            return X, it, its, False, True
        a = rz[active] / pap
        X[:, active] += a * Pa; R[:, active] -= a * AP
        its[active] += 1
        res = np.linalg.norm(R[:, active], axis=0) / nb[active]
        act_idx = np.flatnonzero(active)
        active[act_idx[res < tol]] = False
        if not active.any():
            return X, it, its, True, False
        Ra = R[:, active]
        Za = minv(Ra) if minv else Ra
        rz_new = np.einsum("ij,ij->j", Ra, Za)
        P[:, active] = Za + (rz_new / rz[active]) * P[:, active]
        rz[active] = rz_new
    return X, maxit, its, False, False


def var_sub(ctx, xt, solve_cols, bs=50):
    """v_i = 1 - k_i^T A^{-1} k_i on the N_VAR subset, via W with ||W[:, j]||^2 = k_j^T A^{-1} k_j."""
    out = np.empty(len(ctx.sub))
    for s in range(0, len(ctx.sub), bs):
        check_deadline()
        idx = ctx.sub[s:s + bs]
        W = solve_cols(xt.cols(idx))
        out[s:s + len(idx)] = 1.0 - np.einsum("ij,ij->j", W, W)
    return out


# ----------------------------------------------------------------------------------------------- exact
def exact(ctx, res):
    if ctx.n > C.EXACT_MAX_N:
        raise Skip("exact only for n <= EXACT_MAX_N")
    xt, xv = ctx.xcov()
    n = ctx.n
    with res.timer("fit"):
        K = dense_block(ctx.kernel, ctx.X, ctx.X, ctx.ell)
        K[np.diag_indices_from(K)] += ctx.s2
        Lc = cholesky(K, lower=True, overwrite_a=True, check_finite=False)
        a = cho_solve((Lc, True), ctx.y, check_finite=False)
    with res.timer("mean"):
        res.mt, res.mv = xt.rmatvec(a), xv.rmatvec(a)
    with res.timer("var"):
        res.vsub = var_sub(ctx, xt, lambda Kb: solve_triangular(Lc, Kb, lower=True, check_finite=False))
    res.info["nnz_used"] = n * (n + 1) / 2
    res.flops = n ** 3 / 3 + 2 * n ** 2 * (1 + C.N_VAR)


# ----------------------------------------------------------------------------------------------- banding
def band_banded(ctx, res, c):
    """d=1: sort inputs, LAPACK lower-band storage of K*1{|x-x'|<=c ell} + s2 I, banded Cholesky
    (cholesky_banded = the factorisation inside solveh_banded, kept for reuse), dtbtrs for variances."""
    r = c * ctx.ell
    n = ctx.n
    u_pred = ctx.nnz_row(r) / 2
    if (u_pred + 1) * n > C.NNZ_CAP:
        raise Skip(f"band storage {(u_pred + 1) * n:.1e} > cap")
    xt, xv = ctx.xcov()
    with res.timer("fit"):
        order = np.argsort(ctx.X[:, 0]); xs = ctx.X[order, 0]
        u = int(np.max(np.searchsorted(xs, xs + r, side="right") - np.arange(n) - 1))
        if (u + 1) * n > C.NNZ_CAP:
            raise Skip("band storage cap")
        ab = np.zeros((u + 1, n))
        ab[0] = 1.0 + ctx.s2
        for o in range(1, u + 1):
            if o % 200 == 0:
                check_deadline()
            dist = xs[o:] - xs[:-o]
            ab[o, :n - o] = np.where(dist <= r, kfun(ctx.kernel, dist, ctx.ell), 0.0)
        res.info["bandwidth"] = u
        res.info["nnz_row"] = float(1 + 2 * np.count_nonzero(ab[1:]) / n)
        try:
            cb = cholesky_banded(ab, lower=True, overwrite_ab=True, check_finite=False)
        except np.linalg.LinAlgError as e:
            raise Invalid(f"banded cholesky: {e}")
        a = np.empty(n); a[order] = cho_solve_banded((cb, True), ctx.y[order], check_finite=False)
    with res.timer("mean"):
        res.mt, res.mv = xt.rmatvec(a), xv.rmatvec(a)
    with res.timer("var"):
        res.vsub = var_sub(ctx, xt, lambda Kb: lapack.dtbtrs(cb, np.asfortranarray(Kb[order]), uplo="L")[0])
    res.info["nnz_used"] = n * (u + 1)
    res.flops = n * u ** 2 + 2 * n * u * (1 + C.N_VAR)


def band_chol(ctx, res, c):
    """d=2: sparse K*1{dist<=c ell} + s2 I via cKDTree.sparse_distance_matrix, CHOLMOD (supernodal, AMD/METIS)."""
    r = c * ctx.ell
    if ctx.nnz_row(r) * ctx.n > C.NNZ_CAP:
        raise Skip(f"nnz {ctx.nnz_row(r) * ctx.n:.1e} > cap")
    xt, xv = ctx.xcov()
    with res.timer("fit"):
        K = sparse_kernel_chunked(ctx, r)
        res.info["nnz_row"] = K.nnz / ctx.n
        A = (K + sparse.identity(ctx.n, format="csr") * ctx.s2).tocsc()
        del K
        try:
            F = cholmod.cho_factor(A, lower=True)
        except cholmod.CholmodNotPositiveDefiniteError as e:
            raise Invalid(f"cholmod: {e}")
        del A
        a = F.solve(ctx.y)
    with res.timer("mean"):
        res.mt, res.mv = xt.rmatvec(a), xv.rmatvec(a)
    perm = F.perm
    with res.timer("var"):
        res.vsub = var_sub(ctx, xt, lambda Kb: F.solve(Kb[perm], system="L"))
    cc = np.asarray(F.colcount, float)
    res.info["nnz_used"] = float(F.nnz)
    res.flops = float(np.sum(cc ** 2)) + 2 * F.nnz * (1 + C.N_VAR)


# ----------------------------------------------------------------------------------------------- PCG on exact system
def pivoted_cholesky(ctx, k, stop_maxdiag=None):
    """Greedy pivoted Cholesky of K with exact kernel columns. Stops at numerical rank (residual diag <= 1e-10)
    or when the max residual diagonal falls below stop_maxdiag. Returns (Lt [rank x n], rank, max residual)."""
    n = ctx.n
    Lt = np.empty((k, n)); dg = np.ones(n)
    for i in range(k):
        if i % 25 == 0:
            check_deadline()
        j = int(np.argmax(dg)); dj = dg[j]
        if (stop_maxdiag is not None and dj <= stop_maxdiag) or dj <= 1e-10:
            return Lt[:i], i, float(dj)
        col = dense_block(ctx.kernel, ctx.X, ctx.X[j:j + 1], ctx.ell)[:, 0]
        if i:
            col -= Lt[:i].T @ Lt[:i, j]
        Lt[i] = col / math.sqrt(dj)
        dg -= Lt[i] ** 2
        dg[j] = 0.0
    return Lt, k, float(dg.max())


def pcg(ctx, res, rank):
    n = ctx.n
    if n * rank > C.PIVCHOL_CAP:
        raise Skip("pivoted-Cholesky memory cap")
    nnz_ex = ctx.nnz_row(ctx.R_ex) * n
    if nnz_ex > C.NNZ_CAP and float(n) ** 2 > C.DENSE_MATVEC_CAP:
        raise Skip("exact matvec too expensive (no KeOps/GPU)")
    xt, xv = ctx.xcov()
    with res.timer("fit"):
        if nnz_ex <= C.NNZ_CAP:
            K = sparse_kernel_chunked(ctx, ctx.R_ex)
            A = K + sparse.identity(n, format="csr") * ctx.s2
            del K
            mv = lambda V: A @ V
            nnz_mv = A.nnz; res.info["matvec"] = "sparse_trunc"
        else:
            op = DenseKernelOp(ctx.kernel, ctx.X, ctx.ell)
            mv = lambda V: op @ V + ctx.s2 * V
            nnz_mv = n * n; res.info["matvec"] = "dense_blocked"
        Lt, k, _ = pivoted_cholesky(ctx, rank)
        res.info["rank_used"] = k
        Cw = cho_factor(np.eye(k) * ctx.s2 + Lt @ Lt.T)
        minv = lambda R: (R - Lt.T @ cho_solve(Cw, Lt @ R)) / ctx.s2
        a, it, its, conv, neg = block_cg(mv, ctx.y[:, None], minv, maxit=C.PCG_MAXIT)
        res.info["cg_iters"] = it
        if neg:
            raise Invalid("PCG: negative curvature")
        res.info["cg_converged"] = bool(conv)     # not converged in PCG_MAXIT: still evaluated, flagged
        a = a[:, 0]
    with res.timer("mean"):
        res.mt, res.mv = xt.rmatvec(a), xv.rmatvec(a)
    with res.timer("var"):
        Kb = xt.cols(ctx.sub)
        W, it2, its2, conv2, _ = block_cg(mv, Kb, minv, maxit=C.PCG_MAXIT)
        res.vsub = 1.0 - np.einsum("ij,ij->j", Kb, W)
        res.info["cg_iters_var_mean"] = float(its2.mean())
    res.info["nnz_used"] = float(nnz_mv + n * k)
    res.flops = n * k ** 2 + (it + its2.sum()) * (2 * nnz_mv + 4 * n * k)


# ----------------------------------------------------------------------------------------------- SGPR (Titsias)
def inducing_grid(L, d, M):
    if d == 1:
        return ((np.arange(M) + 0.5) * L / M)[:, None], M
    m = int(round(math.sqrt(M)))
    g = (np.arange(m) + 0.5) * L / m
    return np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2), m * m


def sgpr(ctx, res, M):
    """Titsias collapsed variational (SGPR / Nystrom-DTC predictive) with inducing points on a regular grid."""
    n = ctx.n
    if M > n / 2:
        raise Skip("M > n/2")
    if n * M * M > C.SGPR_FLOP_CAP:
        raise Skip(f"n M^2 = {n * M * M:.1e} > cap")
    with res.timer("fit"):
        Z, M = inducing_grid(ctx.L, ctx.d, M)
        Kuu = dense_block(ctx.kernel, Z, Z, ctx.ell)
        jit = 1e-8
        while True:
            try:
                Lu = cholesky(Kuu + jit * np.eye(M), lower=True); break
            except np.linalg.LinAlgError:
                jit *= 10
        res.info["jitter"] = jit
        if ctx.nnz_row(ctx.R_ex) / n < 0.1:            # kernel local relative to the domain: sparse Kuf
            a = cKDTree(Z).sparse_distance_matrix(ctx.tree, ctx.R_ex, output_type="ndarray")
            Kuf = sparse.csr_matrix((kfun(ctx.kernel, a["v"], ctx.ell), (a["i"], a["j"])), shape=(M, n))
            del a
            KK = (Kuf @ Kuf.T).toarray(); Ky = Kuf @ ctx.y
            nnz_kuf = Kuf.nnz
        else:
            KK = np.zeros((M, M)); Ky = np.zeros(M)
            bs = max(500, int(1e7 // M))
            for s in range(0, n, bs):
                check_deadline()
                Kb = dense_block(ctx.kernel, Z, ctx.X[s:s + bs], ctx.ell)
                KK += Kb @ Kb.T; Ky += Kb @ ctx.y[s:s + bs]
            nnz_kuf = n * M
        check_deadline()
        AAt = solve_triangular(Lu, solve_triangular(Lu, KK, lower=True).T, lower=True).T / ctx.s2
        LB = cholesky(np.eye(M) + 0.5 * (AAt + AAt.T), lower=True)
        cvec = solve_triangular(LB, solve_triangular(Lu, Ky, lower=True), lower=True) / ctx.s2

    def predict(Xs, want_var):
        Kus = dense_block(ctx.kernel, Z, Xs, ctx.ell)
        t1 = solve_triangular(Lu, Kus, lower=True)
        t2 = solve_triangular(LB, t1, lower=True)
        return t2.T @ cvec, (1.0 - np.sum(t1 ** 2, 0) + np.sum(t2 ** 2, 0)) if want_var else None
    with res.timer("mean"):
        res.mt = predict(ctx.Xt, False)[0]; res.mv = predict(ctx.Xv, False)[0]
    with res.timer("var"):
        res.vsub = predict(ctx.Xt[ctx.sub], True)[1]
    res.info["M"] = M
    res.info["nnz_used"] = float(nnz_kuf + M * M)
    res.flops = float(min(nnz_kuf * M, n * M * M) + M ** 3 + (C.N_TEST + C.N_VAL + C.N_VAR) * M * M)


# ----------------------------------------------------------------------------------------------- Vecchia
def approx_maxmin(X, L, rng):
    """Approximate maxmin ordering: coarse-to-fine dyadic grids, one random point per newly occupied cell per level,
    remaining points last in random order (matched exact-maxmin accuracy in our checks)."""
    n, d = X.shape
    key = rng.random(n)
    assigned = np.zeros(n, bool)
    parts = []; s = 0
    while (2 ** s) ** d < n / 2:
        cps = 2 ** s
        idx = np.flatnonzero(~assigned)
        cell = np.minimum((X[idx] / L * cps).astype(np.int64), cps - 1)
        cid = np.ravel_multi_index(cell.T, (cps,) * d)
        o = np.lexsort((key[idx], cid))
        first = np.r_[True, cid[o][1:] != cid[o][:-1]]
        ch = idx[o[first]]
        parts.append(ch[np.argsort(key[ch])]); assigned[ch] = True; s += 1
    rest = np.flatnonzero(~assigned)
    parts.append(rest[np.argsort(key[rest])])
    return np.concatenate(parts)


def _cond(ctx, Xi, XN, nugget):
    """b = (K_NN + nugget I)^{-1} k_N(x_i) and conditional variance, batched over rows."""
    mm = XN.shape[1]
    diff = XN[:, :, None, :] - XN[:, None, :, :]
    KNN = kfun_sq(ctx.kernel, np.sum(diff * diff, -1), ctx.ell)
    KNN[:, np.arange(mm), np.arange(mm)] += nugget
    kN = kfun_sq(ctx.kernel, np.sum((XN - Xi[:, None, :]) ** 2, -1), ctx.ell)
    b = np.linalg.solve(KNN, kN[..., None])[..., 0]
    dv = 1 + nugget - np.einsum("ij,ij->i", kN, b)
    return b, kN, dv


def vecchia_latent(ctx, res, m):
    """Latent-field Vecchia, prediction points ordered last (Katzfuss, Guinness, Gong & Zilber 2020): approximate
    maxmin ordering of training inputs, then test, then validation points; each point conditions on its m nearest
    predecessors (searched in a prefix holding >= 2/3 of them). Posterior precision A^T A + diag(1/s2 on observed)
    is factorised by CHOLMOD. Latent nugget 1e-6 keeps SE conditioning blocks numerically PD."""
    n, nt, nv = ctx.n, len(ctx.Xt), len(ctx.Xv)
    N = n + nt + nv
    if N * m * m > C.NNZ_CAP:
        raise Skip("nnz(A^T A) cap")
    nug = 1e-6
    with res.timer("fit"):
        order = approx_maxmin(ctx.X, ctx.L, np.random.default_rng(12345))
        Z = np.vstack([ctx.X[order], ctx.Xt, ctx.Xv])
        rows, cols, vals = [], [], []
        p0 = min(N, 4 * m + 1)
        rows.append(np.array([0])); cols.append(np.array([0])); vals.append(np.array([1 / math.sqrt(1 + nug)]))
        for p in range(1, p0):
            nb = np.argsort(np.sum((Z[:p] - Z[p]) ** 2, 1))[:m]
            b, _, dv = _cond(ctx, Z[p:p + 1], Z[nb][None], nug)
            sd = 1 / math.sqrt(dv[0])
            rows.append(np.full(len(nb) + 1, p)); cols.append(np.r_[p, nb]); vals.append(np.r_[sd, -b[0] * sd])
        a = p0
        while a < N:
            bnd = min(N, int(math.ceil(a * 1.5)))
            tp = cKDTree(Z[:a])
            for s in range(a, bnd, 20000):
                check_deadline()
                e = min(bnd, s + 20000)
                nb = tp.query(Z[s:e], k=m)[1].reshape(e - s, m)
                b, _, dv = _cond(ctx, Z[s:e], Z[nb], nug)
                if np.any(dv <= 0):
                    raise Invalid("non-positive conditional variance")
                sd = 1 / np.sqrt(dv)
                pi = np.arange(s, e)
                rows.append(np.repeat(pi, m + 1)); cols.append(np.c_[pi, nb].ravel())
                vals.append(np.c_[sd, -b * sd[:, None]].ravel())
            a = bnd
        A = sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))
        dobs = np.zeros(N); dobs[:n] = 1 / ctx.s2
        Q = (A.T @ A + sparse.diags(dobs)).tocsc()
        del A
        check_deadline()
        F = cholmod.cho_factor(Q, lower=True)
        del Q
        rhs = np.zeros(N); rhs[:n] = ctx.y[order] / ctx.s2
    with res.timer("mean"):
        mu = F.solve(rhs)
        res.mt, res.mv = mu[n:n + nt], mu[n + nt:]
    inv = np.empty(N, np.int64); inv[F.perm] = np.arange(N)
    with res.timer("var"):
        idx = n + ctx.sub
        E = np.zeros((N, len(idx))); E[inv[idx], np.arange(len(idx))] = 1.0
        W = F.solve(E, system="L")
        res.vsub = np.einsum("ij,ij->j", W, W)
    cc = np.asarray(F.colcount, float)
    res.info["nnz_used"] = float(F.nnz)
    res.flops = N * m ** 3 / 3 + float(np.sum(cc ** 2)) + 2 * F.nnz * (1 + C.N_VAR)


def vecchia_pred(ctx, res, m):
    """NN-Vecchia / NNGP-response prediction: each prediction point conditions on its m nearest observations
    (prediction points ordered last; no global fit needed with known hyperparameters)."""
    def pred(Xs, want_var):
        check_deadline()
        nb = ctx.tree.query(Xs, k=m)[1].reshape(len(Xs), m)
        b, kN, dv = _cond(ctx, Xs, ctx.X[nb], ctx.s2)
        mu = np.einsum("ij,ij->i", b, ctx.y[nb])
        return mu, (1.0 - np.einsum("ij,ij->i", kN, b)) if want_var else None
    with res.timer("mean"):
        res.mt = pred(ctx.Xt, False)[0]; res.mv = pred(ctx.Xv, False)[0]
    with res.timer("var"):
        res.vsub = pred(ctx.Xt[ctx.sub], True)[1]
    res.info["nnz_used"] = float((C.N_TEST + C.N_VAL) * m)
    res.flops = (C.N_TEST + C.N_VAL) * m ** 3 / 3


# ----------------------------------------------------------------------------------------------- registry
def method_list(d, band_cs):
    """band_cs: radii to run (full sweep on rep 0; c* and next value up on later reps)."""
    L = [("exact", "exact", {}, exact, {})]
    for c in band_cs:
        if d == 1:
            L.append(("band", f"band_c{c}", {"c": c}, band_banded, {"c": c}))
        else:
            L.append(("band", f"band_c{c}", {"c": c}, band_chol, {"c": c}))
    for k in C.PCG_RANKS:
        L.append(("pcg", f"pcg_r{k}", {"rank": k}, pcg, {"rank": k}))
    for M in C.SGPR_MS:
        L.append(("sgpr", f"sgpr_M{M}", {"M": M}, sgpr, {"M": M}))
    for m in C.VECCHIA_MS:
        L.append(("vecchia_pred", f"vecchia_pred_m{m}", {"m": m}, vecchia_pred, {"m": m}))
        L.append(("vecchia_latent", f"vecchia_latent_m{m}", {"m": m}, vecchia_latent, {"m": m}))
    return L
