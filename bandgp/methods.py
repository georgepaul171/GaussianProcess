"""All approximation methods. Each method fills a Result with test/val means, test variances and timings.

Conventions: kernel has unit variance, noise variance s2 = SIGMA^2. Cross-covariances k(X, x*) are *exact*
(entries below TRUNC_TOL dropped) for every method that uses them, as in Kassi et al.
Timings are split into fit / mean / var phases; the shared cross-covariance build time is added by the runner
to every method that uses it.
"""
import math
import time
from contextlib import contextmanager

import numpy as np
from scipy import sparse
from scipy.linalg import cho_factor, cho_solve, solve_triangular, cholesky, lapack
from scipy.linalg import cholesky_banded, cho_solve_banded
from scipy.signal import fftconvolve
from scipy.sparse.linalg import eigsh, LinearOperator
from scipy.spatial import cKDTree
import sksparse.cholmod as cholmod

from . import config as C
from .kernels import kfun, kfun_sq, dense_block, DenseKernelOp, r_trunc


class Skip(Exception):
    """Configuration predicted infeasible under the caps; recorded, not run."""


class Invalid(Exception):
    """Approximation produced an invalid object (e.g. failed factorisation / indefinite system)."""


class Timeout(Exception):
    """Method exceeded the per-method time budget (TIME_CAP seconds, single thread)."""


_STATE = {"deadline": None}   # set by the runner before each method


def check_deadline(deadline=None):
    deadline = deadline if deadline is not None else _STATE["deadline"]
    if deadline is not None and time.perf_counter() > deadline:
        raise Timeout(f"> {C.TIME_CAP}s")


class Result:
    def __init__(self):
        self.t = {"fit": 0.0, "mean": 0.0, "var": 0.0}
        self.mt = self.mv = self.vt = self.vsub = None
        self.var_extrapolated = False
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
        exp_nnz = len(Xs) * min(ctx.n, ctx.rho * C.ball_volume(ctx.d, ctx.R_ex))
        self.sparse = exp_nnz <= C.NNZ_CAP / 4
        if self.sparse:
            ts = cKDTree(Xs)
            a = ctx.tree.sparse_distance_matrix(ts, ctx.R_ex, output_type="ndarray")
            self.M = sparse.csc_matrix((kfun(ctx.kernel, a["v"], ctx.ell), (a["i"], a["j"])), shape=(ctx.n, len(Xs)))
            del a

    def rmatvec(self, a):            # K(Xs, X) a
        if self.sparse:
            return self.M.T @ a
        out = np.empty(len(self.Xs))
        for s in range(0, len(self.Xs), 64):
            out[s:s + 64] = dense_block(self.ctx.kernel, self.Xs[s:s + 64], self.ctx.X, self.ctx.ell) @ a
        return out

    def cols(self, idx):             # dense n x len(idx)
        if self.sparse:
            return self.M[:, idx].toarray()
        return dense_block(self.ctx.kernel, self.ctx.X, self.Xs[idx], self.ctx.ell)

    def batches(self, idx=None, bs=None):
        idx = np.arange(len(self.Xs)) if idx is None else idx
        bs = bs or max(8, min(256, int(4e7 // self.ctx.n)))
        for s in range(0, len(idx), bs):
            yield idx[s:s + bs], self.cols(idx[s:s + bs])


class Ctx:
    def __init__(self, data, kernel, ell, d, n, rho, rep=0):
        self.X, self.y = data["X"], data["y"]
        self.Xt, self.Xv, self.L = data["Xt"], data["Xv"], data["L"]
        self.kernel, self.ell, self.d, self.n, self.rho, self.rep = kernel, ell, d, n, rho, rep
        self.s2 = C.SIGMA ** 2
        self.R_ex = r_trunc(kernel, ell, C.TRUNC_TOL)
        self.sub = np.arange(C.N_SUB)
        t0 = time.perf_counter()
        self.tree = cKDTree(self.X)
        self.t_tree = time.perf_counter() - t0
        self._xc = None
        self.band_cache = {}

    def xcov(self):
        if self._xc is None:
            t0 = time.perf_counter()
            self._xc = (CrossCov(self, self.Xt), CrossCov(self, self.Xv))
            self.t_xcov = time.perf_counter() - t0
        return self._xc

    def nnz_row(self, r):
        return min(self.n, self.rho * C.ball_volume(self.d, r))


def sparse_kernel_chunked(ctx, r):
    """K masked to pairs with distance <= r (incl. diagonal), CSR, built row-chunk-wise via cKDTree."""
    n = ctx.n
    per_row = ctx.nnz_row(r)
    chunk = int(max(1000, min(n, 2e7 / max(per_row, 1))))
    indptr = [np.zeros(1, np.int64)]; indices = []; data = []; off = 0
    for s in range(0, n, chunk):
        tc = cKDTree(ctx.X[s:s + chunk])
        a = tc.sparse_distance_matrix(ctx.tree, r, output_type="ndarray")
        m = sparse.csr_matrix((kfun(ctx.kernel, a["v"], ctx.ell), (a["i"], a["j"])), shape=(min(chunk, n - s), n))
        del a
        indices.append(m.indices.astype(np.int32 if n < 2 ** 31 else np.int64)); data.append(m.data)
        indptr.append(m.indptr[1:].astype(np.int64) + off); off += m.nnz
    K = sparse.csr_matrix((np.concatenate(data), np.concatenate(indices), np.concatenate(indptr)), shape=(n, n))
    K.sort_indices()
    return K


def get_band(ctx, c, res):
    """Banded kernel for radius c*ell; build time is charged to every method using it."""
    r = c * ctx.ell
    if ctx.nnz_row(r) * ctx.n > C.NNZ_CAP:
        raise Skip(f"nnz {ctx.nnz_row(r) * ctx.n:.2e} > cap")
    if c not in ctx.band_cache:
        t0 = time.perf_counter()
        K = sparse_kernel_chunked(ctx, r)
        ctx.band_cache = {c: (K, time.perf_counter() - t0)}   # keep only one radius in memory
    K, tb = ctx.band_cache[c]
    res.t["fit"] += tb
    res.info["nnz_row"] = K.nnz / ctx.n
    return K


# ----------------------------------------------------------------------------------------------- linear algebra
def block_cg(matvec, B, minv=None, tol=C.CG_TOL, maxit=C.CG_MAXIT, deadline=None):
    """Independent CG on each column of B (vectorised). Returns X, iterations, negative-curvature flag."""
    B = B.reshape(len(B), -1)
    X = np.zeros_like(B); R = B.copy()
    Z = minv(R) if minv else R.copy()
    P = Z.copy(); rz = np.einsum("ij,ij->j", R, Z)
    nb = np.linalg.norm(B, axis=0); nb[nb == 0] = 1
    active = np.ones(B.shape[1], bool); negcurv = False
    for it in range(1, maxit + 1):
        check_deadline(deadline)
        AP = matvec(P[:, active]) if not active.all() else matvec(P)
        Pa = P[:, active]
        pap = np.einsum("ij,ij->j", Pa, AP)
        if np.any(pap <= 0):
            negcurv = True
            return X, it, negcurv
        a = rz[active] / pap
        X[:, active] += a * Pa; R[:, active] -= a * AP
        res = np.linalg.norm(R[:, active], axis=0) / nb[active]
        done = res < tol
        act_idx = np.flatnonzero(active)
        active[act_idx[done]] = False
        if not active.any():
            return X, it, negcurv
        Ra = R[:, active]
        Za = minv(Ra) if minv else Ra
        rz_new = np.einsum("ij,ij->j", Ra, Za)
        P[:, active] = Za + (rz_new / rz[active]) * P[:, active]
        rz[active] = rz_new
    return X, maxit, negcurv


def var_from_solve(xc, solve_cols, idx=None, deadline=None):
    """v_i = 1 - k_i^T A^{-1} k_i via a routine returning W with ||W[:,j]||^2 = k_j^T A^{-1} k_j."""
    idx = np.arange(len(xc.Xs)) if idx is None else idx
    out = np.empty(len(idx)); pos = 0
    for ii, Kb in xc.batches(idx):
        check_deadline(deadline)
        W = solve_cols(Kb)
        out[pos:pos + len(ii)] = 1.0 - np.einsum("ij,ij->j", W, W)
        pos += len(ii)
    return out


# ----------------------------------------------------------------------------------------------- exact
def exact(ctx, res):
    if ctx.n > C.EXACT_MAX_N:
        raise Skip("exact only for n <= EXACT_MAX_N")
    xt, xv = ctx.xcov()
    with res.timer("fit"):
        K = dense_block(ctx.kernel, ctx.X, ctx.X, ctx.ell)
        K[np.diag_indices_from(K)] += ctx.s2
        Lc = cholesky(K, lower=True, overwrite_a=True, check_finite=False)
        a = cho_solve((Lc, True), ctx.y, check_finite=False)
    with res.timer("mean"):
        res.mt, res.mv = xt.rmatvec(a), xv.rmatvec(a)
    with res.timer("var"):
        res.vt = var_from_solve(xt, lambda Kb: solve_triangular(Lc, Kb, lower=True, check_finite=False))


# ----------------------------------------------------------------------------------------------- banding
def band_chol(ctx, res, c, lam=0.0, K=None):
    K = get_band(ctx, c, res) if K is None else K
    xt, xv = ctx.xcov()
    with res.timer("fit"):
        A = (K + sparse.identity(ctx.n, format="csr") * (ctx.s2 + lam)).tocsc()
        try:
            F = cholmod.cho_factor(A, lower=True)
        except cholmod.CholmodNotPositiveDefiniteError as e:
            raise Invalid(f"cholmod: {e}")
        del A
        a = F.solve(ctx.y)
        res.info["nnz_L"] = float(F.nnz)
    with res.timer("mean"):
        res.mt, res.mv = xt.rmatvec(a), xv.rmatvec(a)
    perm = F.perm
    with res.timer("var"):
        res.vt = var_from_solve(xt, lambda Kb: F.solve(Kb[perm], system="L"))


def band_cg(ctx, res, c):
    K = get_band(ctx, c, res)
    xt, xv = ctx.xcov()
    with res.timer("fit"):
        A = (K + sparse.identity(ctx.n, format="csr") * ctx.s2).tocsr()
        a, it, neg = block_cg(lambda V: A @ V, ctx.y[:, None])
        res.info["cg_iters"] = it
        if neg:
            raise Invalid("CG: negative curvature (indefinite)")
        if it >= C.CG_MAXIT:
            raise Invalid("CG: no convergence")
        a = a[:, 0]
    with res.timer("mean"):
        res.mt, res.mv = xt.rmatvec(a), xv.rmatvec(a)
    if ctx.rep < C.CG_VAR_REPS:
        with res.timer("var"):
            Kb = xt.cols(ctx.sub)
            W, it2, neg = block_cg(lambda V: A @ V, Kb)
            if neg:
                raise Invalid("CG (variance): negative curvature")
            res.vsub = 1.0 - np.einsum("ij,ij->j", Kb, W)
        res.t["var"] *= C.N_TEST / C.N_SUB
        res.var_extrapolated = True
    else:
        res.t["var"] = np.nan


def band_banded(ctx, res, c):
    """d=1 only: sort, store as LAPACK lower band, banded Cholesky (cholesky_banded + dtbtrs)."""
    if ctx.d != 1:
        raise Skip("d=1 only")
    r = c * ctx.ell
    if ctx.nnz_row(r) * ctx.n > C.NNZ_CAP:
        raise Skip("nnz cap")
    xt, xv = ctx.xcov()
    with res.timer("fit"):
        order = np.argsort(ctx.X[:, 0]); xs = ctx.X[order, 0]; n = ctx.n
        u = int(np.max(np.searchsorted(xs, xs + r, side="right") - np.arange(n) - 1))
        res.info["bandwidth"] = u
        if (u + 1) * n > C.NNZ_CAP:
            raise Skip("band storage cap")
        ab = np.zeros((u + 1, n))
        ab[0] = 1.0 + ctx.s2
        for o in range(1, u + 1):
            dist = xs[o:] - xs[:-o]
            ab[o, :n - o] = np.where(dist <= r, kfun(ctx.kernel, dist, ctx.ell), 0.0)
        res.info["nnz_row"] = float(1 + 2 * np.count_nonzero(ab[1:]) / n)
        try:
            cb = cholesky_banded(ab, lower=True, overwrite_ab=True, check_finite=False)
        except np.linalg.LinAlgError as e:
            raise Invalid(f"banded cholesky: {e}")
        a_s = cho_solve_banded((cb, True), ctx.y[order], check_finite=False)
        a = np.empty(n); a[order] = a_s
    with res.timer("mean"):
        res.mt, res.mv = xt.rmatvec(a), xv.rmatvec(a)

    def solve_cols(Kb):
        W, info = lapack.dtbtrs(cb, np.asfortranarray(Kb[order]), uplo="L")
        return W
    with res.timer("var"):
        res.vt = var_from_solve(xt, solve_cols)


def lambda_rowsum_bound(ctx, c):
    """Rigorous Gershgorin-type bound on lambda_max(K - K_band) = -lambda_min(E), O(n + cells log cells):
    bin points on a grid of cell size h; for i in cell a, sum_{j far} k_ij <= sum_b count_b * k(max(r, mindist(a,b)))
    over cells b with maxdist(a,b) > r. Evaluated for all cells at once by FFT convolution."""
    r = c * ctx.ell; d = ctx.d; L = ctx.L
    h = r / 4
    while math.ceil(L / h) ** d > 3e7:
        h *= 1.5
    nc = int(math.ceil(L / h))
    cell = np.minimum((ctx.X / h).astype(np.int64), nc - 1)
    counts = np.bincount(np.ravel_multi_index(cell.T, (nc,) * d), minlength=nc ** d).reshape((nc,) * d).astype(float)
    S = min(nc - 1, int(math.ceil(ctx.R_ex / h)) + 1)
    off = np.arange(-S, S + 1)
    grids = np.meshgrid(*([off] * d), indexing="ij")
    mind = np.sqrt(sum((h * np.maximum(np.abs(g) - 1, 0)) ** 2 for g in grids))
    maxd = np.sqrt(sum((h * (np.abs(g) + 1)) ** 2 for g in grids))
    w = np.where(maxd > r, kfun(ctx.kernel, np.maximum(mind, r), ctx.ell), 0.0)
    conv = fftconvolve(counts, w, mode="same")
    return float(np.max(conv[counts > 0]))


def lambda_exact(ctx, c, Kband):
    """-lambda_min(E) = lambda_max(K_ex - K_band) by Lanczos (feasible only at moderate n * nnz)."""
    nnz_ex = ctx.nnz_row(ctx.R_ex)
    if ctx.n * nnz_ex > C.EIG_CAP:
        raise Skip("eig cap")
    Kex = sparse_kernel_chunked(ctx, ctx.R_ex)
    Far = (Kex - Kband).tocsr()
    del Kex
    v0 = np.random.default_rng(0).standard_normal(ctx.n)
    lmax = float(eigsh(Far, k=1, which="LA", return_eigenvectors=False, tol=1e-6, v0=v0, maxiter=5000)[0])
    return max(0.0, lmax)


def band_infl(ctx, res, c, which):
    K = get_band(ctx, c, res)
    t0 = time.perf_counter()
    lam = lambda_rowsum_bound(ctx, c) if which == "rs" else lambda_exact(ctx, c, K)
    res.t["fit"] += time.perf_counter() - t0
    res.info["lam"] = lam
    band_chol(ctx, res, c, lam=lam, K=K)


# ----------------------------------------------------------------------------------------------- PCG on exact system
def exact_matvec(ctx):
    nnz_ex = ctx.nnz_row(ctx.R_ex)
    if nnz_ex * ctx.n <= C.NNZ_CAP:
        K = sparse_kernel_chunked(ctx, ctx.R_ex)
        A = K + sparse.identity(ctx.n, format="csr") * ctx.s2
        return (lambda V: A @ V), "sparse_trunc"
    if float(ctx.n) ** 2 <= C.DENSE_MATVEC_CAP:
        op = DenseKernelOp(ctx.kernel, ctx.X, ctx.ell)
        return (lambda V: op @ V + ctx.s2 * V), "dense_blocked"
    raise Skip("exact matvec too expensive (no KeOps/GPU)")


def pivoted_cholesky(ctx, k, stop_maxdiag=None, deadline=None):
    """Greedy pivoted Cholesky of K (exact kernel columns). Stops early at numerical rank (residual diag <= 1e-10)."""
    n = ctx.n
    Lt = np.empty((k, n)); dg = np.ones(n)
    for i in range(k):
        if i % 50 == 0:
            check_deadline(deadline)
        j = int(np.argmax(dg)); dj = dg[j]
        if (stop_maxdiag is not None and dj <= stop_maxdiag) or dj <= 1e-10:
            return Lt[:i], i, dj
        col = dense_block(ctx.kernel, ctx.X, ctx.X[j:j + 1], ctx.ell)[:, 0]
        if i:
            col -= Lt[:i].T @ Lt[:i, j]
        Lt[i] = col / math.sqrt(dj)
        dg -= Lt[i] ** 2
        dg[j] = 0.0
    return Lt, k, float(dg.max())


def pcg(ctx, res, rank):
    if ctx.n * rank > C.PIVCHOL_CAP:
        raise Skip("pivoted Cholesky memory cap")
    if rank >= ctx.n:
        raise Skip("rank >= n")
    xt, xv = ctx.xcov()
    with res.timer("fit"):
        mv, mode = exact_matvec(ctx)
        res.info["matvec"] = mode
        Lt, _, _ = pivoted_cholesky(ctx, rank)
        Cw = cho_factor(np.eye(rank) * ctx.s2 + Lt @ Lt.T)
        minv = lambda R: (R - Lt.T @ cho_solve(Cw, Lt @ R)) / ctx.s2
        a, it, neg = block_cg(mv, ctx.y[:, None], minv)
        res.info["cg_iters"] = it
        if neg or it >= C.CG_MAXIT:
            raise Invalid("PCG failed")
        a = a[:, 0]
    with res.timer("mean"):
        res.mt, res.mv = xt.rmatvec(a), xv.rmatvec(a)
    if ctx.rep < C.CG_VAR_REPS:
        with res.timer("var"):
            Kb = xt.cols(ctx.sub)
            W, it2, neg = block_cg(mv, Kb, minv)
            res.vsub = 1.0 - np.einsum("ij,ij->j", Kb, W)
        res.t["var"] *= C.N_TEST / C.N_SUB
        res.var_extrapolated = True
    else:
        res.t["var"] = np.nan


# ----------------------------------------------------------------------------------------------- SGPR (Titsias)
def inducing_grid(L, d, M):
    if d == 1:
        return ((np.arange(M) + 0.5) * L / M)[:, None], M
    m = int(round(math.sqrt(M)))
    g = (np.arange(m) + 0.5) * L / m
    Z = np.stack(np.meshgrid(g, g, indexing="ij"), -1).reshape(-1, 2)
    return Z, m * m


def sgpr(ctx, res, M):
    if M >= ctx.n / 3:
        raise Skip("M too large for n")
    if ctx.n * M * M > C.SGPR_FLOP_CAP:
        raise Skip("SGPR flop cap")
    with res.timer("fit"):
        Z, M = inducing_grid(ctx.L, ctx.d, M)
        res.info["M"] = M
        Kuu = dense_block(ctx.kernel, Z, Z, ctx.ell)
        jit = 1e-8
        while True:
            try:
                Lu = cholesky(Kuu + jit * np.eye(M), lower=True); break
            except np.linalg.LinAlgError:
                jit *= 10
        res.info["jitter"] = jit
        # accumulate Kuf Kfu and Kuf y without storing Kuf (sparse if kernel support is small)
        nnz_frac = min(1.0, ctx.rho * C.ball_volume(ctx.d, ctx.R_ex) / ctx.n)
        if nnz_frac < 0.1:
            tz = cKDTree(Z)
            a = tz.sparse_distance_matrix(ctx.tree, ctx.R_ex, output_type="ndarray")
            Kuf = sparse.csr_matrix((kfun(ctx.kernel, a["v"], ctx.ell), (a["i"], a["j"])), shape=(M, ctx.n))
            del a
            KK = (Kuf @ Kuf.T).toarray(); Ky = Kuf @ ctx.y
        else:
            KK = np.zeros((M, M)); Ky = np.zeros(M)
            bs = max(1000, int(2e7 // M))
            for s in range(0, ctx.n, bs):
                Kb = dense_block(ctx.kernel, Z, ctx.X[s:s + bs], ctx.ell)
                KK += Kb @ Kb.T; Ky += Kb @ ctx.y[s:s + bs]
        AAt = solve_triangular(Lu, solve_triangular(Lu, KK, lower=True).T, lower=True).T / ctx.s2
        B = np.eye(M) + 0.5 * (AAt + AAt.T)
        LB = cholesky(B, lower=True)
        cvec = solve_triangular(LB, solve_triangular(Lu, Ky, lower=True), lower=True) / ctx.s2

    def predict(Xs, want_var):
        Kus = dense_block(ctx.kernel, Z, Xs, ctx.ell)
        t1 = solve_triangular(Lu, Kus, lower=True)
        t2 = solve_triangular(LB, t1, lower=True)
        mu = t2.T @ cvec
        if want_var:
            return mu, 1.0 - np.sum(t1 ** 2, 0) + np.sum(t2 ** 2, 0)
        return mu, None
    with res.timer("mean"):
        res.mt, _ = predict(ctx.Xt, False)
        res.mv, _ = predict(ctx.Xv, False)
    with res.timer("var"):
        _, res.vt = predict(ctx.Xt, True)


# ----------------------------------------------------------------------------------------------- Vecchia
def approx_maxmin(X, L, rng):
    """Approximate maxmin ordering: coarse-to-fine dyadic grids, one random point per newly occupied cell per level,
    remaining points last in random order."""
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


def _batched_cond(ctx, Xi, XN, yN=None):
    """For each row: b = (K_NN + s2 I)^{-1} k_N(x_i), d = 1 + s2 - k^T b (response model)."""
    m = XN.shape[1]
    diff = XN[:, :, None, :] - XN[:, None, :, :]
    KNN = kfun_sq(ctx.kernel, np.sum(diff * diff, -1), ctx.ell)
    KNN[:, np.arange(m), np.arange(m)] += ctx.s2
    kN = kfun_sq(ctx.kernel, np.sum((XN - Xi[:, None, :]) ** 2, -1), ctx.ell)
    b = np.linalg.solve(KNN, kN[..., None])[..., 0]
    return b, kN


def _vecchia_factor(ctx, Z, order_head, m, nugget, deadline):
    """Vecchia factor A (rows = positions in Z) for a field with covariance K(Z,Z) + nugget*I, conditioning each point
    on its m nearest predecessors (searched in a prefix that holds >= 2/3 of the predecessors). Z is already ordered."""
    N = len(Z)
    rows, cols, vals = [], [], []
    p0 = min(N, 4 * m + 1)

    def cond(Xi, XN):
        mm = XN.shape[1]
        diff = XN[:, :, None, :] - XN[:, None, :, :]
        KNN = kfun_sq(ctx.kernel, np.sum(diff * diff, -1), ctx.ell)
        KNN[:, np.arange(mm), np.arange(mm)] += nugget
        kN = kfun_sq(ctx.kernel, np.sum((XN - Xi[:, None, :]) ** 2, -1), ctx.ell)
        b = np.linalg.solve(KNN, kN[..., None])[..., 0]
        dv = 1 + nugget - np.einsum("ij,ij->i", kN, b)
        if np.any(dv <= 0):
            raise Invalid("non-positive conditional variance")
        return b, 1 / np.sqrt(dv)
    for p in range(p0):
        if p == 0:
            rows.append(np.array([0])); cols.append(np.array([0])); vals.append(np.array([1 / math.sqrt(1 + nugget)])); continue
        nb = np.argsort(np.sum((Z[:p] - Z[p]) ** 2, 1))[:m]
        b, sd = cond(Z[p:p + 1], Z[nb][None])
        rows.append(np.full(len(nb) + 1, p)); cols.append(np.r_[p, nb]); vals.append(np.r_[sd[0], -b[0] * sd[0]])
    a = p0
    while a < N:
        bnd = min(N, int(math.ceil(a * 1.5)))
        tp = cKDTree(Z[:a])
        for s in range(a, bnd, 20000):
            check_deadline(deadline)
            e = min(bnd, s + 20000)
            _, nb = tp.query(Z[s:e], k=m)
            nb = nb.reshape(e - s, m)
            b, sd = cond(Z[s:e], Z[nb])
            pi = np.arange(s, e)
            rows.append(np.repeat(pi, m + 1)); cols.append(np.c_[pi, nb].ravel())
            vals.append(np.c_[sd, -b * sd[:, None]].ravel())
        a = bnd
    return sparse.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cols))), shape=(N, N))


def vecchia_latent(ctx, res, m):
    """Latent-field Vecchia with prediction points ordered last (Katzfuss, Guinness, Gong & Zilber 2020, 'LF-ordered
    last'): approximate maxmin ordering of training inputs, then test, then validation points; each point conditions on
    its m nearest predecessors. Posterior precision A^T A + diag(1/s2 on observed) is factorised by CHOLMOD;
    means from one solve, test variances = diag of the inverse via L-solves. A latent nugget of 1e-6 keeps the
    conditioning blocks of smooth kernels numerically positive definite."""
    n, nt, nv = ctx.n, len(ctx.Xt), len(ctx.Xv)
    N = n + nt + nv
    if N * (m + 1) * m / 2 > C.NNZ_CAP:
        raise Skip("nnz cap (A^T A)")
    deadline = None
    with res.timer("fit"):
        order = approx_maxmin(ctx.X, ctx.L, np.random.default_rng(12345))
        Z = np.vstack([ctx.X[order], ctx.Xt, ctx.Xv])
        A = _vecchia_factor(ctx, Z, None, m, 1e-6, deadline)
        dobs = np.zeros(N); dobs[:n] = 1 / ctx.s2
        Q = (A.T @ A + sparse.diags(dobs)).tocsc()
        del A
        F = cholmod.cho_factor(Q, lower=True)
        res.info["nnz_L"] = float(F.nnz)
        rhs = np.zeros(N); rhs[:n] = ctx.y[order] / ctx.s2
    with res.timer("mean"):
        mu = F.solve(rhs)
        res.mt, res.mv = mu[n:n + nt], mu[n + nt:]
    perm = F.perm
    inv = np.empty(N, np.int64); inv[perm] = np.arange(N)
    with res.timer("var"):
        vt = np.empty(nt); bs = max(8, min(256, int(4e7 // N)))
        for s in range(0, nt, bs):
            check_deadline(deadline)
            idx = np.arange(n + s, n + min(nt, s + bs))
            E = np.zeros((N, len(idx))); E[inv[idx], np.arange(len(idx))] = 1.0   # (P e_i)
            W = F.solve(E, system="L")
            vt[s:s + len(idx)] = np.einsum("ij,ij->j", W, W)
        res.vt = vt


def vecchia_pred(ctx, res, m):
    """Standard Vecchia/NN prediction (prediction points ordered last): condition each prediction point on its
    m nearest observations. No global fit needed with known hyperparameters."""
    def pred(Xs, want_var):
        _, nb = ctx.tree.query(Xs, k=m)
        nb = nb.reshape(len(Xs), m)
        XN = ctx.X[nb]
        b, kN = _batched_cond(ctx, Xs, XN)
        mu = np.einsum("ij,ij->i", b, ctx.y[nb])
        return mu, (1.0 - np.einsum("ij,ij->i", kN, b)) if want_var else None
    with res.timer("mean"):
        res.mt, _ = pred(ctx.Xt, False)
        res.mv, _ = pred(ctx.Xv, False)
    with res.timer("var"):
        _, res.vt = pred(ctx.Xt, True)


# ----------------------------------------------------------------------------------------------- registry
def method_list(d):
    L = [("exact", "exact", {}, exact, {})]
    for c in C.CS:
        L += [("band_chol", f"band_chol_c{c}", {"c": c}, band_chol, {"c": c}),
              ("band_cg", f"band_cg_c{c}", {"c": c}, band_cg, {"c": c})]
        if d == 1:
            L.append(("band_banded", f"band_banded_c{c}", {"c": c}, band_banded, {"c": c}))
        L += [("infl_rs", f"infl_rs_c{c}", {"c": c}, band_infl, {"c": c, "which": "rs"}),
              ("infl_exact", f"infl_exact_c{c}", {"c": c}, band_infl, {"c": c, "which": "exact"})]
    for k in C.PCG_RANKS:
        L.append(("pcg", f"pcg_r{k}", {"rank": k}, pcg, {"rank": k}))
    for M in C.SGPR_MS:
        L.append(("sgpr", f"sgpr_M{M}", {"M": M}, sgpr, {"M": M}))
    for m in C.VECCHIA_MS:
        L.append(("vecchia_latent", f"vecchia_latent_m{m}", {"m": m}, vecchia_latent, {"m": m}))
        L.append(("vecchia_pred", f"vecchia_pred_m{m}", {"m": m}, vecchia_pred, {"m": m}))
    return L
