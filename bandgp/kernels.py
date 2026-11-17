import math
import numpy as np
import numexpr as ne
from scipy import sparse
from scipy.optimize import brentq

SQ3 = math.sqrt(3.0)


def kfun(kernel, r, ell):
    """Unit-variance stationary kernels as functions of distance. SE uses exp(-r^2/ell^2) (feeler convention)."""
    if kernel == "se":
        return ne.evaluate("exp(-(r / ell) ** 2)")
    t = SQ3 / ell
    return ne.evaluate("(1 + t * r) * exp(-t * r)")


def kfun_sq(kernel, r2, ell):
    if kernel == "se":
        return ne.evaluate("exp(-r2 / (ell * ell))")
    t = SQ3 / ell
    return ne.evaluate("(1 + t * sqrt(r2)) * exp(-t * sqrt(r2))")


def r_trunc(kernel, ell, tol):
    if kernel == "se":
        return ell * math.sqrt(math.log(1.0 / tol))
    t = brentq(lambda s: (1 + s) * math.exp(-s) - tol, 1e-6, 200.0)
    return t * ell / SQ3


def tail_integral(kernel, ell, r, d):
    """int_{|u|>r} k(u) du (used for the analytic expected far-field row sum)."""
    from scipy.integrate import quad
    f = (lambda s: 2 * kfun(kernel, np.array(s), ell)) if d == 1 else (lambda s: 2 * math.pi * s * kfun(kernel, np.array(s), ell))
    return quad(lambda s: float(f(s)), r, np.inf, limit=200)[0]


def spectral_density(kernel, w, ell, d):
    if kernel == "se":   # FT of exp(-r^2/ell^2)
        return (math.pi * ell * ell) ** (d / 2) * np.exp(-(w * ell) ** 2 / 4)
    nu = 1.5
    lam2 = 2 * nu / ell ** 2
    C = 2 ** d * math.pi ** (d / 2) * math.gamma(nu + d / 2) * lam2 ** nu / math.gamma(nu)
    return C * (lam2 + w * w) ** (-(nu + d / 2))


def d_eff_continuum(kernel, ell, d, rho, volume, sigma2):
    """Continuum approximation of tr K (K + s2 I)^{-1} for a uniform design of density rho on a domain of given volume."""
    from scipy.integrate import quad
    g = lambda w: rho * spectral_density(kernel, w, ell, d) / (rho * spectral_density(kernel, w, ell, d) + sigma2)
    if d == 1:
        val = 2 * quad(g, 0, np.inf, limit=400)[0] / (2 * math.pi)
    else:
        val = quad(lambda w: 2 * math.pi * w * g(w), 0, np.inf, limit=400)[0] / (2 * math.pi) ** 2
    return volume * val


def sparse_kernel(tree_a, tree_b, R, kernel, ell, shape, exclude=None):
    """Sparse kernel matrix with entries for pairs at distance <= R (cKDTree.sparse_distance_matrix)."""
    a = tree_a.sparse_distance_matrix(tree_b, R, output_type="ndarray")
    i, j, v = a["i"], a["j"], a["v"]
    del a
    data = kfun(kernel, v, ell)
    M = sparse.csr_matrix((data, (i, j)), shape=shape)
    return M


def dense_block(kernel, A, B, ell):
    r2 = np.zeros((A.shape[0], B.shape[0]))
    for k in range(A.shape[1]):
        diff = A[:, k:k + 1] - B[None, :, k]
        r2 += diff * diff
    K = kfun_sq(kernel, r2, ell)
    K[K < 1e-30] = 0.0          # avoid subnormal floats (they slow BLAS/LAPACK ~100x); numerically irrelevant
    return K


class DenseKernelOp:
    """Matrix-free exact kernel matvec, blocked numpy (no KeOps available on this machine)."""
    def __init__(self, kernel, X, ell, block=2048):
        self.kernel, self.X, self.ell, self.block = kernel, X, ell, block
        self.shape = (X.shape[0], X.shape[0])

    def __matmul__(self, V):
        n = self.X.shape[0]
        out = np.empty((n,) + V.shape[1:])
        for s in range(0, n, self.block):
            Kb = dense_block(self.kernel, self.X[s:s + self.block], self.X, self.ell)
            out[s:s + self.block] = Kb @ V
        return out
