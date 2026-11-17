"""Design, truth f0 and noise. f0 is a cosine series on [0,L]^d whose coefficients depend on the
*absolute* frequency pi*j/L:  a_j = s_j * max(|j|/L, 1)^{-(alpha + d/2)}, random signs s_j, normalised to unit
variance. At L=1 this is exactly the feeler's j^{-(alpha+1/2)} (d=1) / (j^2+k^2)^{-(alpha+1)/2} (d=2) series.
The series is evaluated exactly on a fine grid by a type-I DCT and linearly interpolated; the interpolant *is* f0."""
import numpy as np
from scipy.fft import dct, dctn
from . import config as C

J0 = {1: 400, 2: 40}          # terms per unit side length (feeler used 400 / 40 on [0,1])
JMAX = {1: 4_000_000, 2: 2048}
OVERSAMPLE = {1: 8, 2: 4}


def seed_seq(*parts):
    return np.random.SeedSequence([int(round(p * 1000)) if isinstance(p, float) else int(p) for p in parts])


class F0:
    def __init__(self, d, alpha, L, rng):
        self.d, self.L = d, L
        J = int(min(np.ceil(J0[d] * L), JMAX[d]))
        N = OVERSAMPLE[d] * J
        self.N = N
        if d == 1:
            j = np.arange(1, J + 1)
            c = rng.choice([-1.0, 1.0], J) * np.maximum(j / L, 1.0) ** (-(alpha + 0.5))
            c /= np.sqrt(np.sum(c ** 2) / 2)
            a = np.zeros(N + 1)
            a[1:J + 1] = c / 2
            self.grid = dct(a, type=1)               # = sum_j c_j cos(pi j k / N)
        else:
            jj = np.arange(J + 1)
            fr = np.sqrt(jj[:, None] ** 2 + jj[None, :] ** 2)
            c = rng.choice([-1.0, 1.0], (J + 1, J + 1)) * np.maximum(fr / L, 1.0) ** (-(alpha + 1.0))
            c[0, 0] = 0.0
            w = np.full((J + 1, J + 1), 0.25); w[0, :] = 0.5; w[:, 0] = 0.5; w[0, 0] = 0
            c /= np.sqrt(np.sum(c ** 2 * w))
            a = np.zeros((N + 1, N + 1))
            a[:J + 1, :J + 1] = c
            a[1:, 0] /= 2; a[0, 1:] /= 2; a[1:, 1:] /= 4   # dct type-1 doubles interior terms per axis
            self.grid = dctn(a, type=1)
            del a

    def __call__(self, X):
        h = self.L / self.N
        if self.d == 1:
            return np.interp(X[:, 0], np.linspace(0, self.L, self.N + 1), self.grid)
        u = np.clip(X / h, 0, self.N - 1e-9)
        i0 = np.floor(u).astype(np.int64); t = u - i0
        g = self.grid
        i, j = i0[:, 0], i0[:, 1]; tx, ty = t[:, 0], t[:, 1]
        return ((1 - tx) * (1 - ty) * g[i, j] + tx * (1 - ty) * g[i + 1, j]
                + (1 - tx) * ty * g[i, j + 1] + tx * ty * g[i + 1, j + 1])


def make_data(d, alpha, beta, n, rep):
    """Common random numbers: data depend only on (d, alpha, beta, n, rep), not on kernel or lengthscale."""
    rng = np.random.default_rng(seed_seq(d, alpha, beta, n, rep))
    L = C.domain_L(n, d, beta)
    f0 = F0(d, alpha, L, rng)
    X = rng.uniform(0, L, size=(n, d))
    y = f0(X) + C.SIGMA * rng.standard_normal(n)
    Xt = rng.uniform(0.05 * L, 0.95 * L, size=(C.N_TEST, d))
    Xv = rng.uniform(0.05 * L, 0.95 * L, size=(C.N_VAL, d))
    out = dict(X=X, y=y, Xt=Xt, ft=f0(Xt), Xv=Xv, fv=f0(Xv), L=L)
    del f0
    return out
