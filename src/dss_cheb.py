"""
Chebyshev-collocation version of the self-similar-coordinate map.

Why a second discretisation of the same equation: the physical Floquet modes of
a DSS solution are those ANALYTIC at the self-similarity horizon.  Near the
horizon the characteristic speed vanishes linearly, a ~ mu (X - X_h) with
mu > 0, so the pull-back operator has eigenfunctions (X - X_h)^s with
multipliers e^{-s mu t} for every complex s -- a continuum, of which only the
integer s are analytic.  A finite-difference grid represents non-integer s
just as happily as integer s, so its map spectrum contains that whole
continuum; measured at N = 200 vs 400 (src/dss_sweep.py spec) every eigenvalue
except lam0, the two lam = 0 modes and the lam = 1 gauge mode drifts, which is
that continuum being resampled.

Empirically the collocation map does not carry that continuum, while the
banded map does.  We use the difference but do not derive it: the tempting
argument -- "a global polynomial basis can only represent analytic functions"
-- is NOT available, because a limit of polynomials need not be analytic.  What
is being made is a smoothness SELECTION, and it has to be declared as one: the
target class is perturbations smooth across the horizon, chosen because a
continuum of branches cannot be what organises a codimension-one threshold.
The banded scheme reaches the same class by a different route, by filtering
(dss_spec_cheb2), so the two agree at a matched selection rather than
independently of any selection.

Interface matches dss_core.Sim, so dss_newton works against either.
"""
import numpy as np


def _cheb(n, Xmax):
    """CGL points ascending on [0, Xmax] and the spectral d/dX matrix."""
    k = np.arange(n)
    xi = -np.cos(np.pi * k / (n - 1))               # -1 .. 1
    X = 0.5 * Xmax * (1.0 + xi)
    c = np.ones(n); c[0] = c[-1] = 2.0
    c *= (-1.0) ** k
    Xd = np.tile(xi, (n, 1)).T
    dX = Xd - Xd.T
    D = np.outer(c, 1.0 / c) / (dX + np.eye(n))
    D -= np.diag(D.sum(axis=1))
    return X, D * (2.0 / Xmax)


def _cumop(D):
    """B with (B f)_i = int_{X_0}^{X_i} f dX, spectrally exact: solve D F = f
    on rows 1.. together with F_0 = 0."""
    n = D.shape[0]
    M = D.copy()
    M[0, :] = 0.0; M[0, 0] = 1.0
    S = np.eye(n); S[0, 0] = 0.0
    return np.linalg.solve(M, S)


FOURPI = 4.0 * np.pi


class Sim:
    def __init__(self, N=64, Xmax=4.0, **_):
        self.X, self.D = _cheb(N, Xmax)
        self.B = _cumop(self.D)
        self.N, self.Xmax = N, Xmax
        self.Xi = np.empty(N); self.Xi[0] = 0.0
        self.Xi[1:] = 1.0 / self.X[1:]
        self.h = np.diff(self.X).min()

    # -- constraints ------------------------------------------------------
    def fields(self, H):
        hbar = (self.B @ H) * self.Xi
        hbar[0] = H[0]                                # l'Hopital at X = 0
        q = (H - hbar) ** 2 * self.Xi                 # ~ X^3 at the centre
        q[0] = 0.0
        # ln g is capped: a Newton trial state can be wildly unphysical, and
        # an inf here poisons the whole residual instead of merely making the
        # line search back off.  g = e^80 is far inside the black-hole regime.
        g = np.exp(np.minimum(FOURPI * (self.B @ q), 80.0))   # g[0] = 1
        gbar = (self.B @ g) * self.Xi
        gbar[0] = g[0]
        return hbar, g, gbar

    def rhs(self, H):
        hbar, g, gbar = self.fields(H)
        a = self.X - 0.5 * gbar
        s = (g - gbar) * (H - hbar) * self.Xi * 0.5
        s[0] = 0.0
        return -a * (self.D @ H) + s

    def speed(self, H):
        _, _, gbar = self.fields(H)
        return self.X - 0.5 * gbar

    def phi0(self, H):
        return float(H[0])                            # hbar(0) = H(0) = phi(0)

    def dx(self, H):
        return self.D @ H

    # -- evolution --------------------------------------------------------
    def evolve(self, H, Tend, nstep):
        H = np.array(H, dtype=float, copy=True)
        dT = Tend / nstep
        for _ in range(nstep):
            k1 = self.rhs(H)
            k2 = self.rhs(H + 0.5 * dT * k1)
            k3 = self.rhs(H + 0.5 * dT * k2)
            k4 = self.rhs(H + dT * k3)
            H += (dT / 6.0) * (k1 + 2.0 * k2 + 2.0 * k3 + k4)
        return H

    def nstep_cfl(self, H, Tend, cfl=0.35):
        amax = max(np.abs(self.speed(H)).max(), 1e-12)
        return int(np.ceil(Tend / (cfl * self.h / amax)))

    def trace(self, H, Tend, nstep, nout=200):
        H = np.array(H, dtype=float, copy=True)
        k = max(nstep // nout, 1)
        Ts, ph, hm = [0.0], [self.phi0(H)], [np.abs(H).max()]
        dT = Tend / nstep
        done = 0
        while done < nstep:
            m = min(k, nstep - done)
            H = self.evolve(H, m * dT, m)
            done += m
            if not np.all(np.isfinite(H)):
                break
            Ts.append(done * dT); ph.append(self.phi0(H)); hm.append(np.abs(H).max())
        return np.array(Ts), np.array(ph), np.array(hm), H
