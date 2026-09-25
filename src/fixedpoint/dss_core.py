"""
The Choptuik system as an autonomous PDE in self-similar coordinates.

With  T = -ln(u* - u),  X = r/(u* - u),  and the Christodoulou/Garfinkle free
datum h(u, r) = d(r phi)/dr written as H(T, X), the null evolution

    dh/du|_v = (g - gbar)(h - hbar)/(2 r),      dr/du|_v = -gbar/2

becomes

    H_T = -(X - gbar/2) H_X + (g - gbar)(H - Hbar)/(2 X)                  (*)

with the same scale-invariant constraints as in radius,

    Hbar = (1/X) int_0^X H dX,   ln g = 4 pi int_0^X (H-Hbar)^2 dX/X,
    gbar = (1/X) int_0^X g dX.

Three things make (*) the right formulation for a fixed-point solve:

  * it is AUTONOMOUS in T, so the half-period map is simply "integrate for
    time Delta/2" -- no rescaling, no regridding, no resampling;
  * the grid in X is FIXED, so the Lagrangian depletion of the interior that
    limits direct evolution never occurs;
  * both boundaries are outflow provided X_max lies above the self-similarity
    horizon (the repelling periodic orbit of dX/dT = X - gbar/2, whose maximum
    over the cycle is X ~ 2.5 -- measured in src/fixedpoint/dss_probe.py), so the problem
    is closed with NO boundary conditions at all.

H is even in X (regularity at the centre: phi is even in r and H = phi + X
phi_X), which is used to build the derivative stencils near X = 0.

The system has two exact symmetries that survive discretisation, and these
serve as free accuracy checks on any Floquet spectrum computed from (*):
  H -> H + c        (the equation sees H only through H_X and H - Hbar) -> lam = 0
  T -> T + a        (autonomy)                                          -> lam = 0
  u* -> u* + d      (which acts as T -> T - d e^T)                      -> lam = 1
and it is odd, H -> -H, which is what makes the antiperiodic half-period
formulation legitimate.

Both spatial operators are high order and banded: `deriv_stencils` gives
O(h^(w-1)) differentiation, `quad_stencils` an O(h^(w-1)) *cumulative*
quadrature (the parabolic rule in `nullcore.cumquad` is only third order
cumulatively and caps the whole scheme at that).
"""
import numpy as np
from numba import njit

FOURPI = 4.0 * np.pi


# ------------------------------------------------------------ FD / quadrature

def _lagrange_deriv_weights(xs, x0):
    """Weights w_k with sum_k w_k f(x_k) = f'(x0) for the interpolant."""
    w = len(xs)
    out = np.zeros(w)
    for k in range(w):
        s = 0.0
        for m in range(w):
            if m == k:
                continue
            t = 1.0 / (xs[k] - xs[m])
            for q in range(w):
                if q != k and q != m:
                    t *= (x0 - xs[q]) / (xs[k] - xs[q])
            s += t
        out[k] = s
    return out


def _lagrange_int_weights(xs, a, b):
    """Weights w_k with sum_k w_k f(x_k) = int_a^b p(x) dx, p the interpolant.
    Solved as exact moment conditions in a scaled variable for conditioning."""
    w = len(xs)
    c = 0.5 * (a + b)
    s = b - a
    t = (np.asarray(xs) - c) / s
    M = np.vander(t, w, increasing=True).T          # M[m, k] = t_k^m
    m = np.arange(w)
    rhs = (0.5 ** (m + 1) - (-0.5) ** (m + 1)) / (m + 1.0) * s
    return np.linalg.solve(M, rhs)


def deriv_stencils(X, w=5):
    """
    d/dX on the fixed grid X (which excludes X = 0), order w-1.

    Stencils are centred where possible and shifted inward at each end.  H has
    no parity at X = 0 (see the module docstring), so no ghost points are used;
    the shift direction at both ends coincides with the upwind direction, so
    no boundary condition is needed.  Returns (idx, wt) of shape (n, w).
    """
    n = len(X)
    idx = np.zeros((n, w), dtype=np.int32)
    wt = np.zeros((n, w))
    for i in range(n):
        j0 = min(max(i - w // 2, 0), n - w)
        js = np.arange(j0, j0 + w)
        idx[i] = js
        wt[i] = _lagrange_deriv_weights(X[js], X[i])
    return idx, wt


def quad_stencils(X, w=6):
    """
    Cumulative quadrature I_i = int_0^{X_i} f dX, order w-1 cumulatively.

    Row i holds the weights for the single interval [X_{i-1}, X_i] (with
    X_{-1} = 0); the caller accumulates them.  The first row extrapolates the
    integrand to X = 0 from the innermost points, which is legitimate for all
    three integrands here (H, (H-Hbar)^2/X ~ X and g are each smooth there).
    """
    n = len(X)
    idx = np.zeros((n, w), dtype=np.int32)
    wt = np.zeros((n, w))
    for i in range(n):
        a = 0.0 if i == 0 else X[i - 1]
        b = X[i]
        j0 = min(max(i - w // 2, 0), n - w)
        js = np.arange(j0, j0 + w)
        idx[i] = js
        wt[i] = _lagrange_int_weights(X[js], a, b)
    return idx, wt


@njit(cache=True)
def _dx(H, idx, wt, out):
    n = H.shape[0]; w = idx.shape[1]
    for i in range(n):
        s = 0.0
        for k in range(w):
            s += wt[i, k] * H[idx[i, k]]
        out[i] = s


@njit(cache=True)
def _cum(f, idx, wt, out):
    n = f.shape[0]; w = idx.shape[1]
    acc = 0.0
    for i in range(n):
        s = 0.0
        for k in range(w):
            s += wt[i, k] * f[idx[i, k]]
        acc += s
        out[i] = acc


# ---------------------------------------------------------------- constraints

@njit(cache=True)
def metric(X, H, hbar, g, gbar, qi, qw, tmp):
    n = X.shape[0]
    _cum(H, qi, qw, hbar)
    for i in range(n):
        hbar[i] = hbar[i] / X[i]
    for i in range(n):
        d = H[i] - hbar[i]
        tmp[i] = d * d / X[i]
    _cum(tmp, qi, qw, g)
    for i in range(n):
        g[i] = np.exp(FOURPI * g[i])
    _cum(g, qi, qw, gbar)
    for i in range(n):
        gbar[i] = gbar[i] / X[i]


@njit(cache=True)
def rhs(X, H, dH, di, dw, qi, qw, hbar, g, gbar, dHdX, tmp):
    metric(X, H, hbar, g, gbar, qi, qw, tmp)
    _dx(H, di, dw, dHdX)
    n = X.shape[0]
    for i in range(n):
        a = X[i] - 0.5 * gbar[i]
        dH[i] = -a * dHdX[i] + (g[i] - gbar[i]) * (H[i] - hbar[i]) / (2.0 * X[i])


@njit(cache=True)
def rk4(X, H, dT, nstep, di, dw, qi, qw, w):
    n = X.shape[0]
    k1 = w[0]; k2 = w[1]; k3 = w[2]; k4 = w[3]; th = w[4]
    hbar = w[5]; g = w[6]; gbar = w[7]; dHdX = w[8]; tmp = w[9]
    for _ in range(nstep):
        rhs(X, H, k1, di, dw, qi, qw, hbar, g, gbar, dHdX, tmp)
        for i in range(n):
            th[i] = H[i] + 0.5 * dT * k1[i]
        rhs(X, th, k2, di, dw, qi, qw, hbar, g, gbar, dHdX, tmp)
        for i in range(n):
            th[i] = H[i] + 0.5 * dT * k2[i]
        rhs(X, th, k3, di, dw, qi, qw, hbar, g, gbar, dHdX, tmp)
        for i in range(n):
            th[i] = H[i] + dT * k3[i]
        rhs(X, th, k4, di, dw, qi, qw, hbar, g, gbar, dHdX, tmp)
        c = dT / 6.0
        for i in range(n):
            H[i] += c * (k1[i] + 2.0 * k2[i] + 2.0 * k3[i] + k4[i])


class Sim:
    """Fixed-grid integrator of (*).  `nstep` is held fixed across a Newton
    solve so the half-period map is a smooth function of Delta (only
    dT = (Delta/2)/nstep varies)."""

    def __init__(self, N=600, Xmax=4.0, wd=5, wq=6):
        self.h = Xmax / N
        self.X = (np.arange(N) + 1.0) * self.h
        self.N, self.Xmax = N, Xmax
        self.di, self.dw = deriv_stencils(self.X, wd)
        self.qi, self.qw = quad_stencils(self.X, wq)
        self.w = np.zeros((10, N))

    # -- diagnostics ------------------------------------------------------
    def fields(self, H):
        n = self.N
        hbar = np.empty(n); g = np.empty(n); gbar = np.empty(n); tmp = np.empty(n)
        metric(self.X, np.ascontiguousarray(H, dtype=float),
               hbar, g, gbar, self.qi, self.qw, tmp)
        return hbar, g, gbar

    def dx(self, H):
        out = np.empty(self.N)
        _dx(np.ascontiguousarray(H, dtype=float), self.di, self.dw, out)
        return out

    def speed(self, H):
        _, _, gbar = self.fields(H)
        return self.X - 0.5 * gbar

    def phi0(self, H, k=6):
        """phi at the centre.  hbar = phi exactly, so this is hbar(0), reached
        by one-sided Lagrange extrapolation (hbar has no parity at X = 0)."""
        hbar, _, _ = self.fields(H)
        return float(np.polyval(np.polyfit(self.X[:k], hbar[:k], k - 1), 0.0))

    # -- evolution --------------------------------------------------------
    def evolve(self, H, Tend, nstep):
        H = np.ascontiguousarray(H, dtype=float).copy()
        rk4(self.X, H, Tend / nstep, nstep, self.di, self.dw,
            self.qi, self.qw, self.w)
        return H

    def nstep_cfl(self, H, Tend, cfl=0.35):
        amax = max(np.abs(self.speed(H)).max(), 1e-12)
        return int(np.ceil(Tend / (cfl * self.h / amax)))

    def trace(self, H, Tend, nstep, nout=200):
        H = np.ascontiguousarray(H, dtype=float).copy()
        k = max(nstep // nout, 1)
        Ts, ph, hm = [0.0], [self.phi0(H)], [np.abs(H).max()]
        dT = Tend / nstep
        done = 0
        while done < nstep:
            m = min(k, nstep - done)
            rk4(self.X, H, dT, m, self.di, self.dw, self.qi, self.qw, self.w)
            done += m
            if not np.all(np.isfinite(H)):
                break
            Ts.append(done * dT); ph.append(self.phi0(H)); hm.append(np.abs(H).max())
        return np.array(Ts), np.array(ph), np.array(hm), H
