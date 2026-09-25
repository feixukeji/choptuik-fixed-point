"""
Floquet spectroscopy of a critical solution from nonlinear evolution data.

Motivation
----------
Martin-Garcia & Gundlach (1999) could measure the perturbation eigenvalues of
the Choptuik solution only because the critical solution had first been
CONSTRUCTED, so that the equations could be linearised about it and the
resulting linear system evolved.

This module asks whether the least-damped decaying mode lambda_1 can instead
be read directly off a tuned NONLINEAR evolution, using nothing but the
recorded time series of one central observable.  The spherical massless
scalar is the validation case, because
there both Delta = 3.445452402(3) and lambda_0 = 1/gamma = 2.674 are known
independently.

Model
-----
With slow time T = -ln(u* - u), a near-critical solution is

    y(T) = sum_j C_j e^{lambda_j T} f_j(T),      f_j(T + Delta) = f_j(T)

so that every mode contributes a COMB of complex exponentials
lambda_j + 2 pi i n / Delta.  The modes present are

    lambda      = 0          the critical solution itself
    lambda_g    = 1          the gauge mode:  u* -> u* + d  sends
                             T -> T - d e^{T}, i.e. an error in the
                             accumulation point IS a mode with exponent
                             exactly 1.  (The field equations are autonomous
                             in u, so the shifted solution is again a
                             solution.)
    lambda_0    = 1/gamma    the one growing physical mode, amplitude ~ (p-p*)
    lambda_1... the decaying physical modes -- the target.

Because the gauge mode is degenerate with u* itself, u* is not a nuisance
parameter to be eliminated but the amplitude of a mode in the fit: fitting
(u*, Delta) jointly IS nulling the gauge mode.  The scalar field of the
Choptuik solution obeys phi(T + Delta/2) = -phi(T), so its comb contains only
odd n; the even-n amplitudes are fitted as well and serve as a null test.
"""
import numpy as np
from scipy.linalg import cho_factor, cho_solve
from scipy.optimize import minimize

DELTA_REF = 3.445452402      # Gundlach 1997, exact-solution construction
GAMMA_REF = 0.374
LAM0_REF = 1.0 / GAMMA_REF   # = 2.6738, the growing mode
LAM_GAUGE = 1.0


# ---------------------------------------------------------------- design
def _comb(T, Delta, M, parity):
    """cos/sin columns at n = 1,3,5,... ('odd') or n = 0,2,4,... ('even')."""
    ns = range(1, M + 1, 2) if parity == "odd" else range(0, M + 1, 2)
    cols = []
    for n in ns:
        a = (2 * np.pi * n / Delta) * T
        cols.append(np.cos(a))
        if n:
            cols.append(np.sin(a))
    return cols


def design(T, Delta, M=9, Mg=1, lam0=LAM0_REF, Tref=None):
    """
    Background comb (both parities) plus, optionally, a low-order comb
    multiplying e^{lam0 T} for the growing mode.  Returns the matrix and the
    sizes of the odd, even and growing blocks.
    """
    odd = _comb(T, Delta, M, "odd")
    even = _comb(T, Delta, M, "even")
    cols = odd + even
    ng = 0
    if Mg is not None and lam0:
        Tref = T.max() if Tref is None else Tref
        E = np.exp(lam0 * (T - Tref))
        gr = [E * c for c in (_comb(T, Delta, Mg, "odd")
                              + _comb(T, Delta, Mg, "even"))]
        cols += gr
        ng = len(gr)
    return np.vstack(cols).T, len(odd), len(even), ng


def _solve(X, y, ridge=1e-11):
    nrm = np.linalg.norm(X, axis=0)
    nrm[nrm == 0] = 1.0
    Xn = X / nrm
    A = Xn.T @ Xn
    A[np.diag_indices_from(A)] += ridge * np.trace(A) / A.shape[0]
    try:
        c = cho_solve(cho_factor(A), Xn.T @ y)
    except Exception:
        c, *_ = np.linalg.lstsq(Xn, y, rcond=None)
    return c / nrm, y - Xn @ c


# ---------------------------------------------------------------- fitting
def _rss_q(u, y, ub, q, Delta, M, Mg, lam0):
    ustar = ub + np.exp(-q)
    T = -np.log(ustar - u)
    X, *_ = design(T, Delta, M, Mg, lam0, Tref=q)
    _, r = _solve(X, y)
    return float(r @ r / len(y))


def fit_background(u, y, ub, M=9, Mg=1, lam0=LAM0_REF,
                   qgrid=None, Dgrid=None, nstart=4, seed=None):
    """
    Joint estimate of the accumulation point u* and the echoing period Delta
    from the discrete self-similarity of a single time series.

    u* is written as ub + exp(-q), where ub is the last retained sample, so q
    is the slow time at the end of the analysis window and u* is guaranteed to
    lie beyond the window (but, for a subcritical run, generally INSIDE the
    full record: the evolution continues past u* after it leaves the critical
    solution).  The nonlinear search is over (q, Delta) only; the harmonic
    amplitudes are linear and are eliminated exactly.
    """
    u = np.asarray(u, float); y = np.asarray(y, float)
    qgrid = np.arange(4.0, 16.01, 0.25) if qgrid is None else qgrid
    Dgrid = np.arange(3.00, 4.001, 0.010) if Dgrid is None else Dgrid
    if seed is not None:
        seeds = [seed]
        V = None
    else:
        seeds = None
    V = np.empty((len(qgrid), len(Dgrid))) if seeds is None else None
    for i, q in (enumerate(qgrid) if seeds is None else []):
        T = -np.log(ub + np.exp(-q) - u)
        for j, D in enumerate(Dgrid):
            X, *_ = design(T, D, M, Mg, lam0, Tref=q)
            _, r = _solve(X, y)
            V[i, j] = r @ r
    if seeds is None:
        P = np.pad(V, 1, constant_values=np.inf)
        loc = np.ones_like(V, bool)
        for di in (-1, 0, 1):
            for dj in (-1, 0, 1):
                if di or dj:
                    loc &= V <= P[1 + di:1 + di + V.shape[0],
                                  1 + dj:1 + dj + V.shape[1]]
        ii, jj = np.where(loc)
        seeds = [(qgrid[ii[k]], Dgrid[jj[k]])
                 for k in np.argsort(V[ii, jj])[:nstart]]

    f = lambda p: _rss_q(u, y, ub, p[0], p[1], M, Mg, lam0)
    best = None
    for s in seeds:
        r = minimize(f, s, method="Nelder-Mead",
                     options=dict(xatol=1e-12, fatol=1e-20,
                                  maxiter=3000, maxfev=5000))
        if best is None or r.fun < best.fun:
            best = r
    q, D = best.x
    ustar = ub + np.exp(-q)
    T = -np.log(ustar - u)
    X, no, ne, ng = design(T, D, M, Mg, lam0, Tref=q)
    c, res = _solve(X, y)
    a1, b1 = c[0], c[1]
    return dict(ustar=float(ustar), Delta=float(D), q=float(q),
                rms=float(np.sqrt(best.fun)), coef=c, T=T, resid=res,
                nodd=no, neven=ne, ngrow=ng, M=M, Mg=Mg, lam0=lam0, ub=float(ub),
                amp_odd=float(np.sqrt(np.sum(c[:no] ** 2))),
                amp_even=float(np.sqrt(np.sum(c[no:no + ne] ** 2))),
                amp_grow=float(np.sqrt(np.sum(c[no + ne:] ** 2))),
                theta=float(np.arctan2(b1, a1) * D / (2 * np.pi)))


def fit_iterate(u, y, Ta, Tb, ustar0, niter=3, npts=800, **kw):
    """fit_background with the analysis window itself iterated to convergence."""
    u = np.asarray(u, float); y = np.asarray(y, float)
    out = None
    for _ in range(niter):
        ua, ub = ustar0 - np.exp(-Ta), ustar0 - np.exp(-Tb)
        m = (u >= ua) & (u <= ub)
        if m.sum() < 200:
            return None
        uu, yy = u[m], y[m]
        if len(uu) > npts:
            k = np.unique(np.linspace(0, len(uu) - 1, npts).astype(int))
            uu, yy = uu[k], yy[k]
        out = fit_background(uu, yy, ub=float(uu.max()),
                             seed=(out["q"], out["Delta"]) if out else None, **kw)
        if abs(out["ustar"] - ustar0) < 1e-14:
            ustar0 = out["ustar"]; break
        ustar0 = out["ustar"]
    out["window"] = (float(Ta), float(Tb))
    return out


# ------------------------------------------------------- envelope / decay
def period_rms(T, r, Delta, Tq=None, n=400):
    """RMS of r over one echoing period centred on each query time."""
    o = np.argsort(T); T, r = np.asarray(T)[o], np.asarray(r)[o]
    if Tq is None:
        Tq = np.linspace(T.min() + Delta / 2, T.max() - Delta / 2, n)
    c = np.concatenate([[0.0], np.cumsum(np.diff(T) * 0.5 *
                                         (r[1:] ** 2 + r[:-1] ** 2))])
    lo = np.interp(Tq - Delta / 2, T, c)
    hi = np.interp(Tq + Delta / 2, T, c)
    return Tq, np.sqrt(np.maximum(hi - lo, 0.0) / Delta)


def loglinfit(x, y):
    """Slope, intercept and R^2 of ln y against x."""
    x = np.asarray(x, float); y = np.asarray(y, float)
    m = y > 0
    x, ly = x[m], np.log(y[m])
    A = np.vstack([x, np.ones_like(x)]).T
    c, *_ = np.linalg.lstsq(A, ly, rcond=None)
    r = ly - A @ c
    ss = float(np.sum((ly - ly.mean()) ** 2))
    return float(c[0]), float(c[1]), float(1 - r @ r / ss if ss > 0 else np.nan)


def sliding_slope(x, y, W):
    """Local decay rate d ln y / dx in a sliding window of width W."""
    out = []
    for xc in x:
        m = np.abs(x - xc) <= W / 2
        if m.sum() > 8 and np.all(y[m] > 0):
            out.append((xc, loglinfit(x[m], y[m])[0]))
    if not out:
        return np.array([]), np.array([])
    return np.array([o[0] for o in out]), np.array([o[1] for o in out])


def resample_T(u, y, ustar, Tg):
    """Interpolate a record onto a slow-time grid; NaN outside its support."""
    u = np.asarray(u, float); y = np.asarray(y, float)
    m = u < ustar - 1e-15
    T = -np.log(ustar - u[m])
    o = np.argsort(T)
    return np.interp(Tg, T[o], y[m][o], left=np.nan, right=np.nan)
