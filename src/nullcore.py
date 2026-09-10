"""Numba-compiled core of the Christodoulou/Garfinkle null evolution."""
import numpy as np
from numba import njit

FOURPI = 4.0 * np.pi


@njit(cache=True, fastmath=False)
def cumquad(x, y, out):
    """Cumulative integral of y dx (local parabola on each interval)."""
    n = x.shape[0]
    out[0] = 0.0
    if n < 3:
        for i in range(1, n):
            out[i] = out[i - 1] + 0.5 * (x[i] - x[i - 1]) * (y[i] + y[i - 1])
        return
    # first interval, stencil (0,1,2)
    h1 = x[1] - x[0]
    h2 = x[2] - x[1]
    d01 = (y[1] - y[0]) / h1
    d12 = (y[2] - y[1]) / h2
    d012 = (d12 - d01) / (x[2] - x[0])
    out[1] = out[0] + y[0] * h1 + d01 * 0.5 * h1 * h1 - d012 * h1 * h1 * h1 / 6.0
    for i in range(1, n - 1):
        a = x[i - 1]
        hh1 = x[i] - a
        hh2 = x[i + 1] - x[i]
        e01 = (y[i] - y[i - 1]) / hh1
        e12 = (y[i + 1] - y[i]) / hh2
        e012 = (e12 - e01) / (x[i + 1] - a)
        u1 = hh1
        u2 = hh1 + hh2
        du2 = u2 * u2 - u1 * u1
        du3 = u2 * u2 * u2 - u1 * u1 * u1
        out[i + 1] = out[i] + (y[i - 1] * hh2 + e01 * 0.5 * du2
                               + e012 * (du3 / 3.0 - hh1 * 0.5 * du2))


@njit(cache=True)
def _extrap0(x, y):
    """Quadratic Lagrange extrapolation of y to x = 0 from three innermost points."""
    x0 = x[0]; x1 = x[1]; x2 = x[2]
    return (y[0] * (x1 * x2) / ((x0 - x1) * (x0 - x2))
            + y[1] * (x0 * x2) / ((x1 - x0) * (x1 - x2))
            + y[2] * (x0 * x1) / ((x2 - x0) * (x2 - x1)))


@njit(cache=True)
def metric_fields(r, h, hbar, g, gbar, re, ye, I):
    """Fill hbar, g, gbar.  re/ye/I are scratch arrays of length n+1."""
    n = r.shape[0]
    re[0] = 0.0
    for i in range(n):
        re[i + 1] = r[i]
    # hbar
    ye[0] = _extrap0(r, h)
    for i in range(n):
        ye[i + 1] = h[i]
    cumquad(re, ye, I)
    for i in range(n):
        hbar[i] = I[i + 1] / r[i]
    # ln g  =  4 pi \int q dr,  q = (h - hbar)^2 / r
    ye[0] = 0.0
    for i in range(n):
        d = h[i] - hbar[i]
        ye[i + 1] = d * d / r[i]
    cumquad(re, ye, I)
    for i in range(n):
        g[i] = np.exp(FOURPI * I[i + 1])
    # gbar
    ye[0] = 1.0
    for i in range(n):
        ye[i + 1] = g[i]
    cumquad(re, ye, I)
    for i in range(n):
        gbar[i] = I[i + 1] / r[i]


@njit(cache=True)
def rhs(r, h, dr, dh, hbar, g, gbar, re, ye, I):
    metric_fields(r, h, hbar, g, gbar, re, ye, I)
    n = r.shape[0]
    for i in range(n):
        dh[i] = (g[i] - gbar[i]) * (h[i] - hbar[i]) / (2.0 * r[i])
        dr[i] = -0.5 * gbar[i]


@njit(cache=True)
def rk4(r, h, du, w):
    """
    One classical RK4 step in u.  `w` is a (14, n) scratch workspace plus
    two length-(n+1) buffers appended by the caller.
    """
    n = r.shape[0]
    k1r = w[0][:n]; k1h = w[1][:n]; k2r = w[2][:n]; k2h = w[3][:n]
    k3r = w[4][:n]; k3h = w[5][:n]; k4r = w[6][:n]; k4h = w[7][:n]
    tr = w[8][:n]; th = w[9][:n]
    hbar = w[10][:n]; g = w[11][:n]; gbar = w[12][:n]
    re = w[13][:n + 1]; ye = w[14][:n + 1]; I = w[15][:n + 1]
    rhs(r, h, k1r, k1h, hbar, g, gbar, re, ye, I)
    for i in range(n):
        tr[i] = r[i] + 0.5 * du * k1r[i]; th[i] = h[i] + 0.5 * du * k1h[i]
    rhs(tr, th, k2r, k2h, hbar, g, gbar, re, ye, I)
    for i in range(n):
        tr[i] = r[i] + 0.5 * du * k2r[i]; th[i] = h[i] + 0.5 * du * k2h[i]
    rhs(tr, th, k3r, k3h, hbar, g, gbar, re, ye, I)
    for i in range(n):
        tr[i] = r[i] + du * k3r[i]; th[i] = h[i] + du * k3h[i]
    rhs(tr, th, k4r, k4h, hbar, g, gbar, re, ye, I)
    c = du / 6.0
    for i in range(n):
        r[i] += c * (k1r[i] + 2.0 * k2r[i] + 2.0 * k3r[i] + k4r[i])
        h[i] += c * (k1h[i] + 2.0 * k2h[i] + 2.0 * k3h[i] + k4h[i])
