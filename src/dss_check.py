"""
Validation of the self-similar-coordinate core against direct evolution.

A snapshot of a near-critical run is mapped to (T, X), interpolated onto the
fixed X grid, and advanced with `dss_core`.  Because the near-critical solution
is not exactly DSS, the two must agree only until the growing mode separates
them -- but agreement over a full echo period at the 1e-3 level, with 4th-order
convergence under grid refinement, tests every term of the transformed
equation, the boundary treatment and the derivative stencils at once.
"""
import sys
import numpy as np
from scipy.interpolate import CubicSpline
import nullcore as nc, dss_core as dc, floquet as F


PROBE = "out/dss/probe_x10_n2400_K20.npz"


def load_probe(path=PROBE, Ta=3.4, ustar=None):
    d = np.load(path)
    us = d["snap_u"]
    if ustar is None:
        ustar = float(d["ustar"])
    T = -np.log(np.maximum(ustar - us, 1e-300))
    k = int(np.argmin(np.abs(T - Ta)))
    return d, ustar, T, k


def seed_from(path, Ta=3.4, N=400, Xmax=4.0, wd=7, ustar=None):
    """Seed straight from any snapshot file written by dss_probe/dss_seedgen."""
    d, us, T, k = load_probe(path=path, Ta=Ta, ustar=ustar)
    S, H, tau = seed(d, us, k, N=N, Xmax=Xmax, wd=wd)
    return S, H, float(T[k]), float(us)


def seed(d, ustar, k, N=600, Xmax=4.0, wd=7):
    r = d[f"r{k}"]; h = d[f"h{k}"]
    tau = ustar - d["snap_u"][k]
    X = r / tau
    S = dc.Sim(N=N, Xmax=Xmax, wd=wd)
    if S.X[0] < X[0] or S.X[-1] > X[-1]:
        raise ValueError(f"grid [{S.X[0]:.4g},{S.X[-1]:.4g}] outside data "
                         f"[{X[0]:.4g},{X[-1]:.4g}]")
    H = CubicSpline(X, h)(S.X)
    return S, H, float(tau)


def reference_phi0(d, ustar, Tgrid):
    """phi0(T) of the direct run, on the same slow-time grid."""
    u, y = d["u"], d["phi0"]
    m = u < ustar - 1e-300
    Tr = -np.log(ustar - u[m])
    return np.interp(Tgrid, Tr, y[m])


def main(Ta=3.4, span=3.4455, Xmax=4.0, ustar=None):
    for N in (300, 600, 1200):
        d, us_, T, k = load_probe(Ta=Ta, ustar=ustar)
        S, H, tau = seed(d, us_, k, N=N, Xmax=Xmax)
        ns = S.nstep_cfl(H, span, cfl=0.35)
        Ts, ph, hm, Hend = S.trace(H, span, ns, nout=400)
        ref = reference_phi0(d, us_, T[k] + Ts)
        ok = np.isfinite(ph) & np.isfinite(ref)
        e = np.abs(ph - ref)[ok]
        # error at 1/4, 1/2, 3/4, 1 of the span
        q = [np.searchsorted(Ts, span * f) - 1 for f in (0.25, 0.5, 0.75, 1.0)]
        print(f"N={N:5d} nstep={ns:6d} Xmax={Xmax} T0={T[k]:.3f} "
              f"|dphi0| at T0+{span/4:.2f},{span/2:.2f},{3*span/4:.2f},{span:.2f}: "
              + " ".join(f"{np.abs(ph[i]-ref[i]):.2e}" for i in q)
              + f"  max|H|end={np.abs(Hend).max():.4f}")
    np.savez("out/dss/check.npz", Ts=Ts, phi=ph, ref=ref, X=S.X, Hend=Hend)


if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k_, v_ = a.split("="); kw[k_] = float(v_)
    main(**kw)
