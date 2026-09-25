"""Minimum of the ray speed A = X - gbar/2 on a fixed outer boundary.

Sec. II C of the paper needs  min_T A(T, X_max) > 0  for the outer boundary to
be outflow at every phase.  That is NOT implied by X_max > max_T X_h: the
self-similarity horizon is an ORBIT of dX/dT = A, not a zero of A, and the
instantaneous zeros of A reach further out than the orbit does.  This script
measures both, on every converged fixed point.
"""
import os
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from paths import out
from fixedpoint import dss_core as dc, dss_cheb as cb

OUT = out("fixedpoint")


def cycle(kind, N, Xmax, wd=7, nT=800):
    """Sample A(T, .) over one half period; return X, T, A."""
    if kind == "fd":
        f = np.load(f"{OUT}/spec_N{N}_X{Xmax:g}_w{wd}.npz")
        S = dc.Sim(N=N, Xmax=Xmax, wd=wd)
    else:
        f = np.load(f"{OUT}/cheb_N{N}_X{Xmax:g}.npz")
        S = cb.Sim(N=N, Xmax=Xmax)
    H, D = f["H"], float(f["Delta"])
    ns = S.nstep_cfl(H, 0.5 * D, 0.30)
    k = max(ns // nT, 1)
    Hc, done, dT, A, Ts = H.copy(), 0, 0.5 * D / ns, [], []
    while True:
        hb, g, gb = S.fields(Hc)
        A.append(S.X - 0.5 * gb); Ts.append(done * dT)
        if done >= ns:
            break
        m = min(k, ns - done); Hc = S.evolve(Hc, m * dT, m); done += m
    return S.X, np.array(Ts), np.array(A), D


def zeros(X, A):
    """Outermost instantaneous zero of A at each phase."""
    Z = []
    for a in A:
        i = np.where(np.diff(np.sign(a)) != 0)[0]
        if len(i) == 0:
            Z.append(np.nan); continue
        Z.append(brentq(CubicSpline(X, a), X[i[-1]], X[i[-1] + 1]))
    return np.array(Z)


def main():
    runs = [("fd", 800, 4.0)] + [("cheb", 224, x) for x in (3.0, 4.0, 5.0, 6.0)] \
        + [("cheb", 176, 2.5)]
    rows = []
    for kind, N, Xm in runs:
        try:
            X, T, A, D = cycle(kind, N, Xm)
        except FileNotFoundError:
            print(f"  (missing {kind} N={N} X={Xm})"); continue
        Ab = A[:, -1]                      # value ON the boundary node
        Z = zeros(X, A)
        rows.append((kind, N, Xm, Ab.min(), Ab.max(),
                     np.nanmin(Z), np.nanmax(Z)))
        nb = int(np.isnan(Z).sum())
        print(f"{kind:5s} N={N:4d} Xmax={Xm:.1f}   "
              f"min_T A(T,Xmax) = {Ab.min():+.4f}   max_T A = {Ab.max():+.4f}   "
              f"outermost zero of A in [{np.nanmin(Z):.4f}, {np.nanmax(Z):.4f}]"
              + (f"   [no zero in domain at {nb}/{len(Z)} phases]" if nb else ""))

    # the interesting question: as a function of radius, on the N=800 X=4 run,
    # where does min_T A first become positive?
    X, T, A, D = cycle("fd", 800, 4.0)
    mn = A.min(axis=0)
    j = np.where(mn > 0)[0][0]
    xc = brentq(CubicSpline(X, mn), X[j - 1], X[j])
    print(f"\n  min_T A(T,X) > 0 for all X > {xc:.4f}   "
          f"(banded N=800, X_max=4)")
    print(f"  for reference: max_T X_h (orbit) = 1.5608, "
          f"max_T outermost zero = {zeros(X, A).max():.4f}")
    for xq in (2.5, 2.8, 3.0, 3.2, 4.0):
        if xq <= X[-1]:
            v = np.array([CubicSpline(X, a)(xq) for a in A])
            print(f"    X = {xq:.1f}:  min_T A = {v.min():+.5f}")
    np.savez(f"{OUT}/outflow.npz",
             rows=np.array([r[2:] for r in rows], float),
             kinds=np.array([f"{r[0]}{r[1]}" for r in rows]),
             xcrit=xc, X=X, minA=mn)


if __name__ == "__main__":
    main()
