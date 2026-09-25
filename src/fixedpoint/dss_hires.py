"""
High-resolution Chebyshev spectra: push N until the exponents below lam1 either
converge or are shown to be domain-dependent (hence truncation artefacts).

Two changes over dss_spec_cheb:

1. The dense Jacobian is formed in parallel over columns.  Each column is two
   independent map evaluations, so this is embarrassingly parallel; at N = 288
   it turns a 16-hour serial job into about an hour.

2. Newton is warm-started from the converged fixed point of the next lower N
   (cubic spline onto the new collocation points, plus that run's Delta).  The
   *answer* cannot depend on the seed -- dss_seedtest establishes that over
   four initial-data families and eight tuning depths -- so this only buys
   Newton iterations.  The cold-start (probe-seed, Delta0 = 3.30) runs that
   support the no-tuning claim are the ones in dss_sweep/dss_spec_cheb.
"""
import os, sys, time
import numpy as np
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from paths import out
from fixedpoint import dss_cheb as cb, dss_newton as dn, dss_spec_cheb as sc

OUT = out("fixedpoint")
_G = {}


def _col(j):
    S, H, Delta, nstep, sc_ = _G["S"], _G["H"], _G["D"], _G["nstep"], _G["sc"]
    e = np.zeros(S.N); e[j] = sc_
    return j, ((S.evolve(H + e, 0.5 * Delta, nstep)
                - S.evolve(H - e, 0.5 * Delta, nstep)) / (2.0 * sc_))


def jacobian_mp(S, H, Delta, nstep, eps=1e-6, nproc=None):
    import multiprocessing as mp
    n = S.N
    _G.update(S=S, H=H, D=Delta, nstep=nstep,
              sc=eps * max(np.linalg.norm(H) / np.sqrt(n), 1e-12))
    J = np.empty((n, n))
    nproc = nproc or int(os.environ.get("SLURM_CPUS_PER_TASK", 8))
    with mp.Pool(nproc) as pool:
        for j, c in pool.imap_unordered(_col, range(n), chunksize=2):
            J[:, j] = c
    return J


def warm_seed(S, src):
    """Interpolate a saved fixed point onto S's collocation points."""
    f = np.load(src)
    return CubicSpline(f["X"], f["H"])(S.X), float(f["Delta"])


def run(N, Xmax=4.0, src=None, cfl=0.30, nproc=None, jac=True, tol=1e-11):
    S = cb.Sim(N=N, Xmax=Xmax)
    if src and os.path.exists(src):
        H0, D0 = warm_seed(S, src)
        how = f"warm({os.path.basename(src)})"
    else:
        H0, _ = sc.seed_on(S, Ta=3.4)
        D0, how = 3.30, "cold(probe,3.30)"
    nstep = S.nstep_cfl(H0, 0.5 * dn.DELTA_REF, cfl)
    t0 = time.time()
    z, nr, P = dn.solve(S, H0, D0, nstep, verbose=False, maxit=30, tol=tol)
    H, Delta = z[:-1], z[-1]
    print(f"N={N:4d} Xmax={Xmax:g} {how} nstep={nstep} -> "
          f"Delta={Delta:.10f} relerr={(Delta-dn.DELTA_REF)/dn.DELTA_REF:+.3e} "
          f"|R|={nr:.2e} maps={P.nmap} ({time.time()-t0:.0f}s)", flush=True)
    out = dict(N=N, Xmax=Xmax, X=S.X, H=H, Delta=Delta, resid=nr, nstep=nstep)
    if jac and nr < 1e-8:
        t1 = time.time()
        J = jacobian_mp(S, H, Delta, nstep, nproc=nproc)
        nu, lam, V = dn.spectrum(J, Delta)
        out.update(J=J, nu=nu, lam=lam)
        print(f"   N={N:4d} Xmax={Xmax:g} spectrum ({time.time()-t1:.0f}s): " +
              " ".join(f"{l.real:+.5f}{l.imag:+.4f}j" for l in lam[:10]),
              flush=True)
    np.savez_compressed(f"{OUT}/cheb_N{N}_X{Xmax:g}.npz", **out)
    return out


def chain(Ns, Xmax=4.0, start=None, **kw):
    """Sequential warm-started sweep in N at fixed Xmax."""
    src = start
    for N in Ns:
        run(N, Xmax=Xmax, src=src, **kw)
        src = f"{OUT}/cheb_N{N}_X{Xmax:g}.npz"


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1]
    if mode == "N":                 # push resolution at Xmax = 4
        chain([240, 256, 272, 288], Xmax=4.0,
              start=f"{OUT}/cheb_N224_X4.npz")
    elif mode == "X":
        # Domain test at the highest converged N.  Warm seeds are chosen so the
        # spline never extrapolates past the source domain: X = 3 comes from
        # the N = 224 solution on [0, 4], X = 5 and 6 from the N = 176 runs on
        # the same domain.  Then push N = 224 -> 256 at X = 6, where the N = 176
        # value (-0.923) was the one outlier.
        run(224, Xmax=3.0, src=f"{OUT}/cheb_N224_X4.npz")
        run(224, Xmax=5.0, src=f"{OUT}/cheb_N176_X5.npz")
        run(224, Xmax=6.0, src=f"{OUT}/cheb_N176_X6.npz")
        run(256, Xmax=6.0, src=f"{OUT}/cheb_N224_X6.npz")
    else:
        run(int(mode), Xmax=float(sys.argv[2]) if len(sys.argv) > 2 else 4.0)
