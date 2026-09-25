"""
The physical (analytic) Floquet spectrum of the Choptuik solution, from the
Chebyshev-collocation map.

The finite-difference map of dss_core resolves the fixed point superbly but its
map spectrum contains the non-analytic continuum (X - X_h)^s described in
dss_cheb: at N = 200 vs 400 only lam0, the two lam = 0 modes and the lam = 1
gauge mode survive refinement.  Those branches do not appear in the Chebyshev
map spectrum at any resolution run here, which is why the decaying exponents
are read off this side.  That is an observation about the discrete spectra and
not a theorem about the continuum operator: a limit of polynomials need not be
analytic, so the global basis is a smoothness *selection*, matched on the
banded side by the Chebyshev filter of dss_lam1_banded, and not a proof that
the excluded branches are absent from the problem.  The target class is
perturbations smooth across the similarity horizon, chosen because
low-Holder-regularity modes at the horizon are not what smooth initial data
produce -- not because a continuous spectrum could not organize a
codimension-one threshold, which it could.
"""
import os, sys, time
import numpy as np
from scipy.interpolate import CubicSpline
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from paths import out
from fixedpoint import dss_cheb as cb, dss_newton as dn, dss_check as ck

OUT = out("fixedpoint")


def seed_on(S, Ta=3.4, path=None):
    d, us, T, k = ck.load_probe(path=path or ck.PROBE, Ta=Ta)
    r, h = d[f"r{k}"], d[f"h{k}"]
    tau = us - d["snap_u"][k]
    return CubicSpline(r / tau, h)(S.X), float(T[k])


def run(N=96, Xmax=4.0, Delta0=3.30, cfl=0.30, Ta=3.4, jac=True):
    S = cb.Sim(N=N, Xmax=Xmax)
    H0, T0 = seed_on(S, Ta=Ta)
    nstep = S.nstep_cfl(H0, 0.5 * dn.DELTA_REF, cfl)
    t0 = time.time()
    z, nr, P = dn.solve(S, H0, Delta0, nstep, verbose=False, maxit=30)
    H, Delta = z[:-1], z[-1]
    print(f"N={N:4d} nstep={nstep:6d} -> Delta={Delta:.10f} "
          f"relerr={(Delta-dn.DELTA_REF)/dn.DELTA_REF:+.3e} |R|={nr:.2e} "
          f"maps={P.nmap} ({time.time()-t0:.0f}s)", flush=True)
    out = dict(N=N, Xmax=Xmax, X=S.X, H=H, Delta=Delta, resid=nr, nstep=nstep,
               T0=T0)
    if jac and nr < 1e-8:
        t1 = time.time()
        J, _ = dn.jacobian(S, H, Delta, nstep)
        nu, lam, V = dn.spectrum(J, Delta)
        out.update(J=J, nu=nu, lam=lam)
        print(f"   N={N:4d} spectrum ({time.time()-t1:.0f}s): " +
              " ".join(f"{l.real:+.5f}{l.imag:+.4f}j" for l in lam[:10]),
              flush=True)
    np.savez_compressed(f"{OUT}/cheb_N{N}_X{Xmax:g}.npz", **out)
    return out


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    for a in sys.argv[1:]:
        run(N=int(a))
