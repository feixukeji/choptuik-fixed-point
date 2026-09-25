"""
Resolution and finite-difference-step study of the Chebyshev map spectrum.

Two things can move the decaying exponents: not enough Chebyshev modes, and
roundoff in the finite-difference Jacobian.  The map here takes ~3e4 RK4 steps,
so its accumulated roundoff divided by the difference step sets a floor on the
Jacobian entries; varying eps separates that from genuine truncation.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from paths import out
from fixedpoint import dss_cheb as cb, dss_newton as dn, dss_spec_cheb as sc

OUT = out("fixedpoint")


def respec(N, eps, Xmax=4.0):
    """Recompute the spectrum of an already-converged fixed point."""
    f = np.load(f"{OUT}/cheb_N{N}_X{Xmax:g}.npz")
    H, Delta, nstep = f["H"], float(f["Delta"]), int(f["nstep"])
    S = cb.Sim(N=N, Xmax=Xmax)
    t0 = time.time()
    J, _ = dn.jacobian(S, H, Delta, nstep, eps=eps)
    nu, lam, V = dn.spectrum(J, Delta)
    print(f"N={N} eps={eps:.0e} ({time.time()-t0:.0f}s): " +
          " ".join(f"{l.real:+.5f}{l.imag:+.4f}j" for l in lam[:10]), flush=True)
    np.savez_compressed(f"{OUT}/cheb_spec_N{N}_e{eps:.0e}.npz",
                        nu=nu, lam=lam, eps=eps, N=N, Delta=Delta)


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    mode = sys.argv[1]
    if mode == "eps":
        for e in (1e-5, 1e-4):
            respec(int(sys.argv[2]), e)
    elif mode == "xmax":
        # is the leading decaying mode a property of the interior, or of the
        # truncation?  An interior eigenvalue must not move with Xmax; that is
        # necessary, not sufficient, and at fixed N this scan also coarsens the
        # grid near the horizon.  Run it at matched near-horizon spacing (raise
        # N with Xmax) to vary the domain alone -- see make_tables._matched.
        sc.run(N=int(sys.argv[2]), Xmax=float(sys.argv[3]), jac=True)
    else:
        sc.run(N=int(sys.argv[2]), jac=True)
