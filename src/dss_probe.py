"""
Locate the self-similarity horizon of the Choptuik solution in the coordinates
X = r/(u*-u), T = -ln(u*-u).

Rays obey  dX/dT = X - gbar/2.  The horizon is the *repelling periodic orbit*
of that ODE, not a zero of the right-hand side at fixed T.  Its maximum over
the cycle sets the smallest outer boundary at which a fixed-X (Eulerian)
formulation of the DSS fixed-point problem is closed: for X_max above it the
outer boundary is outflow at every phase and no boundary condition is needed.
"""
import os, sys, time
import numpy as np
import collapse, floquet as F, nullcore as nc

OUT = "out/dss"
G1_N600_PSTAR = 0.6823132832191163


def main(x=10.0, n=2400, Kdom=20.0, snap_every=200):
    os.makedirs(OUT, exist_ok=True)
    p = G1_N600_PSTAR * (1.0 - 10.0 ** (-x))
    t0 = time.time()
    R = collapse.Run(p, r0=0.25, sigma=0.05, n=n, profile="gauss",
                     Kdom=Kdom).run(snap_every=snap_every)
    rec = R["rec"]
    print(f"outcome={R['outcome']} steps={R['steps']} nsnap={len(R['snaps'])} "
          f"wall={time.time()-t0:.1f}s", flush=True)
    u, y = rec["u"], rec["phi0"]
    fit = F.fit_iterate(u, y, 4.0, 9.0, u.max() - 1e-4, niter=3)
    print(f"u*={fit['ustar']:.12f} Delta={fit['Delta']:.6f} rms={fit['rms']:.3e}",
          flush=True)
    snaps = R["snaps"]
    us = np.array([s[0] for s in snaps])
    np.savez_compressed(
        f"{OUT}/probe_x{x:g}_n{n}_K{Kdom:g}.npz",
        ustar=fit["ustar"], Delta=fit["Delta"], p=p, x=x, n=n, Kdom=Kdom,
        u=u, phi0=y, rref=rec["rref"], snap_u=us,
        **{f"r{i}": s[1] for i, s in enumerate(snaps)},
        **{f"h{i}": s[2] for i, s in enumerate(snaps)})
    print("saved", flush=True)


if __name__ == "__main__":
    a = sys.argv[1:]
    main(*( [float(a[0])] if a else [] ),
         **({} if len(a) < 2 else dict(n=int(a[1]))),
         )
