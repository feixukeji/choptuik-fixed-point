"""
Extract a seed profile H(X) for the DSS fixed-point solve.

One near-critical subcritical evolution of family G1, with profile snapshots.
u* is fitted from the central-field record (src/floquet.fit_iterate), then the
snapshots are mapped to self-similar coordinates X = r/(u*-u), H = h.
"""
import os, sys, time
import numpy as np
import collapse, floquet as F

OUT = "out/dss"
G1_N600_PSTAR = 0.6823132832191163


def main(x=8.0, n=600, snap_every=100):
    os.makedirs(OUT, exist_ok=True)
    p = G1_N600_PSTAR * (1.0 - 10.0 ** (-x))
    t0 = time.time()
    R = collapse.Run(p, r0=0.25, sigma=0.05, n=n, profile="gauss").run(
        snap_every=snap_every)
    rec = R["rec"]
    print(f"outcome={R['outcome']} steps={R['steps']} "
          f"nsnap={len(R['snaps'])} wall={time.time()-t0:.1f}s", flush=True)

    u, y = rec["u"], rec["phi0"]
    fit = F.fit_iterate(u, y, 4.0, 9.0, u.max() - 1e-4, niter=3)
    ustar = fit["ustar"]
    print(f"u*={ustar:.12f}  Delta={fit['Delta']:.6f}  rms={fit['rms']:.3e}",
          flush=True)

    snaps = R["snaps"]
    us = np.array([s[0] for s in snaps])
    T = -np.log(np.maximum(ustar - us, 1e-300))
    np.savez_compressed(
        f"{OUT}/seed_x{x:g}_n{n}.npz",
        ustar=ustar, Delta=fit["Delta"], p=p, x=x, n=n,
        u=u, phi0=y, rref=rec["rref"], snap_u=us, snap_T=T,
        **{f"r{i}": s[1] for i, s in enumerate(snaps)},
        **{f"h{i}": s[2] for i, s in enumerate(snaps)},
    )
    print("saved", f"{OUT}/seed_x{x:g}_n{n}.npz", flush=True)


if __name__ == "__main__":
    main(*[float(a) for a in sys.argv[1:2]], **({} if len(sys.argv) < 3 else
          dict(n=int(sys.argv[2]))))
