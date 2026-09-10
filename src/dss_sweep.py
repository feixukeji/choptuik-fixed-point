"""Convergence and robustness of the DSS fixed point."""
import sys, time, json
import numpy as np
import dss_core as dc, dss_check as ck, dss_newton as dn

OUT = "out/dss"


def one(N=400, Xmax=4.0, wd=7, Delta0=3.30, Ta=3.4, cfl=0.30, jac=False,
        cflmul=1.0):
    d, us, T, k = ck.load_probe(Ta=Ta)
    S, H0, _ = ck.seed(d, us, k, N=N, Xmax=Xmax, wd=wd)
    nstep = int(round(S.nstep_cfl(H0, 0.5 * dn.DELTA_REF, cfl) * cflmul))
    t0 = time.time()
    z, nr, P = dn.solve(S, H0, Delta0, nstep, verbose=False)
    H, Delta = z[:-1], z[-1]
    r = dict(N=N, Xmax=Xmax, wd=wd, Delta0=Delta0, Ta=float(T[k]), cfl=cfl,
             nstep=nstep, Delta=float(Delta), resid=float(nr),
             wall=time.time() - t0, nmap=P.nmap)
    print(f"N={N:5d} Xmax={Xmax:g} wd={wd} nstep={nstep:6d} D0={Delta0:.2f} "
          f"Ta={T[k]:.3f} -> Delta={Delta:.10f} "
          f"relerr={(Delta-dn.DELTA_REF)/dn.DELTA_REF:+.3e} |R|={nr:.1e} "
          f"({time.time()-t0:.0f}s)", flush=True)
    if jac:
        t1 = time.time()
        J, _ = dn.jacobian(S, H, Delta, nstep)
        nu, lam, V = dn.spectrum(J, Delta)
        r["nu_abs"] = np.abs(nu).tolist()      # complex is not JSON-serialisable
        r["lam_re"] = lam.real.tolist()
        r["lam_im"] = lam.imag.tolist()
        np.savez_compressed(f"{OUT}/spec_N{N}_X{Xmax:g}_w{wd}.npz",
                            X=S.X, H=H, Delta=Delta, J=J, nu=nu, lam=lam)
        print(f"   spectrum ({time.time()-t1:.0f}s): " +
              " ".join(f"{l.real:+.4f}{l.imag:+.4f}j" for l in lam[:8]), flush=True)
    return r


def main(which="conv"):
    res = []
    if which == "conv":
        for N in (100, 200, 400, 800, 1600):
            res.append(one(N=N))
    elif which == "robust":
        for D0 in (2.60, 3.00, 3.80, 4.40):
            res.append(one(N=200, Delta0=D0))
        for Ta in (2.6, 3.0, 3.8, 4.2, 4.6):
            res.append(one(N=200, Ta=Ta))
        for Xmax in (2.8, 3.2, 5.0, 6.0):
            res.append(one(N=200, Xmax=Xmax))
        for wd in (5, 9):
            res.append(one(N=200, wd=wd))
        for m in (0.5, 2.0):
            res.append(one(N=200, cflmul=m))
    elif which == "spec":
        for N in (200, 400, 800):
            res.append(one(N=N, jac=True))
    with open(f"{OUT}/sweep_{which}.json", "w") as f:
        json.dump(res, f, indent=1)


if __name__ == "__main__":
    main(*sys.argv[1:])
