"""
How much tuning does the fixed-point route actually need?

The direct method needs the deepest bisection double precision allows (~15
digits) and still yields Delta to only ~0.1% (docs/08 A).  Here tuning enters
only through the SEED; Newton then solves an equation that knows nothing about
initial data.  So the question is: from how shallow a bisection can a seed
still be built?

The seed needs neither u* nor a prior value of Delta.

  * u*:  on a single slice u = u_snap, X = r/(u* - u_snap) differs from r/rho by
         one constant factor, so a wrong accumulation point is exactly a
         rescaling of X.  rho is therefore a scan parameter, not a quantity to
         estimate -- which removes the u* estimator (and its own tuning
         requirement) from the pipeline.  rho is scanned in units of the
         energy-weighted feature radius r_ref of that snapshot.
  * Delta: the seed is chosen as the (snapshot, rho, Delta_trial) triple that
         minimises the antiperiodicity residual

             q = || S_{Delta_trial/2}[H] + H || / || H ||,

         so the search itself is a crude estimator of Delta, refined by Newton
         to ~1e-8.  Scanning all snapshots matters: q varies by an order of
         magnitude along a run, and an earlier version of this script sampled
         three of ~300 and concluded, wrongly, that shallow seeds never work.

Reference point: the seed used for the main solve (docs/09) has q = 0.19.
"""
import os, sys, glob, json, time
import numpy as np
from scipy.interpolate import CubicSpline
import nullcore as nc, dss_core as dc, dss_newton as dn

OUT = "out/dss"
SEEDS = "out/dss/seeds"
RHOS = (1.8, 2.2, 2.6, 3.0, 3.5, 4.0)
DGRID = np.arange(2.60, 4.401, 0.15)


def rref(r, h):
    """Energy-weighted radius, from the snapshot alone (no record fields)."""
    n = len(r)
    hb = np.empty(n); g = np.empty(n); gb = np.empty(n)
    re = np.empty(n + 1); ye = np.empty(n + 1); I = np.empty(n + 1)
    nc.metric_fields(r, h, hb, g, gb, re, ye, I)
    w = (h - hb) ** 2 / r
    return float(np.trapezoid(w * r, r) / np.trapezoid(w, r))


def scan(path, N=200, Xmax=4.0, wd=7, stride=3, rhos=RHOS, Dg=DGRID,
         nstep_cap=6000):
    """Best (snapshot, rho, Delta_trial) by antiperiodicity residual."""
    d = np.load(path)
    S = dc.Sim(N=N, Xmax=Xmax, wd=wd)
    ns = len(d["snap_u"])
    best = None
    for i in range(0, ns, stride):
        r, h = d[f"r{i}"], d[f"h{i}"]
        if len(r) < 40:
            continue
        rr = rref(r, h)
        for m in rhos:
            X = r / (rr * m)
            if S.X[0] < X[0] or S.X[-1] > X[-1]:
                continue
            H = CubicSpline(X, h)(S.X)
            nb = S.nstep_cfl(H, 0.5 * Dg.max(), 0.30)
            if nb > nstep_cap:
                continue
            nH = np.linalg.norm(H)
            for D in Dg:
                k = max(int(nb * D / Dg.max()), 8)
                He = S.evolve(H, 0.5 * D, k)
                if not np.all(np.isfinite(He)):
                    continue
                q = np.linalg.norm(He + H) / nH
                if best is None or q < best[0]:
                    best = (float(q), i, float(rr * m), float(D), float(m))
    return S, d, best


def refine(S, d, best, k_ms=0, cfl=0.30, D0s=None):
    """
    Newton from the scanned seed, over a grid of Delta_0.

    The scan's own Delta minimiser is only accurate to the scan grid and can
    land outside Newton's basin (which is contained in (3.0, 3.8) at N = 200,
    docs/09 section 5) -- G1_x5 failed that way with a BETTER seed than T1_x5,
    which succeeded.  Separating the two, the acceptance test stays blind: the
    criterion is the residual, never agreement with a published Delta.

    k_ms > 0 also tries multiple shooting, which docs/09 section 5b measured to
    be *worse* here (plain GMRES does not converge on the block-bidiagonal
    system), so it is off by default.
    """
    q, i, rho, Dtrial, m = best
    r, h = d[f"r{i}"], d[f"h{i}"]
    H0 = CubicSpline(r / rho, h)(S.X)
    nstep = S.nstep_cfl(H0, 0.5 * dn.DELTA_REF, cfl)
    if D0s is None:
        D0s = np.arange(2.6, 4.401, 0.2)
    res = {}
    bestn = (np.inf, np.nan)
    for D0 in np.concatenate([[Dtrial], D0s]):
        try:
            z, nr, P = dn.solve(S, H0, float(D0), nstep, verbose=False, maxit=30)
        except Exception:
            continue
        if np.isfinite(nr) and nr < bestn[0]:
            bestn = (float(nr), float(z[-1]))
        if bestn[0] < 1e-10:
            break
    res["single"] = (bestn[1], bestn[0])
    if k_ms:
        try:
            _, D, nr, P = dn.solve_ms(S, H0, Dtrial, nstep, k=k_ms,
                                      verbose=False)
            res["ms"] = (float(D), float(nr))
        except Exception:
            res["ms"] = (np.nan, np.inf)
    else:
        res["ms"] = (np.nan, np.inf)
    return res


def depth(fam="*", N=200, Xmax=4.0, stride=3, k_ms=0):
    rows = []
    for path in sorted(glob.glob(f"{SEEDS}/{fam}_x*.npz"),
                       key=lambda z: (z.split("/")[-1].split("_")[0],
                                      int(z.split("_x")[1][:-4]))):
        tag = os.path.basename(path)[:-4]
        t0 = time.time()
        S, d, best = scan(path, N=N, Xmax=Xmax, stride=stride)
        if best is None:
            print(f"  {tag}: no usable snapshot", flush=True); continue
        q, i, rho, Dtrial, m = best
        res = refine(S, d, best, k_ms=k_ms)
        # Acceptance is on the Newton residual ALONE, as the paper says.
        # No comparison with a reference period enters; over the 28 records
        # taken with the earlier rule (which also required
        # |Delta - DELTA_REF| < 2e-3) the classification is identical.
        ok = {k: (v[1] < 1e-8) for k, v in res.items()}
        rows.append(dict(tag=tag, x=int(d["x"]), nbisect=int(d["nbisect"]),
                         q=q, isnap=i, rho_over_rref=m, Dtrial=Dtrial,
                         single=res["single"], ms=res["ms"],
                         ok_single=ok["single"], ok_ms=ok["ms"]))
        print(f"  {tag}: nbisect={int(d['nbisect']):3d} q={q:.3f} "
              f"snap={i:4d} rho/rref={m:.1f} Dscan={Dtrial:.2f} | "
              f"Newton {res['single'][0]:.8f}/{res['single'][1]:.0e} "
              f"{'OK' if ok['single'] else '--'}"
              + (f" | ms{k_ms} {res['ms'][0]:.8f}/{res['ms'][1]:.0e} "
                 f"{'OK' if ok['ms'] else '--'}" if k_ms else "")
              + f"  ({time.time()-t0:.0f}s)", flush=True)
    return rows


if __name__ == "__main__":
    os.makedirs(OUT, exist_ok=True)
    kw = {k: (int(v) if k in ("N", "stride", "k_ms") else v)
          for k, v in (a.split("=") for a in sys.argv[2:])}
    fam = sys.argv[1] if len(sys.argv) > 1 else "*"
    res = depth(fam=fam, **kw)
    with open(f"{OUT}/seedtest_depth_{fam}.json", "w") as f:
        json.dump(res, f, indent=1, default=float)
