"""
Seeds for the fixed-point solve from SHALLOWLY tuned evolutions.

The whole methodological claim of the fixed-point route is that it needs only a
rough starting profile, not a deeply tuned one.  This script makes that
falsifiable: for each family it bisects the amplitude and stores, for every
tuning depth x = 1, 2, ... digits, a snapshot run at the deepest subcritical
amplitude known at that point in the bisection, together with the u* that the
(u*, Delta) estimator of core/floquet.py extracts from that same (shallow) record.  Newton is then
started from each and the minimum x that still converges is measured.

Note that u* itself is only needed to define X = r/(u*-u); an error dust* in it
rescales X by 1/(1 + du* e^T), which is precisely the exact gauge mode, so the
seed degrades gracefully rather than catastrophically.
"""
import os, sys, time
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from paths import out
from core import collapse, floquet as F

OUT = out("fixedpoint", "seeds")
FAMILIES = {
    "G1": dict(profile="gauss",     r0=0.25, sigma=0.05),
    "T1": dict(profile="tanh",      r0=0.25, sigma=0.05),
    "O1": dict(profile="gauss_odd", r0=0.25, sigma=0.05),
    "G3": dict(profile="gauss",     r0=0.40, sigma=0.05),
}
NGRID, KDOM = 1200, 20.0
DIGITS = (1, 2, 3, 4, 5, 6, 8)


def bracket_history(fam, ndigit=9):
    """Bisect, recording the deepest subcritical p after each step."""
    kw = dict(FAMILIES[fam], n=NGRID, Kdom=KDOM)
    lo, hi = 0.30, 1.30
    # widen until the bracket really straddles the threshold, otherwise the
    # "deepest subcritical amplitude so far" recorded below is meaningless
    while collapse.classify(lo, **kw)[0]:
        hi, lo = lo, 0.3 * lo
    while not collapse.classify(hi, **kw)[0]:
        lo, hi = hi, 3.0 * hi
    hist = []
    for k in range(int(np.ceil(ndigit * np.log2(10))) + 4):
        pm = 0.5 * (lo + hi)
        bh, R = collapse.classify(pm, **kw)
        if R["outcome"] in ("failed", "unresolved"):
            break
        if bh:
            hi = pm
        else:
            lo = pm
        hist.append((lo, hi, R["outcome"]))
        if (hi - lo) / hi < 1e-15:
            break
    return lo, hi, hist, kw


def main(fam="G1"):
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    lo, hi, hist, kw = bracket_history(fam)
    print(f"[{fam}] bisection {len(hist)} steps, final bracket "
          f"{(hi-lo)/hi:.2e}, wall={time.time()-t0:.0f}s", flush=True)
    for x in DIGITS:
        # the first bisection step at which the bracket is narrower than 10^-x
        j = next((i for i, (a, b, _) in enumerate(hist)
                  if (b - a) / b < 10.0 ** (-x)), None)
        if j is None:
            print(f"[{fam}] x={x}: never reached", flush=True)
            continue
        p = hist[j][0]                      # deepest subcritical amplitude then
        R = collapse.Run(p, **kw).run(snap_every=60)
        rec = R["rec"]
        u, y = rec["u"], rec["phi0"]
        Tw = -np.log((hi - p) / hi) / np.log(10) * 0.374 * np.log(10)
        # fit u* on the last two thirds of the usable record
        Tb = max(Tw - 0.3, 2.0); Ta = max(Tb - 3.0, 1.0)
        try:
            fit = F.fit_iterate(u, y, Ta, Tb, u.max() - 1e-6, niter=3)
            ustar, Del = fit["ustar"], fit["Delta"]
        except Exception:
            ustar, Del = np.nan, np.nan      # diagnostic only; see dss_seedtest
        us = np.array([s[0] for s in R["snaps"]])
        np.savez_compressed(
            f"{OUT}/{fam}_x{x}.npz", ustar=ustar, Delta=Del, p=p, x=x,
            n=NGRID, Kdom=KDOM, nbisect=j + 1, u=u, phi0=y,
            rref=rec["rref"], snap_u=us, **{f"r{i}": s[1] for i, s in enumerate(R["snaps"])},
            **{f"h{i}": s[2] for i, s in enumerate(R["snaps"])})
        print(f"[{fam}] x={x}: nbisect={j+1} p={p:.12f} outcome={R['outcome']} "
              f"Twin~{Tw:.2f} u*={ustar:.10f} Delta_fit={Del:.4f} "
              f"nsnap={len(us)}", flush=True)
    print(f"[{fam}] done, wall={time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main(*sys.argv[1:])
