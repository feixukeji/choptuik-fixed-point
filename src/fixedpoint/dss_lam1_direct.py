"""
Validate lam1 against direct nonlinear evolution, using the trivial modes as
instruments.

The Jacobian of the half-period map at the fixed point is strongly nonnormal,
and that is exactly what makes this measurement possible.  For lam1 the RIGHT
eigenvector peaks at the outer boundary (only 26% of its L2 weight lies inside
the self-similarity horizon) while the LEFT eigenvector is LOCALISED at the
horizon: it peaks at X = 1.35 +- 0.04 over N = 176 ... 288 and carries 96% to
99.99% of its |w|^2 dX weight inside max_T X_h = 1.5608 (lowest at any
resolution examined: 95.9%).  That is localisation, not strict support, so the
adjoint is dominated by the DSS region but is not blind to the exterior.

The vector itself has no continuum limit -- it alternates at the grid scale and
its condition number grows as N^2.8 -- but the SPECTRAL PROJECTOR it defines,
P_1 = v w^T / <w,v>, is stable: applied to smooth perturbations defined as
functions of X, ||P_1 f|| agrees to 2.6-6.2% over N = 176 ... 288.  That, not
the envelope, is what licenses the measurement below.

Biorthogonality then does the rest.  With delta H the deviation of the data
from the fixed point at a matching phase,

    a_j(T) = <w_j, delta H>            w_j J = nu_j w_j,  <w_j, v_k> = delta_jk

isolates mode j *exactly*: the growing mode (nu_0 = -100.15), the u* gauge mode
(nu = -e^{Delta/2}) and the two lam = 0 modes drop out identically, however
large they are.  Along the half-period lattice T_m = T_ref + m Delta/2 the
deviation obeys delta H_{m+1} = J delta H_m (the antiperiodicity sign works out
for both parities of m), so the prediction is a pure ratio with no fitting:

    a_1(T_{m+1}) / a_1(T_m) = nu_1 = +0.2141      [lam1 = (2/Delta) ln nu_1]

Four interleaved lattices are used, from reference phases s = 0, Delta/8,
Delta/4, 3Delta/8, each with its own Jacobian and adjoint.
"""
import os, sys, time, json
import numpy as np
import scipy.linalg as sla
from scipy.interpolate import CubicSpline, barycentric_interpolate

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from paths import out
from core import collapse, floquet as F
from fixedpoint import dss_cheb as cb, dss_newton as dn, dss_hires as hr

OUT = out("fixedpoint")
P0 = 0.6823132832191163          # G1 (gauss r0=.25 s=.05) at n=600
XH_MAX = 1.560820                # max over the cycle of the horizon


# ---------------------------------------------------------------- evolution
def bracket(n, Kdom, **kw):
    """Widen about P0 until one amplitude disperses and one collapses."""
    for w in (1e-5, 1e-4, 1e-3, 1e-2, 1e-1):
        lo, hi = P0 * (1 - w), P0 * (1 + w)
        bl, _ = collapse.classify(lo, n=n, Kdom=Kdom, **kw)
        bh, _ = collapse.classify(hi, n=n, Kdom=Kdom, **kw)
        print(f"  bracket w={w:.0e}: lo={'BH' if bl else 'disp'} "
              f"hi={'BH' if bh else 'disp'}", flush=True)
        if (not bl) and bh:
            return lo, hi
    raise RuntimeError("no bracket")


def evolve(n=2400, Kdom=20.0, ndigit=14, snap_every=25, cache=True, **kw):
    f = f"{OUT}/lam1_run_n{n}_K{Kdom:g}.npz"
    if cache and os.path.exists(f):
        return np.load(f)
    kw = dict(r0=0.25, sigma=0.05, profile="gauss", **kw)
    t0 = time.time()
    lo, hi = bracket(n, Kdom, **kw)
    plo, phi, hist = collapse.bisect(lo, hi, ndigit=ndigit, verbose=False,
                                     n=n, Kdom=Kdom, **kw)
    dep = -np.log10(abs(phi - plo) / phi)
    print(f"bisect: {len(hist)} steps, depth={dep:.1f} digits, "
          f"p-={plo:.17g} ({time.time()-t0:.0f}s)", flush=True)
    R = collapse.Run(plo, n=n, Kdom=Kdom, **kw).run(snap_every=snap_every)
    snaps = R["snaps"]
    print(f"subcritical run: outcome={R['outcome']} steps={R['steps']} "
          f"nsnap={len(snaps)}", flush=True)
    u, y = R["rec"]["u"], R["rec"]["phi0"]
    fit = F.fit_iterate(u, y, 4.0, 9.0, u.max() - 1e-6, niter=4)
    print(f"ustar0={fit['ustar']:.12f} (phi0 fit, Delta={fit['Delta']:.4f})",
          flush=True)
    d = dict(ustar0=fit["ustar"], plo=plo, phi=phi, depth=dep, n=n, Kdom=Kdom,
             u=u, phi0=y, snap_u=np.array([s[0] for s in snaps]),
             **{f"r{i}": s[1] for i, s in enumerate(snaps)},
             **{f"h{i}": s[2] for i, s in enumerate(snaps)})
    np.savez_compressed(f, **d)
    return np.load(f)


# ------------------------------------------------------------ data on the grid
class Data:
    """H(T, X) of the evolution, on a fixed X grid, by interpolation."""

    def __init__(self, d, X, kT=8):
        self.X, self.kT = X, kT
        self.su = d["snap_u"]
        self.r = [d[f"r{i}"] for i in range(len(self.su))]
        self.h = [d[f"h{i}"] for i in range(len(self.su))]

    def set_ustar(self, ustar):
        self.ustar = ustar
        tau = ustar - self.su
        ok = tau > 0
        self.T = np.where(ok, -np.log(np.abs(tau) + 1e-300), -np.inf)
        # a snapshot is usable only if its grid covers the whole X range
        # a snapshot is usable when its grid reaches out to X_max; the inner
        # end needs no test because the null grid runs down to the centre and
        # the spline's cubic extrapolation across r_min/tau ~ 1e-3 is harmless
        self.good = np.array([ok[i] and self.r[i].max() / tau[i] >= self.X[-1]
                              for i in range(len(self.su))])
        self._cache = {}

    def _H(self, i):
        if i not in self._cache:
            tau = self.ustar - self.su[i]
            self._cache[i] = CubicSpline(self.r[i] / tau, self.h[i])(self.X)
        return self._cache[i]

    def at(self, Tt):
        """Barycentric interpolation in slow time across kT snapshots."""
        idx = np.where(self.good)[0]
        if len(idx) < self.kT:
            return None
        j = np.searchsorted(self.T[idx], Tt)
        a = max(0, min(j - self.kT // 2, len(idx) - self.kT))
        sel = idx[a:a + self.kT]
        if not (self.T[sel[0]] <= Tt <= self.T[sel[-1]]):
            return None
        return barycentric_interpolate(self.T[sel],
                                       np.array([self._H(i) for i in sel]), Tt)

    def span(self):
        t = self.T[self.good]
        return t.min(), t.max()


# ----------------------------------------------------------------- adjoints
def adjoints(S, H0, Delta, nstep, s, nproc=None):
    """Fixed point at phase s, its Jacobian, and normalised left eigenvectors."""
    Hs = H0 if s == 0.0 else S.evolve(H0, s, max(2, int(nstep * s / (0.5 * Delta))))
    J = hr.jacobian_mp(S, Hs, Delta, nstep, nproc=nproc)
    nu, vl, vr = sla.eig(J, left=True, right=True)
    lam = np.log(nu.astype(complex) ** 2) / Delta
    o = np.argsort(-lam.real)
    nu, vl, vr, lam = nu[o], vl[:, o], vr[:, o], lam[o]
    W = np.array([vl[:, j] / np.conj(np.vdot(vl[:, j], vr[:, j]))
                  for j in range(len(nu))])
    return Hs, nu, lam, W, vr


def coeffs(dat, Hs, W, vr, Delta, Tref, jlist, jdef, inn, mmax=16, sgn0=1):
    """a_j on the lattice Tref + m Delta/2 (sign from antiperiodicity).

    Also returns the norm of the deviation after the exactly-known directions
    in jdef (constant, phase, u* gauge, growing mode) have been projected out
    biorthogonally.  Whatever is left must decay at lam1 asymptotically.
    """
    nrm = np.linalg.norm(Hs[inn])
    out = []
    for m in range(mmax + 1):
        Tt = Tref + m * 0.5 * Delta
        Hd = dat.at(Tt)
        if Hd is None:
            continue
        dH = Hd - sgn0 * (-1.0) ** m * Hs
        df = dH.astype(complex)
        for j in jdef:
            df = df - (W[j] @ dH) * vr[:, j]
        out.append((m, Tt, np.linalg.norm(dH[inn]) / nrm,
                    np.linalg.norm(np.real(df)[inn]) / nrm,
                    [float(np.real(W[j] @ dH)) for j in jlist]))
    return out


def match_phase(dat, S, H0, Delta, nstep, Tfix, ngrid=192, inn=None):
    """Slow time at which the data matches the stored fixed point H*(0).

    The data at the single slow time Tfix is compared against H*(s) for s over
    a FULL period, so the antiperiodic sign is picked up automatically
    (evolving past Delta/2 returns -H*).  If the best match is at s, then
    data(Tfix) ~ H*(s), hence data(T) ~ H*(T - Tfix + s) and the reference
    phase of the lattice is Tref = Tfix - s.

    The misfit is measured only inside the self-similarity horizon: the domain
    of dependence of X is [0, X], so the interior evolves independently of the
    exterior, while the exterior of a near-critical run still carries its
    initial data and never approaches the fixed point at all.
    """
    inn = np.ones(S.N, bool) if inn is None else inn
    Hd = dat.at(Tfix)
    if Hd is None:
        raise RuntimeError("no data at Tfix")
    best = (np.inf, 0.0)
    Hs = H0.copy()
    ds = Delta / ngrid
    ns = max(2, int(nstep * ds / (0.5 * Delta)))
    for k in range(ngrid):
        e = np.linalg.norm((Hd - Hs)[inn]) / np.linalg.norm(Hs[inn])
        if e < best[0]:
            best = (e, k * ds)
        Hs = S.evolve(Hs, ds, ns)
    return best[0], Tfix - best[1]              # (relerr, Tref)


def pick(lam, nu):
    """Growing mode, u* gauge mode, the two lam = 0 modes, and lam1."""
    jg = int(np.argmax(lam.real))
    ja = int(np.argmin(abs(lam.real - 1.0) + abs(lam.imag)))
    z = np.where(abs(lam.real) < 1e-3)[0]
    jc = int(z[np.argmin(abs(nu[z].real - 1.0))])       # nu = +1: constant
    jp = int(z[np.argmin(abs(nu[z].real + 1.0))])       # nu = -1: phase
    cand = np.where((lam.real < -0.05) & (abs(lam.imag) < 1e-6))[0]
    j1 = int(cand[np.argmax(lam.real[cand])])
    return jg, ja, jc, jp, j1


# -------------------------------------------------------------------- driver
LIN = 0.10          # interior deviation at which the linearisation is trusted


def pool(res):
    """Pooled lam1 over the four phase lattices, from the saved rows.

    A half-period ratio spans TWO lattice times, so both of them have to lie in
    the linear window; testing only the endpoint admits transitions that begin
    far outside it, and those are not half-period decay rates of a linear mode.
    Positive ratios are required because nu_1 = +0.214 > 0, so a negative ratio
    means the a_1 signal has been lost rather than that it decayed at some other
    rate; the count of such sign failures is reported, not silently dropped.
    The alternative statistic keeps them via log|ratio| and is quoted as the
    sensitivity of the result to that choice.  Every +- is a sample standard
    deviation over the pooled ratios, not a standard error.
    """
    D = res["Delta"]
    lin, neg, defl = [], [], []
    for L in res["lattices"]:
        rows = L["rows"]
        for i, r in enumerate(rows):
            if i == 0 or not np.isfinite(r["ratio"]):
                continue
            if not (r["dH"] < LIN and rows[i - 1]["dH"] < LIN):
                continue
            (lin if r["ratio"] > 0 else neg).append(r["lam1"])
            if r["ratio"] > 0 and np.isfinite(r["lam1_def"]):
                defl.append(r["lam1_def"])

    def ms(a):
        a = np.asarray(a, float)
        return [float(a.mean()), float(a.std(ddof=1)), len(a)] if len(a) > 1 else None

    res["lam1_direct"] = ms(lin)
    res["lam1_allsign"] = ms(lin + neg)      # sensitivity to the sign cut
    res["lam1_deflated"] = ms(defl)
    res["nsign_fail"] = len(neg)
    res["LIN"] = LIN
    if res["lam1_direct"]:
        v, e, k = res["lam1_direct"]
        print(f"\npooled over {k} half-period ratios with |dH|/|H*|_in < {LIN} "
              f"at BOTH ends ({len(neg)} further ratios rejected on sign):")
        print(f"  lam1(direct evolution) = {v:+.5f} +- {e:.5f}"
              f"   [fixed point: {res['lattices'][0]['lam1_fp']:+.6f}]")
        if res["lam1_allsign"]:
            v, e, k = res["lam1_allsign"]
            print(f"  keeping sign failures via log|ratio| ({k}): "
                  f"{v:+.5f} +- {e:.5f}")
        if res["lam1_deflated"]:
            v, e, k = res["lam1_deflated"]
            print(f"  deflated-norm slope ({k}) = {v:+.5f} +- {e:.5f}")
    return res


def main(N=176, Xmax=4.0, n=2400, Kdom=20.0, nphase=4, mmax=16, nproc=None,
         tag=""):
    os.makedirs(OUT, exist_ok=True)
    fp = np.load(f"{OUT}/cheb_N{N}_X{Xmax:g}.npz")
    S = cb.Sim(N=N, Xmax=Xmax)
    H0, Delta, nstep = fp["H"], float(fp["Delta"]), int(fp["nstep"])
    d = evolve(n=n, Kdom=Kdom)
    dat = Data(d, S.X)
    dat.set_ustar(float(d["ustar0"]))
    T0, T1 = dat.span()
    print(f"fixed point N={N} Delta={Delta:.10f}; data T in "
          f"[{T0:.3f}, {T1:.3f}] ({(T1-T0)/Delta:.2f} echoes), "
          f"depth={float(d['depth']):.1f} digits", flush=True)

    inn = S.X <= XH_MAX
    Tfix = T0 + 0.35 * (T1 - T0)
    err, Tref0 = match_phase(dat, S, H0, Delta, nstep, Tfix, inn=inn)
    print(f"phase match: data(T={Tfix:.3f}) ~ H*(s), Tref={Tref0:.5f} "
          f"(interior ||dH||/||H*|| = {err:.3e})", flush=True)
    # Walk the lattice origin back to the earliest usable slow time.  Each
    # step of Delta/2 flips the sign of the reference, because the solution is
    # antiperiodic: H*(T - Delta/2) = -H*(T).  Forgetting this makes every
    # deviation come out as 2H* instead of the real one.
    sgn0 = 1.0
    while Tref0 - 0.5 * Delta > T0:
        Tref0 -= 0.5 * Delta; sgn0 = -sgn0
    while Tref0 < T0:
        Tref0 += 0.5 * Delta; sgn0 = -sgn0

    res = {"Delta": Delta, "N": N, "Xmax": Xmax, "depth": float(d["depth"]),
           "Tspan": [T0, T1], "phase_err": err, "sgn0": sgn0, "lattices": []}
    for q in range(nphase):
        s = q * 0.5 * Delta / nphase
        t0 = time.time()
        Hs, nu, lam, W, vr = adjoints(S, H0, Delta, nstep, s, nproc=nproc)
        jg, ja, jc, jp, j1 = pick(lam, nu)
        Tref, sg = Tref0 + s, sgn0
        while Tref - 0.5 * Delta > T0:
            Tref -= 0.5 * Delta; sg = -sg
        c = coeffs(dat, Hs, W, vr, Delta, Tref, [jg, ja, j1],
                   [jc, jp, ja, jg], inn, mmax=mmax, sgn0=sg)
        lat = {"s": s, "nu1": complex(nu[j1]).real,
               "lam1_fp": complex(lam[j1]).real, "rows": []}
        print(f"\n--- lattice s={s:.4f}  lam0={lam[jg].real:+.5f} "
              f"lam_g={lam[ja].real:+.5f}  nu1={nu[j1].real:+.6f} "
              f"lam1(fp)={lam[j1].real:+.6f}  ({time.time()-t0:.0f}s)")
        print("   m      T     |dH|_in    |dH_def|_in      a0          ag"
              "          a1       a1 ratio  ->lam1   |dH_def| ->lam1")
        prev = pdef = None
        for (m, Tt, nd, nf, a) in c:
            rat = a[2] / prev if prev not in (None, 0.0) else np.nan
            l1 = (2.0 / Delta) * np.log(abs(rat)) if np.isfinite(rat) else np.nan
            rd = nf / pdef if pdef not in (None, 0.0) else np.nan
            ld = (2.0 / Delta) * np.log(rd) if np.isfinite(rd) else np.nan
            print(f"  {m:3d} {Tt:7.3f}  {nd:.3e}   {nf:.3e}   "
                  f"{a[0]:+.3e}  {a[1]:+.3e}  {a[2]:+.3e}  "
                  f"{rat:+.5f} {l1:+.5f}  {ld:+.5f}")
            lat["rows"].append(dict(m=m, T=Tt, dH=nd, dHdef=nf, a0=a[0],
                                    ag=a[1], a1=a[2], ratio=float(rat),
                                    lam1=float(l1), lam1_def=float(ld)))
            prev, pdef = a[2], nf
        res["lattices"].append(lat)

    pool(res)
    f = f"{OUT}/lam1_direct_N{N}_n{n}{tag}.json"
    json.dump(res, open(f, "w"), indent=1)
    print(f"\nsaved {f}", flush=True)


if __name__ == "__main__":
    if sys.argv[1:2] == ["repool"]:
        # re-apply pool() to already-saved runs; the rows hold everything it
        # needs, so no re-evolution is required.
        for f in sys.argv[2:]:
            res = json.load(open(f))
            print(f"--- {f}")
            json.dump(pool(res), open(f, "w"), indent=1)
        sys.exit()
    kw = {}
    for a in sys.argv[1:]:
        k, v = a.split("=")
        kw[k] = float(v) if "." in v else int(v)
    main(**kw)
