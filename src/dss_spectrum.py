"""
What the Floquet spectrum of the Choptuik fixed point does and does not contain.

Four diagnostics, all run off a converged fixed point (dss_newton / dss_spec_cheb):

  exact   Verify the three closed-form eigenvectors against the numerically
          computed Jacobian.  This tests DIRECTION, not just rate:
            constant  H -> H + c                     nu = +1
            phase     T -> T + a                     nu = -1
            gauge     u* -> u* + d, i.e.
                      dH = -d e^T (H_T + X H_X)      nu = -e^{Delta/2}
          (docs/08 wrote the gauge mode as -d e^T Z*'(T) for the CENTRAL field,
          which is the same thing there because X H_X vanishes at X = 0.)

  match   Pair eigenvalues between two resolutions and report which survive.

  count   How many eigenvalues sit in fixed windows of Re lam, versus N.  A
          discrete set keeps its members; a continuum migrates and fills in.

  decay   Decay rate of smooth perturbations after EXACT removal of the four
          known directions (oblique projection with the left eigenvectors),
          optionally filtered to the first m Chebyshev modes each step to
          suppress grid-scale contamination.  This is the quantity a
          near-critical evolution experiment actually measures (docs/08 D).

  horizon The self-similarity horizon as the repelling periodic orbit of
          dX/dT = X - gbar/2, with its multiplier over half a period.
"""
import os
import sys
import numpy as np
from scipy.optimize import brentq
from scipy.interpolate import CubicSpline
import dss_core as dc, dss_cheb as cb

OUT = "out/dss"


def load(kind, N, Xmax=4.0, wd=7):
    if kind == "fd":
        f = np.load(f"{OUT}/spec_N{N}_X{Xmax:g}_w{wd}.npz")
        S = dc.Sim(N=N, Xmax=Xmax, wd=wd)
        H = f["H"]
        hb, g, gb = S.fields(H)
        dTH = -(S.X - 0.5 * gb) * S.dx(H) + (g - gb) * (H - hb) / (2 * S.X)
        dXH = S.dx(H)
    else:
        f = np.load(f"{OUT}/cheb_N{N}_X{Xmax:g}.npz")
        S = cb.Sim(N=N, Xmax=Xmax)
        H = f["H"]
        dTH = S.rhs(H)
        dXH = S.D @ H
    return S, f["J"], float(f["Delta"]), S.X, H, dTH, dXH


def trivial(S, X, H, dTH, dXH):
    """The three closed-form eigenvectors and their exact multipliers."""
    return {"constant": (np.ones(len(X)), +1.0),
            "phase":    (dTH,             -1.0),
            "gauge":    (dTH + X * dXH,   None)}   # -exp(Delta/2), filled by caller


def check_exact(kind, Ns, Xmax=4.0, wd=7):
    for N in Ns:
        S, J, D, X, H, dTH, dXH = load(kind, N, Xmax, wd)
        ex = trivial(S, X, H, dTH, dXH)
        ex["gauge"] = (ex["gauge"][0], -np.exp(D / 2))
        print(f"-- {kind} N={N} Delta={D:.9f}")
        for k, (v, nu_ex) in ex.items():
            Jv = J @ v
            nu = float(v @ Jv / (v @ v))
            print(f"   {k:9s} nu={nu:+.8f} exact={nu_ex:+.8f} "
                  f"relerr={abs(nu-nu_ex)/abs(nu_ex):.2e} "
                  f"resid={np.linalg.norm(Jv-nu*v)/np.linalg.norm(Jv):.2e}")


def deflator(J, X, H, dTH, dXH):
    """Oblique projector removing the four known directions exactly."""
    n = len(X)
    nu, V = np.linalg.eig(J)
    W = np.linalg.inv(V).T
    kg = int(np.argmax(np.abs(nu)))
    R = [np.ones(n), dTH, dTH + X * dXH, V[:, kg].real]
    L = []
    for r in R:
        j = int(np.argmax(np.abs(V.conj().T @ r)
                          / (np.linalg.norm(V, axis=0) * np.linalg.norm(r))))
        L.append(W[:, j].real)
    R = np.array(R).T; L = np.array(L).T
    return np.eye(n) - R @ np.linalg.solve(L.T @ R, L.T), nu[kg]


def cheb_filter(X, m, Xmax, wts):
    xi = 2 * X / Xmax - 1.0
    B = np.zeros((len(X), m)); B[:, 0] = 1.0
    if m > 1:
        B[:, 1] = xi
    for k in range(2, m):
        B[:, k] = 2 * xi * B[:, k - 1] - B[:, k - 2]
    return B @ np.linalg.solve(B.T * wts @ B, B.T * wts)


def decay(kind, Ns, ms=(None,), Xmax=4.0, wd=7, ntrial=40, nit=10, seed=1):
    for N in Ns:
        S, J, D, X, H, dTH, dXH = load(kind, N, Xmax, wd)
        P, nug = deflator(J, X, H, dTH, dXH)
        wts = (np.full(N, X[1] - X[0]) if kind == "fd" else
               np.gradient(X))                      # crude but only a weight
        for m in ms:
            Op = (P @ J) if m is None else (cheb_filter(X, m, Xmax, wts) @ (P @ J))
            rng = np.random.default_rng(seed); acc = []
            for _ in range(ntrial):
                c = rng.normal(size=8)
                v = sum(c[j] * np.cos(j * np.pi * X / Xmax) for j in range(8))
                v = Op @ v if m is not None else P @ v
                v /= np.linalg.norm(v); rr = []
                for _ in range(nit):
                    u = Op @ v; nn = np.linalg.norm(u)
                    rr.append(2 * np.log(nn) / D); v = u / nn
                acc.append(rr)
            a = np.array(acc)
            print(f"  {kind} N={N} m={str(m):>4s}: " + " ".join(
                f"{a[:,k].mean():+.3f}[{a[:,k].std():.2f}]" for k in range(min(nit, 8))))


def count(kind, Ns, Xmax=4.0, wd=7):
    print("     N   [-1,-0.1]  [-2,-1]  [-4,-2]  [<-4]   Re>0.1   total")
    for N in Ns:
        f = (np.load(f"{OUT}/spec_N{N}_X{Xmax:g}_w{wd}.npz") if kind == "fd"
             else np.load(f"{OUT}/cheb_N{N}_X{Xmax:g}.npz"))
        r = f["lam"].real
        print(f"  {N:5d}    {np.sum((r>-1)&(r<-0.1)):5d}   {np.sum((r>-2)&(r<=-1)):6d}"
              f"   {np.sum((r>-4)&(r<=-2)):6d}  {np.sum(r<=-4):5d}   {np.sum(r>0.1):5d}"
              f"   {len(r):5d}")


def horizon(kind="fd", N=800, Xmax=4.0, wd=7, nT=1200):
    """Repelling periodic orbit of dX/dT = X - gbar/2 and its multiplier.
    gbar is Delta/2-periodic (quadratic in H), so the orbit closes in Delta/2."""
    S, J, D, X, H, _, _ = load(kind, N, Xmax, wd)
    ns = (S.nstep_cfl(H, 0.5 * D, 0.30) if kind == "fd"
          else S.nstep_cfl(H, 0.5 * D, 0.30))
    k = max(ns // nT, 1)
    Ts, A, Ax = [], [], []
    Hc = H.copy(); done = 0; dT = 0.5 * D / ns
    while True:
        hb, g, gb = S.fields(Hc)
        A.append(X - 0.5 * gb); Ax.append(1.0 - (g - gb) / (2 * X))
        Ts.append(done * dT)
        if done >= ns:
            break
        m = min(k, ns - done); Hc = S.evolve(Hc, m * dT, m); done += m
    Ts = np.array(Ts)
    aI = [CubicSpline(X, r) for r in A]; axI = [CubicSpline(X, r) for r in Ax]

    def shoot(x0, want=False):
        x = float(x0); ls = 0.0; xm = x
        for i in range(len(Ts) - 1):
            hh = Ts[i + 1] - Ts[i]
            k1 = aI[i](x); k2 = aI[i + 1](x + hh * k1)
            if want:
                ls += 0.5 * hh * (axI[i](x) + axI[i + 1](x + hh * k1))
            x = x + 0.5 * hh * (k1 + k2); xm = max(xm, x)
        return (x, ls, xm) if want else x

    xs = np.linspace(0.6, 3.5, 60); vals = [shoot(x) - x for x in xs]
    for i in range(len(xs) - 1):
        if vals[i] * vals[i + 1] < 0:
            r = brentq(lambda z: shoot(z) - z, xs[i], xs[i + 1], xtol=1e-12)
            _, ls, xm = shoot(r, want=True)
            print(f"  X_h(0)={r:.6f}  max_T X_h={xm:.6f}  sigma_half={np.exp(ls):.6f}"
                  f"  mu=(2/Delta)ln sigma={2*ls/D:.6f}")


if __name__ == "__main__":
    w = sys.argv[1] if len(sys.argv) > 1 else "all"
    if w in ("exact", "all"):
        print("== exact eigenvectors =="); check_exact("fd", (200, 400, 800))
    if w in ("count", "all"):
        print("\n== eigenvalue counts, finite difference =="); count("fd", (200, 400, 800))
    if w in ("decay", "all"):
        print("\n== smooth-perturbation decay rate ==")
        decay("fd", (400, 800), ms=(None, 40, 60))
    if w in ("horizon", "all"):
        print("\n== self-similarity horizon =="); horizon()


def horizon_scalars(kind="fd", N=800, Xmax=4.0, wd=7, nT=1600):
    """Invariants of the self-similarity horizon as a periodic orbit.

    Reports the Floquet multiplier of the orbit over a FULL period -- that is
    the quantity invariant under the periodic coordinate changes
    x' = phi(tau, x), tau' = tau + psi(tau, x) that Gundlach & Martin-Garcia
    (PRD 68, 024011) use to put every SSH at constant x, whereas mu = (2/Delta)
    ln sigma_half depends on the parameterisation of slow time.

    Also reports the period average of the scalar a^2 = g/gbar = (1 - 2m/r)^{-1}
    along the orbit.  Their Eq. (77) makes a^2 > 2 a necessary condition for a
    further self-similarity horizon to the future, so this is a direct check
    against a published criterion.
    """
    S, J, D, X, H, _, _ = load(kind, N, Xmax, wd)
    ns = S.nstep_cfl(H, 0.5 * D, 0.30)
    k = max(ns // nT, 1)
    Ts, A, Ax, A2, Z = [], [], [], [], []
    Hc = H.copy(); done = 0; dT = 0.5 * D / ns
    while True:
        hb, g, gb = S.fields(Hc)
        a = X - 0.5 * gb
        A.append(a); Ax.append(1.0 - (g - gb) / (2 * X))
        A2.append(g / gb)
        # the INSTANTANEOUS zero of the ray speed, for contrast with the orbit
        i = int(np.where(np.diff(np.sign(a)) != 0)[0][0])
        Z.append(brentq(CubicSpline(X, a), X[i], X[i + 1]))
        Ts.append(done * dT)
        if done >= ns:
            break
        m = min(k, ns - done); Hc = S.evolve(Hc, m * dT, m); done += m
    Ts = np.array(Ts)
    aI = [CubicSpline(X, r) for r in A]
    axI = [CubicSpline(X, r) for r in Ax]
    a2I = [CubicSpline(X, r) for r in A2]

    def shoot(x0, want=False):
        x = float(x0); ls = 0.0; xm = x; a2 = 0.0; xs = [x]
        for i in range(len(Ts) - 1):
            hh = Ts[i + 1] - Ts[i]
            k1 = aI[i](x); k2 = aI[i + 1](x + hh * k1)
            if want:
                ls += 0.5 * hh * (axI[i](x) + axI[i + 1](x + hh * k1))
                a2 += 0.5 * hh * (a2I[i](x) + a2I[i + 1](x + hh * k1))
            x = x + 0.5 * hh * (k1 + k2); xm = max(xm, x); xs.append(x)
        return (x, ls, xm, a2, np.array(xs)) if want else x

    xs = np.linspace(0.6, 3.5, 60); vals = [shoot(x) - x for x in xs]
    out = None
    for i in range(len(xs) - 1):
        if vals[i] * vals[i + 1] < 0:
            r = brentq(lambda z: shoot(z) - z, xs[i], xs[i + 1], xtol=1e-12)
            _, ls, xm, a2, orb = shoot(r, want=True)
            mu = 2 * ls / D
            print(f"  X_h(0)      = {r:.6f}      min_T X_h = {orb.min():.6f}"
                  f"   max_T X_h = {xm:.6f}")
            print(f"  sigma_half  = {np.exp(ls):.6f}   "
                  f"full-period multiplier = {np.exp(2*ls):.6f}   "
                  f"mu*Delta = {mu*D:.6f}")
            print(f"  mu          = {mu:.6f}      1/mu = {1/mu:.6f}")
            Z = np.array(Z)
            print(f"  <a^2>_SSH   = {a2/(0.5*D):.6f}   "
                  f"(GMG03 Eq.(77): a^2 > 2 needed for a further SSH)")
            print(f"  instantaneous zero of A: [{Z.min():.4f}, {Z.max():.4f}] "
                  f"-- wanders by a factor {Z.max()/Z.min():.2f} against "
                  f"{xm/orb.min():.2f} for the orbit")
            out = dict(Xh0=r, Xh_min=float(orb.min()), Xh_max=xm,
                       sigma_half=float(np.exp(ls)),
                       mult_full=float(np.exp(2 * ls)), mu=mu,
                       a2_mean=float(a2 / (0.5 * D)), T=Ts, orbit=orb,
                       zero_min=float(Z.min()), zero_max=float(Z.max()),
                       A_at_Xh0=float(CubicSpline(X, A[0])(r)))
    return out


def flush(kind="fd", N=800, Xmax=4.0, wd=7, nT=1600,
          eps=(0.20, 0.10, 0.05, 0.02, 0.01), Xstop=1e-3, Tmax=25.0):
    """Time for a RAY starting a distance eps inside the horizon to reach 0.

    The horizon is a repelling periodic orbit, so a ray starting at
    X_h(0) - eps drifts inwards, reaches the centre and leaves.  T_f(eps) is
    when it arrives at X = Xstop.

    Linearising the ray equation about the periodic orbit gives
    d(eta)/dT = A_X(T, X_h(T)) eta, whose solution is e^{mu T} times a
    prefactor of period Delta/2.  So the leading behaviour is
    mu^{-1} ln(1/eps), but the O(1) remainder is a *periodic* function of
    T_f, not a constant: it does not settle as eps -> 0.  We fit it as
    c0 + amplitude*cos(2 pi T_f / P + phase) and report the period that the
    data actually select, alongside Delta/2.
    """
    S, J, D, X, H, _, _ = load(kind, N, Xmax, wd)
    ns = S.nstep_cfl(H, 0.5 * D, 0.30)
    k = max(ns // nT, 1)
    Hc, done, dT, A = H.copy(), 0, 0.5 * D / ns, []
    while True:
        hb, g, gb = S.fields(Hc)
        A.append(CubicSpline(X, X - 0.5 * gb))
        if done >= ns:
            break
        m = min(k, ns - done); Hc = S.evolve(Hc, m * dT, m); done += m
    hh = 0.5 * D / (len(A) - 1)
    d = np.load(f"{OUT}/ssh_invariants.npz") if os.path.exists(
        f"{OUT}/ssh_invariants.npz") else None
    Xh0 = float(d["Xh0"]) if d is not None else 1.352471
    mu = float(d["mu"]) if d is not None else 0.613855
    out = []
    for e in eps:
        x, T, i = Xh0 - e, 0.0, 0
        while x > Xstop and T < Tmax:
            k1 = A[i % (len(A) - 1)](x)
            k2 = A[(i + 1) % (len(A) - 1)](max(x + hh * k1, 0.0))
            x = x + 0.5 * hh * (k1 + k2); T += hh; i += 1
        out.append((e, T, T - np.log(1 / e) / mu))
        print(f"  eps={e:.3f}  T_f={T:.3f}  T_f-(1/mu)ln(1/eps)={out[-1][2]:+.3f}")
    c = np.mean([o[2] for o in out])
    sp = max(abs(o[2] - c) for o in out)
    L = np.log(1.0 / np.array([o[0] for o in out]))
    Tf = np.array([o[1] for o in out])
    res = np.array([o[2] for o in out])
    a, b = np.polyfit(L, Tf, 1)
    print(f"  => slope fixed at 1/mu: T_f = (1/mu) ln(1/eps) + {c:.2f} "
          f"(spread +-{sp:.2f}, 1/mu = {1/mu:.3f})")
    print(f"  => free fit:            slope = {a:.3f} "
          f"({100*(a*mu-1):+.1f}% vs 1/mu), intercept = {b:.2f}")

    # The remainder is periodic in T_f with period Delta/2, not constant.
    def harm(P):
        M = np.c_[np.ones_like(Tf), np.cos(2 * np.pi * Tf / P),
                  np.sin(2 * np.pi * Tf / P)]
        co, *_ = np.linalg.lstsq(M, res, rcond=None)
        return ((res - M @ co) ** 2).sum(), co

    Ps = np.linspace(0.30 * D, 0.90 * D, 4001)   # brackets Delta/2
    Pbest = float(Ps[np.argmin([harm(q)[0] for q in Ps])])
    tot = ((res - res.mean()) ** 2).sum()
    r_h, co_h = harm(0.5 * D)
    amp = float(np.hypot(co_h[1], co_h[2]))
    print(f"  => remainder is periodic, not constant: over T_f in "
          f"[{Tf.min():.2f}, {Tf.max():.2f}] = {(Tf.max()-Tf.min())/(0.5*D):.1f} "
          f"half-periods it oscillates by +-{0.5*(res.max()-res.min()):.3f} "
          f"with no decay")
    print(f"     single harmonic at P = Delta/2 = {0.5*D:.4f}: amplitude "
          f"{amp:.4f}, {100*(1-r_h/tot):.0f}% of the variance")
    print(f"     period selected by the data: {Pbest:.4f} "
          f"({100*(Pbest/(0.5*D)-1):+.1f}% from Delta/2)")
    np.savez(f"{OUT}/flush.npz", eps=np.array([o[0] for o in out]), Tf=Tf,
             resid=res, mu=mu, c=c, spread=sp, slope=a, intercept=b,
             Delta=D, c0=float(co_h[0]), amp=amp, Pbest=Pbest,
             varexp=float(1 - r_h / tot))
    return out
