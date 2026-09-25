"""
Discretely self-similar fixed point of the Choptuik system, by Newton-Krylov
on the half-period antiperiodic map -- no tuning of initial data at all.

Unknowns:  H on N grid points in X, and the echo period Delta.
Equations:  R = S_{Delta/2}[H] + H = 0   (N equations)
            plus one phase condition, because T -> T + a is an exact symmetry
            (an exact lam = 0 Floquet mode) and leaves R singular otherwise.

S_{Delta/2} is "integrate the autonomous PDE of dss_core for slow time
Delta/2".  The *half* period with the minus sign uses Choptuik's antiperiodicity
phi(T + Delta/2) = -phi(T), which the system's oddness H -> -H makes exact; it
also removes the period-doubling ambiguity that a full-period condition has.

Conditioning is favourable: the Jacobian is DS + I, whose eigenvalues are
nu_j + 1 for the half-period Floquet multipliers nu_j.  Every decaying mode has
nu_j ~ 0, so almost the whole spectrum clusters at 1 and only the growing mode
(nu ~ 100), the gauge mode (nu ~ -e^{Delta/2}) and the constant mode (nu = 1)
sit apart.  GMRES therefore converges in a handful of iterations.

After convergence the dense N x N Jacobian of the map is formed column by
column and diagonalised, giving the ENTIRE Floquet spectrum in one shot:
    lambda_j = (2/Delta) ln(nu_j^2) / 2 = (1/Delta) ln(nu_j^2),
i.e. Re lambda_j = (2/Delta) ln|nu_j|.  Three exponents are known exactly in
advance and check the whole computation: nu = +1 (the shift H -> H + c),
nu = -1 (the phase mode, lambda = 0) and nu = -e^{Delta/2} (the accumulation
point shift u* -> u* + d, lambda = 1).
"""
import os, sys, time
import numpy as np
from scipy.interpolate import CubicSpline
from scipy.sparse.linalg import LinearOperator, gmres
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from paths import out
from fixedpoint import dss_core as dc

OUT = out("fixedpoint")
DELTA_REF = 3.445452402       # Gundlach 1997
LAM0_REF = 1.0 / 0.374


class Problem:
    """R(z) with z = [H, Delta]."""

    def __init__(self, S, nstep, Hphase=None, Href=None):
        self.S, self.nstep = S, nstep
        if Hphase is None:
            Hphase = np.zeros(S.N)
        self.w = Hphase / max(np.linalg.norm(Hphase), 1e-300)
        self.Href = Href
        self.c0 = float(self.w @ Href) if Href is not None else 0.0
        self.nmap = 0

    def map(self, H, Delta):
        self.nmap += 1
        return self.S.evolve(H, 0.5 * Delta, self.nstep)

    def R(self, z):
        H, D = z[:-1], z[-1]
        return np.concatenate([self.map(H, D) + H, [self.w @ H - self.c0]])

    def jvp(self, z, v, eps=1e-6):
        s = eps * (1.0 + np.linalg.norm(z) / max(np.linalg.norm(v), 1e-300))
        return (self.R(z + s * v) - self.R(z - s * v)) / (2.0 * s)


def solve(S, H0, Delta0, nstep, tol=1e-11, maxit=25, gmres_tol=1e-4,
          verbose=True):
    Hp = S.evolve(H0, 1e-3, 2)
    Hp = (Hp - H0) / 1e-3                       # dH/dT at the seed: phase mode
    P = Problem(S, nstep, Hphase=Hp, Href=H0)
    z = np.concatenate([H0, [Delta0]])
    r = P.R(z); nr = np.linalg.norm(r)
    if verbose:
        print(f"  it  0  |R|={nr:.4e}  Delta={z[-1]:.9f}", flush=True)
    for it in range(1, maxit + 1):
        J = LinearOperator((len(z), len(z)), matvec=lambda v: P.jvp(z, v))
        dz, info = gmres(J, -r, rtol=gmres_tol, atol=0.0, restart=60, maxiter=8)
        if not np.all(np.isfinite(dz)):
            print("  GMRES failed"); break
        # damped update
        lam = 1.0
        for _ in range(12):
            zt = z + lam * dz
            rt = P.R(zt); nt = np.linalg.norm(rt)
            if np.isfinite(nt) and nt < nr:
                break
            lam *= 0.5
        z, r, nr_old, nr = zt, rt, nr, nt
        if verbose:
            print(f"  it {it:2d}  |R|={nr:.4e}  Delta={z[-1]:.9f}  "
                  f"step={lam:.3f}  gmres={info}  nmap={P.nmap}", flush=True)
        if nr < tol or (nr > nr_old and lam < 1e-3):
            break
    return z, nr, P


def jacobian(S, H, Delta, nstep, eps=1e-6, cols=None):
    """Dense N x N Jacobian of the half-period map at (H, Delta)."""
    n = S.N
    J = np.empty((n, n))
    base = S.evolve(H, 0.5 * Delta, nstep)
    sc = eps * max(np.linalg.norm(H) / np.sqrt(n), 1e-12)
    rng = range(n) if cols is None else cols
    for j in rng:
        e = np.zeros(n); e[j] = sc
        J[:, j] = (S.evolve(H + e, 0.5 * Delta, nstep)
                   - S.evolve(H - e, 0.5 * Delta, nstep)) / (2.0 * sc)
    return J, base


def spectrum(J, Delta):
    """Floquet exponents from the half-period multipliers."""
    nu, V = np.linalg.eig(J)
    lam = np.log(nu.astype(complex) ** 2) / Delta
    o = np.argsort(-lam.real)
    return nu[o], lam[o], V[:, o]


def main(N=400, Xmax=4.0, wd=7, Delta0=3.30, Ta=3.4, cfl=0.30,
         jac=True, tag=""):
    from fixedpoint import dss_check as ck
    os.makedirs(OUT, exist_ok=True)
    d, us, T, k = ck.load_probe(Ta=Ta)
    S, H0, tau = ck.seed(d, us, k, N=N, Xmax=Xmax, wd=wd)
    nstep = S.nstep_cfl(H0, 0.5 * max(Delta0, DELTA_REF), cfl)
    print(f"N={N} Xmax={Xmax} wd={wd} nstep={nstep} seed T={T[k]:.3f} "
          f"Delta0={Delta0}", flush=True)
    t0 = time.time()
    z, nr, P = solve(S, H0, Delta0, nstep)
    H, Delta = z[:-1], z[-1]
    print(f"converged |R|={nr:.3e}  Delta={Delta:.10f}  "
          f"(ref {DELTA_REF:.9f}, rel err {(Delta-DELTA_REF)/DELTA_REF:+.2e})  "
          f"maps={P.nmap}  wall={time.time()-t0:.1f}s", flush=True)
    out = dict(N=N, Xmax=Xmax, wd=wd, nstep=nstep, X=S.X, H=H, Delta=Delta,
               resid=nr, Ta=T[k], cfl=cfl)
    if jac:
        t1 = time.time()
        J, _ = jacobian(S, H, Delta, nstep)
        nu, lam, V = spectrum(J, Delta)
        print(f"Jacobian+eig wall={time.time()-t1:.1f}s", flush=True)
        for i in range(min(14, len(lam))):
            print(f"   nu={nu[i]:+.6e}  lam={lam[i].real:+.6f}"
                  f"{lam[i].imag:+.6f}j", flush=True)
        out.update(J=J, nu=nu, lam=lam)
    np.savez_compressed(f"{OUT}/fixedpoint{tag}_N{N}_X{Xmax:g}_w{wd}.npz", **out)
    print("saved", flush=True)


if __name__ == "__main__":
    kw = {}
    for a in sys.argv[1:]:
        k_, v_ = a.split("=")
        kw[k_] = (int(v_) if k_ in ("N", "wd") else
                  v_ if k_ == "tag" else
                  (v_ == "1") if k_ == "jac" else float(v_))
    main(**kw)


# --------------------------------------------------------- multiple shooting

class MSProblem:
    """
    Same fixed point, split into k segments.

    Unknowns  H_0 ... H_{k-1} and Delta.  Equations

        S_{Delta/2k}[H_j] - H_{j+1} = 0,        j = 0 ... k-2
        S_{Delta/2k}[H_{k-1}] + H_0 = 0         (antiperiodic closure)
        <w, H_0 - H_ref> = 0                    (the exact lam = 0 phase mode)

    The point is conditioning, not accuracy.  A single half-period map amplifies
    along the growing mode by e^{lam0 Delta/2} = 100, which is what limits
    Newton's basin; each segment amplifies by only 100^{1/k}.  One residual
    evaluation still costs exactly one half period of integration, so the extra
    unknowns are nearly free -- only GMRES sees the larger system.
    """

    def __init__(self, S, nstep, k, Hphase, Href):
        self.S, self.k = S, k
        self.nseg = max(nstep // k, 2)
        self.w = Hphase / max(np.linalg.norm(Hphase), 1e-300)
        self.c0 = float(self.w @ Href)
        self.nmap = 0
        self.N = S.N

    def seg(self, H, Delta):
        self.nmap += 1
        return self.S.evolve(H, 0.5 * Delta / self.k, self.nseg)

    def pack(self, Hs, Delta):
        return np.concatenate(list(Hs) + [[Delta]])

    def unpack(self, z):
        N, k = self.N, self.k
        return [z[j * N:(j + 1) * N] for j in range(k)], z[-1]

    def R(self, z):
        Hs, D = self.unpack(z)
        out = []
        for j in range(self.k):
            nxt = self.seg(Hs[j], D)
            out.append(nxt - Hs[j + 1] if j < self.k - 1 else nxt + Hs[0])
        out.append(np.array([self.w @ Hs[0] - self.c0]))
        return np.concatenate(out)

    def jvp(self, z, v, eps=1e-6):
        s = eps * (1.0 + np.linalg.norm(z) / max(np.linalg.norm(v), 1e-300))
        return (self.R(z + s * v) - self.R(z - s * v)) / (2.0 * s)


def solve_ms(S, H0, Delta0, nstep, k=4, tol=1e-11, maxit=25, gmres_tol=1e-4,
             verbose=True, restart=40, maxiter=4):
    """Multiple-shooting Newton-Krylov.  Seed segments by evolving H0."""
    Hp = (S.evolve(H0, 1e-3, 2) - H0) / 1e-3
    P = MSProblem(S, nstep, k, Hp, H0)
    Hs = [H0]
    for j in range(1, k):
        Hs.append(P.seg(Hs[-1], Delta0))
    z = P.pack(Hs, Delta0)
    r = P.R(z); nr = np.linalg.norm(r)
    if verbose:
        print(f"  ms k={k} it  0  |R|={nr:.4e}  Delta={z[-1]:.9f}", flush=True)
    for it in range(1, maxit + 1):
        J = LinearOperator((len(z), len(z)), matvec=lambda v: P.jvp(z, v))
        dz, info = gmres(J, -r, rtol=gmres_tol, atol=0.0, restart=restart,
                         maxiter=maxiter)
        if not np.all(np.isfinite(dz)):
            break
        lam = 1.0
        ok = False
        for _ in range(14):
            zt = z + lam * dz
            rt = P.R(zt); nt = np.linalg.norm(rt)
            if np.isfinite(nt) and nt < nr:
                ok = True; break
            lam *= 0.5
        if not ok:
            break
        z, r, nr_old, nr = zt, rt, nr, nt
        if verbose:
            print(f"  ms k={k} it {it:2d}  |R|={nr:.4e}  Delta={z[-1]:.9f}  "
                  f"step={lam:.3f}  nmap={P.nmap}", flush=True)
        if nr < tol:
            break
    Hs, D = P.unpack(z)
    return Hs[0], float(D), nr, P
