"""
Driver for spherically symmetric massless-scalar critical collapse in
Christodoulou/Garfinkle null coordinates, using the numba core.

The null grid is Lagrangian in v and therefore zooms in on the self-similar
structure automatically (Garfinkle 1995): no adaptive mesh refinement.
"""
import numpy as np
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # src/
from core import nullcore as nc

NSCRATCH = 16


class Run:
    def __init__(self, p, r0=0.25, sigma=0.05, vmax=1.0, n=800, cfl=0.25,
                 drop_frac=0.3, ah_tol=0.02, disperse_hi=0.999,
                 strong_lo=0.98, profile="gauss", Kdom=4.0,
                 s=0.0, r1=0.45, sigma1=0.07):
        self.p, self.r0, self.sigma, self.vmax = p, r0, sigma, vmax
        self.s, self.r1, self.sigma1 = s, r1, sigma1
        self.n0, self.cfl, self.drop_frac = n, cfl, drop_frac
        self.ah_tol, self.disperse_hi, self.strong_lo = ah_tol, disperse_hi, strong_lo
        self.Kdom = Kdom
        v = np.linspace(0.0, vmax, n + 1)[1:]
        self.v = v.copy(); self.r = v.copy()
        self.h = self._initial_h(v, profile)
        self.u = 0.0
        self.r_ah = np.nan
        self.snaps = []
        self._alloc(len(v) * 2 + 8)
        self.rec = {k: [] for k in ("u", "phi0", "minfac", "rmax", "rmin", "n", "hmax", "rref")}

    def _initial_h(self, r, profile):
        p, r0, s = self.p, self.r0, self.sigma
        if profile == "gauss":
            E = np.exp(-((r - r0) / s) ** 2)
            Phi = p * r ** 2 * E
            dPhi = p * (2 * r * E + r ** 2 * E * (-2 * (r - r0) / s ** 2))
        elif profile == "tanh":                      # a second, different family
            E = 1.0 / np.cosh((r - r0) / s) ** 2
            Phi = p * r ** 2 * E
            dPhi = p * (2 * r * E - 2 * r ** 2 * E * np.tanh((r - r0) / s) / s)
        elif profile == "gauss_odd":                 # third family
            E = np.exp(-((r - r0) / s) ** 2)
            Phi = p * r ** 3 * E
            dPhi = p * (3 * r ** 2 * E + r ** 3 * E * (-2 * (r - r0) / s ** 2))
        elif profile == "twobump":
            # continuous deformation: a second pulse of relative amplitude s.
            # s = 0 recovers family G1; growing s moves energy outward.
            sc, r1, s1 = self.s, self.r1, self.sigma1
            x0 = (r - r0) / s; x1 = (r - r1) / s1
            E0 = np.exp(-x0 ** 2); E1 = np.exp(-x1 ** 2)
            F = E0 + sc * E1
            dF = E0 * (-2 * x0 / s) + sc * E1 * (-2 * x1 / s1)
            Phi = p * r ** 2 * F
            dPhi = p * (2 * r * F + r ** 2 * dF)
        elif profile == "morph":
            # continuous deformation between the "gauss" and "tanh" families
            sc = self.s
            x = (r - r0) / s
            E = np.exp(-x ** 2); S = 1.0 / np.cosh(x) ** 2
            F = (1.0 - sc) * E + sc * S
            dF = (1.0 - sc) * E * (-2 * x / s) + sc * (-2 * S * np.tanh(x) / s)
            Phi = p * r ** 2 * F
            dPhi = p * (2 * r * F + r ** 2 * dF)
        else:
            raise ValueError(profile)
        return Phi + r * dPhi

    def _alloc(self, m):
        self.w = np.zeros((NSCRATCH, m + 1))
        self.buf = [np.empty(m) for _ in range(3)]
        self.ebuf = [np.empty(m + 1) for _ in range(3)]

    def _fields(self):
        n = len(self.r)
        hb, g, gb = (b[:n] for b in self.buf)
        re, ye, I = (b[:n + 1] for b in self.ebuf)
        nc.metric_fields(self.r, self.h, hb, g, gb, re, ye, I)
        return hb, g, gb

    def _feature_scale(self, hb):
        """
        Energy-weighted radius  <r> = int r w dr / int w dr  with
        w = 4 pi (h - hbar)^2 / r = d(ln g)/dr.   In a self-similar phase this
        tracks the shrinking feature scale, r_ref ~ (u* - u).
        """
        r, h = self.r, self.h
        w = (h - hb) ** 2 / r
        num = np.trapezoid(w * r, r); den = np.trapezoid(w, r)
        return float(num / den) if den > 0 else float(r[-1])

    def _truncate_outer(self, hb):
        """
        Discard outer grid points beyond K * r_ref.  This is EXACT, not an
        approximation: in these coordinates every quantity at grid point i is
        built from cumulative integrals over j <= i only, so removing outer
        points cannot influence the interior.  It is what keeps the resolution
        of the self-similar structure fixed as it shrinks.
        """
        if self.Kdom <= 0:
            return
        rref = self._feature_scale(hb)
        rcut = self.Kdom * rref
        if self.r[-1] > rcut:
            j = int(np.searchsorted(self.r, rcut))
            j = max(j, 16)
            if j < len(self.r):
                self.r, self.h, self.v = self.r[:j], self.h[:j], self.v[:j]

    def _regrid(self):
        r = self.r
        spac = np.diff(r)
        ok = r[:-1] > self.drop_frac * spac
        i0 = int(np.argmax(ok)) if ok.any() else len(r) - 1
        if i0 > 0:
            self.r, self.h, self.v = self.r[i0:], self.h[i0:], self.v[i0:]
        if len(self.r) < self.n0 // 2 and len(self.r) > 8:
            v, r, h = self.v, self.r, self.h
            vm = 0.5 * (v[1:] + v[:-1])
            rm = _cubic(v, r, vm); hm = _cubic(v, h, vm)
            nv = np.empty(2 * len(v) - 1); nv[0::2] = v; nv[1::2] = vm
            nr = np.empty_like(nv); nr[0::2] = r; nr[1::2] = rm
            nh = np.empty_like(nv); nh[0::2] = h; nh[1::2] = hm
            self.v, self.r, self.h = nv, nr, nh
            if len(nv) + 2 > self.w.shape[1]:
                self._alloc(2 * len(nv) + 8)

    def run(self, max_steps=2_000_000, snap=None, snap_every=0):
        out, minfac = "grid_exhausted", 1.0
        it = 0
        for it in range(max_steps):
            # The innermost grid point tracks a null ray that can reach the
            # centre; r decreases monotonically (dr = -gbar/2 < 0), so once
            # r[0] <= 0 that ray has left the domain and must be discarded.
            # Otherwise hbar = I/r divides by zero.
            if len(self.r) and self.r[0] <= 0.0:
                k = int(np.searchsorted(self.r, 0.0, side="right"))
                self.r, self.h, self.v = self.r[k:], self.h[k:], self.v[k:]
            r, h = self.r, self.h
            if len(r) < 12:
                break
            # Double-precision floor.  The Lagrangian grid zooms in on the
            # self-similar structure without bound, so eventually neighbouring
            # rays have r values that are indistinguishable in double
            # precision and the cumulative quadrature divides by zero.  That is
            # the resolution limit of the method, not a failure of the run:
            # stop and report it as such.  Empirically it is reached at about
            # 14 digits of tuning with n = 600.
            sp = np.diff(r)
            bad = sp <= 1e-13 * r[:-1]
            if bad.any():
                # The innermost rays have converged to within double precision
                # of each other; a zero spacing divides by zero in the
                # cumulative quadrature.  Drop the degenerate points -- they
                # carry no further information -- and continue.  Only if that
                # exhausts the grid is the run reported as unresolved.
                keep = np.concatenate([[True], ~bad])
                self.r, self.h, self.v = self.r[keep], self.h[keep], self.v[keep]
                r, h = self.r, self.h
                if len(r) < 12:
                    out = "unresolved"
                    break
            hb, g, gb = self._fields()
            fac = gb / g
            mf = float(fac.min()); minfac = min(minfac, mf)
            self.rec["u"].append(self.u)
            self.rec["phi0"].append(float(nc._extrap0(r, hb)))
            self.rec["minfac"].append(mf)
            self.rec["rmax"].append(float(r[-1]))
            self.rec["rmin"].append(float(r[0]))
            self.rec["rref"].append(self._feature_scale(hb))
            self.rec["n"].append(len(r))
            self.rec["hmax"].append(float(np.abs(h).max()))
            if snap_every and it % snap_every == 0:
                self.snaps.append((self.u, self.r.copy(), self.h.copy()))
            if mf < self.ah_tol:
                self.r_ah = float(r[int(np.argmin(fac))]); out = "black_hole"; break
            if minfac < self.strong_lo and mf > self.disperse_hi:
                out = "dispersed"; break
            du = self.cfl * float(np.diff(r).min())
            if not np.isfinite(du) or du <= 0:
                out = "failed"; break
            if self.u + du == self.u:          # u can no longer advance
                out = "unresolved"; break
            nc.rk4(self.r, self.h, du, self.w)
            self.u += du
            if not (np.all(np.isfinite(self.h)) and np.all(np.isfinite(self.r))):
                out = "failed"; break
            self._truncate_outer(hb)
            self._regrid()
        for k in self.rec:
            self.rec[k] = np.asarray(self.rec[k])
        return dict(outcome=out, min_fac=minfac, u_end=self.u, steps=it,
                    rec=self.rec, r_ah=self.r_ah, mass=0.5 * self.r_ah,
                    snaps=self.snaps)


def _cubic(x, y, xq):
    idx = np.clip(np.searchsorted(x, xq) - 2, 0, len(x) - 4)
    out = np.zeros_like(xq)
    for j in range(4):
        num = np.ones_like(xq); den = np.ones_like(xq); xj = x[idx + j]
        for k in range(4):
            if k != j:
                xk = x[idx + k]; num *= (xq - xk); den *= (xj - xk)
        out += y[idx + j] * num / den
    return out


def classify(p, **kw):
    R = Run(p, **kw).run()
    return R["outcome"] == "black_hole", R


def bisect(plo, phi, ndigit=15, verbose=True, **kw):
    """Bisection for the critical amplitude.  plo disperses, phi collapses."""
    hist = []
    for k in range(int(np.ceil(ndigit * np.log2(10)))):
        pm = 0.5 * (plo + phi)
        bh, R = classify(pm, **kw)
        hist.append((pm, R["outcome"], R["min_fac"], R["steps"]))
        if verbose:
            print(f"  [{k:3d}] p={pm:.17g} {R['outcome']:14s} "
                  f"minfac={R['min_fac']:.4f} steps={R['steps']}", flush=True)
        if R["outcome"] in ("failed", "unresolved"):
            # the run cannot be classified, so the bisection cannot continue;
            # return the bracket achieved so far rather than guessing
            break
        if bh:
            phi = pm
        else:
            plo = pm
        if abs(phi - plo) <= 4 * np.finfo(float).eps * abs(phi):
            break
    return plo, phi, hist
