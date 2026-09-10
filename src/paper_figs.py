"""Figures for paper A (docs/10_paper_plan.md).  Writes paper/figs/*.pdf."""
import os, glob, re, json
import numpy as np
import scipy.linalg as sla
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.interpolate import CubicSpline

OUT, FIG = "out/dss", "paper/figs"
DELTA_REF = 3.445452402
XH0, XHMIN, XHMAX = 1.352471, 1.297337, 1.560820

plt.rcParams.update({
    "font.size": 9, "axes.labelsize": 9, "legend.fontsize": 8,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "lines.linewidth": 1.35,
    "axes.linewidth": 0.7, "xtick.direction": "in", "ytick.direction": "in",
    "xtick.top": True, "ytick.right": True, "figure.dpi": 160,
    "savefig.bbox": "tight", "savefig.pad_inches": 0.04,
    "mathtext.fontset": "cm", "font.family": "serif",
    "text.color": "#222222", "axes.labelcolor": "#222222",
    "xtick.color": "#222222", "ytick.color": "#222222",
    "figure.facecolor": "white", "axes.facecolor": "white",
    "savefig.facecolor": "white", "legend.handlelength": 2.5,
    "hatch.linewidth": 0.45, "pdf.fonttype": 42, "ps.fonttype": 42,
})
# Dark colors supplement line patterns and marker shapes; they never identify
# a series on their own. Embed TrueType fonts so PDF labels remain selectable.
C = dict(fd="#006B99", cb="#A64800", g="#006B54", pu="#76518F",
         k="#222222", gy="#6b6b6b")
DASHED = (0, (4, 2))
DASHDOT = (0, (5, 1.8, 1.2, 1.8))


def _horizon_band(ax):
    ax.axvspan(XHMIN, XHMAX, facecolor="#f0f0f0", edgecolor="#8a8a8a",
               hatch="//", lw=0, zorder=0)


def _eig(J, D):
    nu, vl, vr = sla.eig(J, left=True, right=True)
    lam = np.log(nu.astype(complex) ** 2) / D
    o = np.argsort(-lam.real)
    nu, vl, vr, lam = nu[o], vl[:, o], vr[:, o], lam[o]
    W = np.array([vl[:, j] / np.conj(np.vdot(vl[:, j], vr[:, j]))
                  for j in range(len(nu))])
    return nu, lam, vr, W


def _pick1(lam):
    c = np.where((lam.real < -0.05) & (abs(lam.imag) < 1e-6))[0]
    return int(c[np.argmax(lam.real[c])])


def _pickdec(lam):
    """Leading decaying eigenvalue whatever its type: the rule of Table III.

    Below N ~ 176 it is a complex pair and is not lam_1; picking the leading
    REAL one there would plot a different sequence from the table's.
    """
    c = np.where(lam.real < -0.05)[0]
    return int(c[np.argmax(lam.real[c])])


def cheb_files():
    out = []
    for f in glob.glob(f"{OUT}/cheb_N*_X*.npz"):
        m = re.search(r"_N(\d+)_X([\d.]+)\.npz", f)
        out.append((int(m.group(1)), float(m.group(2).rstrip(".")), f))
    return sorted(out)


# ------------------------------------------------------------------ figure 1
def fig_solution():
    d = np.load(f"{OUT}/spec_N800_X4_w7.npz")
    X, H, D = d["X"], d["H"], float(d["Delta"])
    import dss_core as dc
    S = dc.Sim(N=len(X), Xmax=X[-1], wd=7)
    hb, g, gb = S.fields(H)
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.25),
                           gridspec_kw=dict(wspace=.30))
    a = ax[0]
    _horizon_band(a)
    a.axhline(0, color=C["k"], lw=.6)
    a.plot(X, H, color=C["fd"], label=r"$h$")
    a.plot(X, hb, color=C["cb"], ls=DASHED, label=r"$\bar h=\varphi$")
    a.set_xlim(0, 4); a.set_xlabel(r"$X=r/(u_*-u)$")
    a.set_ylabel("scalar field")
    lo, hi = a.get_ylim()
    a.set_ylim(lo, hi + .30 * (hi - lo))     # headroom for the legend
    a.legend(frameon=False, loc="upper right", ncol=2,
             handlelength=2.4, columnspacing=1.0, borderaxespad=.2)
    a.text(1.43, lo + .06 * (hi - lo), "SSH", ha="center", fontsize=8,
           color=C["k"])
    b = ax[1]
    _horizon_band(b)
    b.plot(X, g, color=C["fd"], label=r"$g$")
    b.plot(X, gb, color=C["cb"], ls=DASHED, label=r"$\bar g$")
    b.plot(X, g / gb, color=C["g"], ls=DASHDOT, label=r"$a^2=g/\bar g$")
    b.axhline(2.0, color=C["k"], lw=.7, ls=":")
    b.text(3.9, 2.06, r"$a^2=2$", fontsize=8, color=C["k"], ha="right")
    b.set_xlim(0, 4); b.set_xlabel(r"$X=r/(u_*-u)$")
    b.set_ylabel("metric functions")
    lo2, hi2 = b.get_ylim()
    b.set_ylim(lo2, hi2 + .26 * (hi2 - lo2))   # headroom for the legend
    b.legend(frameon=False, loc="upper center", ncol=3, handlelength=2.4,
             columnspacing=1.0, borderaxespad=.2)
    fig.savefig(f"{FIG}/solution.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ figure 2
def fig_convergence():
    cv = json.load(open(f"{OUT}/sweep_conv.json"))
    cv = [r for r in cv if r["resid"] < 1e-8]
    Nf = np.array([r["N"] for r in cv]); Df = np.array([r["Delta"] for r in cv])
    cb = [(N, f) for N, X, f in cheb_files() if X == 4.0]
    Nc, Dc = [], []
    for N, f in cb:
        z = np.load(f)
        if float(z["resid"]) < 1e-8:
            Nc.append(N); Dc.append(float(z["Delta"]))
    Nc, Dc = np.array(Nc), np.array(Dc)
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.25),
                           gridspec_kw=dict(wspace=.34))
    a = ax[0]
    a.loglog(Nf[:-1], abs(np.diff(Df)), "o-", color=C["fd"], ms=4,
             label=r"finite difference, $|\Delta_{2N}-\Delta_N|$")
    if len(Nc) > 2:
        o = np.argsort(Nc)
        a.loglog(Nc[o][:-1], abs(np.diff(Dc[o])), marker="s", ls=DASHED,
                 color=C["cb"], ms=4, mfc="white", mew=.9,
                 label=r"Chebyshev, successive $|\delta\Delta|$")
    nn = np.array([200., 1600.])
    a.loglog(nn, 3e-4 * (nn / 200) ** -4, ":", color=C["k"], lw=.95)
    a.text(560, 2.2e-5, r"$N^{-4}$", fontsize=8)
    import matplotlib.ticker as mt0
    a.set_xticks([100, 200, 400, 800, 1600])
    a.xaxis.set_major_formatter(mt0.ScalarFormatter())
    a.xaxis.set_minor_formatter(mt0.NullFormatter())
    a.set_xlabel(r"$N$"); a.set_ylabel(r"$|\delta\Delta|$")
    a.set_ylim(2e-10, 1e-2)
    a.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.10),
             fontsize=7.5, borderaxespad=0)
    b = ax[1]
    b.semilogx(Nf, (Df - DELTA_REF) / DELTA_REF, "o-", color=C["fd"], ms=4,
               label="finite difference")
    if len(Nc):
        o = np.argsort(Nc)
        m = Nc[o] >= 144          # below this the Chebyshev solve is not
        b.semilogx(Nc[o][m], (Dc[o][m] - DELTA_REF) / DELTA_REF,
                   marker="s", ls=DASHED, color=C["cb"], ms=4,
                   mfc="white", mew=.9,
                   label=r"Chebyshev ($N \geq 144$)")
    b.axhline(0, color=C["k"], lw=.6)
    b.set_ylim(-2e-6, 2e-6)
    b.ticklabel_format(axis="y", style="sci", scilimits=(0, 0))
    # explicit decade-ish ticks: the default minor log labels collide
    import matplotlib.ticker as mt
    b.set_xticks([100, 200, 400, 800, 1600])
    b.xaxis.set_major_formatter(mt.ScalarFormatter())
    b.xaxis.set_minor_formatter(mt.NullFormatter())
    b.set_xlabel(r"$N$")
    b.set_ylabel(r"$\delta\Delta/\Delta_{\rm MG03}$")
    b.legend(frameon=False, loc="lower center", bbox_to_anchor=(.5, 1.10),
             fontsize=7.5, borderaxespad=0)
    fig.savefig(f"{FIG}/convergence.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ figure 3
def fig_spectrum():
    N = max(N for N, X, f in cheb_files() if X == 4.0
            and "J" in np.load(f).files)
    f = f"{OUT}/cheb_N{N}_X4.npz"
    z = np.load(f); D = float(z["Delta"])
    nu, lam, vr, W = _eig(z["J"], D)
    j1 = _pick1(lam)
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.55),
                           gridspec_kw=dict(wspace=.32))
    a = ax[0]
    keep = lam.real > -4.5
    a.scatter(lam.real[keep], lam.imag[keep], s=12, facecolor="none",
              edgecolor=C["gy"], lw=.8, label="map spectrum")
    # labels staggered vertically: the four marked modes are all on Im=0
    # and their annotations otherwise overlap into an unreadable run.
    for x, t, dx, dy, ha, marker in [
            (lam[0].real, r"$\lambda_0=+2.674$", 0, 34, "center", "o"),
            (1.0, r"gauge $\lambda=1$", 0, 22, "center", "^"),
            (0.0, r"$\lambda=0$ (2)", 0, 10, "center", "s")]:
        a.plot([x], [0], marker=marker, ms=5, color=C["fd"])
        a.annotate(t, (x, 0), textcoords="offset points", xytext=(dx, dy),
                   fontsize=7.5, ha=ha, color=C["k"],
                   arrowprops=dict(arrowstyle="-", lw=.6, color=C["k"],
                                   shrinkA=0, shrinkB=3))
    a.plot([lam[j1].real], [0], "D", ms=5, color=C["cb"])
    a.annotate(r"$\lambda_1=-0.895$", (lam[j1].real, 0),
               textcoords="offset points", xytext=(16, -34), fontsize=7.5,
               ha="left", color=C["k"],
               arrowprops=dict(arrowstyle="-", lw=.6, color=C["k"],
                               shrinkA=0, shrinkB=3))
    a.axvline(0, color=C["k"], lw=.6); a.axhline(0, color=C["k"], lw=.6)
    a.set_xlim(-4.6, 3.9); a.set_ylim(-1.05, 1.15)
    a.set_xlabel(r"${\rm Re}\,\lambda$")
    a.set_ylabel(r"${\rm Im}\,\lambda$")
    a.set_title(rf"Chebyshev $N={N}$, $X_{{\max}}=4$", fontsize=8.5)
    b = ax[1]
    Ns, L1, Nc, Lc = [], [], [], []
    for n, X, ff in cheb_files():
        if X != 4.0 or n < 128:      # the panel starts where its axis starts
            continue
        zz = np.load(ff)
        if "J" not in zz.files:
            continue
        _, ll, _, _ = _eig(zz["J"], float(zz["Delta"]))
        j = _pickdec(ll)
        if abs(ll[j].imag) < 1e-6:
            Ns.append(n); L1.append(ll[j].real)
        else:
            Nc.append(n); Lc.append(ll[j].real)
    b.plot(Ns, L1, "s-", color=C["cb"], ms=4,
           label=r"$X_{\max}=4$")
    if Nc:
        b.plot(Nc, Lc, "s", color=C["cb"], ms=4, mfc="white", mew=.9,
               label=r"complex pair, not $\lambda_1$")
    # The X_max scan must be shown at ONE resolution, or the panel appears
    # to contradict the text: N=176 is not converged at X_max=6.
    NX = 224
    xs, ys = [], []
    for n, X, ff in cheb_files():
        if X == 4.0 or n != NX:
            continue
        zz = np.load(ff)
        if "J" not in zz.files:
            continue
        _, ll, _, _ = _eig(zz["J"], float(zz["Delta"]))
        xs.append(X); ys.append(ll[_pick1(ll)].real)
    zz = np.load(f"{OUT}/cheb_N{NX}_X4.npz")
    _, ll, _, _ = _eig(zz["J"], float(zz["Delta"]))
    xs.append(4.0); ys.append(ll[_pick1(ll)].real)
    o = np.argsort(xs); xs = np.array(xs)[o]; ys = np.array(ys)[o]
    b.axhline(L1[-1], color=C["k"], lw=.7, ls=":")
    b.set_xlabel(r"$N$"); b.set_ylabel(r"$\lambda_1$")
    b.set_ylim(-1.14, -0.52)   # room for the legend and the complex entries
    b.set_xlim(120, 300)
    # At fixed N the triangles vary the domain AND the near-horizon node
    # spacing.  The second series holds the spacing fixed by raising N with
    # X_max, so that only the boundary moves; without it neither effect is
    # bounded on its own.  The set is chosen by paper_tables._matched so the
    # figure and Table III always show the same solves.
    import paper_tables as pt
    mrow, href = pt._matched(pt._scan())
    mx = [r[1] for r in mrow]
    my = [r[4].real for r in mrow]
    if len(xs):
        bb = b.twiny()
        bb.plot(xs, ys, marker="^", ls=DASHED, color=C["g"], ms=4.5,
                label=rf"$N={NX}$")
        if len(mx) >= 3:
            bb.plot(mx, my, marker="o", ls=DASHDOT, color=C["pu"], ms=4.5,
                    mfc="white", mew=1,
                    label=rf"matched $h={href:.3f}$")
        bb.set_xlim(2.6, 6.4)
        bb.set_xlabel(rf"$X_{{\max}}$", color=C["k"], fontsize=9)
        bb.tick_params(axis="x", colors=C["k"])
        # The two boundary scans collapse onto one line at the outer scale.
        # The inset is where the comparison is visible: the joint scan (fixed
        # N) is flat, the matched-h scan drifts monotonically by 5.4e-3.  The
        # difference between them is the point -- neither is a pure control.
        if len(mx) >= 3:
            ins = b.inset_axes([0.17, 0.58, 0.47, 0.34])
            ins.plot(xs, ys, marker="^", ls=DASHED, color=C["g"], ms=4, lw=1.1)
            ins.plot(mx, my, marker="o", ls=DASHDOT, color=C["pu"], ms=4,
                     lw=1.1, mfc="white", mew=.9)
            ins.set_xlim(2.6, 6.4); ins.set_ylim(-0.9002, -0.8925)
            ins.set_xticks([3, 4, 5, 6])
            ins.set_yticks([-0.899, -0.896, -0.893])
            ins.yaxis.tick_right()     # keep labels off the main y axis
            ins.tick_params(labelsize=7, pad=2)
            ins.set_xlabel(r"$X_{\max}$", fontsize=7, labelpad=1)
    # Short labels fit in the right panel's empty lower-right corner.
    hs, ls = b.get_legend_handles_labels()
    if len(xs):
        h2, l2 = bb.get_legend_handles_labels()
        hs, ls = hs + h2, ls + l2
    b.legend(hs, ls, frameon=False, loc="lower right", fontsize=7.5,
             borderaxespad=.5, labelspacing=.3, handlelength=2.2)
    fig.savefig(f"{FIG}/spectrum.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ figure 4
def fig_lam1_mode():
    N = max(N for N, X, f in cheb_files() if X == 4.0
            and "J" in np.load(f).files)
    z = np.load(f"{OUT}/cheb_N{N}_X4.npz")
    X, D = z["X"], float(z["Delta"])
    nu, lam, vr, W = _eig(z["J"], D)
    j1 = _pick1(lam)
    v = np.real(vr[:, j1]); v /= np.abs(v).max()
    w = np.real(W[j1]); w /= np.abs(w).max()
    fig, ax = plt.subplots(figsize=(3.45, 2.6))
    _horizon_band(ax)
    ax.axhline(0, color=C["k"], lw=.6)
    ax.plot(X, v, color=C["fd"], label=r"$v_1$ (response)")
    # The adjoint alternates at the grid scale -- ~90% of its discrete cosine
    # spectrum is in the upper third -- so the sign structure is not a feature
    # of a continuum function.  Show it, but draw the envelope over it: only
    # integrated functionals of |w_1| are stable under refinement.
    ax.plot(X, w, color="#707070", lw=.65, alpha=.8,
            label=r"$w_i$ (grid scale)")
    ax.plot(X, np.abs(w), color=C["cb"], lw=1.5, ls=DASHED,
            label=r"$|w_1|$ (receptivity)")
    ax.set_xlim(0, X[-1]); ax.set_ylim(-1.15, 1.15)
    ax.set_xlabel(r"$X$"); ax.set_ylabel("normalized eigenvector")
    ax.annotate("SSH", xy=(1.43, .86), xytext=(.55, .96), fontsize=8,
                color=C["k"], ha="left",
                arrowprops=dict(arrowstyle="-", lw=.6, color=C["k"]))
    handles, labels = ax.get_legend_handles_labels()
    order = [0, 2, 1]
    ax.legend([handles[i] for i in order], [labels[i] for i in order],
              frameon=False, loc="upper right", borderaxespad=.5,
              labelspacing=.3, handlelength=2, handletextpad=.6)
    fig.savefig(f"{FIG}/lam1_mode.pdf")
    plt.close(fig)


# ------------------------------------------------------------------ figure 5
def fig_horizon():
    d = np.load(f"{OUT}/ssh_invariants.npz")
    T, orb, mu = d["T"], d["orbit"], float(d["mu"])
    D = 2 * T[-1]
    fig, ax = plt.subplots(1, 2, figsize=(6.9, 2.45),
                           gridspec_kw=dict(wspace=.30))
    a = ax[0]
    a.plot(np.r_[T, T + 0.5 * D] / D, np.r_[orb, orb], color=C["fd"])
    a.axhline(XHMAX, color=C["k"], lw=.7, ls=":")
    a.axhline(XHMIN, color=C["k"], lw=.7, ls=":")
    span = XHMAX - XHMIN
    a.set_ylim(XHMIN - .30 * span, XHMAX + .30 * span)
    a.text(.99, XHMAX + .07 * span, r"$\max_T X_h=1.5608$", fontsize=8,
           ha="right", va="bottom")
    a.text(.99, XHMIN - .07 * span, r"$\min_T X_h=1.2973$", fontsize=8,
           ha="right", va="top")
    a.set_xlabel(r"$T/\Delta$"); a.set_ylabel(r"$X_h(T)$")
    a.set_title(r"self-similarity horizon as a periodic orbit", fontsize=8.5)
    b = ax[1]
    fl = np.load(f"{OUT}/flush.npz")
    L = np.log(1.0 / fl["eps"]); c = float(fl["c0"])
    Tf, res = fl["Tf"], fl["resid"]
    D, amp = float(fl["Delta"]), float(fl["amp"])
    xx = np.linspace(L.min() - .3, L.max() + .3, 10)
    b.plot(xx, xx / mu + c, "-", color=C["fd"], lw=1.35,
           label=rf"$\mu^{{-1}}\ln(1/\epsilon)+{c:.2f}$")
    b.plot(L, Tf, "o", color=C["cb"], ms=4, mfc="white", mew=.9,
           label="characteristics")
    b.set_xlabel(r"$\ln(1/\epsilon)$"); b.set_ylabel(r"$T_f$")
    b.set_title(r"flush law: $\epsilon=X_h(0)-X_0$", fontsize=8.5)
    b.legend(frameon=False, loc="lower right", fontsize=7.5,
             borderaxespad=.7, labelspacing=.35, handlelength=2.2)
    b.set_ylim(min(Tf) - 1.4, max(Tf) + 1.0)
    # Inset: the remainder is periodic in T_f with period Delta/2, not constant.
    ins = b.inset_axes([.16, .66, .35, .25])
    ph = (Tf / (0.5 * D)) % 1.0
    ins.axhline(c, color=C["k"], lw=.7, ls=":")
    tt = np.linspace(0, 1, 200)
    M = np.c_[np.ones_like(Tf), np.cos(2 * np.pi * Tf / (0.5 * D)),
              np.sin(2 * np.pi * Tf / (0.5 * D))]
    co = np.linalg.lstsq(M, res, rcond=None)[0]
    ins.plot(tt, co[0] + co[1] * np.cos(2 * np.pi * tt)
             + co[2] * np.sin(2 * np.pi * tt), "-", color=C["fd"], lw=1.1)
    ins.plot(ph, res, "o", color=C["cb"], ms=3.3, mfc="white", mew=.8)
    ins.set_xlabel(r"$T_f/(\Delta/2)$ mod 1", fontsize=7, labelpad=2)
    ins.xaxis.set_label_position("top")
    ins.set_ylabel(r"remainder", fontsize=7, labelpad=1)
    ins.tick_params(labelsize=7, pad=1.5)
    ins.set_xlim(0, 1); ins.set_xticks([0, .5, 1], labels=["0", "0.5", "1"])
    fig.savefig(f"{FIG}/horizon.pdf")
    plt.close(fig)


if __name__ == "__main__":
    os.makedirs(FIG, exist_ok=True)
    import sys
    which = sys.argv[1:] or ["solution", "convergence", "spectrum",
                             "lam1_mode", "horizon"]
    for w in which:
        globals()[f"fig_{w}"]()
        print("wrote", w, flush=True)
