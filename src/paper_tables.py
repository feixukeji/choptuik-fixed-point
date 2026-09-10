"""LaTeX tables for paper A.  Writes paper/tables/*.tex from out/dss/*."""
import os, re, glob, json
import numpy as np
import scipy.linalg as sla
import dss_core as dc, dss_cheb as cb, dss_spectrum as sp, nullcore as nc

OUT, TAB = "out/dss", "paper/tables"
BS = "\\\\"


def sci(x, nd=2, sign=True):
    """LaTeX scientific notation."""
    if x == 0:
        return "$0$"
    t = f"{x:{'+' if sign else ''}.{nd}e}"
    m, e = t.split("e")
    return f"${m}\\times10^{{{int(e)}}}$"
DREF = 3.445452402
# Reiterer & Trubowitz, arXiv:1203.3766 / CMP 368, 143 (2019): their Eqs. (2)
# and (3) give K = Delta/2 and the invariant mu_RT to 80 digits with rigorous
# error bounds.  Our mu is 4 pi mu_RT / Delta = 2 pi mu_RT / K -- see the paper.
RT_K = 1.72272620111391067498575597278819186222103458805781088634788403570540
RT_MU = 0.16830707896344996951013497904285742072100199080892966476395293134873
MU_RT_EXACT = 2 * np.pi * RT_MU / RT_K
MULT_RT_EXACT = np.exp(4 * np.pi * RT_MU)
# horizon_scalars('fd', N, 4.0, 7, 1600), see docs/03 section O.
MU_N = {200: 0.613884169, 400: 0.613853436, 800: 0.613855182}


def w(name, body):
    open(f"{TAB}/{name}.tex", "w").write(body.rstrip() + "\n")
    print("wrote", name, flush=True)


def cheb_files(Xmax=None):
    o = []
    for f in glob.glob(f"{OUT}/cheb_N*_X*.npz"):
        m = re.search(r"_N(\d+)_X([\d.]+)\.npz", f)
        N, X = int(m.group(1)), float(m.group(2).rstrip("."))
        if Xmax is None or X == Xmax:
            o.append((N, X, f))
    return sorted(o)


def _eig(J, D):
    nu, vl, vr = sla.eig(J, left=True, right=True)
    lam = np.log(nu.astype(complex) ** 2) / D
    o = np.argsort(-lam.real)
    return nu[o], lam[o], vr[:, o], vl[:, o]


def _lam1(lam):
    c = np.where((lam.real < -0.05) & (abs(lam.imag) < 1e-6))[0]
    return int(c[np.argmax(lam.real[c])])


# ------------------------------------------------------------------- conv
def t_conv():
    fd = [r for r in json.load(open(f"{OUT}/sweep_conv.json"))
          if r["resid"] < 1e-8]
    rows = ["\\begin{table*}[!t]", "\\caption{The echoing period.  $\\Delta$ is an unknown of every solve, never an input.  \\emph{Cold} rows are independent Newton solves from $\\Delta_0=3.30$; \\emph{warm} rows are continuations in $N$ seeded from the previous row, the chain starting at the cold $N=224$ row, and are not independent determinations of $\\Delta$.  The two blocks share no spatial code.  $|R|$ is the converged residual of Eq.~\\eqref{eq:R}.}",
            "\\label{tab:conv}", "\\begin{ruledtabular}",
            "\\begin{tabular}{llrrl r l}",
            "scheme & start & $N$ & $n_{\\rm step}$ & $\\Delta$ & "
            "$(\\Delta-\\Delta_{\\rm MG03})/\\Delta_{\\rm MG03}$ & $|R|$\\\\",
            "\\hline"]
    for i, r in enumerate(fd):
        tag = "banded FD" if i == 0 else ""
        rows.append(f"{tag} & cold & {r['N']} & {r['nstep']} & "
                    f"{r['Delta']:.10f} & "
                    + sci((r['Delta'] - DREF) / DREF) + " & "
                    + sci(r['resid'], 0, False) + BS)
    dfd = np.array([r["Delta"] for r in fd])
    rich = dfd[-1] + (dfd[-1] - dfd[-2]) / 15.0
    rows.append("\\hline")
    rows.append(f"\\multicolumn{{4}}{{l}}{{Richardson, $N=800,1600$}} & "
                f"{rich:.10f} & " + sci((rich - DREF) / DREF) + " & " + BS)
    rows.append("\\hline")
    first = True
    for N, X, f in cheb_files(4.0):
        z = np.load(f)
        if float(z["resid"]) > 1e-8 or N < 160:
            continue
        tag = "Chebyshev" if first else ""
        # cold runs record the seed slow time T0; warm continuations do not.
        how = "cold" if "T0" in z.files else "warm"
        rows.append(f"{tag} & {how} & {N} & {int(z['nstep'])} & "
                    f"{float(z['Delta']):.10f} & "
                    + sci((float(z['Delta']) - DREF) / DREF) + " & "
                    + sci(float(z['resid']), 0, False) + BS)
        first = False
    rows += ["\\hline",
             "\\multicolumn{4}{l}{Mart\\'in-Garc\\'ia \\& Gundlach 2003, Eq.~(58)"
             "~\\cite{gundlach2003}} & "
             f"{DREF:.9f}(3) & --- & \\\\",
             "\\end{tabular}", "\\end{ruledtabular}", "\\end{table*}"]
    w("conv", "\n".join(rows))


# Full-width floats carry [!t] and never [p]: a two-column float can only sit
# at a page top or on a float page, and letting them take float pages leaves
# pages holding one figure and its caption (Sec. U of docs/03_progress.md).
# ------------------------------------------------------------------ exact
def t_exact(Ns=(200, 400, 800)):
    res = {}
    for N in Ns:
        S, J, D, X, H, dTH, dXH = sp.load("fd", N)
        ex = sp.trivial(S, X, H, dTH, dXH)
        ex["gauge"] = (ex["gauge"][0], -np.exp(D / 2))
        for k, (v, nx) in ex.items():
            Jv = J @ v
            nu = float(v @ Jv / (v @ v))
            res.setdefault(k, []).append(
                (abs(nu - nx) / abs(nx),
                 np.linalg.norm(Jv - nu * v) / np.linalg.norm(Jv)))
    lab = {"constant": (r"$H\to H+c$", r"$\mathbf 1$", "$+1$"),
           "phase": (r"$T\to T+a$", r"$H_T$", "$-1$"),
           "gauge": (r"$u_*\to u_*+d$", r"$H_T+XH_X$", r"$-e^{\Delta/2}$")}
    rows = ["\\begin{table*}[!t]",
            "\\caption{The three exactly known Floquet modes of Sec.~\\ref{sec:trivial}, checked against the numerical Jacobian of the banded map.  Columns under $N$ give the relative error in the multiplier $\\nu$; the last column is the eigenvalue residual $\\|Jv-\\nu v\\|/\\|Jv\\|$ at $N=800$.}",
            "\\label{tab:exact}", "\\begin{ruledtabular}",
            "\\begin{tabular}{llc" + "c" * len(Ns) + "c}",
            "symmetry & eigenvector & exact $\\nu$ & " +
            " & ".join(f"$N={n}$" for n in Ns) + " & residual\\\\", "\\hline"]
    for k in ("constant", "phase", "gauge"):
        s, v, nx = lab[k]
        rows.append(f"{s} & {v} & {nx} & " +
                    " & ".join(sci(e, 1, False) for e, _ in res[k]) +
                    " & " + sci(res[k][-1][1], 1, False) + BS)
    rows += ["\\end{tabular}", "\\end{ruledtabular}", "\\end{table*}"]
    w("exact", "\n".join(rows))


# ------------------------------------------------------------------- lam1
def _lam2(lam, j1):
    """The next distinct eigenvalue below lam1."""
    k = j1 + 1
    while k < len(lam) - 1 and abs(lam[k].real - lam[j1].real) < 1e-6:
        k += 1
    return k


XH_REF = 1.3525          # the horizon orbit at the reference phase


def _hgap(f):
    """Collocation gap bracketing the horizon, and nodes strictly inside it.

    At fixed N, enlarging Xmax also coarsens the grid near the horizon, so the
    plain Xmax scan varies two things at once.  This is the second of them,
    measured rather than assumed, so that a scan holding it fixed can be built.
    """
    X = np.load(f)["X"]
    k = int(np.searchsorted(X, XH_REF))
    return float(X[k] - X[k - 1]), int((X < XH_REF).sum())


def _matched(sc, tol=0.06, Nmin=176):
    """Outer-boundary scan at (nearly) fixed near-horizon node spacing.

    At fixed N a scan in Xmax moves the boundary AND coarsens the grid near
    the horizon, so it varies two things at once.  Raising N with Xmax holds
    the second fixed.  Every converged solve is tried as the reference gap;
    the one covering the most distinct Xmax (tie-break: tightest spread in h)
    wins, so this improves automatically as matched solves are added.  Rows
    are (N, Xmax, h, nin, lead, is_real, next), sorted by Xmax.
    """
    h = {(N, X): _hgap(f) for N, X, f in cheb_files()}
    cand = [t for t in sc if t[0] >= Nmin]

    def group(href):
        out = []
        for X in sorted({t[1] for t in cand}):
            c = [t for t in cand
                 if t[1] == X and abs(h[(t[0], X)][0] / href - 1) < tol]
            if c:
                t = min(c, key=lambda t: abs(h[(t[0], X)][0] / href - 1))
                g, nin = h[(t[0], X)]
                out.append((t[0], X, g, nin, t[2], t[3], t[4]))
        return out

    best, bref = [], None
    for t in cand:
        g = group(h[(t[0], t[1])][0])
        if not g:
            continue
        sp = max(r[2] for r in g) / min(r[2] for r in g)
        if (len(g), -sp) > (len(best), -(max(r[2] for r in best)
                                         / min(r[2] for r in best))
                            if best else -1e9):
            best, bref = g, h[(t[0], t[1])][0]
    return best, bref


def _lead_decay(lam):
    """(value, is_real) of the leading decaying exponent, real or complex."""
    d = lam[lam.real < -0.05]
    j = int(np.argmax(d.real))
    return d[j], abs(d[j].imag) < 1e-6


def _scan():
    """Leading decaying exponent, leading REAL one, and the next below.

    ONE definition is used for the reported exponent everywhere in the budget
    table: the least-damped decaying eigenvalue, whatever its type.  The
    leading real decaying eigenvalue is carried alongside because it is how
    lam1 first appears -- at N = 144 and 160 it is already near -0.895 while
    the least-damped decaying eigenvalue is still a complex pair -- but it is
    reported in the text, not mixed into the same column.
    """
    out = []
    for N, X, f in cheb_files():
        z = np.load(f)
        if "J" not in z.files:
            continue
        _, lam, _, _ = _eig(z["J"], float(z["Delta"]))
        d, real = _lead_decay(lam)
        jd = int(np.argmin(abs(lam - d)))
        out.append((N, X, d, real, lam[_lam2(lam, jd)], lam[_lam1(lam)].real))
    return out


def _eps_scan():
    """lam1 vs the Jacobian difference step, at each N for which it was run.

    eps = 1e-6 is the default used everywhere else, so the baseline comes from
    the stored spectrum of the fixed point itself.  Only resolutions that
    actually resolve lam1 are useful as a control on lam1; the leading decaying
    eigenvalue is reported whatever its type, so a resolution that has not yet
    produced a real lam1 is visible as such rather than silently compared.
    """
    out = {}
    for f in glob.glob(f"{OUT}/cheb_spec_N*_e*.npz"):
        m = re.search(r"_N(\d+)_e([\deE.+-]+)\.npz", f)
        N, e = int(m.group(1)), float(m.group(2))
        z = np.load(f)
        out.setdefault(N, {})[e] = np.asarray(z["lam"])
    for N in list(out):
        base = f"{OUT}/cheb_N{N}_X4.npz"
        if not os.path.exists(base):
            del out[N]
            continue
        z = np.load(base)
        if "lam" not in z.files:
            del out[N]
            continue
        out[N][1e-6] = np.asarray(z["lam"])
    return out


def _eps_rows():
    """Rows for the difference-step block of the lam1 error budget."""
    sc = _eps_scan()
    rows, first = [], True
    for N in sorted(sc):
        es = sorted(sc[N], reverse=True)          # 1e-4, 1e-5, 1e-6
        vals = []
        for e in sorted(es):
            v, isreal = _lead_decay(sc[N][e])
            vals.append((e, v, isreal))
        for i, (e, v, isreal) in enumerate(vals):
            tag = "difference step" if first else ""
            first = False
            s = (f"${v.real:.5f}$" if isreal
                 else f"${v.real:.5f}\\pm{abs(v.imag):.4f}i$")
            note = "" if isreal else "not $\\lambda_1$"
            rows.append(f"{tag} & $N={N}$, $10^{{{int(round(np.log10(e)))}}}$ & "
                        f"{s} & {note}" + BS)
        v = np.array([x[1].real for x in vals])
        if all(x[2] for x in vals):
            rows.append(f"\\multicolumn{{2}}{{l}}{{spread over $\\epsilon$ at "
                        f"$N={N}$}} & " + sci(np.ptp(v), 1, False) + " & " + BS)
    return rows


def t_lam1():
    sc = _scan()
    Nser = sorted([t for t in sc if t[1] == 4.0], key=lambda t: t[0])
    # the X-scan is quoted at the highest N for which every domain is
    # resolved; at N=176 the X=6 case is not (see text).
    from collections import Counter
    cnt = Counter(t[0] for t in sc if t[1] != 4.0)
    Nx0 = 224 if cnt.get(224, 0) >= 3 else (
        max(cnt, key=lambda n: (cnt[n], n)) if cnt else None)
    Xser = sorted([t for t in sc if t[0] == Nx0], key=lambda t: t[1]) \
        if Nx0 else []
    Xchk = sorted([t for t in sc if t[0] > Nx0 and t[1] != 4.0
                   and any(u[1] == t[1] for u in Xser)], key=lambda t: (t[0], t[1]))
    good = [t[2].real for t in Nser if t[0] >= 176]

    def cell(d, real):
        """One entry of the exponent column, under a single definition."""
        return (f"${d.real:.5f}$" if real else
                f"${d.real:.3f}\\pm{abs(d.imag):.3f}i$")

    def nx(l2):
        """Next-below entry.  The branch is real at some (N, Xmax) and a
        complex pair at others, so the type has to show."""
        return (f"${l2.real:.3f}$" if abs(l2.imag) < 1e-6 else
                f"${l2.real:.3f}\\pm{abs(l2.imag):.2f}i$")

    rows = ["\\begin{table*}[!t]",
            "\\caption{Error budget for $\\lambda_1$.  Both exponent columns are the least-damped decaying eigenvalue of the half-period map and the next one below it in real part, each whatever its symmetry type; a single value is real, $a\\pm bi$ a complex pair.  Blocks: resolution at fixed $X_{\\max}$; boundary at fixed $N$; boundary at matched near-horizon node spacing $h$, with $n_{\\rm in}$ nodes inside the horizon; the banded deflated measurement; and the Jacobian difference step.  Section~\\ref{sec:budget} says what each control does and does not bound.}",
            "\\label{tab:lam1}", "\\begin{ruledtabular}",
            "\\begin{tabular}{llrr}",
            "control & value & leading decaying $\\lambda$ & next below" + BS,
            "\\hline"]
    for i, (n, _, d, real, l2, _lr) in enumerate(Nser):
        tag = "Chebyshev $N$" if i == 0 else ""
        nxt = nx(l2) if real else "not $\\lambda_1$"
        rows.append(f"{tag} & {n} & {cell(d, real)} & {nxt}" + BS)
    if good:
        rows.append("\\hline")
        rows.append(f"\\multicolumn{{2}}{{l}}{{$N\\ge176$: mean, spread}} & "
                    f"${np.mean(good):.5f}$ & "
                    f"$\\pm{0.5*(max(good)-min(good)):.4f}$" + BS)
    if Xser:
        rows.append("\\hline")
        for i, (n, X, d, real, l2, _lr) in enumerate(Xser):
            tag = f"$X_{{\\max}}$ at $N={n}$" if i == 0 else ""
            rows.append(f"{tag} & {X:g} & {cell(d, real)} & "
                        + nx(l2) + BS)
        v1 = [t[2].real for t in Xser]; v2 = [t[4].real for t in Xser]
        rows.append(f"\\multicolumn{{2}}{{l}}{{range over $X_{{\\max}}$}} & "
                    f"${max(v1)-min(v1):.5f}$ & ${max(v2)-min(v2):.4f}$" + BS)
    mrow, href = _matched(sc)
    shown = {(r[0], r[1]) for r in mrow}
    if Xser:
        for i, (n, X, d, real, l2, _lr) in enumerate(
                [t for t in Xchk if (t[0], t[1]) not in shown]):
            tag = "same, resolved further" if i == 0 else ""
            rows.append(f"{tag} & $N={n}$, $X_{{\\max}}={X:g}$ & "
                        f"{cell(d, real)} & "
                        + nx(l2) + BS)
    if len(mrow) >= 3:
        rows.append("\\hline")
        for i, (n, X, g, nin, d, real, l2) in enumerate(mrow):
            tag = "$X_{\\max}$ at matched $h$" if i == 0 else ""
            rows.append(f"{tag} & $N={n}$, $X_{{\\max}}={X:g}$, $h={g:.4f}$, "
                        f"$n_{{\\rm in}}={nin}$ & {cell(d, real)} & "
                        + nx(l2) + BS)
        v1 = [r[4].real for r in mrow]; v2 = [r[6].real for r in mrow]
        rows.append(f"\\multicolumn{{2}}{{l}}{{range at matched $h$ "
                    f"($h$ spread ${100*(max(r[2] for r in mrow)/min(r[2] for r in mrow)-1):.1f}\\%$)}} & "
                    f"${max(v1)-min(v1):.5f}$ & ${max(v2)-min(v2):.4f}$" + BS)
    rows.append("\\hline")
    for i, (X, v) in enumerate([(3.0, -0.8845), (4.0, -0.8939), (5.0, -0.905)]):
        tag = "banded, deflated" if i == 0 else ""
        rows.append(f"{tag} & $X_{{\\max}}={X:g}$ & ${v:.4f}$ & ---" + BS)
    rows.append("\\hline")
    for r in _eps_rows():
        rows.append(r)
    rows += ["\\hline",
             "\\multicolumn{2}{l}{adopted} & $-0.895\\pm0.010$ & "
             "none stable in $N$ and $X_{\\max}$" + BS,
             "\\end{tabular}", "\\end{ruledtabular}", "\\end{table*}"]
    w("lam1", "\n".join(rows))

# ----------------------------------------------------------------- ladder
def t_ladder():
    D = DREF
    mg = [(r"$\ell=0$ growing", 9.21, "MG-G Table I"),
          (r"$\ell=2$ even", -0.07, "MG-G Table I"),
          (r"$\ell=1$ even", -0.30, "MG-G Table I"),
          (r"$\ell=3$ even", -1.65, "MG-G Table I"),
          (r"$\ell=2$ odd", -2.30, "MG-G Table I"),
          (r"$\ell=4$ even", -3.0, "MG-G Table I")]
    rows = ["\\begin{table}[!tbp]",
            "\\caption{$\\lambda_1$ placed among the non-spherical exponents of Table~I of Ref.~\\cite{martingarcia1999}, whose tabulated $\\kappa\\Delta$ we convert by $\\lambda=\\kappa\\Delta/\\Delta$.  The $\\ell=0$ entry of that table is the growing mode; no $\\ell=0$ decaying value appears there.}",
            "\\label{tab:ladder}", "\\begin{ruledtabular}",
            "\\begin{tabular}{lrrl}",
            "sector & $\\kappa\\Delta$ & $\\lambda$ & source\\\\", "\\hline"]
    ent = [(n, k, k / D, s) for n, k, s in mg]
    ent.append((r"\textbf{$\ell=0$ decaying}", None, -0.89482, "this work"))
    for n, k, v, s in sorted(ent, key=lambda t: -t[2]):
        kk = f"${k:+.2f}$" if k is not None else "---"
        rows.append(f"{n} & {kk} & ${v:+.3f}$ & {s}\\\\")
    rows += ["\\end{tabular}", "\\end{ruledtabular}", "\\end{table}"]
    w("ladder", "\n".join(rows))


# ------------------------------------------------------------------- lam2
def t_lam2():
    sc = _scan()
    N4 = sorted([t for t in sc if t[1] == 4.0 and t[0] >= 176],
                key=lambda t: t[0])
    from collections import Counter
    cnt = Counter(t[0] for t in sc if t[1] != 4.0)
    Nx0 = 224 if cnt.get(224, 0) >= 3 else (
        max(cnt, key=lambda n: (cnt[n], n)) if cnt else None)
    X0 = sorted([t for t in sc if t[0] == Nx0], key=lambda t: t[1]) \
        if Nx0 else []
    Xw = max((t[1] for t in X0), default=None)
    chk = sorted([t for t in sc if t[1] == Xw and t[0] > Nx0],
                 key=lambda t: t[0])
    if not N4:
        w("lam2", "")
        return
    v2 = np.array([t[4].real for t in N4])
    mrow, _href = _matched(sc)
    mr1 = mr2 = None
    if len(mrow) >= 3:
        a = [r[4].real for r in mrow]; b = [r[6].real for r in mrow]
        mr1, mr2 = max(a) - min(a), max(b) - min(b)
    # Envelope over the whole (N, Xmax) family, which is what we actually
    # quote: no single scan varies one variable only (see t_lam1's caption).
    # Xmax = 2.5 is excluded because it violates the outflow condition
    # Eq. (outflow): its lowest boundary speed over the cycle is -0.0316
    # (out/dss/outflow.npz), inflow at 40 of 818 sampled phases.  Dropping it
    # leaves the envelope, the next-eigenvalue spread and their ratio
    # unchanged; it changes 16 solves over Xmax = 2.5..6 into 15 over 3..6.
    XCRIT = 2.5404
    env = [t for t in sc if t[0] >= 176 and t[3] and t[1] >= XCRIT]
    ea = [t[2].real for t in env]; eb = [t[4].real for t in env]
    e1, e2, ne = max(ea) - min(ea), max(eb) - min(eb), len(env)
    reals = [t[0] for t in N4 if abs(t[4].imag) < 1e-6]
    types = ("a complex pair at every resolution sampled" if not reals else
             "a complex pair at most resolutions and real at "
             + ", ".join(f"$N={n}$" for n in reals))
    body = ["What lies below $\\lambda_1$ is not a second discrete mode.  "
            "At $X_{\\max}=4$ the next eigenvalue below it moves across the "
            "whole range sampled, "
            + ", ".join(f"${t[4].real:.3f}$ at $N={t[0]}$" for t in N4)
            + f", and changes type on the way: {types}.  We have not paired "
            "eigenvalues between resolutions, so ``the next eigenvalue by real "
            "part'' need not name the same branch at two different $N$."]
    if len(X0) > 2:
        r1 = max(t[2].real for t in X0) - min(t[2].real for t in X0)
        r2 = max(t[4].real for t in X0) - min(t[4].real for t in X0)
        seg = [f"over $X_{{\\max}}={min(t[1] for t in X0):g}$ to "
               f"{max(t[1] for t in X0):g} at fixed $N={X0[0][0]}$, "
               f"{sci(r1, 1, False)} against ${r2:.2f}$"]
        if mr2 is not None:
            seg.append(f"at matched $h$, ${mr1:.5f}$ against ${mr2:.2f}$")
        if chk:
            n1, _, w1c, _r, p1c, _lr = chk[-1]
            w0 = [t[2].real for t in X0 if t[1] == Xw][0]
            p0 = [t[4].real for t in X0 if t[1] == Xw][0]
            seg.append(f"raising $N$ from ${Nx0}$ to ${n1}$ at "
                       f"$X_{{\\max}}={Xw:g}$, "
                       f"{sci(abs(w1c.real - w0), 1, False)} against "
                       f"${abs(p1c.real - p0):.2f}$")
        body.append(
            "Every control separates the two by about two orders of "
            "magnitude, the movement in $\\lambda_1$ against the movement in "
            "the branch: " + "; ".join(seg) + ".  Over the "
            f"${ne}$ converged solves of Sec.~\\ref{{sec:budget}}, at which "
            f"$\\lambda_1$ is real, $\\lambda_1$ spans ${e1:.5f}$ and this "
            f"branch ${e2:.2f}$, a factor of ${e2/max(e1,1e-12):.0f}$.")
        body.append(
            f"So $\\lambda_1$ is the least-damped decaying exponent that "
            "survives these controls.  Below it, out to "
            f"$\\mathrm{{Re}}\\,\\lambda\\simeq"
            f"{min(t[4].real for t in N4):.1f}$ at $N$ up to "
            f"${max(t[0] for t in sc)}$ and $X_{{\\max}}$ up to "
            f"${max(t[1] for t in X0):g}$, nothing is stable in both, which "
            "leaves open a physical mode in that interval that these "
            "resolutions do not reach as well as an origin in the truncated "
            "outgoing region.")
    w("lam2", "  ".join(body))


# -------------------------------------------------------------------- ssh
def t_ssh():
    d = np.load(f"{OUT}/ssh_invariants.npz")
    rows = ["\\begin{table}[!tbp]",
            "\\caption{The self-similarity horizon as the periodic orbit of Eq.~\\eqref{eq:ray}, from the $N=800$ banded fixed point.  The full-period multiplier is the invariant statement; $\\mu$ refers to the slow time of Eq.~\\eqref{eq:auto}.  Middle block: the instantaneous zero of the ray speed, a different object, whose maximum over the sampled cycle estimates the outflow threshold $x_{\\rm crit}$ of Eq.~\\eqref{eq:outflow}.  Last block: comparison with the rigorously bounded invariant of Ref.~\\cite{reiterer2019} through Eq.~\\eqref{eq:rt}, the exact entries being their eighty-digit values truncated.}",
            "\\label{tab:ssh}", "\\begin{ruledtabular}", "\\begin{tabular}{lr}",
            "quantity & value\\\\", "\\hline",
            f"$X_h$ at the phase of the phase condition & {float(d['Xh0']):.6f}\\\\",
            f"$\\min_T X_h$ & {float(d['Xh_min']):.6f}\\\\",
            f"$\\max_T X_h$ & {float(d['Xh_max']):.6f}\\\\",
            f"half-period multiplier $\\sigma$ & {float(d['sigma_half']):.6f}\\\\",
            f"full-period multiplier $e^{{\\mu\\Delta}}$ & "
            f"{float(d['mult_full']):.6f}\\\\",
            f"$\\mu=(2/\\Delta)\\ln\\sigma$ & {float(d['mu']):.6f}\\\\",
            f"$1/\\mu$ & {1/float(d['mu']):.6f}\\\\",
            f"$\\langle a^2\\rangle_{{\\rm SSH}}$ & "
            f"{float(d['a2_mean']):.6f}\\\\",
            "\\hline",
            f"range of the instantaneous zero of $A$ & "
            f"$[{float(d['zero_min']):.4f},\\,{float(d['zero_max']):.4f}]$\\\\",
            f"$A$ at $X_h$, same phase & "
            f"${float(d['A_at_Xh0']):+.4f}$\\\\",
            f"outflow threshold $x_{{\\rm crit}}$, estimated & "
            f"{float(d['zero_max']):.6f}\\\\",
            "\\hline",
            f"$\\mu$, this work, $N=200$ & {MU_N[200]:.9f}\\\\",
            f"\\quad $N=400$ & {MU_N[400]:.9f}\\\\",
            f"\\quad $N=800$ & {MU_N[800]:.9f}\\\\",
            f"\\quad Richardson, $N=400,800$ & "
            f"{MU_N[800] + (MU_N[800]-MU_N[400])/15:.9f}\\\\",
            f"$\\mu=4\\pi\\mu_{{\\rm RT}}/\\Delta$, "
            f"exact~\\cite{{reiterer2019}} & {MU_RT_EXACT:.9f}\\\\",
            f"$e^{{\\mu\\Delta}}=e^{{4\\pi\\mu_{{\\rm RT}}}}$, exact & "
            f"{MULT_RT_EXACT:.7f}\\\\",
            "\\end{tabular}", "\\end{ruledtabular}", "\\end{table}"]
    w("ssh", "\n".join(rows))


# ----------------------------------------------------------------- filter
def t_filter():
    """Stability of the banded deflated iteration against the filter cutoff."""
    D = json.load(open(f"{OUT}/filter_scan.json"))
    Ns = sorted(D, key=int)
    ms = sorted({int(m) for r in D.values() for m in r})
    PLAT = {"400": (60, 60), "800": (80, 160)}
    rows = ["\\begin{table}[!tbp]",
            "\\caption{Stability of the banded, deflated measurement of $\\lambda_1$ against the filter cutoff $m$: the number of Chebyshev modes retained after each application of the half-period map, not after each time step.  Entries are the converged rate, averaged over $20$ smooth random starts, at $X_{\\max}=4$; boldface marks the plateau.}",
            "\\label{tab:filter}", "\\begin{ruledtabular}",
            "\\begin{tabular}{lrr}",
            "$m$ & " + " & ".join(f"$N={n}$" for n in Ns) + BS, "\\hline"]
    for m in ms:
        cells = []
        for n in Ns:
            v = D[n].get(str(m))
            if v is None:
                cells.append("---"); continue
            lo, hi = PLAT[n]
            t = f"{v[0]:.3f}"
            cells.append(f"$\\mathbf{{{t}}}$" if lo <= m <= hi else f"${t}$")
        rows.append(f"{m} & " + " & ".join(cells) + BS)
    rows += ["\\hline",
             "plateau & $-0.893$ & $-0.895\\pm0.002$" + BS,
             "\\multicolumn{3}{l}{Chebyshev, same $X_{\\max}$: $-0.89521$}"
             + BS,
             "\\end{tabular}", "\\end{ruledtabular}", "\\end{table}"]
    w("filter", "\n".join(rows))



# ------------------------------------------------------------------ flush
def t_flush():
    d = np.load(f"{OUT}/flush.npz")
    e, T, r = d["eps"], d["Tf"], d["resid"]
    D, amp = float(d["Delta"]), float(d["amp"])
    c0, Pb, ve = float(d["c0"]), float(d["Pbest"]), float(d["varexp"])
    k = np.linspace(0, len(e) - 1, 7).round().astype(int)      # 7 of the 28
    e, T, r = e[k], T[k], r[k]
    rows = ["\\begin{table*}[!t]",
            "\\caption{Flush time for a ray starting a distance $\\epsilon$ inside the self-similarity horizon, from integrating the characteristics of the converged fixed point; seven of the $28$ values computed, spanning $0.2\\ge\\epsilon\\ge10^{-4}$.  The last row is the remainder after subtracting $\\mu^{-1}\\ln(1/\\epsilon)$, with $\\mu$ taken from the Floquet multiplier of the orbit rather than fitted.}",
            "\\label{tab:flush}", "\\begin{ruledtabular}",
            "\\begin{tabular}{l" + "r" * len(e) + "}",
            "$\\epsilon$ & " + " & ".join(sci(x, 0, False) if x < 1e-2
                                        else f"{x:.3g}" for x in e) + "\\\\",
            "\\hline",
            "$T_f$ & " + " & ".join(f"{x:.3f}" for x in T) + "\\\\",
            "$T_f-\\mu^{-1}\\ln(1/\\epsilon)$ & " +
            " & ".join(f"${x:+.3f}$" for x in r) + "\\\\",
            "\\end{tabular}", "\\end{ruledtabular}", "\\end{table*}"]
    w("flush", "\n".join(rows))


# ------------------------------------------------------------------ depth
def t_depth():
    fam = {"G1": "gaussian, $r_0=0.25$", "T1": "tanh, $r_0=0.25$",
           "O1": "odd gaussian, $r_0=0.25$", "G3": "gaussian, $r_0=0.40$"}
    D, allD = {}, []
    for k in fam:
        f = f"{OUT}/seedtest_depth_{k}.json"
        if not os.path.exists(f):
            continue
        D[k] = json.load(open(f))
        for r in D[k]:
            # recomputed here from the stored residual, so that the criterion
            # is the Newton residual alone whatever rule the run recorded
            r["ok_single"] = r["single"][1] < 1e-8
            if r["ok_single"]:
                allD.append(r["single"][0])
    xs = sorted({r["x"] for v in D.values() for r in v})
    rows = ["\\begin{table*}[!t]",
            "\\caption{Depth of bisection required of the seed.  Entries are the antiperiodicity residual $q$ of the best seed found at depth $x$, in decimal digits, with $\\checkmark$ if Newton then converged ($|R|<10^{-8}$) and $\\times$ if it did not.  All solves are at $N=200$, where the converged period is $3.4451566$; acceptance is on the Newton residual alone.}",
            "\\label{tab:depth}", "\\begin{ruledtabular}",
            "\\begin{tabular}{ll" + "c" * len(xs) + "}",
            "family & & " + " & ".join(f"$x={x}$" for x in xs) + "\\\\",
            "\\hline"]
    for k, name in fam.items():
        if k not in D:
            continue
        m = {r["x"]: r for r in D[k]}
        rows.append(f"{k} ({name}) & $q$ & " + " & ".join(
            f"{m[x]['q']:.2f}" if x in m else "---" for x in xs) + "\\\\")
        rows.append(" & Newton & " + " & ".join(
            ("$\\checkmark$" if m[x].get("ok_single") else "$\\times$")
            if x in m else "---" for x in xs) + "\\\\")
    if allD:
        rows += ["\\hline",
                 f"\\multicolumn{{2}}{{l}}{{{len(allD)} converged runs}} & "
                 f"\\multicolumn{{{len(xs)}}}{{l}}{{"
                 f"$\\Delta\\in[{min(allD):.8f},\\,{max(allD):.8f}]$, "
                 "spread " + sci(max(allD)-min(allD), 1, False) + "}\\\\"]
    rows += ["\\end{tabular}", "\\end{ruledtabular}", "\\end{table*}"]
    w("depth", "\n".join(rows))


# -------------------------------------------------------------------- ops
def t_ops(Ns=(50, 100, 200, 400)):
    f = lambda x: np.exp(-0.7 * x) * np.sin(2.3 * x + 0.4)
    fp = lambda x: np.exp(-0.7 * x) * (2.3 * np.cos(2.3 * x + 0.4)
                                       - 0.7 * np.sin(2.3 * x + 0.4))
    def F(x):        # int_0^x f
        a, b = -0.7, 2.3
        return ((a * np.exp(a * x) * np.sin(b * x + .4)
                 - b * np.exp(a * x) * np.cos(b * x + .4)) / (a * a + b * b)
                - (a * np.sin(.4) - b * np.cos(.4)) / (a * a + b * b))
    res = {}
    for wd in (5, 7, 9):
        for N in Ns:
            S = dc.Sim(N=N, Xmax=4.0, wd=wd)
            res.setdefault(("d", wd), []).append(
                np.abs(S.dx(f(S.X)) - fp(S.X)).max())
    for N in Ns:
        S = dc.Sim(N=N, Xmax=4.0, wd=7)
        out = np.empty(N)
        dc._cum(f(S.X), S.qi, S.qw, out)
        res.setdefault(("q", 6), []).append(np.abs(out - F(S.X)).max())
        # The rule the scheme started with, for the comparison the appendix
        # makes: a local parabola on each interval, error O(h^4) per interval.
        # Same nodes, and X = 0 prepended so that both rules integrate from 0.
        Xz = np.concatenate([[0.0], S.X])
        o3 = np.empty(N + 1)
        nc.cumquad(Xz, f(Xz), o3)
        res.setdefault(("q", 3), []).append(np.abs(o3[1:] - F(S.X)).max())
    def order(v):
        o = [np.log2(v[i] / v[i + 1]) for i in range(len(v) - 1)
             if v[i + 1] > 1e-11]
        return o or [np.log2(v[0] / v[1])]
    rows = ["\\begin{table*}[!t]",
            "\\caption{Measured accuracy of the banded operators on $f=e^{-0.7X}\\sin(2.3X+0.4)$ over $[0,4]$: maximum error and observed order under doubling.  The cumulative quadrature is the one that matters, because $\\bar H$ and $g$ are cumulative integrals, and the last two rows are the comparison of Appendix~\\ref{app:ops}: the same nodes and the same integrand under a local parabola on each interval and under the six-point rule used here.}",
            "\\label{tab:ops}", "\\begin{ruledtabular}",
            "\\begin{tabular}{l" + "r" * len(Ns) + "l}",
            "operator & " + " & ".join(f"$N={n}$" for n in Ns) +
            " & order\\\\", "\\hline"]
    for wd in (5, 7, 9):
        v = res[("d", wd)]
        rows.append(f"$\\partial_X$, $w={wd}$ & " +
                    " & ".join(sci(x, 1, False) for x in v) +
                    f" & {np.mean(order(v)):.1f}\\\\")
    for lab, wq in (("cumulative quadrature, local parabola", 3),
                    ("cumulative quadrature, 6-point", 6)):
        v = res[("q", wq)]
        rows.append(f"{lab} & " +
                    " & ".join(sci(x, 1, False) for x in v) +
                    f" & {np.mean(order(v)):.1f}\\\\")
    rows += ["\\end{tabular}", "\\end{ruledtabular}", "\\end{table*}"]
    w("ops", "\n".join(rows))

# --------------------------------------------------------- lam1 direct
def t_direct():
    """Direct-evolution test of lam1 by biorthogonal projection."""
    runs = [(176, 1200), (176, 2400), (224, 2400)]
    D = {}
    for N, n in runs:
        f = f"{OUT}/lam1_direct_N{N}_n{n}.json"
        if os.path.exists(f):
            D[(N, n)] = json.load(open(f))
    best = D[(224, 2400)]
    L0 = best["lattices"][0]
    nu1 = L0["nu1"]
    rows = ["\\begin{table}[!tbp]",
            "\\caption{Direct-evolution test of $\\lambda_1$.  Upper block: the $s=0$ phase lattice of the $N=224$ run, with $E_{\\rm int}$ the interior deviation of Eq.~\\eqref{eq:Eint} and $a_1$ the amplitude from the left eigenvector; successive $a_1$ should stand in the ratio $\\nu_1=0.2141$, which Eq.~\\eqref{eq:ratio} turns into the one-step $\\lambda_1$ of the last column.  Lower block: pooled over four phase lattices for three runs by the admission rule of Sec.~\\ref{sec:direct}, the penultimate line showing the sensitivity to the sign cut.  Every $\\pm$ is a sample standard deviation, not a standard error.}",
            "\\label{tab:direct}", "\\begin{ruledtabular}",
            # six columns do not fit \columnwidth at 10pt
            "\\footnotesize\\setlength{\\tabcolsep}{2.5pt}",
            "\\begin{tabular}{rrrlrr}",
            "$m$ & $T$ & $E_{\\rm int}$ & $|a_1|$ & "
            "ratio & $\\lambda_1$" + BS, "\\hline"]
    R0 = L0["rows"]
    for i, r in enumerate(R0[:7]):
        fin = r["ratio"] == r["ratio"]
        ok = (fin and r["ratio"] > 0 and r["dH"] < 0.1
              and i > 0 and R0[i - 1]["dH"] < 0.1)
        # 3 decimals where the ratio is meant to be read against
        # nu_1 = 0.214; 1 where it has blown up and only the sign matters.
        rat = ("---" if not fin else
               f"${r['ratio']:+.3f}$" if abs(r["ratio"]) < 10 else
               f"${r['ratio']:+.1f}$")
        lam = f"${r['lam1']:+.2f}$" if ok else "---"
        # the deviation is read against a threshold of 0.1, so a plain
        # decimal is both clearer and much narrower than sci notation.
        rows.append(f"{r['m']} & ${r['T']:.2f}$ & ${r['dH']:.3f}$ & " +
                    sci(abs(r["a1"]), 1, False) + f" & {rat} & {lam}" + BS)
    rows.append("\\hline")
    rows.append("\\multicolumn{6}{l}{pooled, $E_{\\rm int}<10^{-1}$ at "
                "both ends and ratio $>0$:}" + BS)
    rows.append("\\multicolumn{2}{l}{run} & \\multicolumn{1}{l}{ratios} & "
                "\\multicolumn{1}{l}{sign fail} & "
                "\\multicolumn{2}{l}{$\\lambda_1$ (direct)}" + BS)
    for (N, n), d in sorted(D.items()):
        v, e, k = d["lam1_direct"]
        rows.append(f"\\multicolumn{{2}}{{l}}{{$N={N}$, $n={n}$}} & {k} & "
                    f"{d['nsign_fail']} & "
                    f"\\multicolumn{{2}}{{l}}{{${v:+.2f}\\pm{e:.2f}$}}" + BS)
    v, e, k = best["lam1_allsign"]
    rows.append("\\multicolumn{2}{l}{sign failures kept, $N=224$} & "
                f"{k} & --- & \\multicolumn{{2}}{{l}}{{${v:+.2f}\\pm{e:.2f}$}}" + BS)
    fp = L0["lam1_fp"]
    rows.append("\\multicolumn{4}{l}{fixed point, this work} & "
                f"\\multicolumn{{2}}{{l}}{{${fp:+.3f}\\pm0.010$}}" + BS)
    rows += ["\\end{tabular}", "\\end{ruledtabular}", "\\end{table}"]
    w("direct", "\n".join(rows))


if __name__ == "__main__":
    os.makedirs(TAB, exist_ok=True)
    import sys
    for k in (sys.argv[1:] or ["conv", "exact", "lam1", "ladder", "lam2",
                               "ssh", "flush", "depth", "ops",
                               "direct", "filter"]):
        try:
            globals()[f"t_{k}"]()
        except Exception as e:
            print(f"!! {k}: {type(e).__name__}: {e}", flush=True)
