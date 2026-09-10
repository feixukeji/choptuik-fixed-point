"""
Write the DSS solution and its modes as plain tables others can use.

Four files, each with a fixed column count so that numpy.loadtxt works:

  data/choptuik_dss_profile.txt       X, H, Hbar(=phi), g, gbar
  data/choptuik_dss_growing_mode.txt  X, dH_0        (lam_0 = 1/gamma)
  data/choptuik_dss_lam1_mode.txt     X, H, dH_1   (lam_1 right eigenvector,
                                      with its own background H)
  data/choptuik_dss_lam1_adjoint.txt  X, dH_1, w_1, q, H (lam_1 LEFT
                                      eigenvector, on the Chebyshev nodes -- it
                                      is a discrete functional, not a smooth
                                      function; see the header of that file.
                                      H is that solve's background ON THOSE
                                      NODES, so that the projection needs no
                                      interpolation)

The solution table comes from the finite-difference fixed point (N = 800),
whose spectrum resolves lam_0 and the three trivial modes.  lam_1 is not
resolvable there (docs/09 section 4.4), so its eigenvector comes from the
Chebyshev fixed point at N = 224, which sits at a slightly different phase
along the DSS orbit -- the two phase conditions use different inner products.
The offset is 0.0075 slow-time units (0.22% of an echo); after removing it the
two profiles agree pointwise to 1.2e-5.  The lam_1 file therefore carries its
own background H so that mode and background are consistent.
"""
import io
import numpy as np
from scipy.interpolate import CubicSpline
import scipy.linalg as sla
import dss_core as dc, dss_cheb as cb


def _cc_weights(n, Xmax):
    """Clenshaw-Curtis weights on the ascending CGL grid of dss_cheb._cheb.

    The quadrature that makes sum_i q_i f_i approximate int_0^Xmax f dX, so
    that "L2 weight" in the paper has a stated measure and a reader can move
    between the Euclidean pairing and the dX one.
    """
    N = n - 1
    k = np.arange(n)
    j = np.arange(1, N // 2 + 1)
    b = np.where(2 * j == N, 1.0, 2.0)
    S = (b / (4.0 * j * j - 1.0)) @ np.cos(2.0 * np.outer(j, k) * np.pi / N)
    c = np.where(k % N == 0, 0.5, 1.0)
    return 2.0 * c / N * (1.0 - S) * (0.5 * Xmax)


XT = np.concatenate([np.arange(0.0, 0.5, 0.01),
                     np.arange(0.5, 2.0, 0.025),
                     np.arange(2.0, 4.001, 0.05)])
COMMON = """# Discretely self-similar (Choptuik) solution of the Einstein / massless-scalar
# system in spherical symmetry, from the fixed-point equation
#     S_(Delta/2)[H] + H = 0
# solved by black-box Newton-Krylov with no tuning of initial data
# (src/dss_newton.py; see docs/09_dss_fixed_point.md).
#
# Grid:         X is PIECEWISE uniform in the first three files, with step
#               0.01 on [0, 0.5), 0.025 on [0.5, 2) and 0.05 on [2, 4];
#               151 rows.  The adjoint file uses the 224 Chebyshev
#               collocation nodes of its own solve instead, at full double
#               precision, for the reason given in its header (see below).
#
# Coordinates:  T = -ln(u* - u),  X = r/(u* - u)
# State:        H(X) = h = d(r phi)/dr        (dimensionless, scale invariant)
# Constraints:  Hbar = (1/X) int_0^X H dX  ( = phi exactly )
#               ln g = 4 pi int_0^X (H-Hbar)^2 dX/X
#               gbar = (1/X) int_0^X g dX
# Evolution:    H_T = -(X - gbar/2) H_X + (g-gbar)(H-Hbar)/(2X)
# Antiperiodic: H(T + Delta/2, X) = -H(T, X)
#
# Delta   = 3.445452403(4)  value adopted in the paper, from the Chebyshev
#                          sequence; the bracket is that sequence's own
#                          scatter over resolution, not an error bound
#                          Martin-Garcia & Gundlach 2003: 3.445452402(3)
#           Each file also carries the Delta of the solve it came from, with
#           that solve, below:
#             profile, growing mode      finite differences, N = 800
#             lam_1 mode, lam_1 adjoint  Chebyshev, N = 224
# gamma   = 0.3739608      = 1/lam_0, from lam_0 = 2.674077202
# lam_0   = +2.6740772     the only growing mode in this sector
# lam_g   = +1.0000000     gauge mode u* -> u* + d; exactly 1
# lam     =  0 (twice)     phase (T -> T + a) and constant (H -> H + c)
# lam_1   = -0.8948202     least-damped decaying mode; first determination
# mu      =  0.6139        transverse Floquet exponent of the self-similarity
#                          horizon, treated as the repelling periodic orbit of
#                          the ray equation.  A RAY starting a distance eps
#                          inside it reaches the centre at T = (1/mu) ln(1/eps)
#                          + C(T) + o(1) as eps -> 0, with C of period
#                          Delta/2.  Only the log is fixed by mu: at arrival
#                          the separation is O(1), so the linearization does
#                          not reach the endpoint and C also carries the
#                          nonlinear transit to the centre.  Measured over 7.2
#                          half-periods the remainder oscillates between 2.19
#                          and 2.43 without settling; one harmonic of period
#                          Delta/2 accounts for 62% of the variance.  This is a
#                          statement about characteristics only: the size of a
#                          field perturbation inside the horizon is set by the
#                          Floquet spectrum, and the growing, gauge and neutral
#                          directions do not decay (Sec. VI D of the paper).
# X_h(0)  = 1.352471       horizon periodic orbit at the reference phase
# max X_h = 1.560820       its maximum over the cycle
#
# Parity under Delta/2 (from the sign of the half-period multiplier nu):
#   lam_0, lam_g, phase  are ANTIperiodic -> odd harmonics only (like phi)
#   constant, lam_1      are  periodic    -> even harmonics only
#
# Phase is an exact lam = 0 freedom, so every T-translate is equally a solution.
"""


def main():
    f = np.load("out/dss/spec_N800_X4_w7.npz")
    X, H = f["X"], f["H"]
    S = dc.Sim(N=800, Xmax=4.0, wd=7)
    hb, g, gb = S.fields(H)
    sp = {k: CubicSpline(X, v) for k, v in
          (("H", H), ("hb", hb), ("g", g), ("gb", gb))}
    V = {k: sp[k](XT) for k in sp}
    phi0 = float(np.polyval(np.polyfit(X[:6], hb[:6], 5), 0.0))
    # X = 0 is exact: h(0) = hbar(0) = phi(0), g(0) = gbar(0) = 1
    V["H"][0] = V["hb"][0] = phi0
    V["g"][0] = V["gb"][0] = 1.0
    hdr = (COMMON + f"# phi(0) = {phi0:+.8f} at this phase\n"
           "#\n# Finite-difference fixed point, N = 800 on 0 < X <= 4, 7-point\n"
           "# differences, 6-point cumulative quadrature, Newton residual\n"
           f"# 4.4e-12; this solve's own period is Delta = {float(f['Delta']):.10f}.\n"
           "#\n#      X            H              Hbar(=phi)        g              gbar\n")
    with open("data/choptuik_dss_profile.txt", "w") as fh:
        fh.write(hdr)
        for row in zip(XT, V["H"], V["hb"], V["g"], V["gb"]):
            fh.write("".join(f"{c:15.9f}" for c in row) + "\n")

    # growing mode, same fixed point
    nu, W = np.linalg.eig(f["J"])
    k = int(np.argmax(np.abs(nu)))
    v = W[:, k].real
    if v[int(np.argmax(np.abs(v)))] < 0:
        v = -v
    v /= np.abs(v).max()
    Dfd = float(f["Delta"])
    with open("data/choptuik_dss_growing_mode.txt", "w") as fh:
        fh.write(COMMON + f"""#
# Growing-mode eigenvector of the SAME (finite-difference, N = 800) fixed
# point, whose own period is Delta = {Dfd:.10f}: normalized to peak 1, with
# the sign fixed by requiring the peak positive.
# nu_0 = {nu[k].real:.6f} < 0: ANTIperiodic under Delta/2, odd harmonics only.
# 99.7% of its |v|^2 dX weight lies inside the self-similarity horizon.
#
#      X            dH_0
""")
        for x, y in zip(XT, CubicSpline(X, v)(XT)):
            fh.write(f"{x:15.9f}{y:15.9f}\n")

    # lam_1 mode, Chebyshev fixed point.  Two files: the right eigenvector is a
    # smooth function of X and is written on the piecewise-uniform grid like
    # everything else; the left eigenvector is NOT -- it alternates at the grid scale and
    # has no pointwise continuum limit -- so it is written on its own
    # collocation nodes, where the biorthogonal pairing is a plain dot product.
    fc = np.load("out/dss/cheb_N224_X4.npz")
    Xc, Hc, Dc = fc["X"], fc["H"], float(fc["Delta"])
    nu2, VL2, VR2 = sla.eig(fc["J"], left=True, right=True)
    lam2 = np.log(nu2.astype(complex) ** 2) / Dc
    o = np.argsort(-lam2.real)
    k2 = next(j for j in o if lam2[j].real < -0.1)
    v2, w2 = VR2[:, k2].real, VL2[:, k2].real
    if v2[-1] < 0:
        v2, w2 = -v2, -w2
    v2 = v2 / np.abs(v2).max()
    w2 = w2 / (w2 @ v2)                 # biorthogonal: sum_i w_i v_i = 1
    res = np.linalg.norm(fc["J"] @ v2 - nu2[k2].real * v2) / np.linalg.norm(v2)
    cos = float(w2 @ v2 / (np.linalg.norm(w2) * np.linalg.norm(v2)))
    LAM1 = f"""#
# Least-damped decaying mode, from the CHEBYSHEV fixed point at N = 224,
# Xmax = 4 (Delta = {Dc:.10f}).  lam_1 = {lam2[k2].real:.7f},
# nu_1 = {nu2[k2].real:+.7f} > 0 -> PERIODIC under Delta/2, EVEN harmonics.
# Eigen-residual {res:.1e}; eigenvector smooth (1e-13 of its Chebyshev energy
# above mode 2N/3).  N = 208 and N = 224 agree to 6 digits.
#
# On the outer boundary: at FIXED N a scan in Xmax moves the boundary and also
# coarsens the grid near the horizon, so it does not bound the sensitivity to
# either separately -- and at N = 224 the joint spread over Xmax = 3..6
# (5.9e-4) is in fact SMALLER than the fixed-domain resolution spread
# (2.4e-3), i.e. the two effects partly cancel.  Matching the near-horizon
# node spacing h (the gap bracketing X_h = 1.3525) by raising N with Xmax
# comes closer to a pure domain scan:
#     (N=176, Xmax=3, h=0.02682, 83 nodes inside)  lam_1 = -0.893643
#     (N=224, Xmax=4, h=0.02671, 89 nodes inside)  lam_1 = -0.894820
#     (N=262, Xmax=5, h=0.02667, 91 nodes inside)  lam_1 = -0.895916
#     (N=298, Xmax=6, h=0.02651, 94 nodes inside)  lam_1 = -0.898995
# but not to a pure one: the interior node count still grows, and on a global
# collocation grid every node is fixed by (N, Xmax), so no control moves the
# boundary alone.  So the honest number is the ENVELOPE over the whole
# two-parameter family: over the 15 converged solves with N >= 176 whose
# leading decaying eigenvalue is real and whose outer boundary satisfies the
# outflow condition (Xmax = 3..6, N = 176..298), lam_1 is
# in [-0.8990, -0.8936], a spread of 5.4e-3 -- half the quoted +-0.010 -- while
# the next eigenvalue below spans 0.59 over the same solves, a factor of 110.
# Table III of the paper carries this; paper_tables._matched() regenerates it.
#
# This fixed point sits 0.0075 slow-time units (0.22% of an echo) from the
# phase of choptuik_dss_profile.txt.
"""
    with open("data/choptuik_dss_lam1_mode.txt", "w") as fh:
        fh.write(COMMON + LAM1 + """#
# Its own background H is therefore given here alongside the mode, so that mode
# and background are consistent.  It is RESAMPLED onto the piecewise-uniform
# grid; for a projection with the adjoint use the background column of
# choptuik_dss_lam1_adjoint.txt instead, which is on the collocation nodes.
#
# dH_1 is the RIGHT eigenvector, J v = nu_1 v: the response.  Normalised to
# unit maximum; sign fixed by requiring dH_1 > 0 at the outer edge.  Only 26%
# of its |v|^2 dX weight lies inside max_T X_h = 1.5608 (the unweighted node
# sum would say 18%; the dX measure is the one quoted).  The LEFT eigenvector
# -- the receptivity, and the instrument that reads a_1 out of evolution data
# -- is in choptuik_dss_lam1_adjoint.txt; it is not tabulated here because it
# is not a smooth function of X and must not be interpolated.
#
#      X            H              dH_1
""")
        sh = CubicSpline(Xc, Hc)(XT)
        sv = CubicSpline(Xc, v2)(XT)
        for row in zip(XT, sh, sv):
            fh.write("".join(f"{c:15.9f}" for c in row) + "\n")

    # What a reader of the deposit can reproduce.  The mode file carries the
    # background resampled onto XT and rounded to 9 decimals; splining that
    # back onto the collocation nodes and projecting must not fake a signal.
    asig = 6.5e-5 / np.linalg.norm(v2)      # last usable phase, direct test
    Hspl = CubicSpline(XT, np.round(sh, 9))(Xc)
    einterp = float(np.abs(Hspl - Hc).max())
    afalse = float(abs(w2 @ (Hspl - Hc)))
    ratfalse = afalse / asig

    # The node coordinates need the same care as the background.  The nodes are
    # 2e-4 apart at the ends, so rounding X for readability displaces them by
    # 5e-10 -- and evaluating the background at the displaced coordinates fakes
    # a signal too.  Hence the full-precision body below.
    dxr = 9
    Hxr = CubicSpline(Xc, Hc)(np.round(Xc, dxr))
    axr = float(abs(w2 @ (Hxr - Hc)))
    pctxr = 100.0 * axr / asig

    # Write the body first and read it back, so that the checks quoted in the
    # header are checks on what the file actually contains, not on the arrays
    # in memory.
    qw = _cc_weights(len(Xc), float(fc["Xmax"]))
    body = "".join("".join(f"{c:26.17e}" for c in row) + "\n"
                   for row in zip(Xc, v2, w2, qw, Hc))
    Xf, vf, wf, _, Hf = np.loadtxt(io.StringIO(body), unpack=True)
    azero = float(abs(wf @ (Hf - Hc)))
    exf = float(np.abs(Xf - Xc).max())
    anorm = float(abs(wf @ vf - 1.0))
    assert exf == 0.0 and azero == 0.0, "the body no longer round-trips"

    with open("data/choptuik_dss_lam1_adjoint.txt", "w") as fh:
        fh.write(COMMON + LAM1 + f"""#
# LEFT eigenvector w_1 of the half-period map, w J = nu_1 w: the receptivity.
# It is the instrument of the biorthogonal projection a_1 = <w_1, dH> used in
# the paper, and it is normalized here so that
#
#       sum_i w_1[i] * dH_1[i] = 1                (plain dot product)
#
# over the rows of THIS file, to {anorm:.0e}.  Read the following before using it.
#
# w_1 is NOT a sampled smooth function and must not be splined, resampled or
# plotted as a curve.  It alternates in sign at essentially every node: about
# 90% of its discrete cosine spectrum lies in the upper third of the band,
# against 3e-10 for dH_1, and its envelope does not converge pointwise under
# refinement (N = 224 ... 288 differ by O(1) pointwise).  That is the extreme
# nonnormality of this Jacobian stated sharply: w_1 and dH_1 are very nearly
# orthogonal, cos = {cos:.2e}, so the normalization above carries a factor
# ~1e5.  It is a discrete functional attached to THESE collocation nodes and to
# this discretization, not a continuum adjoint eigenfunction.
#
# Neither does the condition number: kappa_1 = |w||v|/|<w,v>| GROWS as N^2.8,
# from 2.9e4 at N = 176 to 1.1e5 at N = 288.  And <w_1,v_1> = 1 is imposed at
# every resolution, so it is a normalization and not a convergence test.
#
# What IS stable under refinement is the SPECTRAL PROJECTOR
#
#       P_1 = v_1 w_1^T / <w_1, v_1>
#
# which is invariant under v -> a v and w -> b w separately and so carries no
# normalization convention.  Applied to smooth perturbations defined as
# functions of X (unit gaussians of width 0.25 at X = 0.5, 1.0, 1.35, 2.0) and
# sampled on each grid, ||P_1 f||_{{L2(dX)}} agrees to 4.1%, 2.6%, 2.6% and 6.2%
# over N = 176 ... 288, with no drift; ratios between centres are stable to
# 0.4%.  For centres in the exterior (2.5, 3.0, 3.5) it is NOT stable, but it
# is four to six orders of magnitude smaller, i.e. at the noise floor.
#
# Integrated weights are stable too: |w_1| peaks at X = 1.35 +- 0.04 over
# N = 176 ... 288 and 96% to 99.99% of its weight lies inside max_T X_h =
# 1.5608 (lowest at any resolution examined: 95.9%).  That is localization,
# not strict support.
#
# X here are the Chebyshev collocation nodes of the N = 224 solve, NOT the
# piecewise-uniform grid used by the other three files.  dH_1 is repeated on these nodes
# so that the normalization above can be checked and reproduced.  q is the
# Clenshaw-Curtis quadrature weight of each node: the pairing above is the
# plain Euclidean one, and a reader preferring the dX inner product should use
# the density w_1[i]/q[i], which gives the SAME value of the pairing.  The
# "weight inside X_h" quoted above is the dX measure, sum q |w|^2; for w_1 the
# unweighted node sum differs by under half a point, though for dH_1 it would
# give 18% rather than 26%.
#
# H is the background of THIS solve on THESE nodes.  Use it, and not the
# resampled H of choptuik_dss_lam1_mode.txt, whenever w_1 is applied to data:
# a_1 = <w_1, dH> carries the ~1e5 normalization factor above, so splining the
# 151-point background onto these nodes leaves an interpolation error of
# {einterp:.1e} that projects to a spurious |a_1| = {afalse:.2e} --
# {ratfalse:.0f} times the smallest signal of the paper's direct test, which is
# {asig:.2e} in the normalization of this file.
#
# The node coordinates need the same care, which is why every column below is
# written at full double precision instead of being rounded for reading.  The
# nodes are 2e-4 apart at the ends, so rounding X to {dxr} decimals moves them by
# 5e-10 -- and a background evaluated at the moved coordinates projects to
# {axr:.2e}, {pctxr:.0f}% of that same signal, by cubic spline, barycentric
# interpolation or Chebyshev resummation alike.  As written all five columns
# read back bit for bit -- max |X - X_node| = {exf:.1e} -- so the
# zero-perturbation check performed on the columns as read gives {azero:.1e}:
# identically zero, not a small residual.
#
#       X                          dH_1                      w_1                       q                         H
""")
        fh.write(body)
    print("wrote 4 tables")


if __name__ == "__main__":
    main()
