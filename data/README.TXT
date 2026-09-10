Supplemental Material for

  "Choptuik critical solution as a fixed point of an evolution map,
   and its l=0 Floquet spectrum"

  Xulei Sun, Department of Physics, Yale University, New Haven, Connecticut 06520, USA
  Email: xulei.sun@yale.edu


Description
-----------
Four plain-text tables giving the discretely self-similar (Choptuik) critical
solution of the spherically symmetric Einstein/massless-scalar system, its
growing Floquet mode, and the least-damped decaying mode with both of its
eigenvectors, at enough resolution to serve as seeds or reference solutions.
The solution is the fixed point of the half-period map S_(Delta/2)[H] + H = 0
of an evolution code, obtained by matrix-free Newton-Krylov with the echoing
period Delta carried as an unknown of the same solve.


Files
-----
choptuik_dss_profile.txt       the fixed point:  X, H, Hbar(=phi), g, gbar
choptuik_dss_growing_mode.txt  the growing mode:  X, dH_0
choptuik_dss_lam1_mode.txt     the lambda_1 mode:  X, H, dH_1
choptuik_dss_lam1_adjoint.txt  its adjoint:  X, dH_1, w_1, q, H

Each file carries a self-contained header giving the field equations, the
normalizations, and the provenance of every number in it.


Variables and units
-------------------
Everything is dimensionless: the field equations are invariant under
r -> k r, u -> k u, and the tabulated quantities are scale invariant.

  T = -ln(u_* - u)          slow time; the solution has period Delta in T
  X = r/(u_* - u)           self-similar radius; the tabulated coordinate
  H(X) = d(r phi)/dr        the evolved state, at one phase of the cycle
  Hbar = (1/X) int_0^X H dX'  = phi exactly
  g    = exp[4 pi int_0^X (H-Hbar)^2 dX'/X'],   gbar = (1/X) int_0^X g dX'
  ds^2 = -g gbar du^2 - 2 g du dr + r^2 dOmega^2

H is antiperiodic over a half period, H(T + Delta/2, X) = -H(T, X), so the
sign of a tabulated profile or mode is fixed only up to the choice of phase.


Grids
-----
The first three files are tabulated on a PIECEWISE uniform grid in X, 151
rows, with step 0.01 on [0, 0.5), 0.025 on [0.5, 2) and 0.05 on [2, 4].

The adjoint file is NOT on that grid.  It is tabulated on the 224
Chebyshev-Gauss-Lobatto collocation nodes of its own solve, because the
adjoint is a discrete functional attached to those nodes and not a smooth
function of X: it alternates at the grid scale and has no pointwise limit
under refinement (Sec. V E of the paper).  It must not be interpolated.
Column q is the Clenshaw-Curtis quadrature weight of each node, and column H
is the background of that same solve on those same nodes, so that a projection
with w_1 needs no interpolation of the background at all.  All five columns are
written at full double precision, and for the same reason: the nodes are 2e-4
apart at the ends of the domain, so even rounding X for readability would move
them enough to fake a signal (see below).  The file reads back bit for bit.


Provenance, and how to combine the files
----------------------------------------
The two groups come from different solves, whose periods differ by
1.26e-6 (relative 3.7e-7), at phases 0.0075 slow-time units apart
-- 0.22% of an echo:

  profile, growing mode      banded finite differences, N = 800
                             (Delta = 3.4454536098)
  lambda_1 mode and adjoint  Chebyshev collocation, N = 224
                             (Delta = 3.4454523478)

Each lambda_1 file carries the background of its own solve, and that
background, not choptuik_dss_profile.txt, is the one to use with it: column 2
of choptuik_dss_lam1_mode.txt for the right eigenvector, column H of
choptuik_dss_lam1_adjoint.txt for the adjoint.  Mixing the groups would
introduce the phase offset as an error.

Do NOT spline the mode file's background onto the collocation nodes to project
with the adjoint, and do not round the adjoint file's coordinates either.  The
projection a_1 = <w_1, dH> carries a ~1e5 normalization factor, and the
smallest signal measured in the paper's direct test is 7.6e-6 in the
normalization of that file.  Against it, the 4.1e-6 interpolation error of the
resampled background projects to a spurious |a_1| = 2.14e-4, some 28 times the
signal; and a background evaluated at coordinates rounded to nine decimals
projects to 1.46e-6, 19% of it.  Used as written, the columns of the adjoint
file give zero identically.  The header of that file states this.


Reading the files
-----------------
All four load directly with, e.g.,

  import numpy as np
  X, H, Hbar, g, gbar = np.loadtxt("choptuik_dss_profile.txt", unpack=True)
  X, v1, w1, q, Hbg   = np.loadtxt("choptuik_dss_lam1_adjoint.txt", unpack=True)

Lines beginning with '#' are comments.  The biorthogonal normalization of the
adjoint file, sum_i w_1[i] dH_1[i] = 1, can be checked directly from its
columns; a reader preferring the dX inner product should use the density
w_1[i]/q[i], which gives the same value of the pairing.
