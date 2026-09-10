# The Choptuik critical solution as a fixed point of an evolution map

Solver, analysis scripts and archived output for

> Xulei Sun, *Choptuik critical solution as a fixed point of an evolution map,
> and its $\ell=0$ Floquet spectrum* (submitted).  arXiv: TBD.

The critical solution is computed as a fixed point of an ordinary null code's
evolution map, with no discretely self-similar ansatz: in slow time
$T=-\ln(u_*-u)$ the Christodoulou–Garfinkle system is autonomous, so the map is
"integrate for half a period and flip the sign" on a fixed grid in
$X=r/(u_*-u)$, with both boundaries outflow at every phase.  Matrix-free
Newton–Krylov, with the period an unknown of the same solve, gives
$\Delta=3.445452403(4)$; linearizing the same integration at the fixed point
returns the half-period propagator, whose square is the Floquet operator, so
the spectrum comes out of the same computation.

Everything reported in the paper is regenerated from the archived output in
`out/dss/` by the scripts in `src/`.

## Layout

    src/            solver, spectrum and analysis; the three generators below
    out/dss/        archived solver output the paper is built from (13 MB)
    data/           the solution and its modes as plain tables, with README.TXT
    paper/          where the generators write their tables and figures

## Reproducing the paper's tables and figures

From the repository root, with the requirements below:

    PYTHONPATH=src python src/dss_tables.py    # data/choptuik_dss_*.txt
    PYTHONPATH=src python src/paper_tables.py  # paper/tables/*.tex  (Tables I-X)
    PYTHONPATH=src python src/paper_figs.py    # paper/figs/*.pdf    (Figs. 1-5)

These read only `out/dss/` and take under a minute.  Every number in the
paper's tables and every figure comes from them; nothing is entered by hand.
Individual items can be selected by name, e.g. `paper_tables.py lam1 lam2` or
`paper_figs.py spectrum`.

## Re-running the solves

`run_dss.sh` records the commands that produced each archive in `out/dss/`,
one target per solve group:

    ./run_dss.sh seeds      # seed generation, four families
    ./run_dss.sh spec       # the Newton solve and its convergence sweep
    ./run_dss.sh cheb       # Chebyshev spectra, N = 96...160
    ./run_dss.sh chebhi     # N = 176...224
    ./run_dss.sh chebeps2   # finite-difference step control
    ./run_dss.sh chebxmax   # outer-boundary scan
    ./run_dss.sh lam1       # direct long-time evolutions

Set `PYTHON` to the interpreter to use.  These are the expensive part: they
were run as 16-core, 64 GB, up-to-16-hour batch jobs, and the dense Jacobians
at the higher resolutions dominate.  The archives are provided so that the
paper can be checked without repeating them.

## The data tables

`data/` holds the converged fixed-point profile $H_*(X)$, the growing mode, and
the $\lambda_1$ mode with both its eigenvectors, at enough resolution to serve
as seeds or reference solutions.  `data/README.TXT` gives the column
definitions, the grids, and which solve each file came from — the two groups
come from different solves (banded $N=800$ and Chebyshev $N=224$) a phase
$0.0075$ apart in slow time, so the mode file carries the background of its own
solve and that is the background to use with it.  The adjoint is a discrete
functional attached to the $N=224$ collocation nodes and must not be
interpolated; it carries that solve's background on those same nodes, which is
what a projection with it must use — splining the resampled background onto
them fakes a signal 28 times the smallest one measured in the paper, and even
rounding the node coordinates for readability fakes one 19% of it, so that
file's five columns are written at full double precision and read back bit for
bit.  `README.TXT` says how to reproduce the normalization and both checks.

## Requirements

Python 3.11 with the versions the results were produced under:

    numpy 2.3.5   scipy 1.17.1   matplotlib 3.11.1   numba 0.65.0

`pip install -r requirements.txt`.  `numba` is used for the inner evolution
loops; nothing else is required.

## Citation

If you use this code or the tabulated solution, please cite the paper above.
`CITATION.cff` carries the same information in machine-readable form.

## Use of generative AI

Claude Opus 5 (Anthropic) and GPT-6 (OpenAI), on the author's prompts, wrote
the solver and the analysis, table and figure scripts, drafted the paper,
surveyed the literature, and criticized successive drafts.  The author posed
the problem, chose the formulation and the numerical strategy, decided what to
accept, and is accountable for the content.  This repository exists so that
that accountability is checkable: every number and figure in the paper is
regenerated here from archived solver output.

## License

Code under the MIT License (`LICENSE`); the tables in `data/` and the archives
in `out/dss/` under CC BY 4.0.
