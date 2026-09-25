# The Choptuik critical solution as a fixed point of an evolution map

Solver, analysis scripts and archived output for

> Xulei Sun, *Choptuik critical solution as a fixed point of an evolution map,
> and its $\ell=0$ Floquet spectrum* (submitted).  arXiv: TBD.

The critical solution is computed as a fixed point of an ordinary null code's
evolution map.  In slow time $T=-\ln(u_*-u)$ the Christodoulou–Garfinkle
system is autonomous, so the map is "integrate for half a period and flip the
sign" on a fixed grid in $X=r/(u_*-u)$, with both boundaries outflow at every
phase.  No data are prescribed at either boundary, no condition is imposed at
the self-similarity horizon, and the horizon need not be a coordinate surface.
Matrix-free Newton–Krylov, with the period an unknown of the same solve, gives
$\Delta=3.445452403(4)$.  Linearizing the same integration at the fixed point
returns the half-period propagator, whose square is the Floquet operator, so
the spectrum comes out of the same computation.

## Layout

    src/paths.py        every path the scripts use is defined here
    src/core/           the characteristic evolution: numba kernel, Run/bisect driver,
                        (u*, Delta) estimator for near-critical evolution data
    src/fixedpoint/     the fixed-point solve, spectrum, horizon, seeds and data tables
    out/fixedpoint/     archived solver output the tables and figures are built from (14 MB)
    data/               the solution and its modes as plain tables, with README.TXT
    paper/              make_tables.py and make_figs.py, which write paper/tables/ and paper/figs/
    run_dss.sh          the commands that produced the archives, one target per solve group

All paths go through `src/paths.py`, so the scripts can be run from any
directory.

## Reproducing the paper's tables and figures

With the requirements below:

    python src/fixedpoint/dss_tables.py   # data/choptuik_dss_*.txt
    python paper/make_tables.py           # paper/tables/*.tex  (Tables I-X)
    python paper/make_figs.py             # paper/figs/*.pdf    (Figs. 1-5)

These read only `out/fixedpoint/` and take well under a minute.  Every number
in the paper's tables and every figure comes from them; nothing is entered by
hand.  Individual items can be selected by name, e.g.
`make_tables.py lam1 lam2` or `make_figs.py spectrum`.  Rerun from this
archive, the tables and the data files are byte-for-byte those of the paper,
and the figures differ only in the PDF creation date.

Three further archives back numbers quoted in the text rather than in a table:
`outflow.npz` (the outflow threshold of Sec. II C, from `dss_outflow.py`),
`sweep_robust.json` (the robustness scan of Sec. VII, from
`dss_sweep.py robust`) and `check.npz` (the comparison with direct evolution
of Sec. IV C, from `dss_check.py`).

## Re-running the solves

`run_dss.sh` records the commands that produced the archives in
`out/fixedpoint/`, one target per solve group, for example

    ./run_dss.sh spec       # the Newton solve and its convergence sweep
    ./run_dss.sh robust     # the robustness scan
    ./run_dss.sh cheb       # Chebyshev spectra, N = 96...160
    ./run_dss.sh chebhi     # N = 176...224
    ./run_dss.sh chebeps2   # finite-difference step control
    ./run_dss.sh chebxmax   # outer-boundary scan
    ./run_dss.sh lam1       # direct long-time evolutions

and the rest of the pipeline is

    python src/fixedpoint/dss_probe.py 10 2400   # wide-domain near-critical probe
    python src/fixedpoint/dss_newton.py N=400 Delta0=3.30
    python src/fixedpoint/dss_spectrum.py        # spectral diagnostics
    python src/fixedpoint/dss_outflow.py         # outflow threshold
    python src/fixedpoint/dss_check.py           # comparison with direct evolution

Set `PYTHON` to the interpreter `run_dss.sh` should use; under Slurm, submit
it from the repository root.  These are the expensive part: they were run as
16-core, 64 GB, up-to-16-hour batch jobs, and the dense Jacobians at the
higher resolutions dominate.  The archives are provided so that the paper can
be checked without repeating them.  Set `OMP_NUM_THREADS=1` when running
several of them side by side.

## The data tables

`data/` holds the converged fixed-point profile $H_*(X)$, the growing mode,
and the $\lambda_1$ mode with both its eigenvectors, at enough resolution to
serve as seeds or reference solutions; they are also the paper's Supplemental
Material.  `data/README.TXT` gives the column definitions, the grids, and
which solve each file came from.  The two groups come from different solves
(banded $N=800$ and Chebyshev $N=224$) a phase $0.0075$ apart in slow time, so
the mode file carries the background of its own solve, and that is the
background to use with it.  The adjoint is a discrete functional attached to
the $N=224$ collocation nodes and must not be interpolated.  It carries that
solve's background on those same nodes, which is what a projection with it
must use, and its five columns are written at full double precision.
`README.TXT` says how to reproduce the normalization and both checks.

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
that accountability is checkable: every table and figure in the paper is
regenerated here from archived solver output.

## License

Code under the MIT License (`LICENSE`); the tables in `data/` and the archives
in `out/` under CC BY 4.0.
