#!/usr/bin/env python3
"""Figures for benchmarks/imaginary_time/logtime_suite.jl -- logarithmic vs uniform time grids.

Two panels per model:

  LEFT   error vs NUMBER OF STEPS.  This is the axis arXiv:2606.02930's claim lives on -- the
         log grid's whole point is that it needs exponentially fewer steps.
  RIGHT  error vs OPERATOR APPLICATIONS.  The honest cost axis, and it is here because a step
         count alone HIDES the price of a big step: a large dt needs more Krylov vectors, so an
         integrator can win on steps and lose on work. Reporting only the left panel would be
         the same mistake as ranking schemes by wall time.

EVERY TITLE NAMES THE PUBLISHED REFERENCE the errors are measured against, because that is the
thing the suite was rewritten to guarantee. A model whose reference is our own output does not
belong in this file.

Run:  python3 benchmarks/imaginary_time/plot_logtime.py
"""
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "results")
# ⛔ SEVERAL CSVs, NOT ONE. The dissipative model runs from its own driver (`dissipative_xx.jl`,
# which owns the fused d=4 stack) and writes its own file; the cluster also splits the suite across
# an array with one file per task. Reading a single hardcoded path silently omitted whichever model
# was not in it -- and a MISSING model looks exactly like a model that was never run.
DEFAULT_CSVS = [os.path.join(HERE, "..", "results", "logtime_suite.csv"),
                os.path.join(HERE, "..", "results", "logtime_xx_dissip.csv")]

TITLES = {
    "heis_chain": "Heisenberg chain, SU(2) -- imaginary time\n"
                  "log grid WORKS   [ref: Bethe ansatz, exact]",
    "heis_cylinder": "Heisenberg square cylinder, SU(2) -- imaginary time\n"
                     "log grid WORKS (2D)   [ref: Loop QMC, arXiv:1705.03222 Tab.1]",
    "neel_d0": "XXZ Neel quench, Delta=0, U(1) -- real time\n"
               "log grid ALIASES   [ref: free fermion; Barmettler arXiv:0810.4845]",
    "neel_d1": "XXZ Neel quench, Delta=1, U(1) -- real time\n"
               "log grid ALIASES   [ref: Barmettler arXiv:0810.4845, no closed form]",
    "xx_dissip": "Boundary-driven XX chain -- COMPLEX frequencies (Lindblad)\n"
                 "log grid CANNOT work   [ref: exact Lyapunov, arXiv:2104.11479]",
}

# Per-model grid parameters, needed to redraw the Nyquist bound on the step-count axis. The values
# must match the Julia drivers: (t_0, t_max, nyquist_step).
#
# ⚠ THE DISSIPATIVE MODEL'S BOUND USES 4J, NOT 2J. Its observable is QUADRATIC in the modes, so it
# beats pairs of rapidities together and its fastest angular frequency is 2*max|Im mu| ~ 4J --
# twice the fastest single-mode frequency. Using 2J would place the bound at twice the step it
# should be and would draw half the aliasing arms as resolving.
NYQ = {"neel_d0": (0.02, 16.0, math.pi / 2),
       "neel_d1": (0.02, 16.0, math.pi / 2),
       "xx_dissip": (0.02, 20.0, math.pi / 4)}
# grid -> (linestyle, marker); scheme -> colour. Grid is the STRUCTURAL variable, so it gets the
# linestyle and the reader can see grid-vs-grid without consulting the legend colour.
#
# ⛔ `panel` MUST HAVE ITS OWN STYLE. It used to fall through to the default ("-", "o"), which is
# IDENTICAL to `log` -- two different grid constructions drawn as the same series, on a figure
# whose entire subject is the difference between grid constructions.
GRID_STYLE = {"log": ("-", "o"), "uniform": ("--", "s"), "panel": ("-.", "^")}
SCHEME_COLOUR = {"bug": "tab:blue", "bug_rsvd": "tab:blue", "bug_par": "tab:cyan",
                 "bug_exact": "tab:purple", "midpoint": "tab:green", "tdvp2": "tab:red"}

# The published asymptote of the Neel quench is cos(2 J t - pi/4) -- period pi/J. At J = 1 a step
# above pi/2 cannot resolve it at all. This is somebody else's measured frequency, which is what
# makes the aliasing claim quantitative rather than an appeal to our own finer grid.
NYQUIST_STEP = math.pi / 2


def load(paths):
    """Merge every CSV that exists, keyed by model. A missing file is reported, not fatal.

    ⚠ A MISSING FILE IS ANNOUNCED RATHER THAN SKIPPED SILENTLY. The models live in two drivers and
    the cluster splits them further, so "this model has no rows" and "I did not read its file" look
    identical on the figure -- and the second one is a plotting mistake being read as a physics
    result.
    """
    rows = defaultdict(list)
    seen = []
    for path in paths:
        if not os.path.exists(path):
            print(f"  (no CSV at {path} -- skipping)")
            continue
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                rows[r["model"]].append(r)
        seen.append(os.path.basename(path))
    if not rows:
        sys.exit("no rows in any CSV -- run: julia --project=. benchmarks/imaginary_time/logtime_suite.jl")
    print("  read:", ", ".join(seen))
    return rows


def _finite(x):
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def panel(ax, rows, xkey, xlabel):
    """Plot one cost axis. Returns the number of rows DROPPED for a non-finite error.

    A NaN error is not a plotting nuisance to be silenced -- at Delta = 1 there is no closed form,
    so `ref` is deliberately NaN and every error with it. Those rows are real runs and their
    absence must be reported on the figure, not hidden by a filter.
    """
    # ⛔ SYMMETRY IS PART OF THE SERIES KEY. The cylinder now runs under BOTH :SU2 and :U1, and
    # grouping on (scheme, grid) alone would splice two different runs into one line -- joining an
    # SU(2) point to a U(1) point and drawing the segment between them as if it were a trend.
    # ⚠ And they are NOT comparable at equal D anyway: `maxdim` counts MULTIPLETS under :SU2 and
    # STATES under :U1, so the honest comparison is at matched `chi`, not matched D.
    dropped = 0
    for (scheme, grid, sym), pts in sorted(
            _group(rows, lambda r: (r["scheme"], r["grid"], r.get("symmetry", ""))).items()):
        xs, ys = [], []
        for r in sorted(pts, key=lambda r: float(r[xkey])):
            e = _finite(r["err"])
            if e is None:
                dropped += 1
                continue
            xs.append(float(r[xkey]))
            # a zero error would vanish on a log axis; floor it at the double-precision noise
            # level rather than dropping the point silently
            ys.append(max(e, 1e-16))
        if not xs:
            continue
        ls, mk = GRID_STYLE.get(grid, ("-", "o"))
        # symmetry only enters the label when the model actually has more than one, so the
        # single-symmetry figures stay uncluttered
        many_syms = len({r.get("symmetry", "") for r in rows}) > 1
        lab = f"{scheme}, {grid}" + (f", {sym}" if many_syms and sym else "")
        ax.plot(xs, ys, ls, marker=mk, ms=5, alpha=0.55 if sym == "U1" and many_syms else 1.0,
                color=SCHEME_COLOUR.get(scheme, "tab:grey"), label=lab)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel(xlabel)
    ax.grid(alpha=0.3, which="both", lw=0.4)
    return dropped


def _group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def annotate_nyquist(ax, model):
    """For the real-time models, mark which step counts are below the published Nyquist limit.

    The log grid's LAST step is the one that aliases, so the relevant quantity per arm is
    `t_max / ...` -- but the CSV stores only the step count, so the bound is redrawn as the
    smallest `nsteps` whose largest log step still fits under pi/2. That is computed from the same
    grid constructor the Julia side uses, so the two cannot drift apart.
    """
    # ⛔ THE PARAMETERS COME FROM `NYQ`, NOT FROM CONSTANTS BAKED IN HERE. This used to hardcode
    # the Neel grid (0.02, 16.0) and `NYQUIST_STEP = pi/2`, so `NYQ` -- including the dissipative
    # model's `pi/4` entry, which is `4J` rather than `2J` because its observable is QUADRATIC in
    # the modes -- was defined and never read. Drawing the Neel bound on the dissipative panel
    # would have placed it at twice the step it belongs at and shown aliasing arms as resolving.
    if model not in NYQ:
        return
    t0, tmax, nyq_step = NYQ[model]
    ok = None
    for n in range(4, 20001):
        r = (tmax / t0) ** (1.0 / (n - 1))
        ts = [t0 * r ** k for k in range(n)]
        biggest = max([ts[0]] + [ts[i + 1] - ts[i] for i in range(n - 1)])
        if biggest <= nyq_step:
            ok = n
            break
    if ok is None:
        # ⚠ SAY SO ON THE FIGURE. For the dissipative model no reachable `n` resolves the grid, and
        # a silently absent line is indistinguishable from "the bound was never computed".
        ax.annotate("no log grid in range resolves\nthis model's fastest frequency",
                    xy=(0.03, 0.06), xycoords="axes fraction", fontsize=7.5, color="0.25")
        return
    ax.axvline(ok, color="k", ls=":", lw=1.2)
    ax.annotate(f"log grid resolves\nthe fastest mode only for n>={ok}",
                xy=(ok, 0.5), xycoords=("data", "axes fraction"),
                fontsize=7.5, ha="left", va="center",
                xytext=(4, 0), textcoords="offset points")


def main():
    # ⛔ `CSV` NEVER EXISTED. This read `load(CSV)` against a name that is defined nowhere in the
    # file -- a NameError on every invocation, i.e. the script could not have produced a figure
    # since `DEFAULT_CSVS` replaced the single hardcoded path. Extra argv entries are treated as
    # additional CSVs so a cluster array's per-task files can be merged in without editing this.
    paths = sys.argv[1:] or DEFAULT_CSVS
    data = load(paths)
    if not data:
        sys.exit(f"no rows in {paths}")
    for model, rows in sorted(data.items()):
        fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6), sharey=True)
        d0 = panel(axes[0], rows, "nsteps", "number of time steps")
        panel(axes[1], rows, "matvec", "operator applications")
        annotate_nyquist(axes[0], model)
        axes[0].set_ylabel("|error| vs published reference")
        axes[0].set_title(TITLES.get(model, model), fontsize=9.5, loc="left")
        axes[1].set_title("same runs, cost axis", fontsize=9.5, loc="left")
        handles, _ = axes[0].get_legend_handles_labels()
        (axes[1] if not handles else axes[0]).legend(fontsize=8, framealpha=0.9)
        if d0:
            # ⚠ NaN NOW HAS TWO CAUSES AND THE CAPTION MUST NOT NAME ONLY ONE. Besides "no closed
            # form" (Delta = 1), the CYLINDER emits `err = NaN` whenever its bulk fit is not yet
            # linear -- a real run whose distance from the QMC number is simply not available at
            # that set of lengths. Captioning those as "no reference" would misreport why they are
            # missing, and the point of printing the count at all is that an absent series must not
            # look like a series that was never run.
            fig.text(0.01, 0.01,
                     f"{d0} run(s) omitted (err = NaN): no closed-form reference, "
                     f"or a bulk fit not yet in the asymptotic regime",
                     fontsize=7.5, color="0.35")
        fig.tight_layout()
        out = os.path.join(OUTDIR, f"logtime_{model}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
