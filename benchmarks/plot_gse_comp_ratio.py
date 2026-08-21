"""How much does the GSE result depend on `comp_ratio`, with and without the seeded eigensolve?

    python3 benchmarks/plot_gse_comp_ratio.py [results/gse_comp_ratio_L*.csv] [outdir]

⛔ THIS FIGURE IS ABOUT ROBUSTNESS, NOT SPEED. The seeded arm pays one extra operator application
per local solve, so it is not expected to win on cost. What it was meant to buy is FREEDOM FROM A
HYPERPARAMETER: `comp_ratio` splits the RSVD probe between complement and isometry, and the right
value is problem-dependent.

⚠ AS MEASURED, THIS IS A NULL RESULT, AND THE FIGURE EXISTS TO SAY SO. `comp_ratio` is inert on
the open Heisenberg chain at L=16 in all three rank regimes scanned:

    maxdim=256 (cap never binds, chi=26)   spread 1.00x per sweep, both arms
    maxdim=12  (cap binds, 38 states)      err 1.189e-10 for all six values, krylov 60
    maxdim=8   (cap binds hard, 22 states) BIT-IDENTICAL err AND krylov (270 / 490)

The parameter is genuinely reaching the code -- at maxdim=256 sweep 3 shows a monotone trend
(0.0/0.25/0.5 -> 9.306e-12, 0.75/1.0 -> 9.313e-12, none -> 9.315e-12), and at maxdim=12 the seeded
arm's krylov count wobbles by 1 across values. It simply changes nothing that matters.

THE LIKELY REASON IS IN THE SOURCE, NOT THE MODEL. `sector_graded_sketch` treats `comp_ratio` as a
TARGET split and then redistributes any shortfall between the halves (`cbe_core.jl:265-281`), so
the probe stays `npre` wide however the split is set, whenever the fused space has the dimensions
for it. That redistribution was added to fix edge blockage -- at `comp_ratio = 1.0, dex = 64` the
probe used to collapse to 0-1 columns near a chain edge and freeze the staircase profile -- and a
side effect is that it largely removes the sensitivity this figure was built to find.

So the honest reading is: on this problem there is no `comp_ratio` dependence for the seeded
eigensolve to remove. That is a statement about THIS model and THIS sketch, not a proof that the
seed is useless in general -- a harder problem, or a sketch without the redistribution, could
still show it. The spreads are printed either way so the outcome cannot be read off the shape of
the lines alone.

⛔ ONE PANEL PER maxdim. The three scans are different experiments and merging them would average
a binding cap with a non-binding one. `maxdim` is a column; older CSVs without it fall back to the
`_chiNN` filename tag, and anything still unresolved is labelled explicitly rather than assumed.
"""

import csv
import glob
import math
import os
import re
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "results", "gse_comp_ratio_L*.csv")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")

STYLE = {
    "rsvd":        ("#1f77b4", "o", "RSVD, start = zero-padded state"),
    "rsvd_seeded": ("#d62728", "D", "RSVD, start = EXPANDED basis"),
}
# `nothing` is not a number; it is the no-projector extreme. Placed to the right of 1.0 and
# labelled, rather than dropped -- it is the setting the tuning is meant to become unnecessary
# against, so leaving it out would remove the most informative point.
NONE_X = 1.25
FLAT = 2.0          # below this spread, a curve is reported as flat rather than as a trend
# ⛔ A RATIO IS THE WRONG INSTRUMENT AT MACHINE PRECISION. The maxdim=256 arms land between
# 5.3e-15 and 2.0e-14, which is a "spread" of 2.8-3.1x made entirely of floating-point noise on a
# fully converged energy -- reporting it as a trend would imply `comp_ratio` matters there, the
# exact opposite of what the run shows. Any arm whose WORST error is still at roundoff is flat by
# definition, whatever the ratio says.
CONVERGED = 1e-12


def maxdim_of(row, path):
    """The cap for this row. Column first; filename tag as the documented fallback."""
    if row.get("maxdim"):
        return int(row["maxdim"])
    m = re.search(r"_chi(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else 256


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append({
                    "arm": r["arm"], "cr": r["comp_ratio"], "sweep": int(r["sweep"]),
                    "err": float(r["err"]), "krylov": int(r["krylov"]),
                    "states": int(r["maxbond_states"]), "maxdim": maxdim_of(r, p),
                })
    return rows


def xof(cr):
    return NONE_X if cr == "none" else float(cr)


def main():
    paths = sorted(glob.glob(PAT)) if any(c in PAT for c in "*?[") else (
        [PAT] if os.path.exists(PAT) else [])
    if not paths:
        sys.exit(f"no CSV matching {PAT} -- run benchmarks/gse_comp_ratio_scan.jl first")
    rows = load(paths)
    if not rows:
        sys.exit("no rows")

    caps = sorted({r["maxdim"] for r in rows}, reverse=True)
    print(f"{len(rows)} rows, maxdim panels: {caps}")
    fig, axes = plt.subplots(len(caps), 2, figsize=(11.5, 4.3 * len(caps)), squeeze=False)
    summary = []

    for ri, cap in enumerate(caps):
        axL, axR = axes[ri][0], axes[ri][1]
        sub = [r for r in rows if r["maxdim"] == cap]
        states = max((r["states"] for r in sub), default=0)
        for arm in sorted({r["arm"] for r in sub}):
            c, m, lab = STYLE.get(arm, ("k", "x", arm))
            pts = []
            for cr in sorted({r["cr"] for r in sub if r["arm"] == arm}, key=xof):
                g = [r for r in sub if r["arm"] == arm and r["cr"] == cr]
                if not g:
                    continue
                last = max(g, key=lambda r: r["sweep"])
                # `krylov` is PER-SWEEP (see plot_gse_seed.py), so the cost of reaching this error
                # is the SUM over sweeps, not the last sweep's reading -- which is the smallest
                # one, since converged solves exit early.
                tot = sum(r["krylov"] for r in g)
                if math.isfinite(last["err"]) and last["err"] > 0:
                    pts.append((xof(cr), last["err"], tot, cr))
            if not pts:
                continue
            errs = [p[1] for p in pts]
            spread = max(errs) / max(min(errs), 1e-300)
            atroundoff = max(errs) < CONVERGED
            summary.append((cap, arm, min(errs), max(errs), spread, atroundoff))
            if atroundoff:
                tag = "FLAT (all at roundoff)"
            elif spread < FLAT:
                tag = "FLAT"
            else:
                tag = f"spread {spread:.1f}×"
            axL.semilogy([p[0] for p in pts], [p[1] for p in pts], color=c, marker=m, ms=7,
                         lw=1.8, markerfacecolor="none", label=f"{lab}  ({tag})")
            axR.plot([p[0] for p in pts], [p[2] for p in pts], color=c, marker=m, ms=7,
                     lw=1.8, markerfacecolor="none", label=lab)

        binds = "cap BINDS" if cap < 256 else "cap never binds"
        for ax in (axL, axR):
            ax.set_xlabel("comp_ratio")
            ax.set_xticks([0.0, 0.25, 0.5, 0.75, 1.0, NONE_X])
            ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1", "none"])
            ax.grid(alpha=0.3, lw=0.5)
            ax.legend(fontsize=8)
        axL.set_ylabel(r"final $|E - E_{\rm Bethe}|$")
        axL.set_title(f"maxdim={cap} ({binds}, {states} states): accuracy vs the hyperparameter",
                      fontsize=9)
        axR.set_ylabel("total operator applications (all sweeps)")
        axR.set_title(f"maxdim={cap}: what the seed costs", fontsize=9)

    fig.tight_layout()
    out = os.path.join(OUTDIR, "gse_comp_ratio.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)
    for cap, arm, lo, hi, sp, ro in summary:
        note = "  <- at roundoff, ratio is meaningless" if ro else ""
        print(f"  maxdim={cap:<4d} {arm:14s} err {lo:.3e} .. {hi:.3e}   spread {sp:.2f}x{note}")
    flat = [s for s in summary if s[5] or s[4] < FLAT]
    if len(flat) == len(summary):
        # ASCII ONLY IN CONSOLE OUTPUT. The Windows console is cp1252 and `print` of a non-ASCII
        # character raises UnicodeEncodeError -- which killed this script AFTER it had written the
        # figure, so the run looked like it produced a plot and silently lost the one line that
        # says the plot is a null result. Warnings must not be the thing that crashes.
        print(f"WARNING: EVERY arm in EVERY rank regime is flat (<{FLAT}x). `comp_ratio` does not "
              "discriminate on this problem, so this measurement cannot support or refute the "
              "seeded scheme -- see the module docstring for why the sketch's shortfall "
              "redistribution is the likely cause.")


if __name__ == "__main__":
    main()
