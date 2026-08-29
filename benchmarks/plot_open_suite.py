#!/usr/bin/env python3
"""Plots for the open-system suite: XX transport, Heisenberg square, Schwinger chain.

Reads the per-task CSVs written by the three drivers (one file per (model, gamma, scheme), which
is how the sbatch array keeps schemes from timing each other) and produces one figure per model.

    python3 benchmarks/plot_open_suite.py [results_dir]

WHAT EACH PANEL IS FOR, because a plot that is not tied to a claim is decoration:

  * XX          transported magnetisation on LOG-LOG axes. The claim is a CHANGE OF EXPONENT --
                ~t at gamma = 0 (ballistic), ~sqrt(t) at gamma > 0 (diffusive), Znidaric NJP 12
                043001 -- and an exponent is a SLOPE, so the axes have to be log-log or the
                reader is asked to eyeball a power law on linear axes. Reference slopes are drawn.
  * square      staggered magnetisation decay and operator entanglement.
  * schwinger   condensate relaxation; the claim (arXiv:2501.13675) is that the relaxation time
                RISES with gamma, eps0 and mu.
  * all         `Tr rho` against time. ⚠ THIS IS NOT A DECORATION PANEL: `|rho>>` is not
                normalised, so `Tr rho = 1` is the conservation law standing in for the norm. A
                drift there invalidates every other curve in the figure, so it is plotted rather
                than asserted once and forgotten.

⚠ COST HAS TWO AXES AND THEY DISAGREE. Measured on an 8x8 cylinder earlier in this study,
`cbe_bug` used 19x FEWER operator applications and took 1.91x LONGER: `krylov` cannot see the CBE
expansion's SVD/QR work. Both are plotted; quoting either alone is a misreport.

⚠ `krylov_cum` IS ALREADY CUMULATIVE in these CSVs (the drivers accumulate a per-sweep counter).
The raw `get_krylov_log()` is PER SWEEP and non-monotonic -- reading that as a total is a trap
this repo has hit before.
"""
import sys
import csv
import math
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

RESULTS = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")


def load(prefix):
    """Every CSV whose basename starts with `prefix`, grouped by (gamma, scheme)."""
    rows = defaultdict(list)
    if not os.path.isdir(RESULTS):
        return rows
    for fn in sorted(os.listdir(RESULTS)):
        if not (fn.startswith(prefix) and fn.endswith(".csv")):
            continue
        with open(os.path.join(RESULTS, fn), newline="") as fh:      # read universal
            for r in csv.DictReader(fh):
                rows[(float(r["gamma"]), r["scheme"])].append(r)
    return rows


def _style(ax, xlabel, ylabel, title):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, fontsize=10)
    ax.grid(alpha=0.3, linewidth=0.5)


def trace_panel(ax, rows):
    for (g, s), rs in sorted(rows.items()):
        t = [float(r["t"]) for r in rs]
        tr = [float(r["trace"]) for r in rs]
        ax.plot(t, tr, lw=1, label=f"g={g:g} {s}")
    ax.axhline(1.0, color="k", ls=":", lw=1)
    _style(ax, "t", r"$\mathrm{Tr}\,\rho$", "trace (conservation law, NOT the norm)")


def cost_panel(ax, rows):
    """Both cost axes on one panel: operator applications against wall clock."""
    for (g, s), rs in sorted(rows.items()):
        k = [int(r["krylov_cum"]) for r in rs]
        e = [float(r["elapsed"]) for r in rs]
        ax.plot(k, e, lw=1, marker=".", ms=3, label=f"g={g:g} {s}")
    _style(ax, "operator applications (cumulative)", "wall clock [s]",
           "the two cost axes disagree; quote both")


def plot_xx():
    rows = load("xx_open_")
    if not rows:
        return
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
    for (g, s), rs in sorted(rows.items()):
        t = [float(r["t"]) for r in rs]
        dm = [float(r["dM"]) for r in rs]
        pts = [(a, b) for a, b in zip(t, dm) if a > 0 and b > 0]
        if pts:
            axes[0].plot([p[0] for p in pts], [p[1] for p in pts], lw=1.2,
                         label=f"$\\gamma$={g:g} {s}")
    axes[0].set_xscale("log"); axes[0].set_yscale("log")
    # reference slopes: an exponent is a slope, so it is drawn, not described
    tt = [0.3, 8.0]
    axes[0].plot(tt, [0.25 * x for x in tt], "k--", lw=1, label=r"$\propto t$ (ballistic)")
    axes[0].plot(tt, [0.25 * math.sqrt(x) for x in tt], "k:", lw=1,
                 label=r"$\propto\sqrt{t}$ (diffusive)")
    _style(axes[0], "t", r"transported $\Delta M$", "XX under dephasing: exponent change")
    axes[0].legend(fontsize=6)
    for (g, s), rs in sorted(rows.items()):
        axes[1].plot([float(r["t"]) for r in rs], [float(r["S_op"]) for r in rs], lw=1,
                     label=f"g={g:g} {s}")
    _style(axes[1], "t", r"$S_{\rm op}$", "operator entanglement (central cut)")
    axes[1].legend(fontsize=6)
    trace_panel(axes[2], rows)
    fig.tight_layout()
    out = os.path.join(RESULTS, "open_xx.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def plot_square():
    rows = load("heis_square_open_")
    if not rows:
        return
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for (g, s), rs in sorted(rows.items()):
        t = [float(r["t"]) for r in rs]
        axes[0].plot(t, [float(r["Mstag"]) for r in rs], lw=1, label=f"g={g:g} {s}")
        axes[1].plot(t, [float(r["S_op"]) for r in rs], lw=1, label=f"g={g:g} {s}")
    _style(axes[0], "t", r"$M_{\rm stag}$", "Heisenberg 5x5: Neel decay under dephasing")
    axes[0].legend(fontsize=6)
    _style(axes[1], "t", r"$S_{\rm op}$", "operator entanglement")
    trace_panel(axes[2], rows)
    cost_panel(axes[3], rows)
    axes[3].legend(fontsize=6)
    fig.tight_layout()
    out = os.path.join(RESULTS, "open_square.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


def plot_lgt():
    rows = load("lgt_open_")
    if not rows:
        return
    fig, axes = plt.subplots(1, 4, figsize=(19, 4.2))
    for (g, s), rs in sorted(rows.items()):
        t = [float(r["t"]) for r in rs]
        axes[0].plot(t, [float(r["condensate"]) for r in rs], lw=1, label=f"g={g:g} {s}")
        axes[1].plot(t, [float(r["S_op"]) for r in rs], lw=1, label=f"g={g:g} {s}")
    _style(axes[0], "t", "chiral condensate", "Schwinger chain: relaxation vs dissipation")
    axes[0].legend(fontsize=6)
    _style(axes[1], "t", r"$S_{\rm op}$", "operator entanglement")
    trace_panel(axes[2], rows)
    cost_panel(axes[3], rows)
    axes[3].legend(fontsize=6)
    fig.tight_layout()
    out = os.path.join(RESULTS, "open_lgt.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    plot_xx()
    plot_square()
    plot_lgt()
    print("NOTE: every panel is a TREND validation against published behaviour, not a number")
    print("reproduction -- different lattice, boundary conditions and bath. Say so in the writeup.")
