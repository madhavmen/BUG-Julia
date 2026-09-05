#!/usr/bin/env python3
"""Plots for the large-chi / large-D campaign, from large_chi_L<L>.csv.

Four figures, each an SVG with no title (captions live in the thesis):

  runtime_vs_mpoD.svg    s/step vs MPO bond dimension D, one line per solver   <- the ask
  runtime_vs_chi.svg     s/step vs MPS bond dimension chi, one line per solver
  alloc_vs_chi.svg       GB allocated per step -- the axis that turned out to
                         track runtime almost exactly (~1 GB/s for every arm)
  speedup_threads.svg    speedup vs serial as contract tasks / BLAS threads rise

⚠ STEP 1 IS DROPPED EVERYWHERE. The first call to a stepper compiles it, and at chi=1024
that overhead is seconds against a ~120 s step -- small, but it is not the quantity being
plotted and it is not reproducible run to run. The CSV keeps every step so the choice stays
visible here rather than being baked into the measurement.
"""

import os
import sys
import csv
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "svg.fonttype": "none",      # keep text as text so the thesis can restyle it
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# One colour and marker per solver, held fixed across every figure so a reader who has
# learned the legend once never has to re-read it.
ARMS = [
    ("bug",    "BUG",              "#c1440e", "D", "-"),
    ("bugmid", "BUG (midpoint)",   "#8a5a00", "^", "-."),
    ("cbe1s",  "1-site TDVP-CBE",  "#2a6558", "s", "--"),
    ("tdvp2",  "2-site TDVP",      "#33507a", "o", "-"),
]

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else RES


def load(path):
    rows = []
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            r["chi"] = int(r["chi"])
            r["mpoD"] = int(r.get("mpoD", 5))
            r["blas"] = int(r["blas"])
            r["cthreads"] = int(r["cthreads"])
            r["step"] = int(r["step"])
            r["seconds"] = float(r["seconds"])
            r["alloc_gb"] = float(r["alloc_gb"])
            rows.append(r)
    return rows


def mean_steps(rows):
    """Mean seconds and alloc over steps >= 2, keyed by (arm, chi, mpoD, blas, cthreads)."""
    acc = defaultdict(lambda: ([], []))
    for r in rows:
        if r["step"] < 2:
            continue                      # compilation lives in step 1
        t, a = acc[(r["arm"], r["chi"], r["mpoD"], r["blas"], r["cthreads"])]
        t.append(r["seconds"]); a.append(r["alloc_gb"])
    return {k: (sum(t) / len(t), sum(a) / len(a)) for k, (t, a) in acc.items() if t}


def _finish(ax, xlabel, ylabel, path, logx=True, logy=True):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if logx:
        ax.set_xscale("log", base=2)
        ax.xaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
    ax.grid(True, which="major", alpha=0.25, linewidth=0.6)
    ax.legend(frameon=False)
    ax.figure.tight_layout()
    ax.figure.savefig(path, format="svg")
    plt.close(ax.figure)
    print("wrote", path)


def plot_vs(means, xkey, fixed, xlabel, fname, value=0):
    """s/step (value=0) or GB/step (value=1) against one axis, one line per solver."""
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    drew = False
    for arm, label, colour, marker, ls in ARMS:
        pts = sorted((k[1] if xkey == "chi" else k[2], v[value])
                     for k, v in means.items()
                     if k[0] == arm and all(k[i] == f for i, f in fixed.items()))
        if not pts:
            continue
        drew = True
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, color=colour, linestyle=ls, label=label,
                linewidth=1.8, markersize=6)
    if not drew:
        plt.close(fig)
        print("skip", fname, "- no rows matched", fixed)
        return
    _finish(ax, xlabel, "GB allocated per step" if value else "seconds per step",
            os.path.join(OUT, fname))


def plot_thread_speedup(means, fname="speedup_threads.svg"):
    """Speedup against the serial-contract, single-BLAS-thread cell of the same arm+chi.

    ⚠ Only meaningful if the job actually swept them. With `julia -t 2` both parallelism
    axes share a two-slot pool and every cell collapses onto the same number, which is a
    real result about the RUN, not about the code -- so say so rather than drawing a flat
    line and letting it read as "threading does not help".
    """
    fig, ax = plt.subplots(figsize=(6.2, 4.4))
    drew = False
    for arm, label, colour, marker, ls in ARMS:
        cells = {(k[3], k[4]): v[0] for k, v in means.items() if k[0] == arm}
        if len(cells) < 2:
            continue
        base_key = min(cells)                       # smallest (blas, cthreads)
        base = cells[base_key]
        pts = sorted((max(b, c if c else b), base / t) for (b, c), t in cells.items())
        if len(pts) < 2:
            continue
        drew = True
        xs, ys = zip(*pts)
        ax.plot(xs, ys, marker=marker, color=colour, linestyle=ls, label=label,
                linewidth=1.8, markersize=6)
    if not drew:
        plt.close(fig)
        print("skip", fname, "- the job did not sweep threads")
        return
    lim = ax.get_xlim()
    ax.plot(lim, lim, color="0.6", linewidth=1.0, linestyle=":", label="ideal", zorder=0)
    ax.set_xlim(lim)
    _finish(ax, "threads (contract tasks or BLAS)", "speedup vs serial",
            os.path.join(OUT, fname), logx=False, logy=False)


def main():
    csvs = [f for f in os.listdir(RES) if f.startswith("large_chi_L") and f.endswith(".csv")]
    if not csvs:
        sys.exit("no large_chi_L*.csv in %s" % RES)
    rows = []
    for f in csvs:
        rows += load(os.path.join(RES, f))
    means = mean_steps(rows)
    print("%d rows, %d (arm,chi,D,blas,cthreads) cells" % (len(rows), len(means)))

    chis = sorted({k[1] for k in means})
    ds = sorted({k[2] for k in means})
    blas = sorted({k[3] for k in means})
    cthr = sorted({k[4] for k in means})
    print("chi=%s  mpoD=%s  blas=%s  cthreads=%s" % (chis, ds, blas, cthr))

    # THE ASK: runtime vs MPO bond dimension, three (four) solvers. Pin chi and the thread
    # settings so the only thing varying along x is D.
    if len(ds) > 1:
        plot_vs(means, "mpoD", {1: chis[-1], 3: blas[-1], 4: cthr[-1]},
                "MPO bond dimension $D$", "runtime_vs_mpoD.svg")
    else:
        print("skip runtime_vs_mpoD.svg - only one MPO D (%s) in the data" % ds)

    if len(chis) > 1:
        plot_vs(means, "chi", {2: ds[0], 3: blas[-1], 4: cthr[-1]},
                "MPS bond dimension $\\chi$", "runtime_vs_chi.svg")
        plot_vs(means, "chi", {2: ds[0], 3: blas[-1], 4: cthr[-1]},
                "MPS bond dimension $\\chi$", "alloc_vs_chi.svg", value=1)
    else:
        print("skip chi figures - only one chi (%s) in the data" % chis)

    plot_thread_speedup(means)


if __name__ == "__main__":
    main()
