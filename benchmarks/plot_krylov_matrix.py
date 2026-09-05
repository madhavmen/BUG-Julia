"""CBE-BUG across KRYLOV DEPTH: does a deeper basis buy anything on the tolerance matrix?

  krylov_matrix_L<n>_T<t>_D<cap>.png
      panel 1  final error, every grid cell, per depth   -- with best/worst marked
      panel 2  final chi, the same cells                 -- "beside it", per the spec
      panel 3  chi(t) for each depth's BEST-error config
      panel 4  chi(t) for each depth's WORST-error config

⛔ THE COST AXIS IS DELIBERATELY NOT `krylov`, AND THIS FIGURE IS THE CASE THAT PROVES WHY.
   `krylov_dims` counts `apply_one_site` inside `_krylov_frame` plus the root solve. At
   `krylov_basis = 0` (`m0`) THERE IS NO KRYLOV FRAME AT ALL -- the basis is `cbe_expand`'s
   output alone, one power of H -- so essentially the arm's whole basis cost sits in the
   counter's blind spot. MEASURED: m0 reports 8.02x fewer matvecs than mdef and runs 1.45x
   SLOWER over 24 matched cells (L=18, chi=64). Plotting error against `krylov` would therefore
   show m0 as a dramatic Pareto winner while it is in fact the slowest arm. The counts are
   annotated with that warning attached and never used as an x-axis.

⚠ THE HEADLINE IS THE FLATNESS. Best-achievable error across m0/m2/m3/mdef spans
  1.26e-04 - 1.32e-04, i.e. NOTHING, across an 8x span of counted operator applications. The
  depth is not a tuning knob on this problem; it is a cost with no return. (`mdef` is the
  ||H||dt/2-derived depth, which at L=18 clamps to 12.)

⚠ `mdef` HAS A LARGER GRID (49 cells) than the others (25) because it was run before the ladder
  was reduced. Panels 1-2 plot cells, not a fixed count, so this is legitimate -- but the SPREAD
  of its cloud is over a wider knob range and is not directly comparable to the others' spread.
  Only the best/worst markers are compared.

usage:  python3 benchmarks/plot_krylov_matrix.py [resultsdir]
"""

import csv
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")

# Depth order is the STORY ORDER: cheapest basis first.
# ⚠ SHORT TICK LABEL **AND** A LONG ONE, because they go in different places. Four long labels
# on a categorical axis overlap into illegibility at this figure width; the description belongs
# in the legend of panels 3-4, where there is room for it on its own line.
ORDER = [("m0",   "m0",   "CBE frame only, 1 power of H", "#c1440e"),
         ("m2",   "m2",   "2 Krylov vectors",             "#c98b3a"),
         ("m3",   "m3",   "3 Krylov vectors",             "#2a6558"),
         ("mdef", "mdef", "derived depth, m=12",          "#33507a")]


def load(path):
    """-> ({(tau, split): final_row}, {(tau, split): (ts, chis)})"""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return {}, {}
    tmax = max(float(r["t"]) for r in rows)
    final, traj = {}, defaultdict(list)
    for r in rows:
        k = (r["tau_trunc"], r["split_cutoff"])
        traj[k].append((float(r["t"]), int(r["maxbond"])))
        if abs(float(r["t"]) - tmax) < 1e-9:
            final[k] = r
    # keep trajectories only for arms that actually finished
    traj = {k: tuple(np.array(a) for a in zip(*sorted(v)))
            for k, v in traj.items() if k in final}
    return final, traj


def figure(cap, tmax, L, dt, per_depth):
    fig, ax = plt.subplots(1, 4, figsize=(21, 5.0))
    fig.subplots_adjust(wspace=.30, top=.76, bottom=.15)

    present = [(k, lab, desc, col) for k, lab, desc, col in ORDER if k in per_depth]
    xs = np.arange(len(present))

    # ── panels 1 & 2: every cell's final error and final chi, per depth ─────────────────
    lo, hi = np.inf, 0.0
    for i, (key, lab, desc, col) in enumerate(present):
        final, _ = per_depth[key]
        errs = np.array([float(r["err_prof"]) for r in final.values()])
        chis = np.array([int(r["maxbond"]) for r in final.values()])
        lo, hi = min(lo, errs.min()), max(hi, errs.max())
        jit = (np.random.RandomState(0).rand(len(errs)) - .5) * .28
        ax[0].scatter(i + jit, errs, s=16, color=col, alpha=.40, edgecolors="none")
        ax[1].scatter(i + jit, chis, s=16, color=col, alpha=.40, edgecolors="none")
        ax[0].plot([i - .3, i + .3], [errs.min()] * 2, color=col, lw=2.6)
        ax[0].plot([i - .3, i + .3], [errs.max()] * 2, color=col, lw=1.4, ls=":")
        ax[0].annotate("%.2e" % errs.min(), xy=(i, errs.min()), xytext=(0, -12),
                       textcoords="offset points", ha="center", fontsize=7.5, color=col)
        ax[1].plot([i - .3, i + .3], [chis.max()] * 2, color=col, lw=2.6)

    ax[0].set_yscale("log")
    # Headroom BELOW for the best-error labels, which otherwise fall off the axis.
    ax[0].set_ylim(lo / 2.2, hi * 1.6)
    ax[0].set_xticks(xs); ax[0].set_xticklabels([l for _, l, _, _ in present], fontsize=9)
    ax[0].set_ylabel(r"final $\max_j|\langle S^z_j\rangle-$exact$|$")
    # ⛔ THE matvec CAVEAT GOES IN THE TITLE. As an annotate() inside the axes it landed on top of
    # the best-error labels at the bottom -- and the bottom is the only free corner, because the
    # clouds span the full height by construction.
    mv = " / ".join("%.1fk" % (np.mean([int(r["krylov"])
                                        for r in per_depth[k][0].values()]) / 1000)
                    for k, _, _, _ in present)
    ax[0].set_title("Final error, EVERY cell of the tolerance matrix.\n"
                    "Solid = best reachable, dotted = worst.\n"
                    "matvec " + mv + " — ⚠ NOT a cost axis (m0's basis cost is invisible)",
                    fontsize=8.8)
    ax[0].grid(True, which="both", axis="y", lw=.4, alpha=.35)

    ax[1].axhline(cap, ls="--", lw=1.2, color="#555")
    ax[1].text(.99, cap, " cap %d " % cap, ha="right", va="bottom", fontsize=8, color="#555",
               transform=ax[1].get_yaxis_transform())
    ax[1].set_xticks(xs); ax[1].set_xticklabels([l for _, l, _, _ in present], fontsize=9)
    ax[1].set_ylabel(r"final $\chi$")
    ax[1].set_title("Final bond dimension, the same cells.  Depth does not\n"
                    "change the rank the state settles at either.", fontsize=9.5)
    ax[1].grid(True, axis="y", lw=.4, alpha=.35)

    # ── panels 3 & 4: chi(t) for the best and worst cell of each depth ──────────────────
    for which, axis, title in ((0, ax[2], "BEST-error config of each depth"),
                               (1, ax[3], "WORST-error config of each depth")):
        for key, lab, desc, col in present:
            final, traj = per_depth[key]
            if not traj:
                continue
            pick = (min if which == 0 else max)(
                traj.keys(), key=lambda k: float(final[k]["err_prof"]))
            ts, chis = traj[pick]
            axis.plot(ts, chis, "-", color=col, lw=2.2,
                      label=r"%s (%s)   $\tau$=%s, split=%s" % (lab, desc, pick[0], pick[1]))
        axis.axhline(cap, ls="--", lw=1.2, color="#555")
        axis.set_xlabel("$t$"); axis.set_ylabel(r"max bond dimension $\chi(t)$")
        axis.set_title(title + ".\nFlat at the cap = the cap sets the error from there on.",
                       fontsize=9.5)
        axis.legend(fontsize=7.5, frameon=False, loc="lower right")
        axis.grid(True, lw=.4, alpha=.35)

    fig.suptitle("CBE-BUG across Krylov depth — XX chain ($\\Delta$=0), U(1), domain wall, "
                 "L=%s, T=%g, dt=%s, cap %d.   Best error is FLAT over an 8$\\times$ span of depth."
                 % (L, tmax, dt, cap), fontsize=11, y=.95)
    out = os.path.join(RES, "krylov_matrix_L%s_T%g_D%d.png" % (L, tmax, cap))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    groups = defaultdict(dict)
    for p in sorted(glob.glob(os.path.join(RES, "heis_grid_*.csv"))):
        b = os.path.basename(p)
        if "_prof_" in b:
            continue
        m = re.match(r"heis_grid_(\w+?)_L(\d+)_dt([\d.]+)_T([\d.]+)_D(\d+|inf)", b)
        if not m:
            continue
        depth, L, dt, T, cap = m.groups()
        if cap == "inf":
            continue
        final, traj = load(p)
        if not final:
            continue
        groups[(int(cap), float(T), L, dt)][depth] = (final, traj)
    for (cap, T, L, dt), per_depth in sorted(groups.items()):
        if len(per_depth) < 2:
            continue
        figure(cap, T, L, dt, per_depth)


if __name__ == "__main__":
    main()
