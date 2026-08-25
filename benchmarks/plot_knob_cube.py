#!/usr/bin/env python3
"""THE THREE-KNOB CUBE: half-sweep cutoff x root-out cutoff x Krylov depth.

There are exactly three rank/accuracy controls left in one `cbe_bug_step!`, and every accuracy
claim about this sweep is a statement about a POINT in this cube:

    x  `split_cutoff`  the half-sweep SVD that splits the stacked Krylov frame into W[i] / Z[j]
    y  `trunc_thresh`  the closing root-out `truncate_recursive!`
    z  `krylov_basis`  the depth of the Krylov basis each half-sweep builds  ("m")

WHY A CUBE AND NOT THREE LINE SCANS. A one-knob scan holds the other two at values that may
already be the binding constraint, in which case it measures the floor and reports "this knob does
nothing". That is exactly how the Krylov depth stayed invisible for so long: every scan of the
truncation knobs was run at a fixed depth that was itself the limiter.

⛔ READ THE HEATMAPS BY WHERE THE GRADIENT POINTS, decided before looking:

    varies along x only        the HALF-SWEEP governs; the closing pass is cleaning up after it
    varies along y only        the ROOT-OUT pass governs; `split_cutoff` has nothing behind it
    varies along max(x, y)     the two are REDUNDANT -- the looser one sets the accuracy
    flat in both, moves in z   NEITHER truncation governs: the KRYLOV DEPTH is the floor
    genuinely 2D               they cut different things and both must be quoted for any claim

The fourth outcome is the one already measured at L=12 and it is why this plot exists.

Run:  python3 benchmarks/plot_knob_cube.py [results_dir] [L]
"""
import csv
import glob
import math
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
L = sys.argv[2] if len(sys.argv) > 2 else "16"

# The depth each sweep name pins. `mdef` is the package default (cap 30 + tol 1e-6) and is
# deliberately NOT called "m=30": the tolerance usually stops it well short of the cap.
ORDER = ["m0", "m2", "m3", "mdef"]
LABEL = {"m0": "m = 0\n(one power of H)", "m2": "m = 2", "m3": "m = 3",
         "mdef": "package default\n(cap 30, tol 1e-6)"}


def load(path):
    """Final-time row per (tau_trunc, split_cutoff). The last sample is the accumulated answer."""
    rows = {}
    with open(path) as fh:
        for r in csv.DictReader(fh):
            key = (float(r["tau_trunc"]), float(r["split_cutoff"]))
            t = float(r["t"])
            if key not in rows or t >= rows[key][0]:
                rows[key] = (t, r)
    return {k: v[1] for k, v in rows.items()}


def cube(tag=""):
    """Only the named depth arms, and only one run's parameters.

    ⛔ THE GLOB ALONE IS NOT ENOUGH. `results/` accumulates CSVs from every earlier campaign, and
    `heis_grid_*_L12_*.csv` happily matches runs with a DIFFERENT `T` and a different tau ladder
    (`heis_grid_default_L12_dt0.05_T1.5.csv` is one). Pooling those into the same panel compares
    arms that never saw the same problem. Restrict to the depth arms this cube is about, and --
    when several runs of them exist -- to the newest `dt`/`T` among them.
    """
    found = {}
    for p in sorted(glob.glob(os.path.join(RES, f"heis_grid_*_L{L}_*.csv"))):
        base = os.path.basename(p)
        parts = base[: -len(".csv")].split("_")      # heis grid <sweep> L<L> dt<dt> T<T>
        sweep = parts[2]
        if sweep not in ORDER:
            continue
        if tag and tag not in base:
            continue
        found.setdefault("_".join(parts[4:]), {})[sweep] = p

    if not found:
        return {}
    # newest parameter set wins, by file mtime of its members
    key = max(found, key=lambda k: max(os.path.getmtime(v) for v in found[k].values()))
    if len(found) > 1:
        print(f"note: {len(found)} parameter sets present; using {key!r} "
              f"and ignoring {sorted(set(found) - {key})}")
    out = {}
    for sweep, path in found[key].items():
        d = load(path)
        if d:
            out[sweep] = d
    return out


def panel(data, field, title, fname, log=True, fmt="{:.1e}"):
    sweeps = [s for s in ORDER if s in data] + [s for s in data if s not in ORDER]
    if not sweeps:
        return
    ys = sorted({k[0] for k in data[sweeps[0]]}, reverse=True)   # root-out, loose at top
    xs = sorted({k[1] for k in data[sweeps[0]]}, reverse=True)   # half-sweep, loose at left

    grids = []
    for s in sweeps:
        g = np.full((len(ys), len(xs)), np.nan)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                r = data[s].get((y, x))
                if r is not None:
                    try:
                        g[i, j] = float(r[field])
                    except (ValueError, KeyError):
                        pass
        grids.append(g)

    finite = np.concatenate([g[np.isfinite(g)].ravel() for g in grids]) if grids else np.array([])
    finite = finite[finite > 0] if log else finite
    if finite.size == 0:
        return
    norm = (LogNorm(vmin=finite.min(), vmax=finite.max()) if log else None)

    fig, axes = plt.subplots(1, len(sweeps), figsize=(4.1 * len(sweeps), 4.4), squeeze=False)
    for ax, s, g in zip(axes[0], sweeps, grids):
        im = ax.imshow(g, cmap="viridis_r" if log else "viridis", norm=norm, aspect="auto")
        ax.set_xticks(range(len(xs)))
        ax.set_xticklabels([f"{x:.0e}" for x in xs], rotation=45, fontsize=7.5)
        ax.set_yticks(range(len(ys)))
        ax.set_yticklabels([f"{y:.0e}" for y in ys], fontsize=7.5)
        ax.set_title(LABEL.get(s, s), fontsize=9)
        ax.set_xlabel("half-sweep  `split_cutoff`", fontsize=8)
        if ax is axes[0][0]:
            ax.set_ylabel("root-out  `trunc_thresh`", fontsize=8)
        for i in range(len(ys)):
            for j in range(len(xs)):
                if np.isfinite(g[i, j]):
                    ax.text(j, i, fmt.format(g[i, j]), ha="center", va="center",
                            fontsize=6.2, color="white")
    fig.colorbar(im, ax=axes[0].tolist(), fraction=0.025, pad=0.015)
    fig.suptitle(title, fontsize=11)
    fig.savefig(os.path.join(RES, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", fname)


def gradients(data):
    """Which knob actually moves the error -- the cube's summary, as one bar chart.

    For each depth: the spread (max/min) of the error along each axis, averaged over the other.
    A knob that governs shows a large spread; one that is inert shows ~1.
    """
    sweeps = [s for s in ORDER if s in data]
    if not sweeps:
        return
    fig, ax = plt.subplots(figsize=(7.2, 4.0))
    w = 0.35
    xs_pos = np.arange(len(sweeps))
    spread = {"half-sweep (`split_cutoff`)": [], "root-out (`trunc_thresh`)": []}
    for s in sweeps:
        ys = sorted({k[0] for k in data[s]})
        xs = sorted({k[1] for k in data[s]})
        g = np.full((len(ys), len(xs)), np.nan)
        for i, y in enumerate(ys):
            for j, x in enumerate(xs):
                r = data[s].get((y, x))
                if r:
                    g[i, j] = float(r["err_prof"])
        with np.errstate(invalid="ignore"):
            # spread ALONG x = vary half-sweep at fixed root-out, then take the worst row
            sx = np.nanmax(np.nanmax(g, axis=1) / np.nanmin(g, axis=1))
            sy = np.nanmax(np.nanmax(g, axis=0) / np.nanmin(g, axis=0))
        spread["half-sweep (`split_cutoff`)"].append(sx)
        spread["root-out (`trunc_thresh`)"].append(sy)
    for k, (name, vals) in enumerate(spread.items()):
        ax.bar(xs_pos + (k - 0.5) * w, vals, w, label=name)
    ax.axhline(1.0, color="0.3", lw=1, ls="--")
    ax.text(len(sweeps) - 0.5, 1.05, "1.0 = knob is INERT", fontsize=8, ha="right", color="0.3")
    ax.set_yscale("log")
    ax.set_xticks(xs_pos)
    ax.set_xticklabels([LABEL.get(s, s).replace("\n", " ") for s in sweeps], fontsize=8)
    ax.set_ylabel("error spread along the axis  (max / min)")
    ax.set_title(f"Which knob moves the error?  Heisenberg L={L}", fontsize=11)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(RES, f"knob_gradients_L{L}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote knob_gradients_L{L}.png")


def pareto(data):
    """Accuracy against COST, every cube point, coloured by depth.

    The cube's practical question is not "which point is most accurate" but "what does accuracy
    cost", and the answer is only visible with all three knobs on one pair of axes.
    """
    sweeps = [s for s in ORDER if s in data]
    if not sweeps:
        return
    fig, ax = plt.subplots(figsize=(7.4, 5.0))
    cols = plt.cm.viridis(np.linspace(0.05, 0.85, len(sweeps)))
    for s, c in zip(sweeps, cols):
        k = [float(r["krylov"]) for r in data[s].values() if float(r["krylov"]) > 0]
        e = [float(r["err_prof"]) for r in data[s].values() if float(r["krylov"]) > 0]
        ax.scatter(k, e, s=34, color=c, label=LABEL.get(s, s).replace("\n", " "),
                   edgecolor="0.25", linewidth=0.4, zorder=3)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlabel("operator applications  (contention-immune cost axis)")
    ax.set_ylabel("max |dSz(j)| against the exact sparse reference")
    ax.set_title(f"Cost of accuracy across the whole cube -- Heisenberg L={L}", fontsize=11)
    ax.grid(alpha=0.25, which="both", lw=0.5)
    ax.legend(fontsize=8)
    fig.savefig(os.path.join(RES, f"knob_pareto_L{L}.png"), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"wrote knob_pareto_L{L}.png")


def main():
    data = cube()
    if not data:
        print(f"no heis_grid_*_L{L}_*.csv in {RES}")
        return 1
    print("sweeps found:", ", ".join(sorted(data)))
    panel(data, "err_prof", f"Error vs the two truncations, per Krylov depth -- Heisenberg L={L}",
          f"knob_cube_err_L{L}.png")
    panel(data, "maxbond", f"Bond dimension -- the price paid -- Heisenberg L={L}",
          f"knob_cube_chi_L{L}.png", log=False, fmt="{:.0f}")
    panel(data, "krylov", f"Operator applications -- Heisenberg L={L}",
          f"knob_cube_cost_L{L}.png", log=True, fmt="{:.0f}")
    gradients(data)
    pareto(data)
    return 0


if __name__ == "__main__":
    sys.exit(main())
