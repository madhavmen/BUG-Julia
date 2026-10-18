"""IT-1 / IT-2 figures (imaginary-time / open-system plan, Tasks 3-4) from the drivers' CSVs.

    python plot_it1.py DIR L [--tb 1e-05] [--seed 1] [--out fig.png]           IT-1, chain length L
    python plot_it1.py DIR 4x4 --model it2 [...]                               IT-2, cylinder LxxLy

Small multiples, one column per grid (the SAME grid for every arm -- the grid ablation):
  row 1  |E(beta) - E_exact(beta)|   the trajectory error against the exact reference
  row 2  E(beta) - E0                the ground-state error (exact or DMRG reference)
  row 3  chi(beta)                   bond-dimension growth
  row 4  cumulative wall vs E - E0   cost to each accuracy target
  row 5  tdvp2's backward-substep amplification per step (when logged: `backamp` column)
Arms bugP at the chosen tol_basis and seed, tdvp2 (no tol_basis). Steps flagged `failed` are
marked with an open ring: a flagged step is a failure, not a converged point.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import NullFormatter

# reference categorical palette, fixed order (dataviz skill, references/palette.md)
SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ARM_ORDER = ["bug2", "bug3", "bug4", "tdvp2"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def load_steps(path):
    rows = defaultdict(list)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            rows[(r["arm"], r["grid"], r["tol_basis"], r["seed"])].append(r)
    return rows


def fnum(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def style(ax, title=None):
    ax.grid(True, color=GRID, linewidth=1, linestyle="-")
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("L", help="chain length (it1) or cylinder tag LxxLy (it2)")
    ap.add_argument("--model", default="it1", choices=("it1", "it2"))
    ap.add_argument("--tb", default="1e-05")
    ap.add_argument("--seed", default="1")
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    tag = f"L{a.L}" if a.model == "it1" else a.L
    steps = load_steps(os.path.join(a.dir, f"{a.model}_steps_{tag}.csv"))
    grids = sorted({k[1] for k in steps}, key=lambda g: (not g.startswith("panel"), g))
    amp = any("backamp" in rs[0] for rs in steps.values())
    nrow = 5 if amp else 4
    fig, axes = plt.subplots(nrow, len(grids), figsize=(5.2 * len(grids), 3 * nrow), squeeze=False)
    for j, g in enumerate(grids):
        for i, arm in enumerate(ARM_ORDER):
            key = (arm, g, "NaN" if arm == "tdvp2" else a.tb, a.seed)
            rs = steps.get(key)
            if not rs:
                continue
            c = SLOTS[i]
            beta = [fnum(r["beta"]) for r in rs]
            deb = [abs(fnum(r["dEbeta"])) for r in rs]
            de0 = [fnum(r["dE0"]) for r in rs]
            chi = [int(r["chi"]) for r in rs]
            wall = [fnum(r["wall"]) for r in rs]
            fail = [r["status"] == "failed" for r in rs]
            lab = arm if arm == "tdvp2" else f"{arm} (tol_basis {a.tb})"
            series = [(beta, deb), (beta, de0), (beta, chi), (wall, de0)]
            if amp:
                series.append((beta, [fnum(r.get("backamp", "nan")) for r in rs]))
            for row, (x, y) in enumerate(series):
                ax = axes[row][j]
                ax.plot(x, y, color=c, linewidth=2, solid_joinstyle="round", label=lab)
                fx = [xx for xx, ff in zip(x, fail) if ff]
                fy = [yy for yy, ff in zip(y, fail) if ff]
                if fx:
                    ax.plot(fx, fy, linestyle="none", marker="o", markersize=8,
                            markerfacecolor="white", markeredgecolor=c, markeredgewidth=2)
        ttl = [f"|E - E_exact(beta)|   {g}", f"E - E0   {g}", f"chi(beta)   {g}",
               f"E - E0 vs cumulative wall [s]   {g}",
               f"tdvp2 backward amplification log(|M'|/|M|) - s<H1>   {g}"]
        for row in range(nrow):
            ax = axes[row][j]
            style(ax, ttl[row])
            ax.set_xscale("log")
            ax.xaxis.set_minor_formatter(NullFormatter())
            if row not in (2, 4):
                ax.set_yscale("log")
            ax.set_xlabel("wall [s]" if row == 3 else "beta", fontsize=8, color=MUTED)
            ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    what = (f"IT-1 Heisenberg chain L={a.L}" if a.model == "it1"
            else f"IT-2 square cylinder {a.L}")
    fig.suptitle(f"{what}, Neel start (open ring = flagged step)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = a.out or os.path.join(a.dir, f"{a.model}_{tag}_tb{a.tb}.png")
    fig.savefig(out, dpi=130, facecolor="#fcfcfb")
    print("wrote", out)


if __name__ == "__main__":
    main()
