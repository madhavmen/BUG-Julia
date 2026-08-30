#!/usr/bin/env python3
"""Plot the two knob studies of `knobs_l20.jl`.

Study A -- BUG's own knobs: what each one costs and what it buys, against the EXACT correlator.
Study B -- the rSVD sketch knobs: the sketch is supposed to move TIME, not the answer.

  python3 plot_knobs_l20.py [A|B] [geom ...]

Reads  results/knobs_l<N>_<study>_<geoms>.csv
Writes results/knobs_l<N>_<study>.png

DESIGN NOTES THAT ARE ALSO CORRECTNESS NOTES:

* The `err` axis is LOG and floored at the exact-propagation tolerance. `exact_correlator` stops
  each `expv` at 1e-12, so an MPS arm reported below ~1e-12 is at the REFERENCE's floor, not at
  its own. Drawing those points as if they were resolved would claim precision the reference
  cannot supply.

* Study A's headline panel is err-vs-SECONDS, not err-vs-knob. The question the campaign asks is
  "which arm is on the Pareto front", and a knob that improves accuracy while costing more time
  is neither better nor worse until both axes are on the same plot.

* `matvec` is NOT plotted as a cost axis for BUG. `krylov_dims` counts `apply_one_site` and is
  blind to `cbe_expand`'s sketch work, so it understates every BUG row. It is in the CSV for
  diagnosis; wall clock is the cost claim.
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

sys.stdout.reconfigure(errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

# The exact reference's own tolerance: `sparse_expv(..., tol=1e-12)`. Nothing below this is a
# measurement of the MPS arm.
REF_FLOOR = 1e-12


def load(study, geoms):
    rows = []
    for fn in os.listdir(RES):
        if not fn.startswith("knobs_l") or not fn.endswith(".csv"):
            continue
        parts = fn[:-4].split("_")
        if len(parts) < 3 or parts[2] != study:
            continue
        with open(os.path.join(RES, fn), encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if geoms and r["geom"] not in geoms:
                    continue
                r["err"] = float(r["err"])
                r["seconds"] = float(r["seconds"])
                r["matvec"] = int(r["matvec"])
                r["chi"] = int(r["chi"])
                # The panel key: geometry AND evolution cap. `dex`/`dover` are absolute budgets
                # and `growth` is proportional, so the two caps do not share a winner and must
                # not share a column.
                r["gkey"] = f'{r["geom"]} D={r["D"]}'
                rows.append(r)
    return rows


def _floor(e):
    return max(e, REF_FLOOR)


def plot_A(rows, out):
    geoms = sorted({r["gkey"] for r in rows})
    fig, axes = plt.subplots(2, len(geoms), figsize=(6.2 * len(geoms), 9.4), squeeze=False)

    for gi, g in enumerate(geoms):
        gr = [r for r in rows if r["gkey"] == g]
        N = gr[0]["N"]

        # ── (top) the Pareto view: accuracy against wall clock ──────────────────────────
        ax = axes[0][gi]
        style = {"reference": ("s", "#444444"), "baseline": ("*", "#c0392b")}
        for r in gr:
            knob = r["knob"]
            mk, col = style.get(knob, ("o", None))
            if col is None:
                col = {"m": "#1f77b4", "krylov_tol(beta)": "#17becf",
                       "stol_pre": "#2ca02c", "split_cutoff": "#ff7f0e",
                       "root_cutoff": "#9467bd", "parallel": "#8c564b"}.get(knob, "#777777")
            ax.scatter(r["seconds"], _floor(r["err"]), marker=mk, s=170 if mk == "*" else 55,
                       color=col, zorder=3, edgecolors="none")
            ax.annotate(r["arm"].replace("bug_", ""), (r["seconds"], _floor(r["err"])),
                        fontsize=6.2, xytext=(3, 3), textcoords="offset points")
        ax.axhline(REF_FLOOR, ls=":", lw=1, color="k")
        ax.text(0.99, REF_FLOOR, " exact-reference floor (1e-12)", va="bottom", ha="right",
                transform=ax.get_yaxis_transform(), fontsize=7)
        ax.set_yscale("log")
        ax.set_xlabel("wall clock over the trajectory  [s]")
        ax.set_ylabel(r"max$_{x,t}\,|G_{\rm MPS}-G_{\rm exact}|$")
        ax.set_title(f"{g}  (N={N})  --  accuracy vs cost\nlower-left is better")
        ax.grid(alpha=0.3)

        # ── (bottom) one line per knob, value on x ──────────────────────────────────────
        ax = axes[1][gi]
        by = defaultdict(list)
        for r in gr:
            if r["knob"] in ("reference", "baseline", "parallel"):
                continue
            try:
                v = float(r["value"])
            except ValueError:
                continue
            by[r["knob"]].append((v, _floor(r["err"]), r["seconds"]))
        for knob, pts in sorted(by.items()):
            pts.sort()
            xs = [p[0] for p in pts]
            es = [p[1] for p in pts]
            # `m` is a count and the tolerances are magnitudes -- normalise the x axis to the
            # knob's own range so several knobs share one panel without one of them being a
            # vertical line.
            if max(xs) > 0 and min(xs) > 0 and max(xs) / max(min(xs), 1e-300) > 50:
                xs = [np.log10(x) for x in xs]
                lbl = f"{knob}  (log10)"
            else:
                lbl = knob
            ax.plot(xs, es, "o-", label=lbl, lw=1.4, ms=5)
        ax.axhline(REF_FLOOR, ls=":", lw=1, color="k")
        ax.set_yscale("log")
        ax.set_xlabel("knob value (log10 where the range spans decades)")
        ax.set_ylabel(r"max$_{x,t}\,|\Delta G|$")
        ax.set_title(f"{g}  --  one factor at a time from the baseline")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

    fig.suptitle("BUG knobs at N=%s: what each buys, measured against the exact correlator" %
                 rows[0]["N"], fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=150)
    print("wrote", out)


def plot_B(rows, out):
    geoms = sorted({r["gkey"] for r in rows})
    fig, axes = plt.subplots(2, len(geoms), figsize=(6.2 * len(geoms), 9.0), squeeze=False)

    for gi, g in enumerate(geoms):
        gr = [r for r in rows if r["gkey"] == g]
        base = next((r for r in gr if r["knob"] == "baseline"), None)

        # ── (top) TIME per knob -- the question study B actually asks ───────────────────
        ax = axes[0][gi]
        by = defaultdict(list)
        for r in gr:
            if r["knob"] in ("baseline", "exact"):
                continue
            try:
                by[r["knob"]].append((float(r["value"]), r["seconds"], _floor(r["err"])))
            except ValueError:
                pass
        for knob, pts in sorted(by.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", label=knob, lw=1.4, ms=5)
        if base:
            ax.axhline(base["seconds"], ls="--", lw=1.2, color="#c0392b",
                       label=f"baseline {base['seconds']:.2f}s")
        exact = next((r for r in gr if r["knob"] == "exact"), None)
        if exact:
            ax.axhline(exact["seconds"], ls="-.", lw=1.2, color="#444444",
                       label=f"exact CBE (A/B arm) {exact['seconds']:.2f}s")
        ax.set_xlabel("knob value")
        ax.set_ylabel("wall clock over the trajectory  [s]")
        ax.set_title(f"{g}  --  rSVD knobs: COMPUTATION TIME")
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

        # ── (bottom) the accuracy control: the sketch must not move the answer ──────────
        ax = axes[1][gi]
        for knob, pts in sorted(by.items()):
            pts.sort()
            ax.plot([p[0] for p in pts], [p[2] for p in pts], "o-", label=knob, lw=1.4, ms=5)
        if base:
            ax.axhline(_floor(base["err"]), ls="--", lw=1.2, color="#c0392b", label="baseline")
        ax.axhline(REF_FLOOR, ls=":", lw=1, color="k")
        ax.set_yscale("log")
        ax.set_xlabel("knob value")
        ax.set_ylabel(r"max$_{x,t}\,|\Delta G|$")
        ax.set_title(f"{g}  --  the control: a sketch knob should move TIME, not this\n"
                     "a row that moves here is a defect, not a trade-off", fontsize=10)
        ax.legend(fontsize=7.5)
        ax.grid(alpha=0.3)

    fig.suptitle("rSVD sketch knobs at N=%s: cost, and the accuracy control" % rows[0]["N"],
                 fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.965))
    fig.savefig(out, dpi=150)
    print("wrote", out)


def main():
    args = [a for a in sys.argv[1:]]
    study = "A"
    for a in list(args):
        if a in ("A", "B"):
            study = a
            args.remove(a)
    geoms = set(args) if args else None

    rows = load(study, geoms)
    if not rows:
        print(f"no rows for study {study} -- run knobs_l20.jl {study} first")
        return
    out = os.path.join(RES, f"knobs_l{rows[0]['N']}_{study}.png")
    (plot_A if study == "A" else plot_B)(rows, out)


if __name__ == "__main__":
    main()
