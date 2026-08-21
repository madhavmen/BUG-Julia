"""Figures for the RSVD parameter scan.

  rsvd_pareto_<model>_L<n>.png    error against BOTH cost axes, one point per arm, both
                                  schemes overlaid -- the plot the scan exists to produce
  rsvd_budget_<model>_L<n>.png    error against the budget parameter, with the phase-1 seed
                                  spread drawn as a band

⛔ THE PARETO PLOT IS THE ANSWER AND THE BUDGET PLOT IS THE DIAGNOSTIC, not the other way round.
"Does dex=16 lower the error" is trivially yes -- a bigger candidate pool cannot do worse. The
claim cbe_bug actually rests on is error per unit of WORK against tdvp_cbe1s, and only a cost
axis can settle that. A budget curve read on its own will talk you into paying for accuracy you
could have bought more cheaply from the other scheme.

⛔ THE SEED BAND IS NOT DECORATION. Any separation between two arms that is smaller than the
phase-1 spread is RNG, not a parameter effect. Arms inside the band are indistinguishable and
must be reported as such.

⛔ BOTH COST AXES, BECAUSE THEIR DISAGREEMENT IS THE SIGNAL. lanczos_expv exits on breakdown
only, so every solve burns maxiter and seconds normally track krylov count almost exactly. The
sketch's cost is QR and matmul instead, so dex/dover can move seconds WITHOUT moving krylov.
Where the two panels disagree, the sketch has become the bottleneck rather than the solves --
which is a fact about the parameter, not about the machine.

usage:  python3 benchmarks/plot_rsvd_scan.py [glob] [outdir]
"""

import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PATTERN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "rsvd_scan_*.csv")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")

COLOR = {"tdvp_cbe1s": "#2ca02c", "cbe_bug": "#d62728"}
LABEL = {"tdvp_cbe1s": "1-site TDVP-CBE", "cbe_bug": "CBE-BUG"}
MARK = {"growth": "o", "dex": "s"}
FLOOR = 1e-16


def load(pattern):
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            rows.extend(dict(r) for r in csv.DictReader(fh))
    if not rows:
        sys.exit("no rows matched %r -- run benchmarks/rsvd_param_scan.jl first" % pattern)
    return rows


def seed_band(rows, model, scheme):
    """The phase-1 spread for this scheme: the resolution limit of every phase-2 comparison."""
    e = [float(r["err"]) for r in rows
         if r["model"] == model and r["scheme"] == scheme and int(r["phase"]) == 1]
    return (min(e), max(e)) if len(e) > 1 else None


def gate(rows, model):
    """Phase 0 verdict, printed rather than plotted -- it is one comparison per scheme."""
    out = []
    for scheme in ("tdvp_cbe1s", "cbe_bug"):
        d = {r["arm"]: float(r["err"]) for r in rows
             if r["model"] == model and r["scheme"] == scheme and int(r["phase"]) == 0}
        if {"sketch", "exact-svd"} <= set(d):
            s, x = d["sketch"], d["exact-svd"]
            rel = abs(s - x) / max(abs(x), FLOOR)
            out.append("  %-11s sketch %.6e   exact-svd %.6e   rel.diff %.2e  -> %s"
                       % (scheme, s, x, rel,
                          "SATURATED (sampling knobs inert)" if rel < 1e-10
                          else "LOSSY (dover/comp_ratio/seed are live)"))
    return out


def main():
    rows = load(PATTERN)
    os.makedirs(OUTDIR, exist_ok=True)
    models = sorted({(r["model"], int(r["L"])) for r in rows})

    for model, L in models:
        g = gate(rows, model)
        if g:
            print("\nPHASE 0 saturation gate -- %s L=%d" % (model, L))
            print("\n".join(g))

        p2 = [r for r in rows
              if r["model"] == model and int(r["L"]) == L and int(r["phase"]) == 2]
        if not p2:
            continue

        # ── Pareto: error vs each cost axis ──────────────────────────────────────────
        fig, axes = plt.subplots(1, 2, figsize=(13, 5))
        for ax, (col, xlab) in zip(axes, (("krylov", "cumulative Krylov applications"),
                                          ("secs", "wall time (s)"))):
            for scheme in ("tdvp_cbe1s", "cbe_bug"):
                pts = [r for r in p2 if r["scheme"] == scheme]
                if not pts:
                    continue
                for kind in ("growth", "dex"):
                    q = [r for r in pts if r["arm"].startswith(kind)]
                    if not q:
                        continue
                    q.sort(key=lambda r: float(r[col]))
                    ax.plot([float(r[col]) for r in q],
                            [max(float(r["err"]), FLOOR) for r in q],
                            marker=MARK[kind], color=COLOR[scheme], linewidth=1.4,
                            markersize=6, alpha=0.9,
                            linestyle="-" if kind == "growth" else "--",
                            label="%s (%s)" % (LABEL[scheme], kind))
                    for r in q:
                        ax.annotate(r["arm"].split("=")[-1],
                                    (float(r[col]), max(float(r["err"]), FLOOR)),
                                    fontsize=6, xytext=(3, 3), textcoords="offset points")
                band = seed_band(rows, model, scheme)
                if band:
                    ax.axhspan(band[0], band[1], color=COLOR[scheme], alpha=0.12, zorder=0)
            ax.set_yscale("log")
            ax.set_xscale("log")
            ax.set_xlabel(xlab)
            ax.set_ylabel(r"$|\mathrm{obs}-\mathrm{exact}|$")
            ax.grid(alpha=0.25, which="both")
            ax.legend(fontsize=7)
        axes[0].set_title("(a) error vs Krylov cost")
        axes[1].set_title("(b) error vs wall time  — cluster runs only")
        fig.suptitle("%s  L=%d  — RSVD budget Pareto (shaded = seed spread, i.e. noise)"
                     % (model, L), fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.93))
        out = os.path.join(OUTDIR, "rsvd_pareto_%s_L%d.png" % (model.lower(), L))
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)

        # ── the diagnostic: error vs the budget knob itself ──────────────────────────
        fig, ax = plt.subplots(figsize=(7, 5))
        for scheme in ("tdvp_cbe1s", "cbe_bug"):
            q = sorted([r for r in p2 if r["scheme"] == scheme and r["arm"].startswith("dex")],
                       key=lambda r: int(r["dex"]))
            if q:
                ax.plot([int(r["dex"]) for r in q],
                        [max(float(r["err"]), FLOOR) for r in q],
                        marker="s", color=COLOR[scheme], label=LABEL[scheme] + " (dex)")
            band = seed_band(rows, model, scheme)
            if band:
                ax.axhspan(band[0], band[1], color=COLOR[scheme], alpha=0.12, zorder=0)
        ax.set_yscale("log")
        ax.set_xlabel("dex  (directions offered per bond per step)")
        ax.set_ylabel(r"$|\mathrm{obs}-\mathrm{exact}|$")
        ax.set_title("%s L=%d — error vs budget at FIXED rank (maxdim binding)" % (model, L))
        ax.grid(alpha=0.25, which="both")
        ax.legend(fontsize=8)
        fig.tight_layout()
        out = os.path.join(OUTDIR, "rsvd_budget_%s_L%d.png" % (model.lower(), L))
        fig.savefig(out, dpi=140)
        plt.close(fig)
        print("wrote", out)


if __name__ == "__main__":
    main()
