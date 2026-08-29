#!/usr/bin/env python3
"""MODEL 3 (open, uniform grid): the boundary-driven XX chain -- accuracy, chi growth, wall-clock.

Four panels from one trajectory per arm:

  (a) ACCURACY vs an ANALYTIC solution. `xx_boundary_sz` is the closed-form Lyapunov result
      (O(N^3), certified to 7.2e-16 against a dense Liouvillian at N=2..5) and is valid at EVERY t,
      so each point is scored against truth at the time the run actually reached -- never
      interpolated, never against our own finest grid.
  (b) BOND DIMENSION vs t. The superket's OPERATOR entanglement, which is what carries transport.
  (c) COST vs t: cumulative matvec (solid) and wall-clock (dotted).
  (d) CURRENT vs t against the exact current, approaching the closed-form NESS value.

⛔ THE STEADY STATE IS NOT REACHED AND THAT IS NOT A DEFECT. The Liouvillian gap of this model
closes as N^-3 -- already ~2.5e-02 at N=8, so a true NESS needs t ~ 236 (about 12,000 steps at
dt = 0.02) and at N=16 roughly t ~ 1300. Every panel here is the TRANSIENT, which is exactly where
chi grows and where the integrators can differ; the NESS value is drawn as an asymptote for
orientation, not claimed as reached. Charging an integrator for the residual physical transient
would report relaxation as integrator error.

⚠ BELIEVE `matvec` OVER `seconds`. Wall-clock carries whatever else shared the machine; matvec
counts do not. Both are plotted so a contended run is visible rather than silently quoted.

Run:  python3 benchmarks/imaginary_time/plot_xx_chigrowth.py [csv ...]
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "..", "results")

STYLE = {
    "tdvp2":    dict(color="#c0392b", ls="-",  label="2-site TDVP"),
    "bug_rsvd": dict(color="#2471a3", ls="-",  label="BUG (rSVD-CBE)"),
    "bug_par":  dict(color="#1e8449", ls="--", label="BUG (parallel)"),
    "bug_exact": dict(color="#8e44ad", ls=":", label="BUG (exact CBE)"),
}


def load(paths):
    runs = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                for k, v in r.items():
                    if k != "arm":
                        runs[r["arm"]][k].append(float(v))
    return runs


def main(paths):
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        sys.exit("no input CSVs; run `julia --project=. benchmarks/imaginary_time/dissipative_xx.jl chigrowth`")
    runs = load(paths)
    if not runs:
        sys.exit("input CSVs contained no rows")
    arms = [a for a in STYLE if a in runs]
    any_run = next(iter(runs.values()))
    L = int(any_run["L"][0])
    dt = any_run["dt"][0]

    fig, axes = plt.subplots(2, 2, figsize=(13.5, 9))
    fig.suptitle(f"Boundary-driven XX chain (open), L={L}, dt={dt:g}, uniform grid — "
                 f"BUG vs 2-site TDVP against the analytic Lyapunov solution", fontsize=11)

    # (a) accuracy against the analytic solution
    ax = axes[0][0]
    for a in arms:
        d = runs[a]
        ax.semilogy(d["t"], [max(v, 1e-17) for v in d["err_sz"]], **STYLE[a])
    ax.set_xlabel("t"); ax.set_ylabel(r"max$_j$ |$\langle S^z_j\rangle$ - exact|")
    ax.set_title("(a) accuracy vs ANALYTIC solution (exact at every t)")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=8)

    # (b) chi growth
    ax = axes[0][1]
    peak = 0.0
    for a in arms:
        d = runs[a]
        ax.plot(d["t"], d["chi"], **STYLE[a])
        peak = max(peak, max(d["chi"]))
    ax.set_xlabel("t"); ax.set_ylabel(r"max bond dimension $\chi$")
    ax.set_title("(b) operator-entanglement growth")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)
    ax.text(0.98, 0.05, f"peak $\\chi$ = {peak:.0f} — if this equals the cap,\n"
                        "the run is truncation-limited, not converged",
            transform=ax.transAxes, ha="right", fontsize=7, color="#c0392b")

    # (c) cost
    ax = axes[1][0]
    for a in arms:
        d = runs[a]
        ax.plot(d["t"], d["matvec"], **STYLE[a])
        sc = max(d["seconds"]) or 1.0
        ax.plot(d["t"], [s * max(d["matvec"]) / sc for s in d["seconds"]],
                color=STYLE[a]["color"], ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel("t"); ax.set_ylabel("cumulative matvec")
    ax.set_title("(c) cost — solid: matvec (quotable);  dotted: wall-clock, rescaled")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    # (d) current, against exact and the closed-form NESS
    ax = axes[1][1]
    for a in arms:
        d = runs[a]
        ax.plot(d["t"], d["j_bulk"], **STYLE[a])
    d0 = runs[arms[0]]
    ax.plot(d0["t"], d0["j_exact"], color="0.25", ls="-", lw=2.2, alpha=0.35,
            label="exact (Lyapunov)", zorder=0)
    ax.axhline(0.4, color="0.4", ls="-.", lw=0.9)
    ax.text(0.99, 0.4, " closed-form NESS  j=0.4", transform=ax.get_yaxis_transform(),
            ha="right", va="bottom", fontsize=7, color="0.35")
    ax.set_xlabel("t"); ax.set_ylabel("bulk current $j$")
    ax.set_title("(d) transport — NESS is an asymptote here, not a reached state")
    ax.grid(alpha=0.25); ax.legend(fontsize=8)

    print(f"\n  L = {L}, dt = {dt:g}   at final t = {d0['t'][-1]:.3f}")
    print(f"    {'arm':<20s} {'matvec':>12s} {'seconds':>10s} {'chi':>5s} {'err_sz':>11s}")
    base = None
    for a in arms:
        d = runs[a]
        mv, sec, chi, err = d["matvec"][-1], d["seconds"][-1], d["chi"][-1], d["err_sz"][-1]
        if a == "tdvp2":
            base = mv
        # the baseline is not "1.00x fewer than" itself; only label the arms being compared
        extra = (f"   {base/mv:5.2f}x fewer than tdvp2"
                 if (base and mv and a != "tdvp2") else "")
        print(f"    {STYLE[a]['label']:<20s} {mv:>12,.0f} {sec:>10.1f} {chi:>5.0f} "
              f"{err:>11.3e}{extra}")

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(RESULTS, f"xx_chigrowth_L{L}.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args and os.path.isdir(RESULTS):
        args = [os.path.join(RESULTS, f) for f in sorted(os.listdir(RESULTS))
                if f.startswith("xx_chigrowth_") and f.endswith(".csv")]
    main(args)
