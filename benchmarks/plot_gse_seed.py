"""Does seeding the local eigensolve from the EXPANDED basis reduce the sweeps?

    python3 benchmarks/plot_gse_seed.py [results/cbe_sweeps_l16_*gse*.csv] [outdir]

THE ONE THING THAT DIFFERS between `dmrg_cbe_rsvd` and `dmrg_cbe_rsvd_reuse` is the Lanczos START
VECTOR. `expand_bond!` widens a bond "leaving the state unchanged", so the admitted directions
arrive with ZERO amplitude and the default start -- the current tensor, zero-padded -- has no
component along exactly what the expansion just paid to find. The seeded arm gives them weight up
front. Selection, operator, truncation and sweep order are identical.

⚠ READ IT AS APPLICATIONS-TO-ACCURACY, NOT SWEEPS-TO-ACCURACY. The seed costs one extra operator
application per local solve. Fewer sweeps at a higher per-sweep price is not automatically a win,
which is why the right-hand panel is against cumulative Krylov applications and not against the
sweep index. If the two panels disagree, the right-hand one is the answer.

`dmrg_cbe_exact` is drawn too where present: it is the A/B arm for the SELECTION, so it sets the
scale for what "no difference" looks like on this plot.
"""

import csv
import glob
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "results", "cbe_sweeps_l16_*gse*.csv")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")

STYLE = {
    "dmrg_cbe_rsvd":       ("#1f77b4", "o", "RSVD, start = zero-padded state"),
    "dmrg_cbe_rsvd_reuse": ("#d62728", "D", "RSVD, start = EXPANDED basis"),
    "dmrg_cbe_exact":      ("#7f7f7f", "s", "exact selection (A/B reference)"),
}


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                if r["suite"] != "gse":
                    continue
                try:
                    rows.append({
                        "scheme": r["scheme"], "cutoff": float(r["cutoff"]),
                        "sweep": float(r["x"]), "err": float(r["err"]),
                        "krylov": int(r["krylov"]),
                        "states": int(r["maxbond_states"]),
                    })
                except (KeyError, ValueError):
                    continue
    return rows


def main():
    paths = sorted(glob.glob(PAT)) if any(c in PAT for c in "*?[") else (
        [PAT] if os.path.exists(PAT) else [])
    paths = [p for p in paths if "SUPERSEDED" not in os.path.basename(p)]
    if not paths:
        sys.exit(f"no CSV matching {PAT} -- run the gse phase first")
    rows = load(paths)
    if not rows:
        sys.exit("no gse rows found")

    have = sorted({r["scheme"] for r in rows})
    print("schemes present:", ", ".join(have))
    if "dmrg_cbe_rsvd_reuse" not in have:
        # Say so rather than drawing a one-armed comparison that looks like a null result.
        # ASCII ONLY: the Windows console is cp1252, and printing a non-ASCII character raises
        # UnicodeEncodeError -- a warning that crashes instead of warning is worse than no warning.
        print("WARNING: dmrg_cbe_rsvd_reuse is ABSENT -- the seeded arm has not been run yet, so this "
              "figure cannot answer the question it exists to ask.")

    cutoffs = sorted({r["cutoff"] for r in rows})
    fig, axes = plt.subplots(len(cutoffs), 2, figsize=(11.0, 4.2 * len(cutoffs)),
                             squeeze=False)

    for ci, cut in enumerate(cutoffs):
        axL, axR = axes[ci][0], axes[ci][1]
        for scheme in have:
            g = sorted((r for r in rows if r["scheme"] == scheme and r["cutoff"] == cut),
                       key=lambda r: r["sweep"])
            g = [r for r in g if math.isfinite(r["err"]) and r["err"] > 0]
            if not g:
                continue
            c, m, lab = STYLE.get(scheme, ("k", "x", scheme))
            axL.semilogy([r["sweep"] for r in g], [r["err"] for r in g],
                         color=c, marker=m, ms=5, lw=1.6, markerfacecolor="none", label=lab)
            # ⛔ `krylov` IS PER-SWEEP, NOT CUMULATIVE. The driver does
            # `push!(kd, info.krylov_dims)` once per sweep (dmrg_cbe1s.jl:311), and the L=16 run
            # reads 63, 172, 138, 59 -- it goes DOWN as the state converges and the eigensolves
            # exit sooner. Plotting error against it directly draws a zig-zag on a non-monotonic
            # axis, which looks like a curve and means nothing. Accumulate to get real cost.
            cum, tot = [], 0
            for r in g:
                tot += r["krylov"]
                cum.append(tot)
            axR.semilogy(cum, [r["err"] for r in g],
                         color=c, marker=m, ms=5, lw=1.6, markerfacecolor="none", label=lab)

        axL.set_xlabel("sweep")
        axR.set_xlabel("cumulative operator applications")
        for ax, t in ((axL, f"error vs SWEEP  (tol={cut:g})"),
                      (axR, f"error vs COST  (tol={cut:g})")):
            ax.set_ylabel(r"$|E - E_{\rm Bethe}|$")
            ax.set_title(t, fontsize=10)
            ax.grid(alpha=0.3, lw=0.5)
            ax.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(OUTDIR, "cbe_l16_gse_seed.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
