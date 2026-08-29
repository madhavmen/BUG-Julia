#!/usr/bin/env python3
"""Fig. 3 of arXiv:2209.00739 -- ground-state energy per site vs J2, triangular J1-J2.

The paper's panel is E0/N against J2 for two cylinder circumferences with variational-QMC stars
for the infinite system. This reproduces that layout from whatever
`paper2209_gs_*_<Lx>x<Ly>.csv` files are present.

STARS ARE THERMODYNAMIC-LIMIT VALUES AND NO FINITE CLUSTER SITS ON THEM. They are drawn because
the paper draws them -- as the target an extrapolation in circumference should approach from
above, not as a per-run accuracy target. The only number a single finite run can be scored on is
the 6x6 TORUS exact-diagonalisation value, which is drawn only when a torus run is present.

Usage:  python3 benchmarks/ground_state/plot_fig3_2209.py [results_dir]
"""
import csv
import glob
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Published anchors, quoted from Iqbal, Hu, Thomale, Poilblanc & Becca, PRB 93, 144411 (2016),
# Tables I and IV. Nothing computed by this repository belongs in this dict.
VMC_INF = {0.0: (-0.545321, 7e-6), 0.125: (-0.51235, 2.0e-4)}
ED_TORUS36 = {0.0: -0.5603734, 0.125: -0.515564}

ARM_STYLE = {"full": dict(marker="o", ls="-", label="CBE-DMRG, full complement"),
             "rsvd": dict(marker="s", ls="--", label="CBE-DMRG, rSVD sketch")}


def load(d):
    rows = []
    for f in sorted(glob.glob(os.path.join(d, "paper2209_gs_*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                r["_file"] = os.path.basename(f)
                rows.append(r)
    return rows


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")
    rows = [r for r in load(d) if r["lattice"] == "triangular"]
    if not rows:
        sys.exit("no triangular paper2209_gs_*.csv rows found in %s" % d)

    # One curve per (geometry, arm). The geometry label carries the CIRCUMFERENCE, because that is
    # the axis the paper's two curves differ along -- Ly, not the site count.
    geos = sorted({(int(r["Lx"]), int(r["Ly"]), r["torus"] == "1") for r in rows})
    fig, axes = plt.subplots(1, 2, figsize=(12.5, 5.0),
                             gridspec_kw=dict(width_ratios=[1.45, 1]))
    ax, axd = axes

    colors = plt.cm.viridis([0.15, 0.55, 0.85])
    for gi, (lx, ly, tor) in enumerate(geos):
        for arm, st in ARM_STYLE.items():
            sel = [r for r in rows
                   if int(r["Lx"]) == lx and int(r["Ly"]) == ly
                   and (r["torus"] == "1") == tor and r["arm"] == arm]
            if not sel:
                continue
            sel.sort(key=lambda r: float(r["J2"]))
            xs = [float(r["J2"]) for r in sel]
            ys = [float(r["E_site"]) for r in sel]
            n = int(sel[0]["N"])
            lab = "C=%d, L=%d (N=%d)%s -- %s" % (
                ly, lx, n, " TORUS" if tor else "", st["label"].split(", ")[1])
            ax.plot(xs, ys, marker=st["marker"], ls=st["ls"], color=colors[gi % len(colors)],
                    mfc="none" if arm == "rsvd" else colors[gi % len(colors)],
                    ms=6, lw=1.6, label=lab)

    for j2, (e, err) in VMC_INF.items():
        ax.errorbar([j2], [e], yerr=[max(err, 1e-4)], marker="*", ms=18, color="crimson",
                    ls="none", zorder=5,
                    label="VMC, infinite 2D (Iqbal 2016)" if j2 == 0.0 else None)
    if any(r["torus"] == "1" for r in rows):
        for j2, e in ED_TORUS36.items():
            ax.plot([j2], [e], marker="X", ms=12, color="k", ls="none", zorder=6,
                    label="ED, 6x6 torus (exact)" if j2 == 0.0 else None)

    ax.set_xlabel(r"$J_2 / J_1$")
    ax.set_ylabel(r"$E_0 / N$")
    ax.set_title("Fig. 3: triangular $J_1$-$J_2$ ground-state energy per site")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="upper left")

    # Right panel: the two selections against each other. They are variational bounds on the SAME
    # space, so the lower one is the better state; a visible gap means neither has converged.
    for gi, (lx, ly, tor) in enumerate(geos):
        f = {float(r["J2"]): float(r["E_site"]) for r in rows
             if int(r["Lx"]) == lx and int(r["Ly"]) == ly and r["arm"] == "full"
             and (r["torus"] == "1") == tor}
        s = {float(r["J2"]): float(r["E_site"]) for r in rows
             if int(r["Lx"]) == lx and int(r["Ly"]) == ly and r["arm"] == "rsvd"
             and (r["torus"] == "1") == tor}
        common = sorted(set(f) & set(s))
        if common:
            axd.semilogy(common, [abs(f[j] - s[j]) + 1e-18 for j in common],
                         marker="o", ms=5, color=colors[gi % len(colors)],
                         label="C=%d, L=%d" % (ly, lx))
    axd.axhline(1e-8, color="k", ls=":", lw=1, label="agreement target")
    axd.set_xlabel(r"$J_2 / J_1$")
    axd.set_ylabel(r"$|E_{\rm full} - E_{\rm rSVD}| / N$")
    axd.set_title("selection A/B: full complement vs rSVD")
    axd.grid(alpha=0.3, which="both")
    axd.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(d, "fig3_2209.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
