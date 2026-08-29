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

# Both the DMRG and the ED drivers write energies as `%.10f`, so a difference below this is not
# resolvable from the files and must not be plotted as if it were.
CSV_RES = 1e-10

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


def load_exact(d):
    """Exact ground energies of the SAME cluster, keyed (Lx, Ly, torus, J2).

    At these sizes this -- not the thermodynamic-limit star -- is the reference a single run is
    scored against, so the deviation panel is against ED wherever ED exists.
    """
    ex = {}
    for f in sorted(glob.glob(os.path.join(d, "paper2209_exact_*.csv"))):
        with open(f) as fh:
            for r in csv.DictReader(fh):
                ex[(int(r["Lx"]), int(r["Ly"]), r["torus"] == "1", round(float(r["J2"]), 4))] = r
    return ex


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

    # Exact ground energies of the same clusters, where available. Drawn as a black line on the
    # left panel: at these sizes THIS is the reference, and the gap to it is the accuracy claim.
    exact = load_exact(d)
    for gi, (lx, ly, tor) in enumerate(geos):
        pts = sorted((j2, float(r["E0_site"]), int(r["dimer_is_gs"]))
                     for (a, b, t, j2), r in exact.items()
                     if a == lx and b == ly and t == tor)
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "-", color="k", lw=2.2,
                    alpha=0.8, zorder=1,
                    label="exact (ED, same cluster)" if gi == 0 else None)
    # ⛔ RE-BUILD THE LEGEND: it was drawn before these lines existed, so the exact curve -- the
    # one the whole right panel is scored against -- was on the plot but absent from the key.
    ax.legend(fontsize=7.5, loc="upper left")

    # Right panel: deviation from EXACT where we have it, else the two selections against each
    # other. A variational solve cannot sit below exact, so anything at or under zero is a bug and
    # anything far above it is a solve that did not converge -- including a FROZEN one.
    drew = False
    for gi, (lx, ly, tor) in enumerate(geos):
        for arm, st in ARM_STYLE.items():
            pts = []
            for r in rows:
                if (int(r["Lx"]), int(r["Ly"])) != (lx, ly) or r["arm"] != arm:
                    continue
                if (r["torus"] == "1") != tor:
                    continue
                k = (lx, ly, tor, round(float(r["J2"]), 4))
                if k in exact:
                    pts.append((float(r["J2"]),
                                float(r["E_site"]) - float(exact[k]["E0_site"])))
            if pts:
                pts.sort()
                drew = True
                # ⛔ FLOOR AT THE CSV's OWN RESOLUTION, NOT AT ZERO. Both files store %.10f, so
                # when the rounded values agree the difference is exactly 0 and a tiny epsilon
                # would draw it at 1e-18 -- claiming eight orders of magnitude more precision
                # than the data carries. Everything on the floor line means "equal to the last
                # digit written", which is all that can be said.
                axd.semilogy([p[0] for p in pts], [max(abs(p[1]), CSV_RES) for p in pts],
                             marker=st["marker"], ls=st["ls"],
                             color=colors[gi % len(colors)],
                             mfc="none" if arm == "rsvd" else colors[gi % len(colors)],
                             ms=6, label="C=%d, L=%d -- %s" % (ly, lx, arm))
    if drew:
        axd.axhline(CSV_RES, color="k", ls=":", lw=1,
                    label=r"CSV resolution ($10^{-10}$/site)")
        axd.set_ylim(CSV_RES / 3, None)
        axd.set_ylabel(r"$|E_{\rm DMRG} - E_{\rm exact}| / N$")
        axd.set_title("accuracy against exact diagonalisation\n"
                      "(on the dotted line = equal to the last digit stored)")
    else:
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
        axd.set_ylabel(r"$|E_{\rm full} - E_{\rm rSVD}| / N$")
        axd.set_title("selection A/B (no ED available)")
    axd.set_xlabel(r"$J_2 / J_1$")
    axd.grid(alpha=0.3, which="both")
    axd.legend(fontsize=8)

    fig.tight_layout()
    out = os.path.join(d, "fig3_2209.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)


if __name__ == "__main__":
    main()
