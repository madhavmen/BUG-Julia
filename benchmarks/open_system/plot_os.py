"""OS-1/2/3 figures (imaginary-time / open-system plan, Tasks 5-7) from the drivers' CSVs.

    python plot_os.py DIR os1 L8_g0 [--profile m] [--scalar j] [--tb 1e-05] [--seed 1] [--dt 0.1]
    python plot_os.py DIR os2 L4_n2_U2_g1_domain --profile n --scalar N
    python plot_os.py DIR os3 L8_g0.1_dw_weak --profile n --scalar autocorr

Row 1  the profile heatmap (site x time) of the first BUG arm, its |error| against the exact
       reference, and the reference itself (plan: a resolved space-time heatmap)
Row 2  the scalar observable vs t (log time) for every arm and the reference; max error vs t
Row 3  chi(t), |Tr rho - 1| before renormalisation, tdvp2's backward amplification
Steps whose ledger was not met are marked with an open ring: a failure, not a converged point.
"""
import argparse
import csv
import os
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100"]
ARM_ORDER = ["bug2", "bug3", "bug4", "tdvp2"]
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def style(ax, title=None):
    ax.grid(True, color=GRID, linewidth=1)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelsize=8)
    if title:
        ax.set_title(title, fontsize=9, color=INK, loc="left")


def key_of(r, a):
    return r["arm"] == "tdvp2" or (r["tol_basis"] == a.tb and r["seed"] == a.seed)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("prefix", choices=("os1", "os2", "os3"))
    ap.add_argument("tag")
    ap.add_argument("--profile", default="m")
    ap.add_argument("--scalar", default="j")
    ap.add_argument("--tb", default="1e-05")
    ap.add_argument("--seed", default="1")
    ap.add_argument("--dt", default=None)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()

    obs = defaultdict(lambda: defaultdict(dict))      # arm -> (obs, i) -> {t: (value, ref)}
    with open(os.path.join(a.dir, f"{a.prefix}_obs_{a.tag}.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if (a.dt and fnum(r["dt"]) != fnum(a.dt)) or not key_of(r, a):
                continue
            obs[r["arm"]][(r["obs"], int(r["i"]))][fnum(r["t"])] = (fnum(r["value"]), fnum(r["ref"]))
    steps = defaultdict(list)
    with open(os.path.join(a.dir, f"{a.prefix}_steps_{a.tag}.csv"), newline="") as f:
        for r in csv.DictReader(f):
            if (a.dt and fnum(r["dt"]) != fnum(a.dt)) or not key_of(r, a):
                continue
            steps[r["arm"]].append(r)
    arms = [x for x in ARM_ORDER if x in obs]
    if not arms:
        raise SystemExit("no rows match --tb/--seed/--dt")

    fig, axes = plt.subplots(3, 3, figsize=(16, 12))
    # row 1: heatmaps of the first BUG arm (or tdvp2), its error, the reference
    arm0 = next((x for x in arms if x.startswith("bug")), arms[0])
    prof = sorted(i for (o, i) in obs[arm0] if o == a.profile)
    ts = sorted(obs[arm0][(a.profile, prof[0])])
    val = [[obs[arm0][(a.profile, i)][t][0] for t in ts] for i in prof]
    ref = [[obs[arm0][(a.profile, i)][t][1] for t in ts] for i in prof]
    err = [[abs(v - r) for v, r in zip(vr, rr)] for vr, rr in zip(val, ref)]
    ext = (ts[0], ts[-1], prof[0] - 0.5, prof[-1] + 0.5)
    for ax, data, ttl, cm in ((axes[0][0], val, f"{a.profile}_i(t)  {arm0}", "viridis"),
                              (axes[0][1], err, f"|{a.profile} - exact|  {arm0}", "magma"),
                              (axes[0][2], ref, f"{a.profile}_i(t)  exact", "viridis")):
        im = ax.imshow(data, aspect="auto", origin="lower", extent=ext, cmap=cm,
                       interpolation="nearest")
        fig.colorbar(im, ax=ax, fraction=0.046)
        style(ax, ttl)
        ax.grid(False)
        ax.set_xlabel("t", fontsize=8, color=MUTED)
        ax.set_ylabel("site / bond", fontsize=8, color=MUTED)

    # row 2: the scalar vs t, and the worst error over all observables vs t
    for c, arm in zip(SLOTS, ARM_ORDER):
        if arm not in obs:
            continue
        idx = sorted(i for (o, i) in obs[arm] if o == a.scalar)
        if idx:
            i = idx[len(idx) // 2]                     # the middle bond / the single entry
            tt = sorted(t for t in obs[arm][(a.scalar, i)] if t > 0)
            axes[1][0].plot(tt, [obs[arm][(a.scalar, i)][t][0] for t in tt], color=c,
                            linewidth=2, label=arm)
            if arm == arm0:
                axes[1][0].plot(tt, [obs[arm][(a.scalar, i)][t][1] for t in tt], color=INK,
                                linewidth=1, linestyle="--", label="exact")
        tt = sorted({t for d in obs[arm].values() for t in d if t > 0})
        worst = [max(abs(d[t][0] - d[t][1]) for d in obs[arm].values() if t in d) for t in tt]
        axes[1][1].plot(tt, worst, color=c, linewidth=2, label=arm)
    style(axes[1][0], f"{a.scalar}(t)")
    axes[1][0].set_xscale("log")
    style(axes[1][1], "max |error| over all observables")
    axes[1][1].set_xscale("log"); axes[1][1].set_yscale("log")
    axes[1][2].axis("off")

    # row 3: chi(t), trace drift, tdvp2 backward amplification
    for c, arm in zip(SLOTS, ARM_ORDER):
        rs = steps.get(arm)
        if not rs:
            continue
        t = [fnum(r["t"]) for r in rs]
        series = ((axes[2][0], [fnum(r["chi"]) for r in rs]),
                  (axes[2][1], [abs(fnum(r["trace_raw"]) - 1) for r in rs]),
                  (axes[2][2], [fnum(r["backamp"]) for r in rs]))
        bad = [r["status"] == "budget_unmet" for r in rs]
        for ax, y in series:
            ax.plot(t, y, color=c, linewidth=2, label=arm)
            fx = [xx for xx, b in zip(t, bad) if b]
            fy = [yy for yy, b in zip(y, bad) if b]
            if fx:
                ax.plot(fx, fy, linestyle="none", marker="o", markersize=7,
                        markerfacecolor="white", markeredgecolor=c, markeredgewidth=2)
    style(axes[2][0], "chi(t)")
    style(axes[2][1], "|Tr rho - 1| before renormalisation")
    axes[2][1].set_yscale("log")
    style(axes[2][2], "tdvp2 backward amplification")
    for ax in (axes[1][0], axes[1][1], axes[2][0], axes[2][1], axes[2][2]):
        ax.set_xlabel("t", fontsize=8, color=MUTED)
        ax.legend(fontsize=7, frameon=False, labelcolor=INK)
    fig.suptitle(f"{a.prefix} {a.tag}  (tol_basis {a.tb}, seed {a.seed}; open ring = unmet step)",
                 fontsize=11, color=INK, x=0.01, ha="left")
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    out = a.out or os.path.join(a.dir, f"{a.prefix}_{a.tag}_tb{a.tb}.png")
    fig.savefig(out, dpi=130, facecolor="#fcfcfb")
    print("wrote", out)


if __name__ == "__main__":
    main()
