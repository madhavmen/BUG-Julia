"""dissipative_t4_figures.py -- the B1 physics smoke plots of dissipative_cbe_bug_implementation_plan.md
Task 4, from the JSON files of benchmarks/dissipative_b1_smoke.jl.

  b1_currents   j_c = j↑ + j↓ and j_s = j↑ - j↓ on the sink's left and right bonds against t,
                one curve per kappa, one panel row per (U, J_link)
  b1_counts     lost particles (2 x mean jump count) and transmitted particles N_R(t) - N_R(0)
  b1_doublons   the final on-site doublon profile <n↑n↓>_j per kappa
  b1_imbalance  the lead imbalances N_L - N_R and S^z_L - S^z_R against t
  b1_depletion  the left-bond charge current against the left lead's population N_L: a current
                that only follows the reservoir's depletion lies on the kappa = 0 curve

Error bars are one standard error over independent trajectories (two-sided bands of +-2 se).
usage: python dissipative_t4_figures.py RUNDIR [OUTDIR]  (OUTDIR default ../figures/dissipative_t4)
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(rundir):
    runs = []
    for f in sorted(glob.glob(os.path.join(rundir, "b1_smoke_*.json"))):
        with open(f) as io:
            r = json.load(io)
        r["mean"], r["stderr"] = np.array(r["mean"]), np.array(r["stderr"])
        r["idx"] = {n: k for k, n in enumerate(r["names"])}
        runs.append(r)
    return runs


def col(r, name):
    k = r["idx"][name]
    return r["mean"][:, k], r["stderr"][:, k]


def groups(runs):
    g = {}
    for r in runs:
        c = r["config"]
        g.setdefault((c["U"], c["J_link"]), []).append(r)
    return {k: sorted(v, key=lambda r: r["config"]["kappa"]) for k, v in sorted(g.items())}


def band(ax, t, m, s, **kw):
    line, = ax.plot(t, m, **kw)
    ax.fill_between(t, m - 2 * s, m + 2 * s, color=line.get_color(), alpha=0.2, lw=0)


def combo(r, a, b, sign):
    ma, sa = col(r, a)
    mb, sb = col(r, b)
    # the components are correlated within a trajectory; the summed band is conservative
    return ma + sign * mb, sa + sb


def fig_currents(g, out):
    rows = len(g)
    fig, axs = plt.subplots(rows, 4, figsize=(14, 2.8 * rows), squeeze=False)
    for i, ((U, Jl), rs) in enumerate(g.items()):
        for r in rs:
            t = np.array(r["times"])
            lab = f"κ = {r['config']['kappa']:g}"
            for j, (side, sign, what) in enumerate([("left", 1, "j_c"), ("left", -1, "j_s"),
                                                    ("right", 1, "j_c"), ("right", -1, "j_s")]):
                m, s = combo(r, f"jup_{side}", f"jdn_{side}", sign)
                band(axs[i, j], t, m, s, label=lab)
                axs[i, j].set_title(f"{what}, {side} sink bond, U = {U:g}, J_link = {Jl:g}", fontsize=9)
        axs[i, 0].legend(fontsize=7)
    for ax in axs[-1]:
        ax.set_xlabel("J t")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "b1_currents.png"), dpi=130)
    plt.close(fig)


def fig_counts(g, out):
    fig, axs = plt.subplots(1, len(g), figsize=(4.5 * len(g), 3.4), squeeze=False)
    for i, ((U, Jl), rs) in enumerate(g.items()):
        ax = axs[0, i]
        for r in rs:
            t = np.array(r["times"])
            k = r["config"]["kappa"]
            band(ax, t, 2 * np.array(r["jumps_mean"]), 2 * np.array(r["jumps_stderr"]),
                 label=f"lost, κ = {k:g}")
            m, s = col(r, "N_R")
            band(ax, t, m - m[0], s, ls="--", label=f"N_R(t) - N_R(0), κ = {k:g}")
        ax.set_title(f"U = {U:g}, J_link = {Jl:g}", fontsize=9)
        ax.set_xlabel("J t")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "b1_counts.png"), dpi=130)
    plt.close(fig)


def fig_doublons(g, out):
    fig, axs = plt.subplots(1, len(g), figsize=(4.5 * len(g), 3.2), squeeze=False)
    for i, ((U, Jl), rs) in enumerate(g.items()):
        ax = axs[0, i]
        for r in rs:
            L = r["config"]["L"]
            m = np.array([r["mean"][-1, r["idx"][f"dbl_{j}"]] for j in range(1, L + 1)])
            s = np.array([r["stderr"][-1, r["idx"][f"dbl_{j}"]] for j in range(1, L + 1)])
            ax.errorbar(np.arange(1, L + 1), m, yerr=2 * s, marker="o", capsize=2,
                        label=f"κ = {r['config']['kappa']:g}")
        ax.axvline(rs[0]["config"]["j0"], color="k", lw=0.6, ls=":")
        ax.set_title(f"<n↑n↓>_j at J t = {rs[0]['config']['T']:g}, U = {U:g}, J_link = {Jl:g}", fontsize=9)
        ax.set_xlabel("site j")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "b1_doublons.png"), dpi=130)
    plt.close(fig)


def fig_imbalance(g, out):
    fig, axs = plt.subplots(2, len(g), figsize=(4.5 * len(g), 5.6), squeeze=False)
    for i, ((U, Jl), rs) in enumerate(g.items()):
        for r in rs:
            t = np.array(r["times"])
            lab = f"κ = {r['config']['kappa']:g}"
            band(axs[0, i], t, *combo(r, "N_L", "N_R", -1), label=lab)
            band(axs[1, i], t, *combo(r, "Sz_L", "Sz_R", -1), label=lab)
        axs[0, i].set_title(f"N_L - N_R, U = {U:g}, J_link = {Jl:g}", fontsize=9)
        axs[1, i].set_title(f"S^z_L - S^z_R", fontsize=9)
        axs[1, i].set_xlabel("J t")
        axs[0, i].legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "b1_imbalance.png"), dpi=130)
    plt.close(fig)


def fig_depletion(g, out):
    fig, axs = plt.subplots(1, len(g), figsize=(4.5 * len(g), 3.4), squeeze=False)
    for i, ((U, Jl), rs) in enumerate(g.items()):
        ax = axs[0, i]
        for r in rs:
            nl, _ = col(r, "N_L")
            jc, _ = combo(r, "jup_left", "jdn_left", 1)
            ax.plot(nl, jc, marker=".", label=f"κ = {r['config']['kappa']:g}")
        ax.set_title(f"left-bond j_c against N_L, U = {U:g}, J_link = {Jl:g}", fontsize=9)
        ax.set_xlabel("N_L(t)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "b1_depletion.png"), dpi=130)
    plt.close(fig)


def main():
    rundir = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "figures", "dissipative_t4")
    os.makedirs(out, exist_ok=True)
    runs = load(rundir)
    if not runs:
        sys.exit(f"no b1_smoke_*.json in {rundir}")
    g = groups(runs)
    for f in (fig_currents, fig_counts, fig_doublons, fig_imbalance, fig_depletion):
        f(g, out)
    for r in runs:
        c = r["config"]
        print(f"U {c['U']:+g} J_link {c['J_link']:g} kappa {c['kappa']:g}: jumps "
              f"{r['jumps_mean'][-1]:.3f} ± {r['jumps_stderr'][-1]:.3f}, prep E {r['prep']['energy'][-1]:.10f} "
              f"(failed {r['prep']['failed']}), flags {r['norm_growth_flags']}, wall {r['wall']:.0f} s")
    print(f"figures in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
