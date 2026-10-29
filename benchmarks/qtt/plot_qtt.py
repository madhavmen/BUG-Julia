"""benchmarks/qtt/plot_qtt.py -- figures of the QTT plan from the drivers' CSVs.

usage: python plot_qtt.py KIND INDIR OUTDIR
  KIND = impressions (first_impressions.jl) | orders | tdvp (task2_study.jl) | ground (gp_ground_state.jl)
         | shedding (vortex_shedding.jl) | heston (heston_pilot.jl) | production (production.jl)
"""
import csv, os, sys, math
from collections import defaultdict
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def rows(path):
    with open(path) as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def save(fig, out, name):
    os.makedirs(out, exist_ok=True)
    fig.tight_layout()
    fig.savefig(os.path.join(out, name), dpi=150)
    plt.close(fig)
    print("wrote", os.path.join(out, name))


def impressions(indir, out):
    R = rows(os.path.join(indir, "qtt_cbe_nosym_ablation.csv"))
    for tag in ("active", "quiet"):
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        groups = defaultdict(list)
        for r in R:
            if r["tag"] == tag and r["trivial"] == "false":
                groups[(r["fixture"], r["arm"])].append(r)
        for (fx, arm), rs in sorted(groups.items()):
            x = [fnum(r["cols_med"]) for r in rs]
            y = [fnum(r["rhoL_med"]) for r in rs]
            t = [fnum(r["t_med"]) for r in rs]
            mk = "x" if any(fnum(r["false_accepts"]) > 0 for r in rs) else "o"
            ax[0].loglog(x, y, mk, label=f"{fx} {arm}")
            ax[1].loglog(y, t, mk, label=f"{fx} {arm}")
        ax[0].set_xlabel("streamed candidate + holdout columns")
        ax[0].set_ylabel("true DD residual rho_L (offline)")
        ax[1].set_xlabel("achieved rho_L")
        ax[1].set_ylabel("CBE call wall time [s]")
        ax[0].set_title(f"{tag} bonds (x = false accepts present)")
        ax[1].legend(fontsize=5, ncol=2)
        save(fig, out, f"impressions_{tag}.png")
    S = rows(os.path.join(indir, "steps.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for (fx, arm), rs in sorted(_group(S, ("fixture", "arm")).items()):
        wall = [sum(fnum(r["wall"]) for r in rs[: i + 1]) for i in range(len(rs))]
        ax[0].semilogy(wall, [fnum(r.get("err_vs_ref", "nan")) for r in rs], ".-", label=f"{fx} {arm}")
        ax[1].plot([fnum(r["t"]) for r in rs], [fnum(r["chi_max"]) for r in rs], ".-", label=f"{fx} {arm}")
    ax[0].set_xlabel("cumulative wall [s]"); ax[0].set_ylabel("error vs ref arm")
    ax[1].set_xlabel("t"); ax[1].set_ylabel("max chi"); ax[1].legend(fontsize=6)
    save(fig, out, "impressions_steps.png")


def _group(R, keys):
    g = defaultdict(list)
    for r in R:
        g[tuple(r[k] for k in keys)].append(r)
    return g


def orders(indir, out):
    R = rows(os.path.join(indir, "orders.csv"))
    for prob, rs in _group(R, ("problem",)).items():
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for (arm,), a in sorted(_group(rs, ("arm",)).items()):
            a = sorted(a, key=lambda r: fnum(r["h"]))
            ax[0].loglog([fnum(r["h"]) for r in a], [fnum(r["err_global"]) for r in a], "o-", label=arm)
            ax[1].loglog([fnum(r["wall"]) for r in a], [fnum(r["err_global"]) for r in a], "o-", label=arm)
        ax[0].set_xlabel("h"); ax[0].set_ylabel("global error at T"); ax[0].legend(fontsize=7)
        ax[1].set_xlabel("wall [s]"); ax[1].set_ylabel("global error")
        ax[0].set_title(prob[0])
        save(fig, out, f"orders_{prob[0]}.png")


def tdvp(indir, out):
    R = rows(os.path.join(indir, "tdvp.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    for (gam, g, arm), a in sorted(_group(R, ("gamma", "g", "arm")).items()):
        a = sorted(a, key=lambda r: fnum(r["h"]))
        lab = f"gamma={gam} g={g} {arm}"
        if arm.startswith("tdvp"):
            ax[0].semilogy([fnum(r["h"]) for r in a], [fnum(r["A_max"]) for r in a], "o-", label=lab)
        ax[1].loglog([fnum(r["h"]) for r in a], [fnum(r["err"]) for r in a], "o-", label=lab)
    ax[0].set_xlabel("h"); ax[0].set_ylabel("A_max (backward substep)"); ax[0].legend(fontsize=6)
    ax[1].set_xlabel("h"); ax[1].set_ylabel("full-step error at T"); ax[1].legend(fontsize=5)
    save(fig, out, "tdvp_mechanism.png")


def ground(indir, out):
    R = rows(os.path.join(indir, "history.csv"))
    for (case,), rs in _group(R, ("case",)).items():
        best = min(fnum(r["E"]) for r in rs)
        fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
        for (meth,), a in sorted(_group(rs, ("method",)).items()):
            a = [r for r in a if r.get("accepted", "true") == "true"]
            w = [fnum(r["wall"]) for r in a]
            ax[0].semilogy(w, [max(fnum(r["E"]) - best, 1e-16) for r in a], ".-", label=meth)
            ax[1].semilogy(w, [fnum(r["R_mu"]) for r in a], ".-", label=meth)
        ax[0].set_xlabel("wall [s]"); ax[0].set_ylabel("E - E_best")
        ax[1].set_xlabel("wall [s]"); ax[1].set_ylabel("R_mu"); ax[1].legend(fontsize=6)
        ax[0].set_title(case)
        save(fig, out, f"ground_{case}.png")


def shedding(indir, out):
    R = rows(os.path.join(indir, "shedding.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    for (arm, h), a in sorted(_group(R, ("arm", "h")).items()):
        t = [fnum(r["t"]) for r in a]
        ax[0].semilogy(t, [fnum(r["err_state"]) for r in a], ".-", label=f"{arm} h={h}")
        ax[1].plot(t, [fnum(r["vortices_plus"]) + fnum(r["vortices_minus"]) for r in a], ".-")
        ax[2].plot(t, [fnum(r["chi_max"]) for r in a], ".-")
    a = R[: len(set(r["t"] for r in R))]
    ax[1].plot([fnum(r["t"]) for r in a], [fnum(r["ref_vortices_plus"]) + fnum(r["ref_vortices_minus"]) for r in a], "k--", label="reference")
    ax[0].set_ylabel("state error vs reference"); ax[1].set_ylabel("vortex count"); ax[2].set_ylabel("max chi")
    for x in ax:
        x.set_xlabel("t")
    ax[0].legend(fontsize=5); ax[1].legend(fontsize=6)
    save(fig, out, "shedding.png")


def heston(indir, out):
    F = rows(os.path.join(indir, "forward.csv"))
    fig, ax = plt.subplots(1, 2, figsize=(11, 4.2))
    t = [fnum(r["t"]) for r in F]
    for k in (1, 2, 3):
        ax[0].semilogy(t, [abs(fnum(r[f"moment{k}_qtt"]) / fnum(r[f"moment{k}_ref"]) - 1) + 1e-17 for r in F], label=f"moment {k}")
    ax[0].semilogy(t, [abs(fnum(r["mass"]) - 1) + 1e-17 for r in F], "k--", label="mass - 1")
    ax[1].plot(t, [fnum(r["chi"]) for r in F], ".-")
    ax[0].set_xlabel("t"); ax[0].set_ylabel("relative moment error"); ax[0].legend(fontsize=7)
    ax[1].set_xlabel("t"); ax[1].set_ylabel("max chi (rises and falls)")
    save(fig, out, "heston_forward.png")


def production(indir, out):
    L = rows(os.path.join(indir, "ledger.csv"))
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    acc = [r for r in L if r["accepted"] == "true"]
    t = [fnum(r["t"]) for r in acc]
    ax[0].plot(t, [fnum(r["h"]) for r in acc], ".", ms=2)
    ax[1].plot(t, [fnum(r["chi_max"]) for r in acc], ".", ms=2)
    ax[2].plot(t, [fnum(r["maxrss_gb"]) for r in acc], ".", ms=2)
    for x, lab in zip(ax, ("accepted h", "max chi", "peak RSS [GB]")):
        x.set_xlabel("t"); x.set_ylabel(lab)
    save(fig, out, "production.png")


if __name__ == "__main__":
    kind, indir, out = sys.argv[1:4]
    {"impressions": impressions, "orders": orders, "tdvp": tdvp, "ground": ground,
     "shedding": shedding, "heston": heston, "production": production}[kind](indir, out)
