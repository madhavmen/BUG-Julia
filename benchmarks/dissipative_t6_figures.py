"""dissipative_t6_figures.py -- the Task 6 (G3) figures of dissipative_cbe_bug_implementation_plan.md
from the JSON runs of benchmarks/dissipative_run.jl. The time-to-accuracy curves (E 1 / ref, F01 and
F10 cost), the tol_trunc ladder (E 2t against E 2base, F02), the controller pilot (E 3, F03) and the
stability runs (E 4, F04) are dissipative_figures.py's; this script runs those and adds:

  F09  bond growth at a binding cap (E 6b, 6p, 6c): D_final(t) and BUG's working D against t, the
       cap, and the accumulated discarded norm, BUG and TDVP2 of one point in one panel
  F10s per-step phase times (DD sketch, QR, local Krylov, root, compression, environments, closure) against
       the step's D_final, with D^2 / D^3 guides (plan Task 6 "measure before scaling")
  F07  B1 pilot (E 6t): charge currents on both sink-adjacent bonds against t, and their time
       average over the second half of the run against kappa, +-2 standard errors
  F08  B2 pilot: conditional decay rate lambda(t) = -d log P0/dt (E 6n, no_jump) and the mean jump
       count (E 6t, traj)
  F05  A1 kappa scan (E 7k): late-time conditional loss rate and doublon number against kappa, which
       locates the Zeno crossover for the Task 7 A1 point
  F11  trajectory-count convergence (E 6t): the running mean and its 95% interval of the final lost
       particles (B1: 2 x jumps) or emitted quanta (B2: jumps) against the number of trajectories

usage: python dissipative_t6_figures.py RUNDIR[,RUNDIR...] [OUTDIR]   (OUTDIR default ../figures/dissipative_t6)
"""
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dissipative_figures as t5

# t_closure (the order-p frontier sweeps) is in none of the other timers (driver header)
PHASES = ("t_dd", "t_qr", "t_kry", "t_root", "t_trunc", "t_env", "t_closure")


def point(r):
    """The physical point of a run, without the integrator and its settings."""
    c = r["config"]
    return (c["model"], c["L"], c["start"], json.dumps({k: v for k, v in c["params"].items()
                                                         if k != "prep"}, sort_keys=True))


def step_times(r):
    s = r["steplog"]
    return np.cumsum(s["h"])


def fig_bonds(runs, out):
    rs = [r for r in runs if r["E"] in ("6b", "6p", "6c") and r["config"]["mode"] == "no_jump"]
    if not rs:
        return []
    groups = {}
    for r in rs:
        groups.setdefault(point(r) + (r["config"]["maxdim"],), []).append(r)
    fig, axs = plt.subplots(2, len(groups), figsize=(5 * len(groups), 7), squeeze=False)
    rows = []
    for j, (g, sel) in enumerate(sorted(groups.items())):
        for r in sel:
            c, s, t = r["config"], r["steplog"], step_times(r)
            lab = f"{c['int']} dt={c['dt']}" + (" ctrl" if c.get("target") not in (None, "none") else "")
            ln, = axs[0, j].plot(t, s["D_final"], "-", label=f"{lab} D_final")
            if "D_stage" in s:
                axs[0, j].plot(t, s["D_stage"], ":", color=ln.get_color(), label=f"{lab} D_stage")
            axs[1, j].semilogy(t, np.maximum(np.cumsum(s["discarded"]), 1e-18), "-", color=ln.get_color(), label=lab)
            cap = t5.capped_at(r)
            rows.append((f"F09 {t5.label(r)} maxdim {c['maxdim']} {lab}",
                         [("D_final max", max(s["D_final"])), ("first capped step ends", cap),
                          ("discarded total", float(np.sum(s["discarded"]))),
                          ("wall", round(sum(s["wall"]))), ("peak_rss GB", round(r["peak_rss"] / 2**30, 2))], []))
        axs[0, j].axhline(g[-1], color="k", lw=0.6)
        axs[0, j].set_title(f"{g[0]} L={g[1]} {g[2]} maxdim {g[-1]}", fontsize=8)
        axs[0, j].set_ylabel("bond dimension")
        axs[1, j].set_ylabel("accumulated discarded norm")
        for ax in axs[:, j]:
            ax.set_xlabel("t")
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F09_bonds.png"), dpi=130)
    plt.close(fig)
    return rows


def fig_scaling(runs, out):
    rs = [r for r in runs if r["E"] in ("6b", "6p") and r["config"]["mode"] == "no_jump"]
    if not rs:
        return []
    fig, axs = plt.subplots(1, len(rs), figsize=(4.5 * len(rs), 4), squeeze=False)
    rows = []
    for ax, r in zip(axs[0], rs):
        s, c = r["steplog"], r["config"]
        D = np.array(s["D_final"], dtype=float)
        keep = np.arange(len(D)) > 0                  # step 1 carries compilation
        names = [p for p in PHASES if p in s] if c["int"] == "bug" else []
        for p in names + ["wall"]:
            y = np.array(s[p], dtype=float)
            m = keep & (y > 0) & (D > 1)
            ax.loglog(D[m], y[m], "." if p != "wall" else "k+", label=p)
            if p in ("t_dd", "wall") and m.sum() >= 3:
                big = m & (D >= 0.25 * D.max())
                if big.sum() >= 3:
                    sl = np.polyfit(np.log(D[big]), np.log(y[big]), 1)[0]
                    rows.append((f"F10s {t5.label(r)} {c['int']} maxdim {c['maxdim']} {p}",
                                 [("slope vs D (D >= max/4)", round(float(sl), 2))], []))
        d = np.array([D[keep].min(), D.max()]) if keep.any() else np.array([1, 2])
        y0 = np.median(np.array(s["wall"])[keep]) if keep.any() else 1.0
        for k, ls in ((2, "--"), (3, ":")):
            ax.loglog(d, y0 * (d / d[-1]) ** k, "k" + ls, lw=0.6, label=f"D^{k}")
        ax.set_title(f"{c['model']} L={c['L']} {c['int']} maxdim {c['maxdim']}", fontsize=8)
        ax.set_xlabel("D_final of the step")
        ax.set_ylabel("seconds per step")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F10s_scaling.png"), dpi=130)
    plt.close(fig)
    return rows


def col(r, name):
    k = r["names"].index(name)
    return np.array(r["mean"])[:, k], np.array(r["stderr"])[:, k]


def fig_b1(runs, out):
    rs = [r for r in runs if r["E"] == "6t" and r["config"]["model"] == "B1"]
    if not rs:
        return []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    rows, byU = [], {}
    for r in sorted(rs, key=lambda r: (r["config"]["params"]["U"], r["config"]["params"]["kappa"])):
        p, t = r["config"]["params"], np.array(r["times"])
        jl = [col(r, n) for n in ("jup_left", "jdn_left")]
        jr = [col(r, n) for n in ("jup_right", "jdn_right")]
        cl, sl = jl[0][0] + jl[1][0], np.hypot(jl[0][1], jl[1][1])
        cr, sr = jr[0][0] + jr[1][0], np.hypot(jr[0][1], jr[1][1])
        ln, = axs[0].plot(t, cl, "-", label=f"U={p['U']} k={p['kappa']} left")
        axs[0].fill_between(t, cl - 2 * sl, cl + 2 * sl, color=ln.get_color(), alpha=0.2)
        axs[0].plot(t, cr, "--", color=ln.get_color())
        late = t >= 0.5 * t[-1]
        n = late.sum()
        # the per-time standard errors are correlated in t: the bar of the average is the mean bar
        avg = [(c[late].mean(), s[late].mean()) for c, s in ((cl, sl), (cr, sr))]
        byU.setdefault(p["U"], []).append((p["kappa"], avg))
        rows.append((f"F07 B1 L={r['config']['L']} U={p['U']} J_link={p['J_link']} kappa={p['kappa']}",
                     [("ntraj", len(r["traj"])), ("late j_c left", avg[0]), ("late j_c right", avg[1]),
                      ("lost (2 x jumps) at T", (2 * r["jumps_mean"][-1], 2 * r["jumps_stderr"][-1])),
                      ("late points", int(n))], []))
    for U, pts in sorted(byU.items()):
        k = [x[0] for x in pts]
        for side, mk in ((0, "o-"), (1, "s--")):
            axs[1].errorbar(k, [x[1][side][0] for x in pts], yerr=[2 * x[1][side][1] for x in pts],
                            fmt=mk, label=f"U={U} {'left' if side == 0 else 'right'}")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("charge current j_c (solid left, dashed right of the sink)")
    axs[1].set_xscale("symlog", linthresh=0.5)
    axs[1].set_xlabel("kappa")
    axs[1].set_ylabel("j_c averaged over the second half (+-2 se)")
    for ax in axs:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F07_b1.png"), dpi=130)
    plt.close(fig)
    return rows


def fig_b2(runs, out):
    nj = [r for r in runs if r["E"] == "6n" and r["config"]["model"] == "B2"]
    tj = [r for r in runs if r["E"] == "6t" and r["config"]["model"] == "B2"]
    if not nj and not tj:
        return []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    rows = []
    for r in nj:
        c, t, lp = r["config"], np.array(r["times"]), np.array(r["logp"])
        lam = -np.gradient(lp, t)
        Qc = np.array([x["Q"] for x in r["records"]])     # <Q>_c = -d log P0/dt, recorded directly
        ln, = axs[0].plot(t, Qc, "-", label=f"L={c['L']} nexc={c['params']['nexc']} G={c['params']['gamma_plus']} {c['int']}")
        axs[0].plot(t, lam, "x", color=ln.get_color(), ms=3)
        cap = t5.capped_at(r)
        # identity per record interval: -dlogP0 / dt against the trapezoid mean of <Q>_c. The first
        # intervals hold the bright burst (<Q>_c drops from G*nexc to ~1e-2 by t ~ 0.5, dr6k3), which
        # the record grid cannot resolve (mismatch 10-80% there, <= 0.8% from t = 1 on).
        fd, trap = -np.diff(lp) / np.diff(t), 0.5 * (Qc[1:] + Qc[:-1])
        rel = np.abs(fd - trap) / np.maximum(np.abs(trap), 1e-300)
        late = t[:-1] >= 1.0
        rows.append((f"F08 B2 conditional L={c['L']} nexc={c['params']['nexc']} {json.dumps(c['params'])}",
                     [("lambda(T)", float(Qc[-1])), ("log P0(T)", float(lp[-1])),
                      ("first-interval -dlogP0/dt vs trapezoid <Q>_c", (float(fd[0]), float(trap[0]))),
                      ("max rel identity mismatch for t >= 1", float(np.max(rel[late])) if late.any() else None),
                      ("first capped step ends", cap)], []))
    for r in tj:
        c, t = r["config"], np.array(r["times"])
        m, s = np.array(r["jumps_mean"]), np.array(r["jumps_stderr"])
        ln, = axs[1].plot(t, m, "-", label=f"L={c['L']} nexc={c['params']['nexc']} G={c['params']['gamma_plus']}")
        axs[1].fill_between(t, m - 2 * s, m + 2 * s, color=ln.get_color(), alpha=0.2)
        rows.append((f"F08 B2 traj L={c['L']} {json.dumps(c['params'])}",
                     [("ntraj", len(r["traj"])), ("emitted at T", (float(m[-1]), float(s[-1])))], []))
    axs[0].set_yscale("log")
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("conditional rate <Q>_c (x: -d log P0/dt)")
    axs[1].set_xlabel("t")
    axs[1].set_ylabel("mean emitted quanta (+-2 se)")
    for ax in axs:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F08_b2.png"), dpi=130)
    plt.close(fig)
    return rows


def fig_kappa(runs, out):
    """F05 pilot (E 7k): the A1 conditional loss rate <Q>_c(t) per kappa, and its late-time average
    (second half) and the conditional doublon number against kappa. The late rate peaks at the
    crossover from rapid doublon decay to Zeno-like slow dynamics, the Task 7 A1 point."""
    rs = sorted((r for r in runs if r["E"] == "7k" and r["config"]["model"] == "A1"),
                key=lambda r: r["config"]["params"]["kappa"])
    if not rs:
        return []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    rows, ks, late, dlate = [], [], [], []
    for r in rs:
        c, rec = r["config"], r["records"]
        t = np.array([x["t"] for x in rec])
        Q = np.array([x["Q"] for x in rec])
        nd = [i for i, n in enumerate(r["names"]) if n.startswith("dbl_")]
        dbl = np.array([sum(x["obs"][i] for i in nd) for x in rec])
        axs[0].semilogy(t, Q, "-", label=f"kappa={c['params']['kappa']}")
        m = t >= 0.5 * t[-1]
        ks.append(c["params"]["kappa"]); late.append(Q[m].mean()); dlate.append(dbl[m].mean())
        rows.append((f"F05 A1 L={c['L']} U={c['params']['U']} kappa={c['params']['kappa']} {c['int']} maxdim {c['maxdim']}",
                     [("late <Q>_c", float(late[-1])), ("<Q>_c(T)", float(Q[-1])), ("late doublons", float(dlate[-1])),
                      ("log P0(T)", float(r["logp"][-1])), ("first capped step ends", t5.capped_at(r)),
                      ("wall", round(sum(r["steplog"]["wall"])))], []))
    axs[1].loglog(ks, late, "o-", label="late <Q>_c")
    axs[1].loglog(ks, dlate, "s--", label="late conditional doublons")
    k = int(np.argmax(late))
    rows.append(("F05 A1 kappa scan", [("kappa of the largest late <Q>_c", ks[k])], []))
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("conditional loss rate <Q>_c = -d log P0/dt")
    axs[1].set_xlabel("kappa")
    axs[1].set_ylabel("average over the second half")
    for ax in axs:
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F05_kappa.png"), dpi=130)
    plt.close(fig)
    return rows


def fig_samples(runs, out):
    rs = [r for r in runs if r["E"] == "6t" and r["traj"] and "njumps_t" in r["traj"][0]]
    if not rs:
        return []
    fig, ax = plt.subplots(figsize=(6, 4))
    rows = []
    for r in rs:
        c = r["config"]
        f = 2.0 if c["model"] == "B1" else 1.0           # B1 loses a pair per jump
        x = f * np.array([tr["njumps"] for tr in r["traj"]], dtype=float)
        n = np.arange(1, len(x) + 1)
        m = np.cumsum(x) / n
        sd = np.array([x[:k].std(ddof=1) if k > 1 else np.nan for k in n])
        ln, = ax.plot(n, m, "-", label=f"{c['model']} L={c['L']} {json.dumps(c['params'])[:40]}")
        ax.fill_between(n, m - 1.96 * sd / np.sqrt(n), m + 1.96 * sd / np.sqrt(n), color=ln.get_color(), alpha=0.2)
        rows.append((f"F11 {c['model']} L={c['L']} {json.dumps(c['params'])}",
                     [("ntraj", len(x)), ("mean", float(m[-1])), ("95% half-width", float(1.96 * sd[-1] / np.sqrt(len(x)))),
                      ("ntraj for a 5% half-width", int(np.ceil((1.96 * sd[-1] / (0.05 * m[-1])) ** 2)) if m[-1] > 0 else None)], []))
    ax.set_xlabel("trajectories")
    ax.set_ylabel("lost particles / emitted quanta at T (95% band)")
    ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F11_samples.png"), dpi=130)
    plt.close(fig)
    return rows


def main():
    rundirs = sys.argv[1].split(",")
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "figures", "dissipative_t6")
    os.makedirs(out, exist_ok=True)
    runs = t5.load(rundirs)
    refs = t5.references(runs)
    nojump = [r for r in runs if r["config"]["mode"] == "no_jump"]
    rows = []
    for f in (lambda: t5.fig_order(nojump, refs, out), lambda: t5.fig_tolerances(nojump, refs, out),
              lambda: t5.fig_control(nojump, refs, out), lambda: t5.fig_cost(nojump, refs, out),
              lambda: t5.fig_stability(nojump, refs, out),
              lambda: fig_bonds(runs, out), lambda: fig_scaling(runs, out), lambda: fig_b1(runs, out),
              lambda: fig_b2(runs, out), lambda: fig_kappa(runs, out), lambda: fig_samples(runs, out)):
        rows += f() or []
    with open(os.path.join(out, "summary.txt"), "w") as io:
        for name, pts, sl in rows:
            line = f"{name}: {pts}" + (f"  slopes {np.round(sl, 2).tolist()}" if len(sl) else "")
            print(line)
            io.write(line + "\n")
    print(f"{len(runs)} runs, figures in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
