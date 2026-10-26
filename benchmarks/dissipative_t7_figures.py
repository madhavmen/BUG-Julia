"""dissipative_t7_figures.py -- the Task 7 (G4) truncation study of dissipative_cbe_bug_implementation_plan.md
from the JSON runs of benchmarks/dissipative_run.jl: tol_trunc in {1e-5, 1e-7, 1e-9} (E 7) at one point
per model with tol_basis, tol_root and dt fixed, plus a dt-halved arm at 1e-7 (E 7dt), for BUG and TDVP2,
and where the maxdim cap binds a larger-maxdim arm at 1e-7 (E 7cap) that measures the cap error.
Differences are also reported before the first capped step of either run.

  F09  per point: D_final(t) (solid) and BUG's trial frame D_stage (dotted) for the three tol_trunc, the
       accumulated discarded norm, the conditional rate <Q>_c(t) = -d log P0/dt, and the differences
       |run - tightest| of log P0 and the observables against t: each looser tol_trunc against 1e-9
       (compression), the dt-halved arm against its dt partner (time step), BUG against TDVP2
  Rows: the plan's Task 7 acceptance -- which compression differences lie above the time-step
       difference (resolved) and which below it (converged to within the time error) -- and the
       physics at T (log P0, <Q>_c, S, doublons for A1)

usage: python dissipative_t7_figures.py RUNDIR[,RUNDIR...] [OUTDIR]   (OUTDIR default ../figures/dissipative_t7)
"""
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

import dissipative_figures as t5
from dissipative_t6_figures import point, step_times

TOLS = (1e-5, 1e-7, 1e-9)


def diff(r, ref):
    """(t, |log P0 - ref|, max |obs - ref|) on the common output times."""
    t, tr = np.round(r["times"], 9), np.round(ref["times"], 9)
    common = [x for x in t if x in set(tr)]
    i = [list(t).index(x) for x in common]
    j = [list(tr).index(x) for x in common]
    lp = np.abs(np.array(r["logp"])[i] - np.array(ref["logp"])[j])
    ob = np.max(np.abs(t5.obs_matrix(r)[i] - t5.obs_matrix(ref)[j]), axis=1)
    return np.array(common), lp, ob


def window(r, ref):
    """The output times before the first capped step of either run (plan Task 7: a capped curve is
    marked, and its agreement with another is not truncation convergence)."""
    caps = [c for c in (t5.capped_at(r), t5.capped_at(ref)) if c is not None]
    return min(caps, default=np.inf)


def maxdiff(r, ref, tmax=np.inf):
    t, lp, ob = diff(r, ref)
    m = t < tmax - 1e-9
    return (float(lp[m].max()), float(ob[m].max())) if m.any() else None


def physics(r):
    rec, c = r["records"], r["config"]
    t = np.array([x["t"] for x in rec])
    Q = np.array([x["Q"] for x in rec])
    late = t >= 0.5 * t[-1]
    item = [("T", float(t[-1])), ("log P0(T)", float(r["logp"][-1])), ("<Q>_c(T)", float(Q[-1])),
            ("late <Q>_c", float(Q[late].mean())), ("S(T)", float(rec[-1]["S"]))]
    if c["model"] == "A1":
        nd = [i for i, n in enumerate(r["names"]) if n.startswith("dbl_")]
        item.append(("doublons(T)", float(sum(rec[-1]["obs"][i] for i in nd))))
    return item


def cost(r):
    s = r["steplog"]
    item = [("D_final max", max(s["D_final"])), ("discarded total", float(np.sum(s["discarded"]))),
            ("wall", round(sum(s["wall"]))), ("peak_rss GB", round(r["peak_rss"] / 2**30, 2)),
            ("first capped step ends", t5.capped_at(r))]
    if "rank_limited" in s:                 # the ledger's maxdim cap, apart from the working-rank ceiling
        item += [("rank-limited steps", int(sum(s["rank_limited"]))), ("work-limited steps", int(sum(s["work_limited"])))]
    return item


def fig_point(g, sel, out, idx):
    fig, axs = plt.subplots(1, 4, figsize=(22, 4.2))
    rows = []
    ladder = {(r["config"]["int"], r["config"]["tol_trunc"]): r for r in sel if r["E"] == "7"}
    checks = {r["config"]["int"]: r for r in sel if r["E"] == "7dt"}
    for (integ, tol), r in sorted(ladder.items()):
        c, s, t = r["config"], r["steplog"], step_times(r)
        ls = "-" if integ == "bug" else "--"
        lab = f"{integ} dt={c['dt']} tolT={tol:g}"
        ln, = axs[0].plot(t, s["D_final"], ls, label=lab)
        if "D_stage" in s:                    # the frame actually used; D_work is only its ceiling
            axs[0].plot(t, s["D_stage"], ":", color=ln.get_color())
        axs[1].semilogy(t, np.maximum(np.cumsum(s["discarded"]), 1e-18), ls, color=ln.get_color(), label=lab)
        tr = np.array([x["t"] for x in r["records"]])
        axs[2].semilogy(tr, np.maximum([x["Q"] for x in r["records"]], 1e-18), ls, color=ln.get_color(), label=lab)
        rows.append((f"F09 {t5.label(r)} {lab}", cost(r) + physics(r), []))
    # compression: each looser tol against the tightest of the same integrator
    for integ in ("bug", "tdvp2"):
        tight = ladder.get((integ, TOLS[-1]))
        if tight is None:
            continue
        item = []
        for tol in TOLS[:-1]:
            r = ladder.get((integ, tol))
            if r is None:
                continue
            t, lp, ob = diff(r, tight)
            axs[3].semilogy(t, np.maximum(np.maximum(lp, ob), 1e-16), "-" if integ == "bug" else "--",
                            label=f"{integ} tolT {tol:g} vs {TOLS[-1]:g}")
            item.append((f"tolT {tol:g} vs {TOLS[-1]:g} (logP0, obs)", (float(lp.max()), float(ob.max()))))
            w = window(r, tight)
            if np.isfinite(w):
                item.append((f"  same, before the first cap (t < {w:g})", maxdiff(r, tight, w)))
        chk, base = checks.get(integ), ladder.get((integ, 1e-7))
        if chk is not None and base is not None:
            t, lp, ob = diff(chk, base)
            axs[3].semilogy(t, np.maximum(np.maximum(lp, ob), 1e-16), ":", lw=2,
                            label=f"{integ} dt {chk['config']['dt']} vs {base['config']['dt']} (tolT 1e-7)")
            dt_err = max(float(lp.max()), float(ob.max()))
            # the compression differences over the same window as the dt check
            comp = {}
            for tol in TOLS[:-1]:
                r = ladder.get((integ, tol))
                if r is not None:
                    tc, lpc, obc = diff(r, tight)
                    m = tc <= t[-1] + 1e-9
                    comp[tol] = max(float(lpc[m].max()), float(obc[m].max()))
            item += [("dt-halved vs dt at 1e-7 (logP0, obs)", (float(lp.max()), float(ob.max()))),
                     ("  same, before the first cap", (window(chk, base), maxdiff(chk, base, window(chk, base)))),
                     ("dt check window T", float(t[-1])),
                     ("compression diffs in that window", comp),
                     # a tol_trunc whose difference sits below the dt error has converged to within it
                     ("compression diff resolved above the dt error", {k: v > dt_err for k, v in comp.items()})]
        # cap refinement (E 7cap): a larger maxdim at the same tol_trunc and dt measures the cap error
        for r in sel:
            if r["E"] == "7cap" and r["config"]["int"] == integ:
                base = ladder.get((integ, r["config"]["tol_trunc"]))
                if base is not None:
                    t, lp, ob = diff(r, base)
                    axs[3].semilogy(t, np.maximum(np.maximum(lp, ob), 1e-16), "-.", lw=1.5,
                                    label=f"{integ} maxdim {r['config']['maxdim']} vs {base['config']['maxdim']}")
                    item += [(f"maxdim {r['config']['maxdim']} vs {base['config']['maxdim']} at tolT "
                              f"{r['config']['tol_trunc']:g} (logP0, obs)", (float(lp.max()), float(ob.max()))),
                             (f"  maxdim {r['config']['maxdim']} first capped step ends", t5.capped_at(r))]
        rows.append((f"F09 {g[0]} L={g[1]} {integ} truncation study", item, []))
    # integrators against each other at matched tol_trunc
    for tol in TOLS:
        b, d = ladder.get(("bug", tol)), ladder.get(("tdvp2", tol))
        if b is not None and d is not None:
            t, lp, ob = diff(b, d)
            rows.append((f"F09 {g[0]} L={g[1]} BUG vs TDVP2 at tolT {tol:g}",
                         [("max (logP0, obs)", (float(lp.max()), float(ob.max()))),
                          ("before the first cap of either", (window(b, d), maxdiff(b, d, window(b, d)))),
                          ("wall BUG / TDVP2", round(sum(b["steplog"]["wall"]) / sum(d["steplog"]["wall"]), 2))], []))
            if tol == TOLS[-1]:
                axs[3].semilogy(t, np.maximum(np.maximum(lp, ob), 1e-16), "k-", lw=0.8, label=f"BUG vs TDVP2 tolT {tol:g}")
    # USER 10-26: a work-limited BUG step is a failed accuracy gate, so A1 is compared at matched error,
    # i.e. each cap-refinement arm against every arm of the other integrator at its tol_trunc
    caps = [r for r in sel if r["E"] == "7cap"]
    for r in caps:
        c = r["config"]
        rows.append((f"F09 {t5.label(r)} {c['int']} maxdim {c['maxdim']} dt={c['dt']} tolT={c['tol_trunc']:g}",
                     cost(r) + physics(r), []))
        other = [d for d in sel if d["config"]["int"] != c["int"] and d["E"] in ("7", "7cap")
                 and d["config"]["tol_trunc"] == c["tol_trunc"]]
        for d in other:
            t, lp, ob = diff(r, d)
            w = window(r, d)
            rows.append((f"F09 {g[0]} L={g[1]} {c['int']} maxdim {c['maxdim']} vs {d['config']['int']} maxdim "
                         f"{d['config']['maxdim']} at tolT {c['tol_trunc']:g}",
                         [("max (logP0, obs)", (float(lp.max()), float(ob.max()))),
                          ("before the first cap of either", (w, maxdiff(r, d, w))),
                          ("wall ratio", round(sum(r["steplog"]["wall"]) / sum(d["steplog"]["wall"]), 2))], []))
    axs[0].axhline(sel[0]["config"]["maxdim"], color="k", lw=0.6)
    axs[0].set_title(f"{g[0]} L={g[1]} {g[2]}: D_final (solid BUG, dashed TDVP2), BUG D_stage dotted", fontsize=8)
    axs[1].set_title("accumulated discarded norm", fontsize=8)
    axs[2].set_title("conditional rate <Q>_c = -d log P0/dt", fontsize=8)
    axs[3].set_title("max |difference| of log P0 and observables", fontsize=8)
    for ax in axs:
        ax.set_xlabel("t")
        ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"F09_t7_{g[0]}_L{g[1]}.png"), dpi=130)
    plt.close(fig)
    return rows


def same_record(a, b, tol):
    """Two trajectories of one seed took the same jumps: same channels in order, times within `tol`."""
    ea, eb = a["events"], b["events"]
    return len(ea) == len(eb) and all(x["channel"] == y["channel"] and abs(x["t"] - y["t"]) <= tol
                                      for x, y in zip(ea, eb))


def traj_diff(r, ref):
    """Ensemble and pathwise differences of two trajectory runs on their common times and seeds:
    max |mean - mean| over observables and the jump count, the same in units of the combined standard
    error, and over the seeds whose jump records agree (so the paths are comparable) max |obs - obs|.
    A different truncation or step can move a jump past a grid point; those seeds are only counted."""
    t, tr = np.round(r["times"], 9), np.round(ref["times"], 9)
    common = [x for x in t if x in set(tr)]
    i = [list(t).index(x) for x in common]
    j = [list(tr).index(x) for x in common]

    byseed = {x["seed"]: x for x in ref["traj"]}
    pairs = [(x, byseed[x["seed"]]) for x in r["traj"] if x["seed"] in byseed]

    def ens(trajs, k):
        # on the COMMON seeds only: a 5-trajectory cap arm against the first 5 of a 20-trajectory run
        a = np.array([np.column_stack([np.array(x["obs"])[k], np.array(x["njumps_t"])[k]]) for x in trajs])
        return a.mean(axis=0), a.std(axis=0, ddof=1) / np.sqrt(len(trajs)) if len(trajs) > 1 else 0 * a[0]
    (ma, sa), (mb, sb) = ens([x for x, _ in pairs], i), ens([y for _, y in pairs], j)
    d = np.abs(ma - mb)
    se = np.hypot(sa, sb)
    # t = 0 and conserved observables are deterministic: their spread is roundoff and carries no z
    z = d[se > 1e-10] / se[se > 1e-10]
    tol = max(r["config"]["dt"], ref["config"]["dt"]) + 1e-9
    match = [(x, y) for x, y in pairs if same_record(x, y, tol)]
    path = max((float(np.max(np.abs(np.array(x["obs"])[i] - np.array(y["obs"])[j]))) for x, y in match),
               default=None)
    return np.array(common), d.max(axis=1), {"max |d mean|": float(d.max()),
                                            "max z": float(z.max()) if z.size else None,
                                            "seeds matched": f"{len(match)}/{len(pairs)}", "max |d path|": path}


def traj_cost(r):
    s, nt = r["steplog"], len(r["traj"])
    n = len(s["h"])
    item = [("ntraj", nt), ("steps", n), ("D_final max", max(s["D_final"]))]
    if "work_limited" in s:                 # TDVP2 has no ledger; capped_at falls back to D_final >= maxdim
        item += [("rank-limited steps", int(sum(s["rank_limited"]))), ("work-limited steps", int(sum(s["work_limited"])))]
    # the steplog runs the trajectories in sequence, each n / nt fixed steps: the earliest capped step
    # over the trajectories is where the run's accuracy gate fails (plan §4.4)
    if n % nt == 0:
        caps = [t5.capped_at({"config": r["config"], "steplog": {k: v[i * (n // nt):(i + 1) * (n // nt)]
                                                                 for k, v in s.items()}}) for i in range(nt)]
        item += [("first capped step ends (min over trajectories)", min((c for c in caps if c is not None), default=None))]
    return item + [("wall", round(sum(s["wall"]))), ("peak_rss GB", round(r["peak_rss"] / 2**30, 2))]


def traj_physics(r):
    c, t = r["config"], np.array(r["times"])
    late = t >= 0.5 * t[-1]
    m, s = np.array(r["jumps_mean"]), np.array(r["jumps_stderr"])
    f = 2.0 if c["model"] == "B1" else 1.0           # B1 loses a pair per jump
    item = [("T", float(t[-1])), ("lost/emitted at T", (f * float(m[-1]), f * float(s[-1])))]
    if c["model"] == "B1":
        mean, se = np.array(r["mean"]), np.array(r["stderr"])
        for side in ("left", "right"):
            k = [r["names"].index(f"j{x}_{side}") for x in ("up", "dn")]
            jc, sc = mean[:, k].sum(axis=1), np.hypot(se[:, k[0]], se[:, k[1]])
            item.append((f"late j_c {side}", (float(jc[late].mean()), float(sc[late].mean()))))
    return item


def fig_traj(g, sel, out):
    """Task 7 on a trajectory ensemble (B1, B2 traj): the same ladder, dt and cap comparisons as
    `fig_point`, graded against the sampling error: a difference is resolved only above ~2 combined
    standard errors, and the pathwise difference on matched seeds measures it below that."""
    fig, axs = plt.subplots(1, 3, figsize=(17, 4.2))
    rows = []
    ladder = {(r["config"]["int"], r["config"]["tol_trunc"]): r for r in sel if r["E"] == "7"}
    checks = {r["config"]["int"]: r for r in sel if r["E"] == "7dt"}
    caps = [r for r in sel if r["E"] == "7cap"]
    for r in sorted(sel, key=lambda r: (r["config"]["int"], r["E"], r["config"]["tol_trunc"])):
        c = r["config"]
        lab = f"{c['int']} {r['E']} dt={c['dt']} tolT={c['tol_trunc']:g} maxdim {c['maxdim']}"
        t, m, s = np.array(r["times"]), np.array(r["jumps_mean"]), np.array(r["jumps_stderr"])
        ln, = axs[0].plot(t, m, "-" if c["int"] == "bug" else "--", label=lab)
        axs[0].fill_between(t, m - 2 * s, m + 2 * s, color=ln.get_color(), alpha=0.15)
        axs[2].plot(np.arange(len(r["steplog"]["D_final"])), r["steplog"]["D_final"], ",", color=ln.get_color())
        rows.append((f"F09 {t5.label(r)} traj {lab}", traj_cost(r) + traj_physics(r), []))

    def compare(name, a, b, ls):
        t, d, item = traj_diff(a, b)
        axs[1].semilogy(t, np.maximum(d, 1e-16), ls, label=name)
        rows.append((f"F09 {g[0]} L={g[1]} traj {name}", list(item.items()), []))
    for integ in ("bug", "tdvp2"):
        tight = ladder.get((integ, TOLS[-1]))
        for tol in TOLS[:-1]:
            if tight is not None and (integ, tol) in ladder:
                compare(f"{integ} tolT {tol:g} vs {TOLS[-1]:g}", ladder[(integ, tol)], tight, "-")
        base = ladder.get((integ, 1e-7))
        if base is not None and integ in checks:
            compare(f"{integ} dt {checks[integ]['config']['dt']} vs {base['config']['dt']}", checks[integ], base, ":")
        for r in caps:
            if r["config"]["int"] == integ and base is not None:
                compare(f"{integ} maxdim {r['config']['maxdim']} vs {base['config']['maxdim']}", r, base, "-.")
    for tol in TOLS:
        if ("bug", tol) in ladder and ("tdvp2", tol) in ladder:
            compare(f"BUG vs TDVP2 tolT {tol:g}", ladder[("bug", tol)], ladder[("tdvp2", tol)], "k-")
    axs[0].set_title(f"{g[0]} L={g[1]}: mean jumps (+-2 se)", fontsize=8)
    axs[1].set_title("max |difference of ensemble means| (observables, jumps)", fontsize=8)
    axs[2].set_title("D_final per step (all trajectories in sequence)", fontsize=8)
    for ax in axs[:2]:
        ax.legend(fontsize=5)
    axs[0].set_xlabel("t")
    axs[1].set_xlabel("t")
    axs[2].set_xlabel("step")
    fig.tight_layout()
    fig.savefig(os.path.join(out, f"F09_t7_{g[0]}_L{g[1]}_traj.png"), dpi=130)
    plt.close(fig)
    return rows


def main():
    rundirs = sys.argv[1].split(",")
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "figures", "dissipative_t7")
    os.makedirs(out, exist_ok=True)
    allruns = [r for r in t5.load(rundirs) if r["E"] in ("7", "7dt", "7cap")]
    runs = [r for r in allruns if r["config"]["mode"] == "no_jump"]
    groups, tgroups = {}, {}
    for r in runs:
        groups.setdefault(point(r), []).append(r)
    for r in allruns:
        if r["config"]["mode"] == "traj":
            tgroups.setdefault(point(r), []).append(r)
    rows = []
    for idx, (g, sel) in enumerate(sorted(groups.items())):
        rows += fig_point(g, sel, out, idx)
    for g, sel in sorted(tgroups.items()):
        rows += fig_traj(g, sel, out)
    with open(os.path.join(out, "summary.txt"), "w") as io:
        for name, pts, _ in rows:
            line = f"{name}: {pts}"
            print(line)
            io.write(line + "\n")
    print(f"{len(allruns)} runs, figures in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
