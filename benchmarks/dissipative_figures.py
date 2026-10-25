"""dissipative_figures.py -- the Task 5 figures of dissipative_cbe_bug_implementation_plan.md
(F01-F04, F10) from the JSON runs of benchmarks/dissipative_run.jl (one array directory: out/<k>.json
plus configs.txt, whose E= tag names the experiment).

  F01  error at fixed T against dt, BUG (order 3) and TDVP2, with dt^3 / dt^2 guides; the order-3
       ablation (lossless; tol_basis, tol_trunc, tol_root loosened one at a time)
  F02  tol_basis / tol_root / tol_trunc ladders: error against the requested setting, frame width,
       root iterations, discarded norm, final rank, wall per step
  F03  the eq. (7) controller: accepted step against t, its error against the fixed-step reference
  F04  strong dissipation: error against dt and against t for BUG and TDVP2, the largest dt per
       error target, TDVP2's backward-substep amplification, P0 monotonicity, and the energy balance
       (3) in integrated form relative to the reference on the same grid
  F10  cost at matched error: total wall against error, BUG and TDVP2

The error of a run is the largest |log P0 - log P0_ref| and the largest observable difference over the
common output times, against the matching reference run (E tag ending in "ref": same model, size,
parameters and start, the finest step) or, where the run carries one, against ED (err_state).
usage: python dissipative_figures.py RUNDIR[,RUNDIR...] [OUTDIR]   (OUTDIR default ../figures/dissipative_t5)
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def load(rundirs):
    """Runs of one or more array directories (a later array can reuse an earlier one's reference)."""
    runs = []
    for d, rundir in enumerate(rundirs):
        cfg = open(os.path.join(rundir, "configs.txt")).read().split("\n")
        for f in glob.glob(os.path.join(rundir, "out", "*.json")):
            k = int(os.path.basename(f).split(".")[0])
            with open(f) as io:
                r = json.load(io)
            line = cfg[k - 1].split()
            r["E"] = next((x[2:] for x in line if x.startswith("E=")), "")
            r["k"] = (d, k)
            r["dir"] = os.path.basename(os.path.normpath(rundir))
            runs.append(r)
    return sorted(runs, key=lambda r: r["k"])


def key(r):
    c = r["config"]
    p = json.dumps({k: v for k, v in c["params"].items() if k != "prep"}, sort_keys=True)
    return (c["model"], c["L"], c["start"], p, c["T"], c.get("warm", 0.0))


def obs_matrix(r):
    return np.array([x["obs"] for x in r["records"]])


def capped_at(r):
    """End time of the first fixed step that was rank-limited, or None: for BUG, the ledger missed
    its tolerance (`budget_met` = 0: the working-rank ceiling 2 maxdim bound); for TDVP2, which has
    no ledger, a bond reached `maxdim`. Plan U12 / §4.4: such a run is marked, not fitted silently
    (dr5e: the warm A1 L = 12 BUG errors were all capped steps)."""
    s = r.get("steplog", {})
    if "work_limited" in s:                       # runs that log StepRecord.limits: caps only
        flags = [bool(w or k) for w, k in zip(s["work_limited"], s["rank_limited"])]
    elif "budget_met" in s:                       # older runs: any missed budget
        flags = [not b for b in s["budget_met"]]
    else:
        flags = [d >= r["config"]["maxdim"] for d in s.get("D_final", [])]
    t = 0.0
    for h, f in zip(s.get("h", []), flags):
        t += h
        if f:
            return t
    return None


def error(r, ref, tmax=None):
    """(log P0 error, observable error) over the common output times (before `tmax` if given)."""
    t, tr = np.round(r["times"], 9), np.round(ref["times"], 9)
    common = [x for x in t if x in set(tr) and (tmax is None or x < tmax - 1e-9)]
    i = [list(t).index(x) for x in common]
    j = [list(tr).index(x) for x in common]
    lp = np.max(np.abs(np.array(r["logp"])[i] - np.array(ref["logp"])[j]))
    ob = np.max(np.abs(obs_matrix(r)[i] - obs_matrix(ref)[j]))
    return lp, ob


def references(runs):
    refs = {}
    for r in runs:
        if r["E"].endswith("ref"):
            refs[key(r)] = r
    return refs


def ref_error(r, refs, window=False):
    """Error against the reference; `window`: only the output times before the first capped step of
    the run or of its reference (None if no time after t = 0 is left)."""
    if "err_state" in r["records"][-1]:
        return r["records"][-1]["err_state"], abs(r["logp"][-1] - r["records"][-1]["logp_exact"])
    ref = refs.get(key(r))
    if ref is None or ref is r:
        return None
    if not window:
        return error(r, ref)
    caps = [c for c in (capped_at(r), capped_at(ref)) if c is not None]
    tmax = min(caps, default=None)
    if tmax is not None and sum(1 for x in r["times"] if 1e-9 < x < tmax - 1e-9) == 0:
        return None
    return error(r, ref, tmax)


def label(r):
    c = r["config"]
    w = f" warm {c['warm']:g}" if c.get("warm", 0) else ""
    return f"{c['model']} L={c['L']} {c['start']}{w} {json.dumps(c['params'])}"


def fig_order(runs, refs, out):
    groups = {}
    for r in runs:
        if r["E"] in ("1", "4"):
            groups.setdefault((r["E"],) + key(r), []).append(r)
    if not groups:
        return
    fig, axs = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    rows = []
    for ax, (g, rs) in zip(axs[0], sorted(groups.items())):
        for integ, mk in (("bug", "o-"), ("tdvp2", "s--")):
            sel = sorted((r for r in rs if r["config"]["int"] == integ), key=lambda r: r["config"]["dt"])
            sel = [r for r in sel if ref_error(r, refs) is not None]
            if not sel:
                continue
            dt = np.array([r["config"]["dt"] for r in sel])
            e = np.array([max(ref_error(r, refs)) for r in sel])
            cap = [capped_at(r) for r in sel]
            ln, = ax.loglog(dt, e, mk, label=integ)
            hollow = [i for i, c in enumerate(cap) if c is not None]
            ax.loglog(dt[hollow], e[hollow], "o", mfc="white", color=ln.get_color(),
                      label=f"{integ} rank-limited" if hollow else None)
            sl = np.diff(np.log2(e)) / np.diff(np.log2(dt))
            name = f"F{'01' if g[0] == '1' else '04'} {label(rs[0])} {integ}"
            capnote = "" if not hollow else "  RANK-LIMITED from t = " + str(
                {float(dt[i]): round(cap[i], 4) for i in hollow})
            rows.append((name + capnote, list(zip(dt, e)), list(sl)))
            win = [ref_error(r, refs, window=True) for r in sel]
            if hollow or any(capped_at(refs[key(r)]) is not None for r in sel if key(r) in refs):
                wpts = [(d, max(w)) for d, w in zip(dt, win) if w is not None]
                wsl = list(np.diff(np.log2([p[1] for p in wpts])) / np.diff(np.log2([p[0] for p in wpts])))
                rows.append((name + "  (before the first capped step)", wpts, wsl))
        d = np.array(sorted({r["config"]["dt"] for r in rs}))
        e0 = ax.get_ylim()[1] / 10
        ax.loglog(d, e0 * (d / d[-1]) ** 3, "k:", lw=0.8, label="dt^3")
        ax.loglog(d, e0 * (d / d[-1]) ** 2, "k-.", lw=0.8, label="dt^2")
        ax.set_title(label(rs[0]), fontsize=8)
        ax.set_xlabel("dt")
        ax.set_ylabel("max error (log P0, observables)")
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F01_F04_order.png"), dpi=130)
    plt.close(fig)
    return rows


def fig_ablation(runs, refs, out):
    ab = [r for r in runs if r["E"] == "1ab"]
    if not ab:
        return []
    groups = {}
    for r in ab:
        c = r["config"]
        arm = ("lossless" if c["exact"] else f"b{c['tol_basis']:g} r{c['tol_root']:g} t{c['tol_trunc']:g}")
        groups.setdefault(key(r), {}).setdefault(arm, []).append(r)
    fig, axs = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    rows = []
    for ax, (g, arms) in zip(axs[0], sorted(groups.items())):
        for arm, rs in sorted(arms.items()):
            pts = sorted((r["config"]["dt"], ref_error(r, refs)) for r in rs)
            pts = [(dt, e) for dt, e in pts if e is not None]
            if not pts:
                continue
            dt = np.array([p[0] for p in pts])
            e = np.array([max(p[1]) for p in pts])
            ax.loglog(dt, e, "o-", label=arm)
            rows.append((f"F01 ablation {g[0]} L={g[1]} {arm}", list(zip(dt, e)),
                         list(np.diff(np.log2(e)) / np.diff(np.log2(dt)))))
        ax.set_title(f"order-3 ablation, {g[0]} L={g[1]}", fontsize=8)
        ax.set_xlabel("dt")
        ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F01_ablation.png"), dpi=130)
    plt.close(fig)
    return rows


def same_dt_baseline(r, runs):
    """The tight-tolerance BUG run of the same point and step (E 1 or 2base): against it a ladder
    entry's error is its tolerance's own, not the time error the fine-step reference also sees
    (dr5b: the A1 L = 8 ladders were flat at the dt = 0.05 time error, 1.4e-6)."""
    c = r["config"]
    cands = [x for x in runs if x is not r and x["E"] in ("1", "2base") and key(x) == key(r)
             and x["config"]["int"] == "bug" and x["config"]["dt"] == c["dt"]]
    return min(cands, key=lambda x: x["config"]["tol_basis"], default=None)


def fig_tolerances(runs, refs, out):
    lad = {"2b": "tol_basis", "2r": "tol_root", "2t": "tol_trunc"}
    rs = [r for r in runs if r["E"] in lad]
    if not rs:
        return []

    def own_error(r):
        b = same_dt_baseline(r, runs)
        return max(error(r, b)) if b is not None else np.nan
    models = sorted({r["config"]["model"] for r in rs})
    fig, axs = plt.subplots(len(models), 6, figsize=(22, 3.6 * len(models)), squeeze=False)
    rows = []
    for i, m in enumerate(models):
        for tag, tol in lad.items():
            sel = sorted((r for r in rs if r["E"] == tag and r["config"]["model"] == m),
                         key=lambda r: r["config"][tol])
            if not sel:
                continue
            x = [r["config"][tol] for r in sel]
            err = [max(ref_error(r, refs) or (np.nan, np.nan)) for r in sel]
            own = [own_error(r) for r in sel]
            sl = [r["steplog"] for r in sel]
            series = [("error vs reference (o) / vs same-dt tight run (x)", err),
                      ("frame width r_aug", [np.mean(s.get("r_aug", [np.nan])) for s in sl]),
                      ("root iterations", [np.mean(s.get("expv_iters", [np.nan])) for s in sl]),
                      ("discarded", [np.max(s.get("discarded", [np.nan])) for s in sl]),
                      ("D_final", [np.max(s.get("D_final", [np.nan])) for s in sl]),
                      ("wall / step [s]", [np.mean(s.get("wall", [np.nan])[1:]) for s in sl])]
            for j, (name, y) in enumerate(series):
                axs[i, j].loglog(x, y, "o-", label=tol)
                axs[i, j].set_title(f"{m}: {name}", fontsize=8)
                axs[i, j].set_xlabel("requested tolerance")
            axs[i, 0].loglog(x, own, "x--", label=f"{tol}, same dt")
            rows.append((f"F02 {m} {tol} (vs reference, vs same-dt tight)", list(zip(x, err, own)), []))
        for ax in axs[i]:
            ax.legend(fontsize=6)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F02_tolerances.png"), dpi=130)
    plt.close(fig)
    return rows


def fig_control(runs, refs, out):
    rs = [r for r in runs if r["E"] == "3"]
    if not rs:
        return []
    fig, axs = plt.subplots(1, 2, figsize=(11, 4))
    rows = []
    for r in rs:
        c = r["config"]
        ch = r.get("checks", [])
        st = r.get("steps", [])                   # (t, h) of every accepted step
        e = ref_error(r, refs)
        unmet = sum(1 for b in r["steplog"].get("budget_met", []) if not b)
        lab = (f"{c['model']}{' warm' if c.get('warm', 0) else ''} tau={c['target']} "
               f"check {c['check_every']} [{r['dir']}]")
        ln, = axs[0].step([x[0] for x in st], [x[1] for x in st], where="post", label=lab)
        rej = [x for x in ch if not x["accepted"]]
        axs[0].plot([x["t"] for x in rej], [x["h"] for x in rej], "x", color=ln.get_color())
        axs[1].plot([c["target"]], [max(e) if e else np.nan], "o", label=lab)
        rows.append((f"F03 {lab}", [("error", max(e) if e else None), ("rejected", r.get("nreject")),
                                     ("checks", len(ch)), ("accepted", len(st)),
                                     ("h range", (min(x[1] for x in st), max(x[1] for x in st)) if st else None),
                                     ("estimates", [x["err"] for x in ch]),
                                     ("rank-limited steps (incl. check substeps)", unmet),
                                     ("wall", sum(r["steplog"]["wall"]))], []))
    axs[0].set_xlabel("t")
    axs[0].set_ylabel("accepted step (x: rejected attempt)")
    axs[0].set_yscale("log")
    axs[1].set_xscale("log")
    axs[1].set_yscale("log")
    axs[1].set_xlabel("target step error")
    axs[1].set_ylabel("error against the fixed-step reference")
    for ax in axs:
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F03_control.png"), dpi=130)
    plt.close(fig)
    return rows


def balance_residual(r, times=None):
    """Energy balance (3) in integrated form on the run's output grid (or the subset `times`):
    E(t) - E(0) - int_0^t (-Re<HQ>_c + E<Q>_c) by the trapezoid rule, as an array over the grid.
    On a coarse grid the quadrature error dominates; it is the same for a run and its reference on
    the same grid, so only their difference measures the integrator (dr5d: the finite-difference
    residual was 0.7-24, identical for BUG and TDVP2 and for every dt)."""
    rec = r["records"]
    t = np.round([x["t"] for x in rec], 9)
    keep = np.ones(len(t), bool) if times is None else np.isin(t, np.round(times, 9))
    t, rec = t[keep], [x for x, k in zip(rec, keep) if k]
    E = np.array([x["E"] for x in rec])
    rhs = np.array([-x["HQ"] + x["E"] * x["Q"] for x in rec])
    integ = np.concatenate([[0.0], np.cumsum(0.5 * (rhs[1:] + rhs[:-1]) * np.diff(t))])
    return t, E - E[0] - integ


def error_vs_time(r, ref):
    """(t, max over observables and log P0 of |run - reference|) on the common output times."""
    t, tr = np.round(r["times"], 9), np.round(ref["times"], 9)
    common = [x for x in t if x in set(tr)]
    i = [list(t).index(x) for x in common]
    j = [list(tr).index(x) for x in common]
    lp = np.abs(np.array(r["logp"])[i] - np.array(ref["logp"])[j])
    ob = np.max(np.abs(obs_matrix(r)[i] - obs_matrix(ref)[j]), axis=1)
    return np.array(common), np.maximum(lp, ob)


def fig_stability(runs, refs, out):
    """Plan Task 5 stability diagnostics on the E 4 runs: TDVP2's backward-substep amplification,
    the error against t (does it grow?), the largest dt meeting a fixed error, P0 monotonicity (the
    unnormalised norm may only fall) and the energy balance (3) relative to the reference."""
    rs = [r for r in runs if r["E"] == "4"]
    if not rs:
        return []
    fig, axs = plt.subplots(1, 4, figsize=(21, 4))
    rows, best = [], {}
    for r in sorted(rs, key=lambda r: (key(r), r["config"]["int"], r["config"]["dt"])):
        c = r["config"]
        ref = refs.get(key(r))
        amp = max(r["steplog"].get("backward_amp", [0.0]))
        lab = f"{c['model']} {json.dumps(c['params'])[:40]} {c['int']} dt={c['dt']}"
        n2 = np.array(r["steplog"].get("norm2", []))
        p0_up = int(np.sum(n2 > 1 + 1e-12))                   # a step that raised P0
        item = [("backward amp", amp), ("P0 growth flags", len(r.get("flags", []))),
                ("steps raising P0", p0_up), ("first capped step ends", capped_at(r))]
        if ref is not None and ref is not r:
            t, e = error_vs_time(r, ref)
            ln, = axs[1].semilogy(t, np.maximum(e, 1e-16), "-" if c["int"] == "bug" else "--", label=lab)
            _, res = balance_residual(r)
            _, rres = balance_residual(ref, r["times"])
            m = min(len(res), len(rres))
            item += [("max error", float(e.max())), ("error at T", float(e[-1])),
                     ("balance residual (own grid)", float(np.max(np.abs(res)))),
                     ("balance excess over reference", float(np.max(np.abs(res[:m] - rres[:m]))))]
            best.setdefault((key(r), c["int"]), []).append((c["dt"], float(e.max()), capped_at(r)))
        rows.append((f"F04 {lab}", item, []))
        if c["int"] == "tdvp2":
            axs[0].semilogy(c["dt"], amp, "s", label=lab)
    for (g, integ), pts in sorted(best.items()):
        for tol in (1e-3, 1e-4, 1e-6):
            ok = [d for d, e, cap in pts if e <= tol]
            rows.append((f"F04 largest dt with error <= {tol:g}: {g[0]} L={g[1]} {g[3]} {integ}",
                         [("dt", max(ok) if ok else None),
                          ("capped runs", [d for d, _, cap in pts if cap is not None])], []))
        d, e = zip(*sorted((p[0], p[1]) for p in pts))
        axs[2].loglog(d, e, "o-" if integ == "bug" else "s--", label=f"{g[0]} L={g[1]} {integ}")
    for r in refs.values():
        if any(key(r) == key(x) for x in rs):
            t, res = balance_residual(r)
            axs[3].semilogy(t[1:], np.maximum(np.abs(res[1:]), 1e-16), label=f"reference {label(r)[:40]}")
    axs[0].set_title("TDVP2 backward-substep amplification", fontsize=9)
    axs[0].set_xlabel("dt")
    axs[1].set_title("error against t (solid BUG, dashed TDVP2)", fontsize=9)
    axs[1].set_xlabel("t")
    axs[2].set_title("max error against dt", fontsize=9)
    axs[2].set_xlabel("dt")
    axs[3].set_title("energy balance (3), integrated, reference run", fontsize=9)
    axs[3].set_xlabel("t")
    for ax in axs:
        ax.legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F04_stability.png"), dpi=130)
    plt.close(fig)
    return rows


def fig_cost(runs, refs, out):
    rs = [r for r in runs if r["E"] == "1"]
    if not rs:
        return []
    groups = {}
    for r in rs:
        groups.setdefault(key(r), []).append(r)
    fig, axs = plt.subplots(1, len(groups), figsize=(5 * len(groups), 4), squeeze=False)
    rows = []
    for ax, (g, sel) in zip(axs[0], sorted(groups.items())):
        for integ, mk in (("bug", "o-"), ("tdvp2", "s--")):
            pts = sorted((max(ref_error(r, refs) or (np.nan,)), sum(r["steplog"]["wall"])) for r in sel
                         if r["config"]["int"] == integ)
            ax.loglog([p[0] for p in pts], [p[1] for p in pts], mk, label=integ)
            capd = sorted(r["config"]["dt"] for r in sel if r["config"]["int"] == integ and capped_at(r) is not None)
            rows.append((f"F10 {label(sel[0])} {integ}" + (f"  RANK-LIMITED at dt {capd}" if capd else ""), pts, []))
        ax.set_xlabel("error")
        ax.set_ylabel("integrator wall [s]")
        ax.set_title(f"cost at matched error, {g[0]} L={g[1]}", fontsize=8)
        ax.legend(fontsize=7)
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F10_cost.png"), dpi=130)
    plt.close(fig)
    return rows


def main():
    rundirs = sys.argv[1].split(",")
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "figures", "dissipative_t5")
    os.makedirs(out, exist_ok=True)
    runs = load(rundirs)
    refs = references(runs)
    rows = []
    for f in (lambda: fig_order(runs, refs, out), lambda: fig_ablation(runs, refs, out),
              lambda: fig_tolerances(runs, refs, out), lambda: fig_control(runs, refs, out),
              lambda: fig_stability(runs, refs, out), lambda: fig_cost(runs, refs, out)):
        rows += f() or []
    with open(os.path.join(out, "summary.txt"), "w") as io:
        for name, pts, sl in rows:
            line = f"{name}: {pts}" + (f"  slopes {np.round(sl, 2).tolist()}" if len(sl) else "")
            print(line)
            io.write(line + "\n")
    for r in runs:
        if "err_state" in r["records"][-1]:
            print(f"ED check {label(r)} {r['config']['int']}: err_state {r['records'][-1]['err_state']:.2e}")
    print(f"{len(runs)} runs, figures in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
