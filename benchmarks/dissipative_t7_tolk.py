"""dissipative_t7_tolk.py -- the recycled local Krylov tolerance (cbe_bug_step!'s tol_krylov, USER 2026-10-27)
at one point of dissipative_cbe_bug_implementation_plan.md: BUG at one dt over a tol_krylov ladder ("none" =
tol_basis, the default) plus the default at dt/2 as the reference, all in one job on one node
(benchmarks/dissipative_run.jl, DR_TOLK).

  Per arm: max |difference| of log P0 and the observables against the reference, next to the default's own
       (time step plus, when work-limited, the ceiling's cut), so a looser tol_krylov is accepted while it
       stays below that; the trial frame
       D_stage, the kept Krylov directions n_krylov, work-limited steps, D_final, matvecs, step wall, closure
       time, process wall and MaxRSS from /usr/bin/time -v
  F12  error against the reference and step wall against tol_krylov

  Several run directories (comma-separated, the same point on the same start state) are pooled; the
  arms between the ladder's dt and the reference's are listed against the reference too.

usage: python dissipative_t7_tolk.py RUNDIR[,RUNDIR...] [OUTDIR]   (RUNDIR/out/k<K>_dt<dt>.json, RUNDIR/log/<same>.time)
"""
import glob
import json
import os
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from dissipative_t7_figures import diff
from dissipative_t7_parallel import gnu_time


def load(rundir):
    runs = {}
    for f in glob.glob(os.path.join(rundir, "out", "*.json")):
        tag = os.path.basename(f)[:-len(".json")]
        with open(f) as io:
            r = json.load(io)
        tf = os.path.join(rundir, "log", tag + ".time")
        r["time"] = gnu_time(tf) if os.path.exists(tf) else None
        runs[(str(r["config"]["tol_krylov"]), r["config"]["dt"])] = r
    return runs


def total(x):
    return float(np.sum(np.concatenate([np.ravel(v) for v in x]))) if len(x) else 0.0


def account(r, ref):
    s = r["steplog"]
    t, lp, ob = diff(r, ref)
    item = [("t0 |x - ref|", max(float(lp[0]), float(ob[0]))), ("max |x - ref|", max(float(lp.max()), float(ob.max()))),
            ("steps", len(s["h"])), ("D_stage max / mean", (max(s["D_stage"]), round(float(np.mean(s["D_stage"])), 1))),
            ("n_krylov mean", round(float(np.mean(s["n_krylov"])), 1)), ("n_selected mean", round(float(np.mean(s["n_selected"])), 1)),
            ("work-limited steps", int(sum(s["work_limited"]))), ("D_final max", max(s["D_final"])),
            ("matvecs", int(total(s["matvecs"]))), ("step wall s", round(sum(s["wall"]), 1)),
            ("t_closure s", round(sum(s["t_closure"]), 1)), ("peak_rss GB", round(r["peak_rss"] / 2**30, 2))]
    if r["time"] is not None:
        item += [("process wall s", round(r["time"][0], 1)), ("MaxRSS GB", round(r["time"][2] / 2**30, 2))]
    return item


def main():
    rundir = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "figures", "dissipative_t7", "tol_krylov")
    os.makedirs(out, exist_ok=True)
    runs = {}
    for d in rundir.split(","):
        runs.update(load(d))
    dt = max(d for _, d in runs)
    ref = runs[("none", min(d for _, d in runs))]
    tolb = ref["config"]["tol_basis"]
    arms = sorted((k for k, d in runs if d == dt), key=lambda k: tolb if k == "none" else float(k))
    rows, pts = [], []
    for k in arms:
        r = runs[(k, dt)]
        item = account(r, ref)
        rows.append((f"tol_krylov={k} dt={dt}", item))
        pts.append((tolb if k == "none" else float(k), dict(item)["max |x - ref|"], dict(item)["step wall s"]))
    terr = dict(rows[0][1])["max |x - ref|"]           # the default's own difference at this dt
    rows.append(("accepted (below the default's error)", [(k, dict(it)["max |x - ref|"] <= terr) for (k, it) in
                 zip(arms, (it for _, it in rows))]))
    for k, d in sorted(((k, d) for k, d in runs if dt > d > ref["config"]["dt"]), key=lambda x: (-x[1], x[0])):
        rows.append((f"tol_krylov={k} dt={d}", account(runs[(k, d)], ref)))
    rows.append((f"reference tol_krylov=none dt={ref['config']['dt']}", account(ref, ref)[2:]))
    with open(os.path.join(out, "summary.txt"), "w") as io:
        for name, p in rows:
            line = f"{name}: {p}"
            print(line)
            io.write(line + "\n")
    x, e, w = (np.array(v) for v in zip(*pts))
    fig, axs = plt.subplots(1, 2, figsize=(10, 4))
    axs[0].loglog(x, e, "o-")
    axs[0].axhline(terr, ls=":", c="k", label="default")
    axs[0].set(xlabel="tol_krylov", ylabel="max |x - x_ref|", title=f"error vs dt={ref['config']['dt']} reference")
    axs[0].legend()
    axs[1].semilogx(x, w, "o-")
    axs[1].set(xlabel="tol_krylov", ylabel="step wall s", title=f"BUG dt={dt}, tol_basis={tolb}")
    fig.tight_layout()
    fig.savefig(os.path.join(out, "F12_tol_krylov.png"), dpi=120)
    print(f"{len(runs)} runs, summary in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
