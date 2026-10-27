"""dissipative_t7_parallel.py -- Task 7's efficiency/parallel subset of dissipative_cbe_bug_implementation_plan.md:
one point, BUG with cbe_bug_step!'s `parallel` = serial (julia -t 1), half_sweeps (-t 2) and full (the K/L/S
integrator, -t 3), over a dt ladder, all in one job on one node (benchmarks/dissipative_run.jl, DR_PARALLEL).

  Per (arm, dt): step wall (JIT warm-up excluded), process wall, CPU time (user + sys) and core-hours, MaxRSS
       from /usr/bin/time -v, D_final max, work-limited steps, and for `full` the K/L/S account (largest
       omitted block ||M11||/||psi||, fallbacks, new directions, K/L stage time)
  Against serial at the same dt: wall speedup, CPU and memory amplification, and max |difference| of log P0
       and the observables, which must lie below serial's dt-halving difference to count as matched accuracy
  Per arm: the dt-halving differences and the observed order log(d(4h, 2h) / d(2h, h)) / log 2 (3 expected)

usage: python dissipative_t7_parallel.py RUNDIR [OUTDIR]   (RUNDIR/out/<arm>_dt<dt>.json, RUNDIR/log/<same>.time)
"""
import glob
import json
import os
import sys

import numpy as np

from dissipative_t7_figures import diff

ARMS = ("serial", "half_sweeps", "full")


def gnu_time(path):
    """/usr/bin/time -v: (wall s, user + sys s, MaxRSS bytes)."""
    v = {}
    for line in open(path):
        k, _, x = line.strip().rpartition(": ")
        v[k] = x
    hms = [float(x) for x in v["Elapsed (wall clock) time (h:mm:ss or m:ss)"].split(":")]
    wall = sum(x * 60**i for i, x in enumerate(reversed(hms)))
    cpu = float(v["User time (seconds)"]) + float(v["System time (seconds)"])
    return wall, cpu, 1024 * int(v["Maximum resident set size (kbytes)"])


def load(rundir):
    runs = {}
    for f in glob.glob(os.path.join(rundir, "out", "*.json")):
        tag = os.path.basename(f)[:-len(".json")]
        with open(f) as io:
            r = json.load(io)
        tf = os.path.join(rundir, "log", tag + ".time")
        r["time"] = gnu_time(tf) if os.path.exists(tf) else None
        runs[(r["config"]["parallel"], r["config"]["dt"])] = r
    return runs


def maxd(a, b):
    _, lp, ob = diff(a, b)
    return max(float(lp.max()), float(ob.max()))


def account(r):
    s, nt = r["steplog"], r["config"]["threads"]
    item = [("threads", nt), ("steps", len(s["h"])), ("step wall s", round(sum(s["wall"]), 1)),
            ("D_final max", max(s["D_final"])), ("work-limited steps", int(sum(s.get("work_limited", [0])))),
            ("peak_rss GB", round(r["peak_rss"] / 2**30, 2))]
    if r["time"] is not None:
        w, cpu, rss = r["time"]
        item += [("process wall s", round(w, 1)), ("CPU s", round(cpu, 1)), ("CPU / wall", round(cpu / w, 2)),
                 ("core-h used / allocated", (round(cpu / 3600, 3), round(nt * w / 3600, 3))),
                 ("MaxRSS GB", round(rss / 2**30, 2))]
    if "kls_missing" in s:
        miss = np.array(s["kls_missing"])
        item += [("K/L/S steps", len(miss)), ("max ||M11||/||psi||", float(miss.max())),
                 ("steps with ||M11|| > 0", int(sum(miss > 0))), ("fallback steps", int(sum(s["kls_fallback"])))]
        if "kls_new_l" in s:        # per side; runs before it logged only the sum `kls_new`
            nl, nr = np.array(s["kls_new_l"]), np.array(s["kls_new_r"])
            item += [("steps with new directions on both sides", int(sum((nl > 0) & (nr > 0)))),
                     ("left only / right only", (int(sum((nl > 0) & (nr == 0))), int(sum((nl == 0) & (nr > 0))))),
                     ("mean new directions (l, r)", (round(float(nl.mean()), 2), round(float(nr.mean()), 2)))]
        else:
            new = np.array(s["kls_new"])
            item += [("steps with new directions", int(sum(new > 0))), ("mean new directions", round(float(new.mean()), 2))]
        item.append(("t_K + t_L s", round(sum(s["t_K"]) + sum(s["t_L"]), 1)))
    return item


def main():
    rundir = sys.argv[1]
    here = os.path.dirname(os.path.abspath(__file__))
    out = sys.argv[2] if len(sys.argv) > 2 else os.path.join(here, "..", "figures", "dissipative_t7", "parallel")
    os.makedirs(out, exist_ok=True)
    runs = load(rundir)
    dts = sorted({dt for _, dt in runs}, reverse=True)
    rows = []
    for arm in ARMS:
        for dt in dts:
            if (arm, dt) in runs:
                rows.append((f"{arm} dt={dt}", account(runs[(arm, dt)])))
    # the time-step error of each arm: successive dt halvings, and the observed order from two of them
    terr = {}
    for arm in ARMS:
        d = [(h, maxd(runs[(arm, h)], runs[(arm, h2)])) for h, h2 in zip(dts, dts[1:])
             if (arm, h) in runs and (arm, h2) in runs]
        for h, x in d:
            terr[(arm, h)] = x
        p = [round(float(np.log(a[1] / b[1]) / np.log(a[0] / b[0])), 2) for a, b in zip(d, d[1:])]
        rows.append((f"{arm} dt ladder", [("max |x(dt) - x(dt/2)|", dict(d)), ("observed order", p)]))
    # each parallel arm against serial at the same dt: cost ratios and matched accuracy
    for arm in ARMS[1:]:
        for dt in dts:
            r, b = runs.get((arm, dt)), runs.get(("serial", dt))
            if r is None or b is None:
                continue
            x = maxd(r, b)
            item = [("max |x - x_serial|", x), ("serial dt-halving difference", terr.get(("serial", dt))),
                    ("matched (below the time error)", x < terr[("serial", dt)] if ("serial", dt) in terr else None),
                    ("step-wall speedup", round(sum(b["steplog"]["wall"]) / sum(r["steplog"]["wall"]), 2))]
            if r["time"] is not None and b["time"] is not None:
                item += [("process-wall speedup", round(b["time"][0] / r["time"][0], 2)),
                         ("CPU amplification", round(r["time"][1] / b["time"][1], 2)),
                         ("memory amplification (MaxRSS)", round(r["time"][2] / b["time"][2], 2))]
            rows.append((f"{arm} vs serial dt={dt}", item))
    with open(os.path.join(out, "summary.txt"), "w") as io:
        for name, pts in rows:
            line = f"{name}: {pts}"
            print(line)
            io.write(line + "\n")
    print(f"{len(runs)} runs, summary in {os.path.abspath(out)}")


if __name__ == "__main__":
    main()
