"""MATCHED-ERROR GRADING of the correlator records `bug_profile_2d.jl` writes with BP_OBS=1
(simulation_plan_2 §8 "Matched-error performance protocol"; Phase 6 pilot).

The reference is the MEAN of two independent higher-rank runs (e.g. tdvp2e and bug2 at D=1024)
and its uncertainty their disagreement, max_x |G_a - G_b|, per time: with no exact solution at
40 sites, an arm's error is only resolved where it exceeds that uncertainty, and the table says
so. Each test arm's error is max_x |G(x,t) - G_ref(x,t)| on the reference's time grid; its cost
is the wall per unit simulated time from bp_cost_*.csv (steps only -- the observable is untimed).
Errors are absolute: G(c,0) = 1/4 sets the scale, and a relative error would divide by
correlators that pass through zero.

Run:  python3 matched_error.py --ref A.csv B.csv --cost bp_cost_10x4.csv TEST.csv [TEST.csv ...]
      (--times 1,2,5,10 picks the reported times; TEST files are bp_G_<Lx>x<Ly>_<label>_D<D>.csv)
"""
import argparse
import csv
import os
import re
from collections import defaultdict


def load_g(path):
    """{t: {site: complex}} from one bp_G CSV."""
    g = defaultdict(dict)
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            g[round(float(r["t"]), 6)][int(r["site"])] = complex(float(r["ReG"]), float(r["ImG"]))
    return g


def maxdiff(a, b):
    return max(abs(a[x] - b[x]) for x in a)


def wall_per_time(cost_path):
    """{(D, label): (wall per unit time, JIT overhead)} with label as bug_profile_2d's `label`
    (search/tol/fold). Step 0 is the timed warm-up: its excess over the mean step is the JIT
    overhead, reported beside the steady-state rate rather than folded into it."""
    acc = defaultdict(lambda: [0.0, 0.0, 0, None])
    with open(cost_path, newline="") as f:
        for r in csv.DictReader(f):
            if r["search"]:
                lab = "%s_%s_tb%.0e%s" % (r["arm"], r["search"], float(r["tol_basis"]),
                                          "_fold" if r["fold"] == "true" else "")
            else:
                lab = r["arm"]
            a = acc[(int(r["D"]), lab)]
            if int(r["step"]) == 0:
                a[3] = float(r["wall"])
                continue
            a[0] += float(r["wall"])
            a[1] += float(r["dt"])
            a[2] += 1
    out = {}
    for k, (w, t, n, warm) in acc.items():
        if t > 0:
            out[k] = (w / t, max(warm - w / n, 0.0) if warm is not None else None)
    return out


def parse_name(path):
    m = re.match(r"bp_G_\d+x\d+_(.+)_D(\d+)\.csv$", os.path.basename(path))
    if not m:
        raise SystemExit("not a bp_G record: %s" % path)
    return m.group(1), int(m.group(2))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", nargs=2, required=True)
    ap.add_argument("--cost", nargs="*", default=[])
    ap.add_argument("--times", default="1,2,5,10")
    ap.add_argument("tests", nargs="+")
    a = ap.parse_args()
    ga, gb = load_g(a.ref[0]), load_g(a.ref[1])
    ts = sorted(set(ga) & set(gb))
    ref = {t: {x: (ga[t][x] + gb[t][x]) / 2 for x in ga[t]} for t in ts}
    unc = {t: maxdiff(ga[t], gb[t]) / 2 for t in ts}
    want = [float(s) for s in a.times.split(",")]
    pick = [min(ts, key=lambda t: abs(t - w)) for w in want]
    walls = {}
    for c in a.cost:
        walls.update(wall_per_time(c))

    print("reference: mean of %s and %s, %d times up to t=%.2f" %
          (os.path.basename(a.ref[0]), os.path.basename(a.ref[1]), len(ts), ts[-1]))
    print("reference uncertainty max_x|G_a-G_b|/2: " +
          "  ".join("t=%.1f %.2e" % (t, unc[t]) for t in pick) +
          "  (max over t %.2e)" % max(unc.values()))
    print("%-24s %5s %12s %9s %s %12s  %s" % ("arm", "D", "wall/unit t", "JIT", " ".join(
        "%10s" % ("err t=%.1f" % t) for t in pick), "max err", "resolved"))
    for path in a.tests:
        lab, D = parse_name(path)
        g = load_g(path)
        common = [t for t in ts if t in g]
        err = {t: maxdiff(g[t], ref[t]) for t in common}
        emax = max(err.values())
        res = "yes" if emax > 3 * max(unc[t] for t in common) else "NO (within 3x ref unc.)"
        w, jit = walls.get((D, lab), (None, None))
        print("%-24s %5d %12s %9s %s %12.3e  %s" % (
            lab, D, "%.1f s" % w if w is not None else "n/a",
            "%.1f s" % jit if jit is not None else "n/a",
            " ".join("%10s" % ("%.2e" % err[t] if t in err else "-") for t in pick), emax, res))


if __name__ == "__main__":
    main()
