"""BOND GROWTH AND ERROR VS EXACT, per arm, from `fig45_2209.jl` records at a non-binding cap
(agent_handoff §5.2: CBE-BUG vs tdvp2e at the same `tol_trunc`, cap above the reachable rank).

Reads, for one tag, `fig45_G_<tag>_exact.csv` and every `fig45_G_<tag>_<arm>.csv` /
`fig45_steps_<tag>_<arm>.csv` beside it. Records may be PARTIAL (the driver writes them per
step), so each arm is graded on the times it has reached and the table says how far that is.
Error is max_x |G - G_exact| (absolute; G(c,0) = 1/4 sets the scale).

Run:  python3 growth_vs_exact.py DIR TAG [--times 0.5,1,2,4,8] [--plot out.png]
"""
import argparse
import csv
import glob
import os

from matched_error import load_g, maxdiff


def load_steps(path):
    rows = []
    with open(path, newline="") as f:
        for r in csv.DictReader(f):
            bonds = [int(b) for b in r["bonds"].split(";")] if r.get("bonds") else []
            rows.append(dict(t=round(float(r["t"]), 6), wall=float(r["wall"]), chi=int(r["chi"]),
                             disc=float(r["discarded"]), unmet=r["unmet"], bonds=bonds))
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dir")
    ap.add_argument("tag")
    ap.add_argument("--times", default="0.5,1,2,4,8")
    ap.add_argument("--plot", default=None)
    a = ap.parse_args()

    ex = load_g(os.path.join(a.dir, "fig45_G_%s_exact.csv" % a.tag))
    pre = os.path.join(a.dir, "fig45_G_%s_" % a.tag)
    arms = sorted(p[len(pre):-4] for p in glob.glob(pre + "*.csv") if not p.endswith("_exact.csv"))
    want = [float(x) for x in a.times.split(",")]

    res = {}
    for arm in arms:
        g = load_g(pre + arm + ".csv")
        # only complete time columns: a record cut mid-write has a short last column
        ts = sorted(t for t in g if t in ex and len(g[t]) == len(ex[t]))
        err = [(t, maxdiff(g[t], ex[t])) for t in ts]
        sp = os.path.join(a.dir, "fig45_steps_%s_%s.csv" % (a.tag, arm))
        steps = load_steps(sp) if os.path.exists(sp) else []
        res[arm] = (err, steps)

    print("tag %s   (error = max_x |G - G_exact|; chi = max bond dim after the step)" % a.tag)
    print("%-8s %7s  " % ("arm", "t_max") + "  ".join("err@%-5g chi@%-5g" % (t, t) for t in want))
    for arm, (err, steps) in res.items():
        e = dict(err)
        c = {s["t"]: s["chi"] for s in steps}
        cells = []
        for t in want:
            et = e.get(round(t, 6))
            ct = c.get(round(t, 6))
            cells.append("%-9s %-9s" % ("%.2e" % et if et is not None else "-",
                                         ct if ct is not None else "-"))
        tmax = err[-1][0] if err else 0.0
        print("%-8s %7.2f  " % (arm, tmax) + "  ".join(cells))
    for arm, (err, steps) in res.items():
        if steps:
            print("%-8s max err %.2e  max chi %d  sum discarded %.2e  wall %.0f s  unmet steps %d"
                  % (arm, max(x for _, x in err) if err else float("nan"),
                     max(s["chi"] for s in steps), sum(s["disc"] for s in steps),
                     sum(s["wall"] for s in steps), sum(1 for s in steps if s["unmet"])))

    if a.plot:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
        for arm, (err, steps) in res.items():
            if err:
                ax[0].semilogy([t for t, _ in err], [max(x, 1e-16) for _, x in err], label=arm)
            if steps:
                ax[1].plot([s["t"] for s in steps], [s["chi"] for s in steps], label=arm)
                ax[2].semilogy([s["t"] for s in steps], [max(s["disc"], 1e-16) for s in steps],
                               label=arm)
        ax[0].set(xlabel="t", ylabel=r"max$_x |G-G_{\rm exact}|$", title="error vs exact")
        ax[1].set(xlabel="t", ylabel=r"$\chi_{\max}$", title="bond dimension (non-binding cap)")
        ax[2].set(xlabel="t", ylabel="discarded norm per step", title="truncation")
        for x in ax:
            x.legend()
            x.grid(alpha=0.3)
        fig.suptitle(a.tag)
        fig.tight_layout()
        fig.savefig(a.plot, dpi=130)
        print("wrote", a.plot)


if __name__ == "__main__":
    main()
