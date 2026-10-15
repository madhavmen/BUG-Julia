"""Tables (and, where matplotlib is installed, figures) for benchmarks/krylov_closure_l16.jl.

usage:  python3 benchmarks/analyze_krylov_closure.py OUTDIR [FIGDIR]

Reads OUTDIR/kl16_<phase>.csv (+ _closure.csv, _profile.csv) for every phase present and prints
markdown: IT1 exact state, IT2 convergence with observed slopes, IT3 conservation, IT4 rank
adaptation, IT5 work at matched accuracy. Figures go to FIGDIR (default OUTDIR/fig).
"""
import csv
import math
import os
import sys
from collections import defaultdict

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
except ImportError:          # the cluster python has no matplotlib: tables only
    plt = None


def load(path):
    if not os.path.exists(path):
        return []
    with open(path) as fh:
        return list(csv.DictReader(fh))


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def fmt(x):
    if isinstance(x, str):
        return x
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    if isinstance(x, float):
        return f"{x:.2e}"
    return str(x)


def table(header, rows):
    out = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    out += ["| " + " | ".join(fmt(c) for c in r) + " |" for r in rows]
    return "\n".join(out)


def arm_key(r):
    return (r["arm"], num(r["dt"]))


def runs(rows):
    """Group step rows into runs, keyed by (arm, dt, maxdim), steps in order."""
    g = defaultdict(list)
    for r in rows:
        g[(r["arm"], num(r["dt"]), r["maxdim"])].append(r)
    for k in g:
        g[k].sort(key=lambda r: int(r["step"]))
    return g


def vec(s):
    return [num(x) for x in s.split(";")] if s else []


def slopes(e, dts):
    return [f"{math.log(e[i] / e[i + 1]) / math.log(dts[i] / dts[i + 1]):.2f}"
            if e[i] > 0 and e[i + 1] > 0 else "" for i in range(len(dts) - 1)]


def rho(outdir):
    """What the UNCHANGED order-1 step already closes, beside its observed temporal order."""
    rows = load(os.path.join(outdir, "kl16_rho.csv"))
    if not rows:
        return
    print("## rho -- global Krylov closure of the unchanged order-1 step (T = 0.8)\n")
    print("rho_j = ||(I-P) H F_j|| / ||F_j|| on the step's own half-sweep frames (upper bound "
          "sqrt(left^2 + right^2), exact pieces for nearest-neighbour H); leak_j = |dt| ||(I-P) "
          "H F_j|| / ||psi|| with F_j = (dt^j/j!) B^j S0. Max and median over the steps.\n")
    rr = runs(rows)
    for prof in sorted({a for a, _, _ in rr}):
        fin = {dt: rs for (a, dt, _), rs in rr.items() if a == prof}
        dts = sorted(fin, reverse=True)
        body = []
        for dt in dts:
            rs = fin[dt]
            R = [vec(r["rho_initial"]) for r in rs]
            Lk = [vec(r["leak_initial"]) for r in rs]
            m = max(len(x) for x in R)
            mx = lambda V, j: max(v[j] for v in V if len(v) > j)
            md = lambda V, j: sorted(v[j] for v in V if len(v) > j)[len(V) // 2]
            body.append([dt, num(rs[-1]["wf_err_aligned"]), max(int(r["chi_max"]) for r in rs)]
                        + [mx(R, j) for j in range(m)] + [md(R, j) for j in range(m)]
                        + [mx(Lk, j) for j in range(m)])
        m = (len(body[0]) - 3) // 3
        print(f"### {prof}\n")
        print(table(["dt", "wf err(T) aligned", "chi"] + [f"max rho_{j}" for j in range(m)]
                    + [f"med rho_{j}" for j in range(m)] + [f"max leak_{j}" for j in range(m)],
                    body))
        e = [b[1] for b in body]
        print("\nobserved slopes " + ", ".join(f"{dts[i]:g}->{dts[i+1]:g}: {s}"
                                              for i, s in enumerate(slopes(e, dts))) + "\n")


def it1(outdir):
    rows = [r for r in load(os.path.join(outdir, "kl16_it1.csv")) if not math.isnan(num(r["wf_err"]))]
    if not rows:
        return
    print("## IT1 -- exact real-time state (dt = 0.05)\n")
    print(table(["arm", "t", "wf err", "wf err (phase-aligned)", "max |dSz| vs exact",
                 "vs free fermions", "left-half M err", "centre current err", "chi", "r_aug",
                 "passes", "leak (j=0..p-1)", "defect", "status"],
                [[r["arm"], num(r["t"]), num(r["wf_err"]), num(r["wf_err_aligned"]),
                  num(r["err_sz"]), num(r["err_sz_ff"]), num(r["err_mag_left"]),
                  num(r["err_current_mid"]), r["chi_max"], r["r_aug"], r["n_passes"],
                  r["leak"], num(r["defect"]), r["status"] + (f" ({r['unmet']})" if r["unmet"] else "")]
                 for r in rows]))
    print()


def it2(outdir, figdir):
    rows = load(os.path.join(outdir, "kl16_it2.csv"))
    if not rows:
        return
    fin = {}
    for (arm, dt, _), rs in runs(rows).items():
        fin[(arm, dt)] = rs[-1]
    arms = sorted({a for a, _ in fin}, key=lambda a: (a == "midpoint", a))
    dts = sorted({d for _, d in fin}, reverse=True)
    print("## IT2 -- temporal convergence at T = 0.8 (compression off, tight tolerances)\n")
    for col, name in (("wf_err", "wavefunction error, global phase included"),
                      ("wf_err_aligned", "phase-aligned wavefunction error"),
                      ("err_sz", "max |Sz - exact|")):
        hdr = ["arm"] + [f"dt={d:g}" for d in dts] + [f"slope {dts[i]:g}->{dts[i+1]:g}" for i in range(len(dts) - 1)]
        body = []
        for a in arms:
            e = [num(fin[(a, d)][col]) if (a, d) in fin else math.nan for d in dts]
            sl = [math.log(e[i] / e[i + 1]) / math.log(dts[i] / dts[i + 1])
                  if e[i] > 0 and e[i + 1] > 0 else math.nan for i in range(len(dts) - 1)]
            body.append([a] + e + [f"{s:.2f}" if not math.isnan(s) else "" for s in sl])
        print(f"### {name}\n")
        print(table(hdr, body))
        print()
    print("### step statistics at the final step\n")
    print(table(["arm", "dt", "chi", "r_aug", "passes", "leak", "defect", "expv err", "status"],
                [[a, d, fin[(a, d)]["chi_max"], fin[(a, d)]["r_aug"], fin[(a, d)]["n_passes"],
                  fin[(a, d)]["leak"], num(fin[(a, d)]["defect"]), num(fin[(a, d)]["expv_err"]),
                  fin[(a, d)]["status"] + (f" ({fin[(a, d)]['unmet']})" if fin[(a, d)]["unmet"] else "")]
                 for a in arms for d in dts if (a, d) in fin]))
    print()
    if plt is not None:
        fig, ax = plt.subplots(figsize=(5.5, 4.2))
        for a in arms:
            xs = [d for d in dts if (a, d) in fin]
            ys = [num(fin[(a, d)]["wf_err"]) for d in xs]
            ax.loglog(xs, ys, "o-", label=a)
        for p, ls in ((1, ":"), (2, "--"), (3, "-."), (4, (0, (1, 3)))):
            x0, y0 = dts[0], max(num(fin[(a, dts[0])]["wf_err"]) for a in arms if (a, dts[0]) in fin)
            ax.loglog(dts, [y0 * (d / x0) ** p for d in dts], color="gray", ls=ls, lw=0.8)
        ax.set_xlabel("dt")
        ax.set_ylabel("||psi(T) - psi_exact(T)||")
        ax.set_title("IT2: XX L=16 domain wall, T = 0.8")
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "it2_convergence.png"), dpi=150)
        plt.close(fig)


def it3(outdir, figdir):
    rows = load(os.path.join(outdir, "kl16_it3.csv"))
    if not rows:
        return
    print("## IT3 -- conservation to t = 2 (dt = 0.05)\n")
    body = []
    for (arm, dt, _), rs in runs(rows).items():
        g = [r for r in rs if not math.isnan(num(r["wf_err"]))]
        body.append([arm, max(abs(num(r["norm_drift"])) for r in rs),
                     max(abs(num(r["energy_drift"])) for r in rs),
                     max(abs(num(r["nup_drift"])) for r in rs),
                     max(num(r["iso_defect"]) for r in rs),
                     max(num(r["err_sz"]) for r in rs),
                     max(num(r["wf_err_aligned"]) for r in g) if g else math.nan,
                     sum(r["status"] == "met" for r in rs), len(rs)])
    print(table(["arm", "max |norm - 1|", "max |E - E0|", "max |dN_up|", "max isometry defect",
                 "max |Sz - exact|", "max wf err (aligned)", "steps met", "steps"], body))
    print()
    prof = load(os.path.join(outdir, "kl16_it3_profile.csv"))
    if plt is not None and prof:
        for arm in sorted({p["arm"] for p in prof}):
            pr = [p for p in prof if p["arm"] == arm]
            ts = sorted({num(p["t"]) for p in pr})
            L = max(int(p["site"]) for p in pr)
            sel = [ts[int(k)] for k in (len(ts) * f // 4 - 1 for f in (1, 2, 3, 4))]
            fig, ax = plt.subplots(figsize=(5.5, 4.2))
            for t in sel:
                xs = list(range(1, L + 1))
                m = {int(p["site"]): p for p in pr if num(p["t"]) == t}
                line, = ax.plot(xs, [num(m[j]["sz_mps"]) + 0.5 for j in xs], "o", ms=4, label=f"MPS t={t:g}")
                ax.plot(xs, [num(m[j]["sz_exact"]) + 0.5 for j in xs], "-", color=line.get_color(), lw=1)
            ax.set_xlabel("site j")
            ax.set_ylabel("n_j = <Sz_j> + 1/2")
            ax.set_title(f"IT3: domain-wall front, {arm} (dots MPS, lines exact)")
            ax.legend(fontsize=7)
            fig.tight_layout()
            fig.savefig(os.path.join(figdir, f"it3_front_{arm}.png"), dpi=150)
            plt.close(fig)


def it4(outdir, figdir):
    rows = load(os.path.join(outdir, "kl16_it4.csv"))
    if not rows:
        return
    clo = load(os.path.join(outdir, "kl16_it4_closure.csv"))
    print("## IT4 -- rank adaptation (dt = 0.05, T = 4, tol_basis 1e-8, tol_trunc 1e-10)\n")
    body = []
    rr = runs(rows)
    for (arm, dt, md), rs in sorted(rr.items(), key=lambda kv: (int(kv[0][2]), kv[0][0])):
        g = [r for r in rs if not math.isnan(num(r["wf_err"]))]
        reasons = sorted({u for r in rs for u in r["unmet"].split("+") if u})
        cs = [c for c in clo if c["arm"] == arm]
        body.append([arm, md, max(int(r["chi_max"]) for r in rs), max(int(r["r_aug"]) for r in rs),
                     sum(r["status"] == "met" for r in rs), len(rs), "+".join(reasons),
                     any(r["workcap"] == "true" for r in rs), any(r["maxdim_reached"] == "true" for r in rs),
                     max(num(r["discarded_cum"]) for r in rs),
                     max(num(r["defect"]) for r in rs),
                     max(num(r["max_sample"]) for r in rs), max(num(r["max_tail"]) for r in rs),
                     sum(int(c["added"]) for c in cs if c["added"]),
                     sum(int(c["new_sectors"]) for c in cs if c["new_sectors"]),
                     "; ".join(f"{num(r['t']):g}:{num(r['wf_err_aligned']):.1e}" for r in g)])
    print(table(["arm", "maxdim", "max chi", "max D_work", "steps met", "steps", "unmet reasons",
                 "workcap", "maxdim hit", "max discarded (cum)", "max defect", "max Test B",
                 "max Test A", "closure dirs added", "new sectors", "wf err (aligned) by t"], body))
    print()
    if plt is not None:
        fig, axs = plt.subplots(1, 2, figsize=(10, 4))
        for (arm, dt, md), rs in sorted(rr.items(), key=lambda kv: (int(kv[0][2]), kv[0][0])):
            t = [num(r["t"]) for r in rs]
            axs[0].plot(t, [int(r["chi_max"]) for r in rs], label=f"{arm}")
            axs[1].semilogy(t, [max(num(r["err_sz"]), 1e-16) for r in rs], label=f"{arm}")
        axs[0].set_xlabel("t"); axs[0].set_ylabel("D_max(t)"); axs[0].legend(fontsize=7)
        axs[1].set_xlabel("t"); axs[1].set_ylabel("max |Sz - exact|")
        axs[0].set_title("IT4: bond dimension"); axs[1].set_title("IT4: error")
        fig.tight_layout()
        fig.savefig(os.path.join(figdir, "it4_rank.png"), dpi=150)
        plt.close(fig)


def it5(outdir):
    rows = load(os.path.join(outdir, "kl16_it5.csv"))
    if not rows:
        return
    print("## IT5 -- work at matched accuracy (ONE process, T = 2, tol_basis 1e-8)\n")
    body = []
    for (arm, dt, _), rs in sorted(runs(rows).items(), key=lambda kv: (-kv[0][1], kv[0][0])):
        f = rs[-1]
        s = lambda c: sum(num(r[c]) for r in rs)
        body.append([arm, dt, num(f["wf_err_aligned"]), max(num(r["err_sz"]) for r in rs),
                     s("t_step"), s("t_closure"), s("alloc_bytes") / 2**30, int(s("krylov_dims")),
                     int(s("n_h1")), int(s("n_h1_closure")), int(s("n_root_frontier")),
                     int(s("n_h2_leak")), int(s("n_probes")), int(s("n_holdout")),
                     int(s("expv_iters")), s("n_passes") / len(rs), max(int(r["chi_max"]) for r in rs),
                     sum(r["status"] == "met" for r in rs)])
    print(table(["arm", "dt", "wf err(T) aligned", "max |dSz|", "wall s", "closure s", "alloc GiB",
                 "operator applications", "one-site H (half-sweeps)", "one-site H (closure)",
                 "root frontier applications", "two-site leak applications", "CBE probes",
                 "holdout probes", "root expv iterations", "passes/step", "max chi", "steps met"],
                body))
    print()


def main():
    outdir = sys.argv[1] if len(sys.argv) > 1 else "."
    figdir = sys.argv[2] if len(sys.argv) > 2 else os.path.join(outdir, "fig")
    if plt is not None:
        os.makedirs(figdir, exist_ok=True)
    rho(outdir)
    it1(outdir)
    it2(outdir, figdir)
    it3(outdir, figdir)
    it4(outdir, figdir)
    it5(outdir)


if __name__ == "__main__":
    main()
