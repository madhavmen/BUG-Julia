"""Summarise the adaptive XX L=20 campaign (`adaptive_xx_l20.jl`).

usage:  python benchmarks/analyze_adaptive_xx.py CSV [CSV ...] [--md OUT.md]

Reads the streamed per-step CSVs and prints (and optionally writes as Markdown):

  * per sweep (basis / exact / mid / trunc / expv): the final-time error against the swept
    tolerance, the work it bought (operator applications, probes, Krylov directions, rank) and
    the ledger verdict -- i.e. which component limits accuracy, and how loose each tolerance can
    be before the error moves;
  * the loosest tolerance per sweep that still reaches each accuracy target;
  * profiles (IT01-IT04, IT06): error after 1 and NSTEP steps, spread over seeds, rank, status;
  * baseline (IT05): the legacy fixed-knob solver, for cost at matched accuracy.

Standard library only (csv, statistics), so it runs anywhere.
"""
import csv
import sys
from collections import defaultdict
from statistics import mean, pstdev

TARGETS = (1e-2, 1e-3, 1e-4, 1e-5)
SWEEP_KNOB = {"basis": "tol_basis", "exact": "tol_basis", "mid": "tol_basis",
              "trunc": "tol_trunc", "expv": "tol_expv"}


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            rows.extend(csv.DictReader(fh))
    return rows


def num(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def runs(rows):
    """Group step rows into runs keyed by everything that identifies an arm."""
    out = defaultdict(list)
    for r in rows:
        key = (r["phase"], r["arm"], r["state"], r["scheme"], r["seed"], r["path"], r["exact"],
               r["tol_basis"], r["tol_trunc"], r["tol_expv"], r["maxdim"])
        out[key].append(r)
    for k in out:
        out[k].sort(key=lambda r: int(r["step"]))
    return out


def summary(steps):
    last = steps[-1]
    tot = lambda col: sum(num(s[col]) for s in steps if s[col] not in ("", None))
    worst_status = "budget_unmet" if any(s["status"] == "budget_unmet" for s in steps) else last["status"]
    unmet = sorted({u for s in steps for u in s["unmet"].split("+") if u})
    return {
        "err1": num(steps[0]["err_sz"]), "errT": num(last["err_sz"]),
        "errmax": max(num(s["err_sz"]) for s in steps),
        "chi": int(num(last["chi_max"])), "raug": max(num(s["r_aug"]) for s in steps),
        "matvec": tot("krylov_dims"), "h1": tot("n_h1"), "probes": tot("n_probes"),
        "holdout": tot("n_holdout"), "kry": tot("n_krylov"), "cpass": tot("n_c_pass"),
        "dafter": tot("n_d_after_c"), "expv_it": tot("expv_iters"),
        "disc": max(num(s["discarded_cum"]) for s in steps),
        "norm": num(last["norm_drift"]), "szd": num(last["sz_drift"]),
        "status": worst_status, "unmet": "+".join(unmet), "tstep": tot("t_step"),
        "tcbe": tot("t_cbe"), "tkry": tot("t_kry"), "troot": tot("t_root"),
        "nsteps": len(steps),
    }


def fmt(x, w=9):
    if isinstance(x, float):
        return f"{x:{w}.2e}" if x == x else " " * (w - 3) + "nan"
    return f"{x:>{w}}"


def report(rows):
    R = runs(rows)
    lines = []
    out = lines.append

    # ── sweeps: one component at a time ──────────────────────────────────────────────────
    sweeps = defaultdict(list)
    for k, steps in R.items():
        if k[0] == "sweeps":
            sweeps[k[1]].append((num(steps[0][SWEEP_KNOB[k[1]]]), summary(steps)))
    if sweeps:
        out("## Sweeps: one tolerance at a time (Neel, seed 1; others tight)\n")
        out("| sweep | tol | err(1 step) | err(T) | chi | r_aug | H applications | probes | "
            "Krylov dirs | C pass / D after C | expv iters | discarded | status |")
        out("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
        for name in ("basis", "exact", "mid", "trunc", "expv"):
            for tol, s in sorted(sweeps.get(name, []), key=lambda t: -t[0]):
                out(f"| {name} | {tol:.0e} | {s['err1']:.2e} | {s['errT']:.2e} | {s['chi']} | "
                    f"{int(s['raug'])} | {int(s['matvec'])} | {int(s['probes'])} | "
                    f"{int(s['kry'])} | {int(s['cpass'])} / {int(s['dafter'])} | "
                    f"{int(s['expv_it'])} | {s['disc']:.1e} | {s['status']}"
                    f"{' (' + s['unmet'] + ')' if s['unmet'] else ''} |")
        out("")
        out("### Loosest tolerance reaching each accuracy target (max |Sz - exact| over the run)\n")
        out("| sweep | " + " | ".join(f"err <= {t:.0e}" for t in TARGETS) + " |")
        out("|---|" + "---|" * len(TARGETS))
        for name in ("basis", "exact", "mid", "trunc", "expv"):
            pts = sweeps.get(name, [])
            if not pts:
                continue
            cells = []
            for t in TARGETS:
                ok = [tol for tol, s in pts if s["errmax"] <= t]
                cells.append(f"{max(ok):.0e}" if ok else "none")
            out(f"| {name} | " + " | ".join(cells) + " |")
        floor = min((s["errmax"] for pts in sweeps.values() for _, s in pts), default=float("nan"))
        out(f"\nError floor over all sweep runs: {floor:.2e} (what the tightest settings reach "
            f"at this dt -- the integrator/time-step error no tolerance removes).\n")

    # ── profiles ────────────────────────────────────────────────────────────────────────
    prof = defaultdict(list)
    for k, steps in R.items():
        if k[0] == "profiles":
            prof[(k[1], k[3], k[2])].append(summary(steps))
    if prof:
        out("## Profiles (IT01-IT04, IT06): mean +- spread over seeds\n")
        out("| profile | scheme | state | seeds | err(1 step) | err(T) mean | err(T) max | chi | "
            "H applications | norm drift | Sz drift | unmet runs |")
        out("|---|---|---|---|---|---|---|---|---|---|---|---|")
        for (p, sch, st), ss in sorted(prof.items()):
            e = [s["errT"] for s in ss]
            out(f"| {p} | {sch} | {st} | {len(ss)} | {mean(s['err1'] for s in ss):.2e} | "
                f"{mean(e):.2e} +- {pstdev(e):.1e} | {max(e):.2e} | "
                f"{max(s['chi'] for s in ss)} | {mean(s['matvec'] for s in ss):.0f} | "
                f"{max(abs(s['norm']) for s in ss):.1e} | {max(abs(s['szd']) for s in ss):.1e} | "
                f"{sum(s['status'] == 'budget_unmet' for s in ss)} |")
        out("")

    # ── combos / maxdim ─────────────────────────────────────────────────────────────────
    for ph, title in (("combos", "Combined loose settings (do they compound?)"),
                      ("maxdim", "Rank cap binding (IT07: failures from maxdim are visible)")):
        sel = [(k, summary(steps)) for k, steps in R.items() if k[0] == ph]
        if not sel:
            continue
        out(f"## {title}\n")
        out("| scheme | tol_basis | tol_trunc | tol_expv | maxdim | err(T) | chi | H applications | "
            "discarded | status |")
        out("|---|---|---|---|---|---|---|---|---|---|")
        for k, s in sorted(sel, key=lambda t: (t[0][3], -num(t[0][7]), -num(t[0][8]), -num(t[0][9]), int(t[0][10]))):
            out(f"| {k[3]} | {num(k[7]):.0e} | {num(k[8]):.0e} | {num(k[9]):.0e} | {k[10]} | "
                f"{s['errT']:.2e} | {s['chi']} | {int(s['matvec'])} | {s['disc']:.1e} | "
                f"{s['status']}{' (' + s['unmet'] + ')' if s['unmet'] else ''} |")
        out("")

    # ── matched accuracy (one process) ──────────────────────────────────────────────────
    mt = [(k, summary(steps)) for k, steps in R.items() if k[0] == "matched"]
    if mt:
        out("## Matched accuracy, one process (IT05 / simulation_plan acceptance)\n")
        out("| scheme | state | arm | tol_basis | err(T) | H applications | trial width r_aug | "
            "chi | t_step sum | status |")
        out("|---|---|---|---|---|---|---|---|---|---|")
        for k, s in sorted(mt, key=lambda t: (t[0][3], t[0][2], t[1]["errT"])):
            tb = "" if k[7] == "" else f"{num(k[7]):.0e}"
            out(f"| {k[3]} | {k[2]} | {k[1]} | {tb} | {s['errT']:.2e} | {int(s['matvec'])} | "
                f"{int(s['raug'])} | {s['chi']} | {s['tstep']:.1f}s | {s['status']} |")
        out("")

    # ── baseline ────────────────────────────────────────────────────────────────────────
    base = [(k, summary(steps)) for k, steps in R.items() if k[0] == "baseline"]
    if base:
        out("## Baseline (IT05): legacy fixed-knob solver\n")
        out("| arm | scheme | state | err(1 step) | err(T) | chi | H applications | t_step sum |")
        out("|---|---|---|---|---|---|---|---|")
        for k, s in sorted(base):
            out(f"| {k[1]} | {k[3]} | {k[2]} | {s['err1']:.2e} | {s['errT']:.2e} | {s['chi']} | "
                f"{int(s['matvec'])} | {s['tstep']:.1f}s |")
        out("")
    return "\n".join(lines)


def main(argv):
    md = None
    if "--md" in argv:
        i = argv.index("--md")
        md = argv[i + 1]
        argv = argv[:i] + argv[i + 2:]
    if not argv:
        print(__doc__)
        return 2
    text = report(load(argv))
    print(text)
    if md:
        with open(md, "w") as fh:
            fh.write(text + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main(sys.argv[1:]))
