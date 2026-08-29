#!/usr/bin/env python3
"""MODEL 3 on a UNIFORM grid: BUG against 2-site TDVP, dissipative Heisenberg chain.

Reproduces the observables of Hryniuk & Szymanska (Comms Phys 2026, 9:45) Fig. 3a-c -- bulk
<sigma^a(t)> and the CONNECTED correlators C^aa_d = <s^a_i s^a_{i+d}> - <s^a_i><s^a_{i+d}> for
d = 1, 2 -- and puts the cost of each integrator beside them.

⛔ THE ACCURACY PANEL IS A SELF-CONVERGENCE PANEL, NOT AN ERROR PANEL. There is no exact reference
at L = 16 (`4^16` is 4.3 billion amplitudes), so nothing here is a deviation from truth. Panel (d)
plots each arm against the SAME arm at the finest dt in the input set. Reading it as an error
against an exact solution would overstate exactly what this campaign refuses to overstate
elsewhere -- and our own finest grid is not an anchor.

⚠ THE PAPER IS THE ANCHOR AND THE COMPARISON IS QUALITATIVE. Theirs is N = 200 with chi = 20;
ours is N = 16. Curves should share shape, sign and rough relaxation time. A numeric deviation
between the two is dominated by the size difference and means nothing about either integrator.

⚠ COST IS QUOTED ON `krylov`, NOT `seconds`. Wall-clock carries whatever else shared the node;
matvec counts do not. `seconds` is drawn, faintly, only so a wildly contended run is visible.

Run:  python3 benchmarks/plot_dissip_heis.py [csv ...]
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RESULTS = os.path.join(HERE, "results")

STYLE = {
    "tdvp2":    dict(color="#c0392b", ls="-",  marker="o", label="2-site TDVP"),
    "bug_rsvd": dict(color="#2471a3", ls="-",  marker="s", label="BUG (rSVD-CBE)"),
    "bug_par":  dict(color="#1e8449", ls="--", marker="^", label="BUG (parallel)"),
}
COMPS = [("x", "sx", "cxx"), ("y", "sy", "cyy"), ("z", "sz", "czz")]


def load(paths):
    """-> {(arm, dt): {column: [values in t order]}}"""
    runs = defaultdict(lambda: defaultdict(list))
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                key = (r["arm"], float(r["dt"]))
                for k, v in r.items():
                    if k == "arm":
                        continue
                    runs[key][k].append(float(v))
    return runs


def main(paths):
    paths = [p for p in paths if os.path.exists(p)]
    if not paths:
        sys.exit("no input CSVs found; run `julia --project=. "
                 "benchmarks/dissipative_heisenberg.jl protocol` first")
    runs = load(paths)
    if not runs:
        sys.exit("input CSVs contained no rows")
    dts = sorted({dt for _, dt in runs})
    fine = min(dts)
    arms = [a for a in STYLE if any(k[0] == a for k in runs)]
    L = int(next(iter(runs.values()))["L"][0])

    fig, axes = plt.subplots(2, 3, figsize=(16.5, 9))
    fig.suptitle(
        f"Dissipative Heisenberg chain, L={L}, uniform grid  "
        f"(Jx,Jy,Jz)=(-1.0,-0.9,-1.2), h=-1, gamma=1, start |+x>^L\n"
        f"BUG vs 2-site TDVP -- protocol of Hryniuk & Szymanska, Comms Phys 2026 9:45 (their N=200)",
        fontsize=11)

    # ── (a) bulk magnetisations ────────────────────────────────────────────────────────────────
    ax = axes[0][0]
    for arm in arms:
        d = runs.get((arm, fine))
        if not d:
            continue
        st = dict(STYLE[arm]); st.pop("label")
        for i, (nm, col, _) in enumerate(COMPS):
            ax.plot(d["t"], d[col], **st, alpha=[1.0, 0.55, 0.3][i], markersize=3,
                    markevery=5, label=f"{STYLE[arm]['label']}  $\\sigma^{nm}$")
    ax.set_xlabel("t"); ax.set_ylabel(r"bulk $\langle\sigma^a\rangle$")
    ax.set_title("(a) magnetisations")
    ax.axhline(0, color="0.7", lw=0.6); ax.grid(alpha=0.25)
    ax.legend(fontsize=6, ncol=2)

    # ── (b, c) connected correlators ───────────────────────────────────────────────────────────
    for panel, dist in ((axes[0][1], 1), (axes[0][2], 2)):
        for arm in arms:
            d = runs.get((arm, fine))
            if not d:
                continue
            st = dict(STYLE[arm]); st.pop("label")
            for i, (nm, _, cc) in enumerate(COMPS):
                col = f"{cc}{dist}"
                if col not in d:
                    continue
                panel.plot(d["t"], d[col], **st, alpha=[1.0, 0.55, 0.3][i], markersize=3,
                           markevery=5, label=f"{STYLE[arm]['label']}  $C^{{{nm}{nm}}}_{dist}$")
        panel.set_xlabel("t"); panel.set_ylabel(f"$C^{{aa}}_{dist}$ (connected)")
        panel.set_title(f"({'bc'[dist - 1]}) correlators, d = {dist}")
        panel.axhline(0, color="0.7", lw=0.6); panel.grid(alpha=0.25)
        panel.legend(fontsize=6, ncol=2)

    # ── (d) self-convergence in dt ─────────────────────────────────────────────────────────────
    ax = axes[1][0]
    if len(dts) < 2:
        ax.text(0.5, 0.5, "one dt only --\nno convergence statement available",
                ha="center", va="center", transform=ax.transAxes, fontsize=9, color="#c0392b")
    else:
        for arm in arms:
            ref = runs.get((arm, fine))
            if not ref:
                continue
            for dt in dts:
                if dt == fine:
                    continue
                d = runs.get((arm, dt))
                if not d:
                    continue
                n = min(len(d["t"]), len(ref["t"]))
                dev = [max(abs(d[c][i] - ref[c][i]) for _, c, _ in COMPS) for i in range(n)]
                st = dict(STYLE[arm]); st.pop("label"); st.pop("marker")
                ax.semilogy(d["t"][:n], [max(v, 1e-16) for v in dev], **st,
                            alpha=0.9 if dt == max(dts) else 0.5,
                            label=f"{STYLE[arm]['label']}  dt={dt:g}")
    ax.set_xlabel("t"); ax.set_ylabel(f"max$_a$ |$\\sigma^a$(dt) - $\\sigma^a$(dt={fine:g})|")
    ax.set_title(f"(d) SELF-convergence vs own finest dt -- NOT an error")
    ax.grid(alpha=0.25, which="both"); ax.legend(fontsize=7)

    # ── (e) bond dimension ─────────────────────────────────────────────────────────────────────
    ax = axes[1][1]
    peak = 0.0
    for arm in arms:
        d = runs.get((arm, fine))
        if not d:
            continue
        st = dict(STYLE[arm]); st.pop("marker")
        ax.plot(d["t"], d["chi"], **st)
        peak = max(peak, max(d["chi"]))
    ax.set_xlabel("t"); ax.set_ylabel(r"max bond dimension $\chi$")
    ax.set_title("(e) rank growth")
    ax.grid(alpha=0.25); ax.legend(fontsize=7)
    # ⚠ THE CAP IS NOT IN THE CSV, so this cannot decide saturation on its own -- it reports the
    # peak and says what to check. Asserting "converged" from a curve that merely looks flat is
    # how a truncation artefact gets reported as physics: a superket truncated too hard keeps a
    # smooth <sigma^a(t)> and a perfect Tr rho while its connected correlators decay numerically.
    ax.text(0.98, 0.04, f"peak $\\chi$ = {peak:.0f} -- if this equals DH_CAP,\n"
                        "the correlators are NOT converged", transform=ax.transAxes,
            ha="right", fontsize=7, color="#c0392b")

    # ── (f) cost ───────────────────────────────────────────────────────────────────────────────
    ax = axes[1][2]
    for arm in arms:
        d = runs.get((arm, fine))
        if not d:
            continue
        st = dict(STYLE[arm]); st.pop("marker")
        ax.plot(d["t"], d["krylov"], **st)
        ax.plot(d["t"], [s * d["krylov"][-1] / max(d["seconds"][-1], 1e-9) for s in d["seconds"]],
                color=st["color"], ls=":", lw=0.8, alpha=0.4)
    ax.set_xlabel("t"); ax.set_ylabel("cumulative matvec (krylov)")
    ax.set_title("(f) cost -- solid: matvec (quotable);  dotted: wall-clock, rescaled")
    ax.grid(alpha=0.25); ax.legend(fontsize=7)

    # the headline ratio, printed rather than drawn, so it cannot be misread off an axis
    print(f"\n  cumulative matvec at t_max  (L = {L}, dt = {fine:g})")
    base = None
    for arm in arms:
        d = runs.get((arm, fine))
        if not d:
            continue
        k = d["krylov"][-1]
        if arm == "tdvp2":
            base = k
        print(f"    {STYLE[arm]['label']:<20s} {k:>12,.0f}"
              + (f"   {base / k:6.2f}x fewer than tdvp2" if base and k else ""))

    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(RESULTS, "dissip_heis_protocol.png")
    fig.savefig(out, dpi=150)
    print(f"\nwrote {out}")


if __name__ == "__main__":
    args = sys.argv[1:]
    if not args:
        args = [os.path.join(RESULTS, f) for f in sorted(os.listdir(RESULTS))
                if f.startswith("dissip_heis_protocol_") and f.endswith(".csv")] \
            if os.path.isdir(RESULTS) else []
    main(args)
