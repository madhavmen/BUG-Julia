#!/usr/bin/env python3
"""Figures for the open long-range runs -- Hryniuk & Szymanska, Comms Phys (2026) 9:45.

    python3 benchmarks/plot_dissipative_longrange.py [results/dissip_*.csv ...]

One PNG per CSV. The left half is the PAPER COMPARISON, the right half is ours:

    (a) magnetizations <sigma^a>          their Figs 3a / 4a / 6a
    (b) C^aa_{d=1}                        their Figs 3b / 4b / 6b
    (c) C^aa_{d=2}                        their Fig 3c
    (d) S_zz(q), q = 0, pi, 2pi/N         their Fig 5 observable, here vs TIME
    (e) bond dimension chi(t)
    (f) cumulative operator applications
    (g) cumulative wall clock
    (h) ARM-vs-ARM SPREAD -- see below

--- READING RULES ------------------------------------------------------------------------

* COLOUR IS THE PAULI COMPONENT, LINESTYLE IS THE ARM. That is deliberate: the paper's own
  panels colour by component (x/y/z), so keeping that mapping makes the visual comparison
  with the published figure direct, and the arms then have to be told apart by dash pattern.
  A colour-by-arm figure cannot be laid next to the paper's at all.

* ⛔ PANEL (h) IS THE ACCURACY STATEMENT, AND IT IS A SPREAD, NOT AN ERROR. There is no exact
  reference at L = 16 -- `4^16` is out of reach -- so no arm can be scored. What CAN be said is
  how far the three arms drift apart on the same grid: if they agree to 1e-3 while the paper's
  own curves differ from its exact N = 10 check by more than that, the integrator is not the
  limiting factor. ⚠ Mutual agreement is NOT proof of correctness: three arms sharing one
  systematic error agree perfectly. The generator is pinned separately, against a dense
  Liouvillian at L = 3-4 (`certify_lr`), and THAT is what makes the curves trustworthy.

* ⚠ THE PAPER RUNS N = 200 (1D) AND 4x4 (2D) AND WE RUN L = 16. Agreement is expected to be
  QUALITATIVE -- the shape of the relaxation, the sign and ordering of the correlators, the
  competition-induced structure -- not curve-on-curve. The paper's own 2D panel shows visible
  finite-size discrepancy against its N = 3x3 exact check for the same reason.
"""
import csv
import sys
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# Component colours follow the paper's panels (green / orange / blue for x / y / z).
COMP = [("sx", "#2ca02c", r"$\langle\sigma^x\rangle$"),
        ("sy", "#ff9e1b", r"$\langle\sigma^y\rangle$"),
        ("sz", "#1f77b4", r"$\langle\sigma^z\rangle$")]
# ⛔ WIDE-AND-PALE FIRST, THIN-DASHED ON TOP. The arms routinely agree to several digits on
# these models, and two solid lines of equal width make the second one vanish -- which reads
# as a MISSING ARM rather than as agreement. Drawn this way, coincidence shows as a dashed
# line inside a broad band and divergence shows as two separate curves.
ARM = {
    "tdvp2":      dict(ls="-",  lw=4.2, alpha=0.35, z=1, label="2-site TDVP"),
    "tdvp_cbe1s": dict(ls="--", lw=1.5, alpha=1.0,  z=3, label="1-site CBE-TDVP"),
    "cbe_bug":    dict(ls=":",  lw=2.0, alpha=1.0,  z=4, label="CBE-BUG (rSVD, par)"),
    "bug_rsvd":   dict(ls=":",  lw=2.0, alpha=1.0,  z=4, label="CBE-BUG (rSVD)"),
    "bug_par":    dict(ls="-.", lw=1.5, alpha=1.0,  z=5, label="CBE-BUG (rSVD, par)"),
}
CORR = [("xx", "#2ca02c"), ("yy", "#ff9e1b"), ("zz", "#1f77b4")]


def f(row, key):
    v = row.get(key, "")
    if v in ("", "NaN", "nan", "Inf", "-Inf", None):
        return float("nan")
    return float(v)


def load(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def plot_csv(path):
    rows = load(path)
    if not rows:
        return None
    arms = []
    for r in rows:
        if r["arm"] not in arms:
            arms.append(r["arm"])
    by = {a: sorted([r for r in rows if r["arm"] == a], key=lambda r: f(r, "t")) for a in arms}
    L = int(rows[0]["L"])
    shape = rows[0].get("shape", f"chain{L}")
    dt = f(rows[0], "dt")
    has_szz = "szz_q0" in rows[0]

    fig, axes = plt.subplots(2, 4, figsize=(21.5, 9.0))
    fig.suptitle(f"{path.stem}   —   L = {L} ({shape}), dt = {dt:g}, symmetry = none (fused $d$=4)"
                 "   —   2-site TDVP vs 1-site CBE-TDVP vs CBE-BUG",
                 fontsize=13, y=0.985)
    (ax_m, ax_c1, ax_c2, ax_s), (ax_chi, ax_kry, ax_wall, ax_dev) = axes

    for a in arms:
        rs = by[a]
        st = ARM.get(a, dict(ls="-", lw=1.5, alpha=1.0, z=2, label=a))
        t = [f(r, "t") for r in rs]
        base = dict(ls=st["ls"], lw=st["lw"], alpha=st["alpha"], zorder=st["z"])
        for key, col, lab in COMP:
            ax_m.plot(t, [f(r, key) for r in rs], color=col, **base)
        for d, ax in ((1, ax_c1), (2, ax_c2)):
            for cn, col in CORR:
                ax.plot(t, [f(r, f"c{cn}{d}") for r in rs], color=col, **base)
        if has_szz:
            for key, col in (("szz_q0", "#333333"), ("szz_qpi", "#c0392b"),
                             ("szz_q2pin", "#8e44ad")):
                ax_s.plot(t, [f(r, key) for r in rs], color=col, **base)
        ax_chi.plot(t, [f(r, "chi") for r in rs], color="#444444", **base)
        ax_kry.plot(t, [f(r, "krylov") for r in rs], color="#444444", **base)
        ax_wall.plot(t, [f(r, "seconds") for r in rs], color="#444444", **base)

    # ── (h) the spread across arms, at each saved time ──────────────────────────────────────
    keys = [k for k, _, _ in COMP] + [f"c{c}{d}" for d in (1, 2) for c, _ in CORR]
    if len(arms) >= 2:
        ts = [f(r, "t") for r in by[arms[0]]]
        n = min(len(by[a]) for a in arms)
        spread = []
        for i in range(n):
            m = 0.0
            for k in keys:
                vals = [f(by[a][i], k) for a in arms]
                vals = [v for v in vals if v == v]
                if len(vals) >= 2:
                    m = max(m, max(vals) - min(vals))
            spread.append(max(m, 1e-18))
        ax_dev.semilogy(ts[:n], spread, color="#a03030", marker="o", ms=3, lw=1.5)
        ax_dev.set(xlabel="$t$", ylabel="max spread over arms & observables",
                   title="(h) ARM SPREAD — not an error, see caption")
    else:
        ax_dev.text(0.5, 0.5, "only one arm in this file", ha="center", va="center",
                    transform=ax_dev.transAxes, color="#666666")
        ax_dev.set(title="(h) ARM SPREAD")
    ax_dev.grid(alpha=0.3, which="both")

    # Legends: one for components, one for arms, so neither has 9 entries.
    comp_h = [plt.Line2D([], [], color=c, lw=2.5, label=lab) for _, c, lab in COMP]
    arm_h = [plt.Line2D([], [], color="#444444", ls=ARM.get(a, {}).get("ls", "-"),
                        lw=ARM.get(a, {}).get("lw", 1.5),
                        alpha=ARM.get(a, {}).get("alpha", 1.0),
                        label=ARM.get(a, {}).get("label", a)) for a in arms]
    ax_m.set(xlabel="$t$", ylabel=r"$\langle\sigma^a\rangle$",
             title="(a) magnetizations — the paper's Fig. 3a / 4a / 6a")
    ax_m.legend(handles=comp_h + arm_h, fontsize=7.5, ncol=2)
    ax_m.grid(alpha=0.3)

    for d, ax in ((1, ax_c1), (2, ax_c2)):
        h = [plt.Line2D([], [], color=c, lw=2.5, label=rf"$C^{{{cn[0]}{cn[1]}}}_{{d={d}}}$")
             for cn, c in CORR]
        ax.set(xlabel="$t$", ylabel=rf"$C^{{aa}}_{{d={d}}}$",
               title=f"({'b' if d == 1 else 'c'}) connected correlators, $d$ = {d}")
        ax.legend(handles=h, fontsize=8)
        ax.grid(alpha=0.3)
        ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)

    if has_szz:
        h = [plt.Line2D([], [], color=c, lw=2.5, label=lab) for c, lab in
             (("#333333", "$q=0$ (ferro)"), ("#c0392b", "$q=\\pi$ (antiferro)"),
              ("#8e44ad", "$q=2\\pi/N$ (modulated)"))]
        ax_s.set(xlabel="$t$", ylabel=r"$S_{zz}(q)$",
                 title="(d) structure factor — the paper's Fig. 5 observable")
        ax_s.legend(handles=h, fontsize=8)
    else:
        ax_s.text(0.5, 0.5, "no S_zz(q) column in this file\n(the Fig. 3 driver predates it)",
                  ha="center", va="center", transform=ax_s.transAxes, color="#666666")
        ax_s.set(title="(d) structure factor")
        ax_s.set_xticks([]); ax_s.set_yticks([])
    ax_s.grid(alpha=0.3)

    ax_chi.set(xlabel="$t$", ylabel=r"max bond dimension $\chi$", title="(e) rank growth")
    cap_hit = max((f(r, "chi") for r in rows), default=0)
    ax_chi.legend(handles=arm_h, fontsize=8, title=f"max $\\chi$ reached = {cap_hit:.0f}",
                  title_fontsize=8)
    ax_chi.grid(alpha=0.3)
    ax_kry.set(xlabel="$t$", ylabel="cumulative operator applications",
               title="(f) STRUCTURAL cost — mixed units")
    ax_kry.legend(handles=arm_h, fontsize=8)
    ax_kry.grid(alpha=0.3)
    ax_wall.set(xlabel="$t$", ylabel="cumulative step time [s]",
                title="(g) WALL CLOCK — the honest cost axis")
    ax_wall.legend(handles=arm_h, fontsize=8)
    ax_wall.grid(alpha=0.3)

    fig.text(0.5, 0.006,
             "(f) mixes units: 2-site, 1-site and 0-site matvecs are not the same work, and "
             "CBE-BUG's count EXCLUDES the expansion's H·Theta — read it with (g). (h) is a SPREAD "
             "between arms, not an error: there is no exact reference at L=16, and three arms "
             "sharing one systematic error would agree perfectly. The generator is pinned "
             "separately against a dense Liouvillian at L=3-4.",
             ha="center", fontsize=8, style="italic", color="#444444")
    fig.tight_layout(rect=(0, 0.028, 1, 0.955))
    out = RESULTS / f"{path.stem}.png"
    fig.savefig(out, dpi=145)
    plt.close(fig)
    return out


def main():
    args = sys.argv[1:]
    paths = [Path(a) for a in args] if args else sorted(
        list(RESULTS.glob("dissip_lr_*.csv")) + list(RESULTS.glob("dissip_heis_protocol_*.csv")))
    if not paths:
        sys.exit("no dissip_*.csv in benchmarks/results")
    for p in paths:
        out = plot_csv(p)
        if out:
            print("wrote", out.relative_to(HERE.parent))


if __name__ == "__main__":
    main()
