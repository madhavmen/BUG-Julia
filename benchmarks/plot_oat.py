"""Figures for the one-axis twisting benchmark -- arXiv:2208.10972 Fig. 2, recreated at L = 10.

Reads the streamed CSV from `oat_l10.jl` and writes two figures:

  oat_panels_L<n>.png    the paper's Fig. 2 layout, one panel per quantity:
                      (a) S^x_tot(t)          against the closed form (L/2)cos^(L-1)(t/2)
                      (b) entanglement entropy against the closed-form Schmidt spectrum
                      (c) bond dimension D(t)  against the EXACT rank min(n,L-n)+1
                      (d) error analysis       delta s^x_tot (solid), delta E (dashed),
                                               discarded weight xi (dotted), log scale
  oat_convergence_L<n>.png   max_t |delta S^x_tot| against maxdim, per scheme -- the panel the paper
                      cannot draw, because at L = 100 there is no exact rank to converge to.

EVERY EXACT CURVE IS A CLOSED FORM, not a second numerical scheme. The reference is
`tests/common/analytic_reference.jl`, which shares no code with the integrators being measured;
OAT's Hamiltonian is diagonal in the computational basis, so the state, its Schmidt spectrum and
its entropy are all known analytically for every L and t.

THE RANK CEILING IS WHY PANEL (c) HAS A FLAT REFERENCE LINE. OAT's only entangling factor is
exp(-i t a b) in the two block magnetisations, so the exact Schmidt rank at the centre is
min(n, L-n) + 1 for ALL time -- 6 at L = 10. A run reporting more than that is reporting noise,
not physics, which is a statement the L = 100 figure in the paper cannot make.

usage:  python3 benchmarks/plot_oat.py [results.csv] [outdir]
"""

import csv
import glob
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "oat_L*.csv")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")

# Same colour per integrator as plot_cbe_sweeps_l16.py, so a scheme is the same colour in every
# figure in the repo.
COLOR = {"tdvp_cbe1s": "#2ca02c", "tdvp2": "#7f4fc0", "cbe_bug": "#d62728"}
LABEL = {"tdvp_cbe1s": "1-site TDVP-CBE", "tdvp2": "2-site TDVP",
         "cbe_bug": "CBE-BUG (interleaved)"}
# Linestyle keys on the rank cap. Solid is the one at or above the exact ceiling.
DASH = {2: ":", 3: "-.", 4: "--", 6: "-", 8: "-", 16: "-"}

# ⛔ A PER-SCHEME DASH PATTERN AND WIDTH, BECAUSE THE CURVES GENUINELY COINCIDE. 1-site TDVP-CBE
# and 2-site TDVP agree here to ~1e-14 (measured: S^x_tot 4.785508001962157 vs 4.7855080019621 at
# t = 0.2) -- at these bond dimensions CBE's expansion is COMPLETE, so the two span the same
# subspace and that agreement is the paper's central claim rather than an accident. Drawn with one
# style they hide each other exactly, and a reader cannot tell "three curves agree" from "one
# curve was plotted". Widest-first with narrowing dashes on top makes every arm visible.
SCHEME_STYLE = {          # (linestyle, linewidth, z-order)
    "tdvp2":      ("-",           3.2, 3),
    "tdvp_cbe1s": ((0, (6, 3)),   1.9, 4),
    "cbe_bug":    ((0, (1.5, 2)), 1.5, 5),
}


def load(pattern):
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    if not paths:
        sys.exit("no CSV matched %r -- run benchmarks/oat_l10.jl first" % pattern)
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            rows.extend(dict(r) for r in csv.DictReader(fh))
    if not rows:
        sys.exit("CSV(s) %s are empty" % paths)
    # `scheme` and `sym` are LABELS, not numbers. Coercing `sym` ("Z2"/"none") through float()
    # would turn it into NaN and quietly destroy the only column that says which symmetry an arm
    # was run under -- and two arms that differ ONLY in symmetry would then be indistinguishable.
    for r in rows:
        for k, v in list(r.items()):
            if k in ("scheme", "sym"):
                continue
            try:
                r[k] = float(v)
            except (TypeError, ValueError):
                r[k] = float("nan")
    print("loaded %d rows from %s" % (len(rows), ", ".join(os.path.basename(p) for p in paths)))
    return rows


def series(rows):
    """(scheme, maxdim) -> rows, time-ordered."""
    out = defaultdict(list)
    for r in rows:
        out[(r["scheme"], int(r["maxdim"]))].append(r)
    for k in out:
        out[k].sort(key=lambda r: r["t"])
    return out


def col(rs, name):
    return [r[name] for r in rs]


def panels(rows, outpath):
    S = series(rows)
    L = int(rows[0]["L"])
    ceiling = min(L // 2, L - L // 2) + 1
    caps = sorted({int(r["maxdim"]) for r in rows})
    # The full-rank arm is what panels (a)-(c) show per scheme: at or above the ceiling the sweep
    # is exact, so these are the curves that test the pipeline rather than the truncation.
    full = min((c for c in caps if c >= ceiling), default=max(caps))

    fig, ax = plt.subplots(2, 2, figsize=(11.5, 7.6))
    (a, b), (c, d) = ax

    # ── (a) S^x_tot(t) ───────────────────────────────────────────────────────
    ref = None
    for (sch, cap), rs in sorted(S.items()):
        if cap == full:
            ref = rs
            break
    if ref:
        a.plot(col(ref, "t"), col(ref, "sx_exact"), color="k", lw=5.5, alpha=0.22,
               label=r"exact  $(L/2)\cos^{L-1}(t/2)$", zorder=1, solid_capstyle="round")
    for (sch, cap), rs in sorted(S.items()):
        if cap != full:
            continue
        ls, lw, z = SCHEME_STYLE.get(sch, ("-", 1.3, 3))
        a.plot(col(rs, "t"), col(rs, "sx"), color=COLOR.get(sch, "gray"),
               ls=ls, lw=lw, label="%s, $D_{max}$=%d" % (LABEL.get(sch, sch), cap), zorder=z)
    # One truncated arm, to show what a rank below the ceiling costs.
    trunc = min(caps)
    if trunc < ceiling:
        for (sch, cap), rs in sorted(S.items()):
            if cap == trunc and sch == "tdvp_cbe1s":
                a.plot(col(rs, "t"), col(rs, "sx"), color=COLOR.get(sch, "gray"),
                       ls=":", lw=1.6, label="%s, $D_{max}$=%d" % (LABEL.get(sch, sch), cap),
                       zorder=2)
    a.set_xlabel("$t$")
    a.set_ylabel(r"$S^{tot}_x(t)$")
    a.set_title("(a) total spin: collapse and revival, $L=%d$" % L, loc="left", fontsize=10)
    a.axhline(0.0, color="0.8", lw=0.6, zorder=0)
    a.legend(fontsize=7.5, loc="lower left")

    # ── (b) entanglement entropy ─────────────────────────────────────────────
    if ref:
        b.plot(col(ref, "t"), col(ref, "ee_exact"), color="k", lw=5.5, alpha=0.22,
               label="exact (closed-form Schmidt spectrum)", zorder=1, solid_capstyle="round")
    for (sch, cap), rs in sorted(S.items()):
        if cap != full:
            continue
        ls, lw, z = SCHEME_STYLE.get(sch, ("-", 1.3, 3))
        b.plot(col(rs, "t"), col(rs, "ee"), color=COLOR.get(sch, "gray"), ls=ls, lw=lw,
               label=LABEL.get(sch, sch), zorder=z)
    b.axhline(math.log(ceiling), color="crimson", lw=0.9, ls="--",
              label=r"$\ln(\min(n,L{-}n){+}1) = \ln %d$" % ceiling)
    b.set_xlabel("$t$")
    b.set_ylabel("entanglement entropy (centre cut)")
    b.set_title("(b) entanglement entropy", loc="left", fontsize=10)
    b.legend(fontsize=7.5, loc="lower right")

    # ── (c) bond dimensions ──────────────────────────────────────────────────
    #
    # ONE ARM PER SCHEME AT FULL RANK, plus the truncated caps of a SINGLE scheme. Plotting all
    # 15 arms is what the data supports and what the legend cannot: 15 entries covered the axes
    # and the reader could not tell which curve saturated where. The cap dependence is not lost --
    # it is exactly what the convergence figure and panel (d) show, with an axis built for it.
    if ref:
        c.plot(col(ref, "t"), col(ref, "rank_exact"), color="k", lw=5.5, alpha=0.22,
               label="exact rank", zorder=1, solid_capstyle="round")
    for (sch, cap), rs in sorted(S.items()):
        if cap != full:
            continue
        ls, lw, z = SCHEME_STYLE.get(sch, ("-", 1.3, 3))
        c.plot(col(rs, "t"), col(rs, "maxbond"), color=COLOR.get(sch, "gray"), ls=ls, lw=lw,
               label="%s, $D_{max}$=%d" % (LABEL.get(sch, sch), cap), zorder=z)
    for (sch, cap), rs in sorted(S.items()):
        if sch != "tdvp_cbe1s" or cap >= ceiling:
            continue
        c.plot(col(rs, "t"), col(rs, "maxbond"), color=COLOR.get(sch, "gray"),
               ls=DASH.get(cap, "--"), lw=1.0, alpha=0.7,
               label="   ″ , $D_{max}$=%d" % cap, zorder=2)
    c.axhline(ceiling, color="crimson", lw=0.9, ls="--",
              label="rank ceiling = %d" % ceiling)
    c.set_xlabel("$t$")
    c.set_ylabel("$D(t)$")
    c.set_title("(c) bond dimension vs the exact rank ceiling", loc="left", fontsize=10)
    c.legend(fontsize=7, loc="lower right", ncol=2)

    # ── (d) error analysis, the paper's panel (d) ────────────────────────────
    for (sch, cap), rs in sorted(S.items()):
        if cap not in (trunc, full):
            continue
        base = COLOR.get(sch, "gray")
        lw = 1.5 if cap == full else 1.0
        d.semilogy(col(rs, "t"), [max(v, 1e-18) for v in col(rs, "sx_density_err")],
                   color=base, ls="-", lw=lw,
                   label=r"$\delta s^{tot}_x$  %s, $D$=%d" % (LABEL.get(sch, sch), cap))
        d.semilogy(col(rs, "t"), [max(v, 1e-18) for v in col(rs, "dE")],
                   color=base, ls="--", lw=lw * 0.8, alpha=0.75)
        xi = [max(v, 1e-18) for v in col(rs, "discarded")]
        if max(xi) > 1e-17:
            d.semilogy(col(rs, "t"), xi, color=base, ls=":", lw=lw * 0.8, alpha=0.75)
    d.set_xlabel("$t$")
    d.set_ylabel("error")
    d.set_title(r"(d) $\delta s^{tot}_x$ (solid), $\delta E$ (dashed), $\xi$ (dotted)",
                loc="left", fontsize=10)
    d.legend(fontsize=6.5, loc="lower right")
    d.set_ylim(1e-17, 1e1)

    fig.suptitle("One-axis twisting, $H=(\\sum_\\ell S^z_\\ell)^2/2$, "
                 "$|\\Psi(0)\\rangle=\\otimes|{\\rightarrow}\\rangle$ "
                 "-- arXiv:2208.10972 Fig. 2 recreated at $L=%d$, $\\delta t=%g$"
                 % (L, rows[0]["dt"]), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    fig.savefig(outpath, dpi=150)
    print("wrote", outpath)


def convergence(rows, outpath):
    """max_t |delta S^x_tot| vs maxdim. The exact rank ceiling must show as a cliff."""
    L = int(rows[0]["L"])
    ceiling = min(L // 2, L - L // 2) + 1
    worst = defaultdict(dict)
    for r in rows:
        k, cap = r["scheme"], int(r["maxdim"])
        worst[k][cap] = max(worst[k].get(cap, 0.0), r["sx_err"])

    fig, ax = plt.subplots(figsize=(6.6, 4.6))
    allcaps = sorted({c for per in worst.values() for c in per})
    for sch, per in sorted(worst.items()):
        caps = sorted(per)
        ls, lw, z = SCHEME_STYLE.get(sch, ("-", 1.3, 3))
        # Same overlap problem as the panels: the two TDVP arms coincide to ~1e-14, so distinct
        # dashes and marker sizes are what keep both readable.
        ax.semilogy(caps, [max(per[c], 1e-18) for c in caps], ls=ls, lw=lw, marker="o",
                    ms=9 - 2 * z + 6, color=COLOR.get(sch, "gray"),
                    label=LABEL.get(sch, sch), zorder=z)
    ax.axvline(ceiling, color="crimson", ls="--", lw=1.0,
               label="exact rank ceiling = %d" % ceiling)
    ax.set_xticks(allcaps)
    ax.set_xticklabels([str(c) for c in allcaps])
    ax.set_xlabel("$D_{max}$")
    ax.set_ylabel(r"$\max_t |S^{tot}_x - S^{tot,exact}_x|$")
    ax.set_title("OAT $L=%d$: error against the closed form vs rank cap" % L,
                 loc="left", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.25, which="both")
    fig.tight_layout()
    fig.savefig(outpath, dpi=150)
    print("wrote", outpath)


def main():
    rows = load(CSV)
    os.makedirs(OUTDIR, exist_ok=True)

    # ⛔ PARTITION BY `L` BEFORE PLOTTING ANYTHING. The default pattern is `oat_L*.csv`, which
    # globs EVERY system size, but the series key is `(scheme, maxdim)` with no `L` in it. So the
    # moment a second size exists, rows from different `L` land in the SAME curve -- and the
    # damage is silent, not loud:
    #
    #   * `sx_exact` is `(L/2)cos^(L-1)(t/2)`, a DIFFERENT curve per `L`, so the "exact" reference
    #     drawn is whichever size happened to sort first;
    #   * `panels()` reads `L = rows[0]["L"]` and derives the rank ceiling from it, so the ceiling
    #     line and the title would label one size while the data is a mixture;
    #   * caps overlap only partly (2/4/16 exist at both L=10 and L=12, but the ceiling arm is 6 at
    #     L=10 and 7 at L=12), so the mixture is uneven across curves rather than uniformly wrong.
    #
    # Nothing about that announces itself in the figure -- it just draws a plausible plot of two
    # different physical systems. So each `L` gets its own figure, and the filename carries it.
    by_L = defaultdict(list)
    for r in rows:
        by_L[int(r["L"])].append(r)
    if len(by_L) > 1:
        print("found %d system sizes (%s) -- writing one figure set per L"
              % (len(by_L), ", ".join("L=%d" % k for k in sorted(by_L))))
    for L, rs in sorted(by_L.items()):
        panels(rs, os.path.join(OUTDIR, "oat_panels_L%d.png" % L))
        convergence(rs, os.path.join(OUTDIR, "oat_convergence_L%d.png" % L))


if __name__ == "__main__":
    main()
