"""IS CBE-BUG'S ERROR FLOOR A TIME-ORDER DEFICIT OR A LARGER CONSTANT? The dt sweep that the
tolerance campaign could not answer, because every arm in it ran at ONE dt (0.05).

  dt_order_L<n>_T<t>_D<cap>.png
      panel 1  err_prof(t_max) vs dt, log-log, fitted slope per scheme + slope-1/slope-2 guides
      panel 2  THE ANSWER PANEL: the bug/TDVP error RATIO vs dt

⛔ PANEL 2 IS WHY THIS FIGURE EXISTS, AND PANEL 1 CANNOT SUBSTITUTE FOR IT.
   MEASURED at dt=0.05 (L=18, cap 128, tau=1e-7): the bug/tdvp_cbe1s err_prof ratio is 3.82-4.07 at
   EVERY sampled time from t=0.5 (chi=6, nothing truncated) to t=20 (chi=128) -- a 20x span of rank
   with no movement. So the 4x gap is the LOCAL STEP ERROR, not the basis, not truncation, not the
   cap. That kills the explanations the tolerance campaign was built to test but does NOT identify
   the cause, because AT FIXED dt a difference in ORDER and a difference in CONSTANT both show up
   as a constant ratio. Only sweeping dt separates them:

     ratio FLAT in dt          -> same order, BUG's local constant is ~4x larger
                                  => the fix must shrink the constant; raising the order will move
                                     both arms together and the gap survives.
     ratio GROWS as dt shrinks -> BUG is a LOWER ORDER (order 1 vs 2 predicts the ratio DOUBLES per
                                  halving of dt: (C1*dt)/(C2*dt^2) = (C1/C2)/dt)
                                  => the fix is the 2402.08607 midpoint step or composite root
                                     steps, and it should close the gap outright.

⚠ THE SLOPES IN PANEL 1 ARE FITTED ON THE FINEST THREE POINTS ONLY. The coarse end of a dt ladder
  leaves the asymptotic regime (the dt=0.2 arm is 10 steps for the whole trajectory), and including
  it drags a genuine order-2 slope toward 1.5. The fit window is stated on the figure, and every
  point is plotted so the reader can see where the curve straightens.

⚠ SCORED AT t_max=2, NOT t_max=20, AND THAT IS DELIBERATE. At t=2 the effect is already fully
  developed (ratio 3.98) while chi is ~12 against a cap of 64 -- so NOTHING is truncated and the
  measurement is pure time-discretisation error. At T=20 the arms pin at the cap and truncation
  would contaminate the slope. See `dt-order-tests-need-rank-limited-regime`.

⚠ THE dt LADDER MUST DIVIDE `sample_every`. `run_arm` compares against a reference built on a TIME
  grid; a dt that does not land on it was silently compared against the exact profile at a
  DIFFERENT time until this was fixed (2026-09-03). `sample_every=0.1` admits the whole ladder.

usage:  python3 benchmarks/plot_dt_order.py [resultsdir]
"""

import csv
import glob
import os
import re
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")

STYLE = [("tdvp2",           "2-site TDVP",     "#33507a", "o"),
         ("tdvp_cbe1s",      "1-site TDVP-CBE", "#2a6558", "s"),
         ("bug_interleaved", "CBE-BUG",         "#c1440e", "D")]

NFIT = 3   # fit the slope on the finest NFIT points only -- see the docstring


def load(path):
    """-> ({scheme: {dt: err}}, {scheme: {dt: chi}}, tmax)"""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return {}, {}, None
    tmax = max(float(r["t"]) for r in rows)
    err, chi = defaultdict(dict), defaultdict(dict)
    for r in rows:
        if abs(float(r["t"]) - tmax) < 1e-9:
            err[r["scheme"]][float(r["dt"])] = float(r["err_prof"])
            chi[r["scheme"]][float(r["dt"])] = int(r["maxbond"])
    return err, chi, tmax


def slope(dts, es, n=NFIT):
    """least-squares log-log slope over the n FINEST points; None if too few."""
    pts = sorted(zip(dts, es))[:n]
    if len(pts) < 2:
        return None
    x = np.log(np.array([p[0] for p in pts]))
    y = np.log(np.array([p[1] for p in pts]))
    return float(np.polyfit(x, y, 1)[0])


def figure(path):
    err, chi, tmax = load(path)
    present = [s for s in STYLE if s[0] in err and len(err[s[0]]) >= 2]
    if not present:
        return
    cap = re.search(r"_D(\d+|inf)", os.path.basename(path))
    cap = cap.group(1) if cap else "?"
    Lm = re.search(r"_L(\d+)", os.path.basename(path))
    L = Lm.group(1) if Lm else "?"

    fig, ax = plt.subplots(1, 2, figsize=(13.0, 5.2))
    fig.subplots_adjust(wspace=.26, top=.80, bottom=.15)

    # ── panel 1: convergence ────────────────────────────────────────────────────────────────
    allmin, allmax = np.inf, 0.0
    for key, lab, col, mk in present:
        d = sorted(err[key])
        e = [err[key][x] for x in d]
        allmin, allmax = min(allmin, min(e)), max(allmax, max(e))
        p = slope(d, e)
        ax[0].loglog(d, e, mk + "-", color=col, lw=2.4, ms=6,
                     label="%s   slope %s" % (lab, ("%.2f" % p) if p is not None else "n/a"))
    # Slope guides, anchored BELOW the lowest curve so they never collide with the data. The
    # slope-2 guide is drawn last and faint: it lies exactly ON the measured lines, which is the
    # point, so it must not be mistaken for one of them. Labels go at the LEFT end -- the right end
    # is where the coarse-dt markers are.
    dref = min(min(err[k]) for k, *_ in present)
    eref = min(err[k][min(err[k])] for k, *_ in present) / 3.0
    dd = np.array([dref, dref * 16])
    for p, ls in ((1, ":"), (2, "--")):
        ax[0].loglog(dd, eref * (dd / dref) ** p, ls, color="#999", lw=1.1, zorder=0)
        # ⚠ LABEL AT THE GUIDE'S OWN RIGHT END, NOT THE SHARED LEFT ANCHOR. Both guides start at
        # the same point, so labelling there stacks the two labels on top of each other AND puts
        # the slope-2 line straight through its own label. At the right end they are a factor 16
        # apart and both sit below every data curve.
        ax[0].annotate(r"$\propto dt^%d$" % p, xy=(dd[-1], eref * 16 ** p), xytext=(-3, 3),
                       textcoords="offset points", fontsize=8, color="#999", ha="right")

    # ⚠ THE MATCHED-ACCURACY STATEMENT, DRAWN RATHER THAN ASSERTED. Both arms are order 2, so a 4x
    # constant is EXACTLY one halving of dt -- and on a log-log plot that is a horizontal line
    # cutting the two curves at two dt values a factor 2 apart. This is the only cost-relevant
    # claim in the figure that does not go through the broken `krylov` counter.
    bugk = "bug_interleaved"
    if bugk in err and "tdvp_cbe1s" in err:
        db = sorted(err[bugk])
        # the finest bug point whose error also appears on the cbe1s curve at ~2x the dt
        for x in db:
            tgt = err[bugk][x]
            hit = [y for y in err["tdvp_cbe1s"] if abs(err["tdvp_cbe1s"][y] / tgt - 1) < .05]
            if hit and x < max(db):
                y = hit[0]
                ax[0].plot([x, y], [tgt, tgt], "-", color="#444", lw=1.0, alpha=.8, zorder=4)
                ax[0].plot([x, y], [tgt, tgt], "|", color="#444", ms=7, zorder=4)
                ax[0].annotate("same error at $dt$ and $2dt$:\nBUG needs 2$\\times$ the steps",
                               xy=(np.sqrt(x * y), tgt), xytext=(0, 7),
                               textcoords="offset points", ha="center", va="bottom",
                               fontsize=7.8, color="#444")
                break
    ax[0].set_xlabel("$dt$")
    ax[0].set_ylabel(r"$\max_j|\langle S^z_j\rangle-$exact$|$ at $t=%g$" % tmax)
    ax[0].set_title("Convergence in $dt$.  Slope fitted on the %d FINEST points\n"
                    "(the coarse end leaves the asymptotic regime)." % NFIT, fontsize=9.5)
    ax[0].legend(fontsize=8.5, frameon=False, loc="upper left")
    ax[0].grid(True, which="both", lw=.4, alpha=.35)

    # ── panel 2: the ratio -- the panel that actually decides ───────────────────────────────
    bug = "bug_interleaved"
    drew = []
    # ⛔ LOG y, NOT LINEAR. The rejected hypothesis' prediction runs to ~80 while the data sits at
    # 4-5, so on a linear axis the prediction sets the scale and squashes BOTH measured curves into
    # one indistinguishable line at the bottom -- the figure would show the contrast and hide the
    # result. On a log axis the flatness is legible AND the divergence is still obvious.
    if bug in err:
        for key, lab, col, mk in present:
            if key == bug:
                continue
            ds = sorted(set(err[bug]) & set(err[key]))
            if len(ds) < 2:
                continue
            rt = [err[bug][x] / err[key][x] for x in ds]
            ax[1].loglog(ds, rt, mk + "-", color=col, lw=2.4, ms=6,
                         label="BUG / %s   (%.2f-%.2f)" % (lab, min(rt), max(rt)))
            drew.append((ds, rt, col))
        if drew:
            ds = sorted(err[bug])
            # ⚠ ANCHOR THE REJECTED HYPOTHESIS AT THE **COARSEST** dt, NOT THE FINEST. Both
            # hypotheses agree there by construction (it is the anchor), and the prediction then
            # RISES toward fine dt -- which is the actual claim: "if BUG were one order lower,
            # REFINING dt would make the gap worse". Anchored at the finest point instead it
            # descends to the right and reads backwards, as if the rejected curve were the lower one.
            r0 = err[bug][ds[-1]] / min(err[k][ds[-1]] for k, *_ in present if k != bug)
            pred = [r0 * (ds[-1] / x) for x in ds]
            ax[1].loglog(ds, pred, "--", color="#999", lw=1.3, zorder=0,
                         label="if BUG were order 1 (rejected)")
            # The legend already names this line, so the annotation carries ONLY the payload --
            # what the rejected hypothesis would have predicted. Repeating "one order lower" here
            # put this text against the legend's own entry for the same line.
            ax[1].annotate("would reach %.0f$\\times$ here" % max(pred),
                           xy=(ds[0], max(pred)), xytext=(8, -6), textcoords="offset points",
                           fontsize=8, color="#777", ha="left", va="top")
            # ⚠ THE ONLY FREE REGION IS THE WEDGE between the rising prediction (top-left to
            # bottom-right) and the flat data lines. "Below the flat lines" is a thin strip -- the
            # data sits near the bottom of the range by construction -- and putting four lines of
            # text there collided with both the lines and the legend.
            ax[1].annotate("MEASURED: FLAT over 16$\\times$ in $dt$\n"
                           "$\\Rightarrow$ same order, ~4$\\times$ larger constant\n"
                           "$\\Rightarrow$ a higher-order BUG moves BOTH arms;\n"
                           "the gap must come out of the CONSTANT",
                           xy=(0.04, 0.34), xycoords="axes fraction", fontsize=8.2,
                           color="#c1440e", ha="left", va="center")
            ax[1].set_ylim(min(min(r) for _, r, _ in drew) / 1.9, max(pred) * 1.7)
            # Plain numbers: over a 2-100 range matplotlib's default "6x10^0" ticks are unreadable.
            for axis in (ax[1].yaxis,):
                axis.set_major_formatter(matplotlib.ticker.ScalarFormatter())
                axis.set_minor_formatter(matplotlib.ticker.NullFormatter())
            ax[1].set_yticks([2, 3, 4, 5, 10, 20, 40, 80])
    ax[1].set_xlabel("$dt$")
    ax[1].set_ylabel("error ratio  BUG / TDVP")
    ax[1].set_title("THE ANSWER PANEL.  At fixed $dt$ an order deficit and a larger\n"
                    "constant are indistinguishable; only this sweep separates them.",
                    fontsize=9.5)
    # Upper right: the prediction descends to the bottom-right, so this corner is the free one.
    ax[1].legend(fontsize=8.2, frameon=False, loc="upper right")
    ax[1].grid(True, which="both", lw=.4, alpha=.35)

    chis = sorted({v for k, *_ in present for v in chi[k].values()})
    fig.suptitle("Time-discretisation order — XX chain ($\\Delta$=0), U(1), domain wall, "
                 "L=%s, T=%g, cap %s, $\\tau$=1e-8.   $\\chi$ reached %d-%d $\\ll$ cap, "
                 "so NOTHING is truncated." % (L, tmax, cap, chis[0], chis[-1]),
                 fontsize=10.5, y=.955)
    out = os.path.join(RES, "dt_order_L%s_T%g_D%s.png" % (L, tmax, cap))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)

    # ── the numbers, because the figure is a summary of them ────────────────────────────────
    print("\n%-18s %-9s %-11s %-6s" % ("scheme", "dt", "err_prof", "chi"))
    for key, lab, col, mk in present:
        for d in sorted(err[key]):
            print("%-18s %-9g %-11.4e %-6d" % (key, d, err[key][d], chi[key][d]))
        p = slope(sorted(err[key]), [err[key][x] for x in sorted(err[key])])
        print("%-18s fitted slope (finest %d): %s\n" % (key, NFIT,
              ("%.3f" % p) if p is not None else "n/a"))
    if bug in err:
        print("%-9s %-12s %-12s" % ("dt", "BUG/cbe1s", "BUG/tdvp2"))
        for d in sorted(err[bug]):
            a = err["tdvp_cbe1s"].get(d)
            b = err["tdvp2"].get(d)
            print("%-9g %-12s %-12s" % (
                d,
                "%.3f" % (err[bug][d] / a) if a else "-",
                "%.3f" % (err[bug][d] / b) if b else "-"))


def main():
    pats = sorted(glob.glob(os.path.join(RES, "heis_dt_*.csv")))
    if not pats:
        print("no heis_dt_*.csv in", RES)
        return
    for p in pats:
        if "_prof_" in os.path.basename(p):
            continue
        figure(p)


if __name__ == "__main__":
    main()
