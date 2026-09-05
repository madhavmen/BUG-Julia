"""WHERE DOES CBE-BUG BEAT BOTH TDVP ARMS, AND ON WHICH AXES? The tolerance scan, read as a
comparison rather than as a convergence check.

  bug_wins_L<n>_T<t>_D<cap>.png
      panel 1  L-inf error vs tau_trunc, BUG-wins region shaded
      panel 2  final chi vs tau -- the win is only meaningful AT EQUAL RANK
      panel 3  the cleanest cell as bars: equal chi, error and counted matvec side by side

⛔ THE HEADLINE IS "MORE ACCURATE AT EQUAL RANK", NOT "CHEAPER", AND THE PANELS ARE BUILT TO
   KEEP THOSE APART. Two of the four available axes cannot carry the claim:

   * `krylov` IS NOT A COST AXIS. It counts `apply_one_site` in `_krylov_frame` plus the root
     solve, and is BLIND to every operator application inside `cbe_expand` -- which is ~50% of
     BUG's step and 0% of a TDVP step. So the ~7x fewer matvecs is a LOWER BOUND ON BUG'S COST,
     i.e. an UPPER bound on its advantage, and it is drawn as an annotation with that stated
     rather than as a Pareto front. A Pareto on this axis flatters BUG by construction.

   * `seconds` IS CONTAMINATED ON THIS BOX AND THE DATA SAYS SO ON ITS FACE: BUG's own wall clock
     runs 1290 -> 1741 -> 928 -> 1019 s as tau TIGHTENS, i.e. non-monotonic on strictly
     increasing work. Not plotted at all. (Modern Standby cannot be disabled here and the same
     grid cell has been measured 65x apart.)

⚠ SO THE WIN IS STATED WHERE IT IS CHECKABLE: at tau >= 1e-5 every arm sits at the SAME chi (64,
  the cap) and BUG has the lowest error. That is a comparison with no free parameters left.

⚠ AND THE LOSS IS PLOTTED TOO, at the same weight. Below tau ~ 1e-6 both TDVP arms overtake and
  BUG floors at ~1.3e-04 while they reach 3.0e-05. A figure showing only the left half of this
  scan would be a different claim than the data supports.

usage:  python3 benchmarks/plot_bug_wins.py [resultsdir]
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

# BUG last and widest so a coincidence shows as a halo, and in the ACCENT colour -- the knee
# figure drew it in grey, which reads as "the arm being ignored".
STYLE = [("tdvp2",           "2-site TDVP",     "#33507a", 2.6, 2, "o"),
         ("tdvp_cbe1s",      "1-site TDVP-CBE", "#2a6558", 2.6, 2, "s"),
         ("bug_interleaved", "CBE-BUG",         "#c1440e", 3.2, 3, "D")]


def load(path):
    """-> {scheme: {tau: row}} keeping only rows that reached t_max."""
    rows = list(csv.DictReader(open(path, encoding="utf-8")))
    if not rows:
        return {}, None
    tmax = max(float(r["t"]) for r in rows)
    out = defaultdict(dict)
    for r in rows:
        if abs(float(r["t"]) - tmax) < 1e-9:
            out[r["scheme"]][float(r["tau_trunc"])] = r
    return out, tmax


def figure(path):
    data, tmax = load(path)
    present = [s for s in STYLE if s[0] in data]
    if len(present) < 2:
        return
    taus = sorted({t for s in present for t in data[s[0]]}, reverse=True)
    if len(taus) < 3:
        return

    fig, ax = plt.subplots(1, 3, figsize=(16.5, 5.0))
    fig.subplots_adjust(wspace=.28, top=.78, bottom=.14)

    def series(key, col):
        d = data[key]
        return [float(d[t][col]) if t in d else np.nan for t in taus]

    # ── which taus does BUG actually win, AND at what rank? Computed, never assumed ──────
    # ⚠ THE RANK TEST IS `<= 1.1x`, NOT `<=`, AND THE SLACK IS THE POINT. A strict `<=` threw
    # away the BEST cell in the scan: at tau=1e-4 BUG carries chi=40 against tdvp2's 37 and is
    # 2.6x MORE ACCURATE. Calling that a non-win because of 8% more bond dimension would be a
    # criterion chosen to produce a tidy statement rather than a true one. The actual rank ratio
    # is printed per cell so a reader applies their own threshold.
    bug = "bug_interleaved"
    wins = []
    for t in taus:
        if t not in data.get(bug, {}):
            continue
        others = [data[k][t] for k, *_ in present if k != bug and t in data[k]]
        if not others:
            continue
        eb = float(data[bug][t]["err_prof"])
        cb = int(data[bug][t]["maxbond"])
        co = min(int(o["maxbond"]) for o in others)
        if eb < min(float(o["err_prof"]) for o in others) and cb <= 1.1 * co:
            wins.append(t)

    # ── panel 1: error vs tau ────────────────────────────────────────────────────────────
    if wins:
        ax[0].axvspan(min(wins) / 3.16, max(wins) * 3.16, color="#c1440e", alpha=.08, zorder=0)
        # ⚠ THE LABEL MUST MATCH THE CRITERION ABOVE. It read "rank <= both" after the test was
        # relaxed to `<= 1.1x`, i.e. the figure asserted a stricter condition than it applied --
        # the same class of defect as a hardcoded caption, and just as invisible.
        ax[0].annotate("BUG wins here\n(lowest error, rank within 10%)",
                       xy=(np.sqrt(min(wins) * max(wins)), .93), xycoords=("data", "axes fraction"),
                       ha="center", va="top", fontsize=8.5, color="#c1440e")
    for key, lab, col, lw, z, mk in present:
        ax[0].loglog(taus, series(key, "err_prof"), mk + "-", color=col, lw=lw, ms=5, zorder=z,
                     label=lab)
    ax[0].invert_xaxis()
    ax[0].set_xlabel(r"root tolerance $\tau_{\rm trunc}$  (tightening $\rightarrow$)")
    ax[0].set_ylabel(r"$\max_j|\langle S^z_j\rangle-$exact$|$")
    ax[0].set_title("Accuracy.  BUG is best at LOOSE tolerance and floors\n"
                    "at ~1.3e-04, where both TDVP arms keep improving.", fontsize=9.5)
    ax[0].legend(fontsize=8.5, frameon=False, loc="lower left")
    ax[0].grid(True, which="both", lw=.4, alpha=.35)

    # ── panel 2: rank ────────────────────────────────────────────────────────────────────
    # ⛔ WITHOUT THIS PANEL THE WIN IS UNINTERPRETABLE. "Lower error" bought with more bond
    # dimension is not a win, and BUG has been measured carrying 1.2-2.3x the rank of the TDVP
    # arms under other settings. Here every arm pins at the cap for tau <= 1e-5, which is what
    # makes those cells a clean comparison.
    for key, lab, col, lw, z, mk in present:
        ax[1].semilogx(taus, series(key, "maxbond"), mk + "-", color=col, lw=lw, ms=5, zorder=z,
                       label=lab)
    cap = re.search(r"_D(\d+|inf)_", os.path.basename(path))
    cv = None
    if cap and cap.group(1) != "inf":
        cv = float(cap.group(1))
        ax[1].axhline(cv, ls="--", lw=1.2, color="#555")
        ax[1].text(.99, cv, " cap %.0f " % cv, ha="right", va="bottom", fontsize=8,
                   color="#555", transform=ax[1].get_yaxis_transform())
    ax[1].invert_xaxis()
    ax[1].set_xlabel(r"root tolerance $\tau_{\rm trunc}$")
    ax[1].set_ylabel(r"final $\chi$")
    # ⛔ THIS TITLE IS DERIVED, NOT WRITTEN. It used to assert "all three PIN AT THE CAP for
    # tau <= 1e-5" -- true of the cap-64 run it was authored against, and FALSE at cap 128, where
    # the same tolerance leaves them at chi = 40/71/72 and they only reach the cap at 1e-7. A
    # hardcoded caption survives the data changing underneath it and is invisible in review.
    pinned = sorted([t for t in taus
                     if cv is not None
                     and all(t in data[k] and abs(int(data[k][t]["maxbond"]) - cv) < 1
                             for k, *_ in present)], reverse=True)
    if pinned:
        ax[1].set_title("Rank.  All three PIN AT THE CAP for $\\tau\\leq$%g, so the\n"
                        "accuracy comparison there has no free parameter left." % max(pinned),
                        fontsize=9.5)
    else:
        ax[1].set_title("Rank.  NO tolerance here pins every arm at the cap, so every\n"
                        "accuracy claim below must be read against this panel.", fontsize=9.5)
    ax[1].legend(fontsize=8.5, frameon=False, loc="lower right")
    ax[1].grid(True, which="both", lw=.4, alpha=.35)

    # ── panel 3: the cleanest equal-rank cell, error and matvec side by side ─────────────
    # EVERY winning cell, coarsest first -- not one hand-picked one. The rank is printed under
    # each group because it is the qualifier that decides whether the cell counts.
    cells = sorted(wins, reverse=True)[:3]
    if cells:
        a = ax[2]
        keys = [k for k, *_ in present]
        cols = [c for _, _, c, *_ in present]
        w = 0.26
        gx = np.arange(len(cells))
        for j, (key, lab, col, *_r) in enumerate(present):
            y = [float(data[key][t]["err_prof"]) if t in data[key] else np.nan for t in cells]
            a.bar(gx + (j - 1) * w, y, w, color=col, alpha=.85, label=lab,
                  edgecolor="white", linewidth=.5)
            for i, v in enumerate(y):
                if not np.isnan(v):
                    a.text(gx[i] + (j - 1) * w, v * 1.06, "%.2e" % v, ha="center",
                           fontsize=6.8, rotation=90, va="bottom")
        a.set_yscale("log")
        a.set_ylabel(r"$\max_j|\langle S^z_j\rangle-$exact$|$")
        # Tick labels carry tau, rank and the matvec ratio -- the three numbers that make the
        # cell interpretable, on the axis rather than in prose.
        ticks = []
        for t in cells:
            rk = "/".join(str(int(data[k][t]["maxbond"])) for k in keys if t in data[k])
            mvb = int(data[bug][t]["krylov"])
            mvo = min(int(data[k][t]["krylov"]) for k in keys if k != bug and t in data[k])
            adv = min(float(data[k][t]["err_prof"])
                      for k in keys if k != bug and t in data[k]) / float(data[bug][t]["err_prof"])
            ticks.append(r"$\tau$=%g" % t + "\n$\\chi$ = %s" % rk +
                         "\n%.1f$\\times$ better\n%.0f$\\times$ fewer matvec" % (adv, mvo / mvb))
        a.set_xticks(gx); a.set_xticklabels(ticks, fontsize=7.5)
        # With a single winning cell the axis auto-scales to the one bar group and the bars render
        # as three absurdly wide slabs. Pad the x range so one group looks like one group.
        a.set_xlim(-0.62, len(cells) - 0.38)
        a.set_ylim(min(v for t in cells for k in keys if t in data[k]
                       for v in [float(data[k][t]["err_prof"])]) / 2.5,
                   max(v for t in cells for k in keys if t in data[k]
                       for v in [float(data[k][t]["err_prof"])]) * 6)
        a.set_title("The cells BUG wins.  $\\chi$ order = %s.\n"
                    "⚠ matvec is BLIND to cbe_expand: a LOWER bound on BUG's cost."
                    % ", ".join(l for _, l, *_ in present), fontsize=8.5)
        a.legend(fontsize=8, frameon=False, loc="upper right")
        a.grid(True, axis="y", which="both", lw=.4, alpha=.35)

    L = data[present[0][0]][taus[0]]["L"]
    fig.suptitle("Where CBE-BUG beats both TDVP arms — XX chain ($\\Delta$=0), U(1), "
                 "domain wall, L=%s, T=%g, dt=%s"
                 % (L, tmax, data[present[0][0]][taus[0]]["dt"]), fontsize=11, y=.95)
    tag = re.sub(r"^heis_tau|\.csv$", "", os.path.basename(path))
    out = os.path.join(RES, "bug_wins%s.png" % tag)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    pats = sorted(glob.glob(os.path.join(RES, "heis_tau_L*.csv")))
    if not pats:
        print("no heis_tau_*.csv in", RES)
        return
    for p in pats:
        if "_prof_" in os.path.basename(p):
            continue
        figure(p)


if __name__ == "__main__":
    main()
