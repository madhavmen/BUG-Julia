"""The two independent CBE-BUG speedups against bond dimension: SKETCH x FOLD x PARALLEL.

  rsvd_parallel_<model>.png
      panel 1  t_cbe vs chi, SERIAL arms only     -- the scaling claim
      panel 2  total step seconds vs chi          -- what it is worth end to end
      panel 3  parallel speedup per CBE variant   -- serial/parallel pairs
      panel 4  accuracy: dprof and err_t2         -- that none of it costs anything

⛔ PANEL 1 IS SERIAL-ARMS-ONLY AND THAT IS NOT A PRESENTATION CHOICE. The phase timers are CPU
   seconds SUMMED OVER BOTH HALF-SWEEPS while `secs`/`t_step` is WALL clock (cbe_bug.jl:955).
   Serially they are the same clock; under `parallel = true` two sweeps run at once and spend
   MORE CPU than the step lasts. So a parallel arm's `t_cbe` is systematically inflated and
   plotting it beside a serial arm's would read as a regression -- MEASURED, `exact parallel`
   shows t_cbe 2.514 against `exact serial`'s 1.979 at chi=256 while being 1.34x FASTER. Panel 3
   is where the parallel arms belong, as a RATIO of wall clocks.

⛔ THE `krylov` COLUMN IS NOT PLOTTED AS A COST AXIS ANYWHERE. It counts `apply_one_site` in
   `_krylov_frame` plus the root solve and is BLIND to every operator application inside
   `cbe_expand`. It is used here only to assert the CBE arms did IDENTICAL work (the annotation
   in panel 2); if that assertion fails the seconds are not comparable and the panel says so.

⚠ tdvp2 IS DRAWN BUT ITS SECONDS ARE NOT A COMPARISON. `rsvd_parallel.jl` pins the CBE arms at
  `maxiter = 20, tol = 0` so they do matched operator work, and deliberately EXEMPTS tdvp2, which
  keeps its own convergence exit -- it runs ~1980 matvecs per step against the CBE arms' 74, i.e.
  a ~10x deeper solve than the campaign's m=3. It is the ACCURACY reference (`err_t2`). Its bar is
  hatched and labelled to stop it being read as a 5x win.

⚠ SINGLE-RUN NUMBERS ON A CONTENDED BOX. `t_root` is pinned work and still spreads 3x between
  arms at chi=64; two runs of the identical chi=128 configuration once disagreed on the SIGN of
  the sketch-vs-exact difference. What this figure is for is the SHAPE ACROSS chi -- flat vs
  growing -- which is a trend over three ranks and survives what any single ratio does not.

usage:  python3 benchmarks/plot_rsvd_parallel.py [resultsdir]
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")
CSVP = os.path.join(RES, "rsvd_parallel.csv")

# arm key -> (label, colour, marker). Keys are the CSV's underscored `arm` strings.
SERIAL = [("exact__serial_OLD",   "exact, pre-fix (3x HΘ)", "#8a8a8a", "v"),
          ("exact__serial",       "exact CBE",                   "#33507a", "o"),
          ("sketch_serial",       "rSVD sketch",                 "#2a6558", "s"),
          ("sketch_serial_FOLD",  "rSVD sketch + FOLD",          "#c1440e", "D")]
PAIRS  = [("exact__serial",      "exact__parallel",      "exact CBE",  "#33507a"),
          ("sketch_serial",      "sketch_parallel",      "sketch",     "#2a6558"),
          ("sketch_serial_FOLD", "sketch_parallel_FOLD", "sketch+FOLD", "#c1440e")]
T2 = "tdvp2_(reference)"


def load(model):
    """-> {arm: {chi: row}} for one model, newest row per (arm, chi)."""
    out = defaultdict(dict)
    with open(CSVP, encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            if r.get("model") != model:
                continue
            try:
                out[r["arm"]][int(r["chi"])] = r
            except (TypeError, ValueError):
                continue
    return out


def figure(model, data):
    chis = sorted({c for d in data.values() for c in d})
    if len(chis) < 2:
        print("  %s: need at least two chi values, have %s" % (model, chis))
        return
    x = np.arange(len(chis))

    fig, ax = plt.subplots(1, 4, figsize=(19.5, 5.0))
    fig.subplots_adjust(wspace=.30, top=.74, bottom=.14)

    # ── 1. t_cbe against chi, SERIAL ONLY (see the header) ───────────────────────────────
    for key, lab, col, mk in SERIAL:
        d = data.get(key, {})
        y = [float(d[c]["t_cbe"]) if c in d else np.nan for c in chis]
        ax[0].plot(x, y, mk + "-", color=col, lw=2.0, ms=6, label=lab)
    ax[0].set_xticks(x); ax[0].set_xticklabels(chis)
    ax[0].set_xlabel(r"bond dimension $\chi$")
    ax[0].set_ylabel(r"$t_{\rm cbe}$  (s, CPU, per step)")
    ax[0].set_title("Expansion cost.  FLAT vs GROWING is the claim:\n"
                    r"folding $\Omega$ in first makes it $O(\chi^2)$, not $O(\chi^3)$.",
                    fontsize=9.5)
    ax[0].legend(fontsize=8, frameon=False)
    ax[0].grid(True, lw=.4, alpha=.35)

    # ── 2. total step seconds, every arm ────────────────────────────────────────────────
    w = 0.14
    allarms = [(k, l, c, m) for k, l, c, m in SERIAL] + \
              [(b, l + " ∥", c, "^") for _, b, l, c in PAIRS]
    for i, (key, lab, col, _mk) in enumerate(allarms):
        d = data.get(key, {})
        y = [float(d[c]["secs"]) if c in d else np.nan for c in chis]
        par = key.find("parallel") >= 0
        ax[1].bar(x + (i - len(allarms) / 2) * w, y, w, color=col, label=lab,
                  alpha=.95 if par else .65,
                  hatch="//" if par else None, edgecolor="white", linewidth=.4)
    # ⛔ THE TWO CAVEATS GO IN THE TITLE, NOT AS annotate() INSIDE THE AXES. With eight bars
    # across three groups there is no free corner: the first version put both notes at the top
    # left and the legend on top of them, and the panel was illegible. The title has room and
    # cannot be overlapped.
    d2 = data.get(T2, {})
    t2txt = ""
    if d2:
        # ⚠ KEEP THIS SHORT. A one-line title wider than the axes overruns into the panels on
        # BOTH sides -- matplotlib centres it on the axes and does not clip it.
        t2txt = ("\ntdvp2 off-axis (depth unmatched): "
                 + " / ".join("%.1f" % float(d2[c]["secs"]) for c in chis if c in d2) + " s")
    # The matched-work assertion, CHECKED rather than asserted: if the CBE arms did different
    # operator work their seconds are not comparable and the title has to say so.
    krys = {int(r["kry"]) for k, dd in data.items() if k != T2 for r in dd.values()}
    krytxt = ("\n!! CBE arms did DIFFERENT operator work (%s matvec) — seconds NOT comparable"
              % "/".join(str(k) for k in sorted(krys))) if len(krys) > 1 else \
             "\nCBE arms matched at %d matvec ✓" % next(iter(krys), 0)
    ax[1].set_xticks(x); ax[1].set_xticklabels(chis)
    ax[1].set_xlabel(r"bond dimension $\chi$"); ax[1].set_ylabel("wall clock per step (s)")
    ax[1].set_title("Whole step.  Hatched = parallel half-sweeps." + krytxt + t2txt,
                    fontsize=8.5)
    # Headroom so the legend clears the tallest bar instead of sitting on it.
    top = np.nanmax([float(r["secs"]) for k, dd in data.items() if k != T2
                     for r in dd.values()])
    ax[1].set_ylim(0, top * 1.42)
    ax[1].legend(fontsize=6.5, frameon=False, ncol=2, loc="upper left")
    ax[1].grid(True, axis="y", lw=.4, alpha=.35)

    # ── 3. parallel speedup, as a ratio of WALL clocks within each CBE variant ──────────
    for i, (ser, par, lab, col) in enumerate(PAIRS):
        ds, dp = data.get(ser, {}), data.get(par, {})
        y = [float(ds[c]["secs"]) / float(dp[c]["secs"])
             if (c in ds and c in dp) else np.nan for c in chis]
        ax[2].bar(x + (i - 1) * 0.26, y, 0.26, color=col, label=lab, alpha=.85,
                  edgecolor="white", linewidth=.5)
    ax[2].axhline(1.0, ls="--", lw=1.2, color="#555")
    ax[2].axhline(2.0, ls=":", lw=1.0, color="#999")
    ax[2].text(.99, 2.0, " ideal (2 workers) ", ha="right", va="bottom", fontsize=7.5,
               color="#999", transform=ax[2].get_yaxis_transform())
    ax[2].set_xticks(x); ax[2].set_xticklabels(chis)
    ax[2].set_xlabel(r"bond dimension $\chi$"); ax[2].set_ylabel(r"serial / parallel  (wall)")
    ax[2].set_title("Parallel half-sweeps.  Below 2 because the root solve\n"
                    "and closing truncation are OUTSIDE the parallel region.", fontsize=9.5)
    ax[2].legend(fontsize=8, frameon=False)
    ax[2].grid(True, axis="y", lw=.4, alpha=.35)

    # ── 4. accuracy: none of this is allowed to cost anything ───────────────────────────
    # ⛔ SKIP THE FIRST TWO ARMS: `dprof` is measured against the FIRST arm in `ARMS` order, so
    # for `exact serial OLD` (and effectively for `exact serial`, which is bit-identical to it)
    # it is zero BY CONSTRUCTION. Plotting it draws a flat line pinned to the bottom of the log
    # axis that looks like a measured result — the same defect as reading the gap-to-tdvp2 off
    # the tdvp2 row. Only the arms that actually DIFFER from the reference belong here.
    for key, lab, col, mk in SERIAL[2:]:
        d = data.get(key, {})
        y = [max(float(d[c]["dprof"]), 1e-17) if c in d else np.nan for c in chis]
        ax[3].semilogy(x, y, mk + "-", color=col, lw=2.0, ms=6,
                       label=lab + "  vs exact CBE")
    # ⛔ THE GAP TO tdvp2 COMES FROM A **CBE** ROW, NOT FROM THE tdvp2 ROW. `err_t2` is measured
    # AGAINST tdvp2, so on the tdvp2 row it is identically 0 -- tdvp2 compared with itself. Read
    # from there it plots as a flat line pinned to the bottom of a log axis, which is exactly what
    # the first version drew, and it silently mislabelled "zero by construction" as "the two
    # integrators agree to machine precision".
    dref = data.get("exact__serial", {})
    if dref:
        y = [float(dref[c]["err_t2"]) if c in dref else np.nan for c in chis]
        ax[3].semilogy(x, y, "k--", lw=1.4,
                       label="BUG vs tdvp2 (discretisation)")
    ax[3].set_xticks(x); ax[3].set_xticklabels(chis)
    ax[3].set_xlabel(r"bond dimension $\chi$")
    ax[3].set_ylabel(r"max$_j|\langle S^z_j\rangle$ difference$|$")
    ax[3].set_title("Cost is free.  Sketch and FOLD sit at ROUNDOFF against the\n"
                    "exact expansion, orders below the integrator's own error.", fontsize=9.5)
    ax[3].legend(fontsize=7.5, frameon=False, loc="lower left")
    ax[3].grid(True, which="both", lw=.4, alpha=.35)

    row = next(iter(next(iter(data.values())).values()))
    fig.suptitle("CBE-BUG: randomised sketch, fold-$\\Omega$-first and parallel half-sweeps "
                 "— %s, L=%s, MPO virtual dim %s, one step from the SAME state, best of 5"
                 % (model, row["L"], row["mpodim"]), fontsize=11, y=.965)
    out = os.path.join(RES, "rsvd_parallel_%s.png" % model)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    if not os.path.exists(CSVP):
        print("no", CSVP)
        return
    with open(CSVP, encoding="utf-8") as fh:
        models = sorted({r["model"] for r in csv.DictReader(fh) if r.get("model")})
    for m in models:
        figure(m, load(m))


if __name__ == "__main__":
    main()
