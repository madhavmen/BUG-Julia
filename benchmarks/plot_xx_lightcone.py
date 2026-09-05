"""Three-scheme XX domain-wall comparison: light cone, chi(t) and error(t).

  heis_lightcone_L<n>_D<cap>_m<iter>.png
      row 1   the EXACT light cone, then each scheme's OWN <Sz_j(t)> cone, shared colour scale
      row 2   max bond dimension against t | L-infinity error against t | wall clock

⚠ ROW 1 PLOTS THE OBSERVABLE, NOT THE ERROR, BY REQUEST -- and the reader must be told what that
  does and does not show. At these tolerances every arm tracks the exact profile to ~1e-4 on a
  colour axis spanning [-0.5, +0.5]: a 5000:1 ratio, i.e. under one part in a thousand of one
  colour step. THE FOUR PANELS ARE THEREFORE EXPECTED TO BE INDISTINGUISHABLE BY EYE, and that
  agreement IS the result being shown -- all three integrators reproduce the same light cone.
  What they do NOT show is which one is right for the wrong reasons, or by how much they differ.
  That lives in the error panel of row 2, which is quantitative and is where any claim about
  relative accuracy must come from. The panels say so on their faces rather than leaving a reader
  to conclude "all four look the same, so the schemes are equivalent".

⛔ ALL FOUR CONES SHARE ONE COLOUR SCALE, FIXED AT [-0.5, +0.5]. Per-panel autoscaling is the
   classic way to make one scheme look like another -- each panel would renormalise to its own
   extremes, so a scheme that had lost norm or overshot would be redrawn as if it had not. The
   scale is the physical range of <S^z>, not the data's range.

⚠ THE REFERENCE IS EXACT, NOT A FINE-GRID PROXY. `sz_exact` is the closed-form free-fermion
  profile at Delta=0, so these are true errors. See plot_heis_profiles.py.

usage:  python3 benchmarks/plot_xx_lightcone.py [resultsdir]
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
import matplotlib.gridspec as gridspec
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")

FLOOR = 1e-16
# Draw order puts the widest first so a coincidence shows as a halo rather than an erased curve.
STYLE = [("tdvp2",              "2-site TDVP",      "#33507a", 4.2, 1),
         ("tdvp_cbe1s",         "1-site TDVP-CBE",  "#2a6558", 2.4, 2),
         ("bug_interleaved_m3", "CBE-BUG (m3)",     "#c1440e", 1.6, 3)]


def fmt(x):
    "1e-06 rather than 1.0000000000000001e-06, and without trailing zeros."
    return "%g" % float(x)


def load_prof(path):
    """-> {scheme: (ts, sites, sz, exact)} with sz/exact as (nt, nsite) arrays."""
    cells = defaultdict(dict)
    for r in csv.DictReader(open(path)):
        cells[r["scheme"]][(float(r["t"]), int(r["site"]))] = (float(r["sz_mps"]),
                                                               float(r["sz_exact"]))
    out = {}
    for sch, d in cells.items():
        ts = sorted({t for t, _ in d})
        sites = sorted({s for _, s in d})
        sz = np.full((len(ts), len(sites)), np.nan)
        ex = np.full((len(ts), len(sites)), np.nan)
        for i, t in enumerate(ts):
            for j, s in enumerate(sites):
                if (t, s) in d:
                    sz[i, j], ex[i, j] = d[(t, s)]
        out[sch] = (np.array(ts), np.array(sites), sz, ex)
    return out


def load_scalar(path):
    """-> {scheme: (ts, maxbond, err_prof, krylov, seconds)}"""
    rows = defaultdict(list)
    for r in csv.DictReader(open(path)):
        rows[r["scheme"]].append((float(r["t"]), int(r["maxbond"]),
                                  float(r["err_prof"]), int(r["krylov"]),
                                  float(r["seconds"])))
    return {s: tuple(np.array(x) for x in zip(*sorted(v))) for s, v in rows.items()}


def figure(prof_path, scalar_path, tag):
    prof = load_prof(prof_path)
    scal = load_scalar(scalar_path)
    present = [(s, lab, c, w, z) for s, lab, c, w, z in STYLE if s in prof]
    if not present:
        return

    # 12 columns so row 1 divides into 4 heatmaps (3 each) and row 2 into 3 plots (4 each).
    fig = plt.figure(figsize=(17.5, 8.8))
    gs = gridspec.GridSpec(2, 12, figure=fig, height_ratios=[1.0, 0.85],
                           hspace=.40, wspace=1.9)

    # ── row 1, panel 1: the light cone itself, from the EXACT solution ────────────────────
    # ⛔ TAKE THE LONGEST ARM, NOT `present[0]`. Arms finish at different times and this figure is
    # regenerated while the run is live, so the first scheme in draw order can be one that is only
    # a quarter done. Using its time grid for every OTHER scheme's error map is a shape mismatch
    # (MEASURED: C was (41,18) against Y of length 22) -- and had the lengths happened to agree it
    # would have silently plotted one arm's errors against another arm's times.
    longest = max(present, key=lambda p: len(prof[p[0]][0]))[0]
    ts, sites, _, ex = prof[longest]
    ax = fig.add_subplot(gs[0, 0:3])
    im = ax.pcolormesh(sites, ts, ex, cmap="RdBu_r", vmin=-.5, vmax=.5, shading="nearest")
    ax.set_title("EXACT  $\\langle S^z_j(t)\\rangle$\n(analytic free fermions)", fontsize=9.5)
    ax.set_xlabel("site $j$"); ax.set_ylabel("$t$")
    fig.colorbar(im, ax=ax, fraction=.046, pad=.03)
    # The front edge: v = J = 1 for the XX chain, so the wall spreads |j - L/2| = t.
    mid = (sites[0] + sites[-1]) / 2
    for sgn in (+1, -1):
        ax.plot(mid + sgn * ts, ts, "k--", lw=1.0, alpha=.55)
    ax.set_xlim(sites[0] - .5, sites[-1] + .5); ax.set_ylim(ts[0], ts[-1])

    # ── row 1, panels 2-4: each scheme's OWN light cone, ONE shared colour scale ──────────
    # ⚠ THE MAX DEVIATION IS PRINTED ON EACH PANEL, because the panel alone cannot show it. At
    # ~1e-4 against a [-0.5, +0.5] axis the difference from the exact cone is ~1/5000 of the
    # colour range -- invisible by construction. Without the number, four identical-looking maps
    # invite exactly the conclusion the data does not support: that the schemes are equally
    # accurate. They agree on the PHYSICS; the error panel below is what ranks them.
    for k, (s, lab, c, w, z) in enumerate(present):
        ts_s, sites_s, sz_s, ex_s = prof[s]              # this arm's OWN grid -- see above
        ax = fig.add_subplot(gs[0, 3 * (k + 1):3 * (k + 2)])
        im = ax.pcolormesh(sites_s, ts_s, sz_s, cmap="RdBu_r", vmin=-.5, vmax=.5,
                           shading="nearest")
        dev = np.nanmax(np.abs(sz_s - ex_s))
        partial = "  (to t=%g)" % ts_s[-1] if len(ts_s) < len(ts) else ""
        ax.set_title("%s%s\n$\\langle S^z_j(t)\\rangle$   (max dev %.1e)" % (lab, partial, dev),
                     fontsize=9.5, color=c)
        ax.set_xlabel("site $j$")
        for sgn in (+1, -1):                             # the same v = J = 1 front as the exact
            ax.plot(mid + sgn * ts_s, ts_s, "k--", lw=1.0, alpha=.55)
        ax.set_xlim(sites[0] - .5, sites[-1] + .5)
        ax.set_ylim(ts[0], ts[-1])                       # same t range on every panel
        if k == 0:
            ax.set_ylabel("$t$")
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03)

    # ── row 2, left: rank growth ─────────────────────────────────────────────────────────
    a1 = fig.add_subplot(gs[1, 0:4])
    cap = re.search(r"_D(\d+|inf)_", tag)
    for s, lab, c, w, z in present:
        t, mb, _, _, _ = scal[s]
        a1.plot(t, mb, "o-", ms=3.5, lw=w, color=c, zorder=z,
                label="%s   (final $\\chi$=%d)" % (lab, mb[-1]))
    if cap and cap.group(1) != "inf":
        cv = float(cap.group(1))
        a1.axhline(cv, ls="--", lw=1.2, color="#555")
        a1.text(.01, cv, " cap %.0f" % cv, va="bottom", ha="left", fontsize=8, color="#555",
                transform=a1.get_yaxis_transform())
    a1.set_xlabel("$t$"); a1.set_ylabel(r"max bond dimension $\chi(t)$")
    a1.set_title("Rank growth.  Flat at the cap = the cap sets the error from there on.",
                 fontsize=9.5)
    a1.legend(fontsize=8.5, frameon=False, loc="lower right")
    a1.grid(True, lw=.4, alpha=.35)

    # ── row 2, right: error growth ───────────────────────────────────────────────────────
    a2 = fig.add_subplot(gs[1, 4:8])
    for s, lab, c, w, z in present:
        t, _, e, kry, _ = scal[s]
        a2.semilogy(t, np.maximum(e, FLOOR), "o-", ms=3.5, lw=w, color=c, zorder=z,
                    label="%s   (%d matvec)" % (lab, kry[-1]))
    a2.set_xlabel("$t$")
    a2.set_ylabel(r"$\max_j|\langle S^z_j\rangle - $ exact$|$")
    a2.set_title("Error growth.  Cost in the legend is total operator applications.",
                 fontsize=9.5)
    a2.legend(fontsize=8.5, frameon=False, loc="lower right")
    a2.grid(True, which="both", lw=.4, alpha=.35)

    # ── row 2, right: wall clock ─────────────────────────────────────────────────────────
    # ⚠ WALL CLOCK ON THIS MACHINE IS NOT A MEASUREMENT AND THE PANEL SAYS SO ON ITS FACE.
    # Modern Standby suspends the box unpredictably and cannot be prevented (three fixes tried),
    # and other jobs share the cores: the SAME grid cell, bit-identical to 4 matvecs, has been
    # measured at 520 s and at 33867 s -- 65x. `matvec` in the middle panel is the cost axis that
    # survives. This is drawn because it was asked for and because the SHAPE (linear vs
    # accelerating) is still informative; the absolute values are not comparable across runs.
    # ⛔ AND WHEN BUG RAN PARALLEL, THE PANEL IS NOT COMPARING LIKE WITH LIKE -- IT SAYS SO.
    # BUG gets two workers, both TDVP arms get one, because TDVP's sweep has no independent
    # halves to give a second worker to. That is a fair statement about the METHODS and a unfair
    # one about the code, and which of the two a reader takes away depends entirely on the panel
    # admitting it.
    par = "0"
    if os.path.exists(scalar_path):
        for r in csv.DictReader(open(scalar_path)):
            if r["scheme"].startswith("bug"):
                par = r.get("parallel", "0") or "0"
                break
    a3 = fig.add_subplot(gs[1, 8:12])
    for s, lab, c, w, z in present:
        t, _, _, _, secs = scal[s]
        note = "  [2 workers]" if (par == "1" and s.startswith("bug")) else ""
        a3.plot(t, secs, "o-", ms=3.5, lw=w, color=c, zorder=z,
                label="%s   (%.0f s)%s" % (lab, secs[-1], note))
    a3.set_xlabel("$t$"); a3.set_ylabel("cumulative step time (s)")
    a3.set_title("Wall clock -- SHAPE ONLY, values are contention- and\n"
                 "standby-contaminated on this box (65x measured). Cost = matvec."
                 + ("\nBUG: 2 workers; TDVP: 1 (its sweep has no parallel halves)."
                    if par == "1" else ""),
                 fontsize=9.0)
    a3.legend(fontsize=8.5, frameon=False, loc="upper left")
    a3.grid(True, lw=.4, alpha=.35)

    # ⛔ THE TOLERANCES COME FROM THE DATA, NEVER FROM A HARDCODED STRING. This read
    # "root 1e-6 / half-sweep 1e-4" and was printed verbatim over the MATCHED run, which used
    # half-sweep 1e-6 -- a figure whose caption asserts the opposite of what it plots. Exactly the
    # defect fixed in plot_knee's "Delta=1, Neel, uncapped" title. `tau_trunc` and `split_cutoff`
    # are columns; use them.
    # ⛔ AND SO DO THE CBE PATH AND THE WORKER COUNT, WHICH ARE THE TWO SWITCHES THAT REDEFINE
    # WHAT THE WALL-CLOCK PANEL MEANS. `exact_cbe` decides whether the randomised sketch ran at
    # all (`exact=true` selects `full_local_basis`, which applies H to `d*chi_r` columns instead
    # of the sketch's `Dpre`) and `parallel` decides whether BUG's two half-sweeps overlapped.
    # Neither is in the filename -- `knobtag()` carries L, dt, T, maxdim, maxiter and nothing
    # else -- so before these columns existed the ONLY record of which run a figure came from was
    # a `tag=` string in a shell script. An entire campaign was quoted as an rSVD cost result
    # while `exact_cbe=true` had the sketch switched off, and nothing in any figure could have
    # contradicted it. `.get` with a default keeps older CSVs (written before the columns) from
    # crashing the plotter -- they report "unrecorded", which is exactly what they are.
    knobs = ""
    if os.path.exists(scalar_path):
        for r in csv.DictReader(open(scalar_path)):
            if r["scheme"].startswith("bug"):
                knobs = ("BUG: root %s / half-sweep %s;  TDVP: tol %s"
                         % (fmt(r["tau_trunc"]), fmt(r["split_cutoff"]), fmt(r["tau_trunc"])))
                ex, par = r.get("exact_cbe", ""), r.get("parallel", "")
                nth = r.get("nthreads", "")
                if ex not in ("", None):
                    knobs += ";  CBE: %s" % ("exact (rSVD OFF)" if ex == "1"
                                             else "randomised sketch")
                if par not in ("", None):
                    knobs += ";  BUG half-sweeps: %s" % (
                        "PARALLEL on %s workers (TDVP is sequential: 1)" % nth if par == "1"
                        else "serial")
                break
    fig.suptitle("XX chain ($\\Delta$=0), U(1), domain-wall quench%s   %s"
                 % (tag.replace("_", "  "), knobs), fontsize=10.5, y=.99)
    out = os.path.join(RES, "heis_lightcone%s.png" % tag)
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    pats = sorted(glob.glob(os.path.join(RES, "heis_tau_prof_L*_T10_*.csv")))
    if not pats:
        print("no T=10 tau profile CSVs in", RES)
        return
    for p in pats:
        scalar = p.replace("_tau_prof_", "_tau_")
        if not os.path.exists(scalar):
            print("  no scalar CSV beside", os.path.basename(p))
            continue
        tag = re.sub(r"^heis_tau_prof|\.csv$", "", os.path.basename(p))
        figure(p, scalar, tag)


if __name__ == "__main__":
    main()
