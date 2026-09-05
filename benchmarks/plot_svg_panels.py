"""FIVE STANDALONE SVG PANELS for the thesis: three light cones, bond-dimension growth, error
growth. No titles -- every panel is captioned in the document, not in the file.

  lightcone_bug.svg        <S^z_j(t)> map, BUG
  lightcone_cbe1s.svg      <S^z_j(t)> map, 1-site TDVP-CBE
  lightcone_tdvp2.svg      <S^z_j(t)> map, 2-site TDVP
  bond_growth.svg          chi(t), all three arms
  error_growth.svg         max_j |<S^z_j> - exact|(t), all three arms

CONFIG: tau_trunc = 1e-4, the cell where BUG is BEST RELATIVE TO BOTH TDVP ARMS at L=18/T=20
(3.05e-03 vs 8.01e-03 / 8.02e-03, i.e. 2.6x) AND the only one where the rank cap never activates
(chi 40 / 37 / 38 against a cap of 128), so `bond_growth.svg` shows genuine growth rather than a
curve flattening onto a cap.
⚠ THIS IS BUG'S *RELATIVE* OPTIMUM, NOT ITS LOWEST ABSOLUTE ERROR. That is tau=1e-7 (1.275e-04),
where all three arms pin at chi=128 and BUG is 4x WORSE than tdvp_cbe1s -- a worse figure on both
counts. The choice is stated here so it is auditable rather than implicit.

⛔ THE THREE LIGHT CONES SHARE ONE FIXED COLOUR RANGE (-0.5, +0.5), SET FROM THE PHYSICS AND NOT
   FROM THE DATA. Three separately-autoscaled maps would each use their own limits, so identical
   colours would mean different values across the three files and the panels could not be compared
   side by side in the document -- while looking perfectly fine individually.

⚠ `svg.fonttype = "none"` keeps text as TEXT rather than converting it to paths, so the labels stay
  editable and searchable in Illustrator/Inkscape and inherit the document's font. The cost is that
  the viewer must have the font; the fallback stack is set for that reason.

⚠⚠ THE WALL TIMES IN THE GROWTH LEGENDS ARE **ONE MEASUREMENT PER ARM, RUN SEQUENTIALLY** on a
   shared laptop (bug -> cbe1s -> tdvp2), so each arm saw a different window of machine load and
   the spread is NOT bounded by these runs. They are quotable here for two specific reasons and no
   others: (1) the Windows event log shows NO Kernel-Power 506/507 in the 06:30-07:30 window when
   they ran, so no Modern Standby suspension hit them; (2) at this config the wall clock is the
   ONLY honest cost axis -- `krylov` says BUG is 7.3x cheaper than tdvp2 while the clock says 1.85x,
   because the counter cannot see `cbe_expand` (measured here: t_cbe 230.7 s vs t_kry 122.7 s, so
   the UNCOUNTED phase is 1.9x the counted one). A publication number belongs on the cluster.
   Read from the CSV's `seconds` column, so the legend follows the data automatically.

usage:  python3 benchmarks/plot_svg_panels.py [tau] [resultsdir]
"""

import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
matplotlib.rcParams["svg.fonttype"] = "none"      # keep text as text, not outlines
# ⚠ NO "Helvetica" IN THIS STACK. It is absent on this box and matplotlib emits one
# `findfont: Font family 'Helvetica' not found.` per text element -- ~250 lines per run, which
# buries any real warning. Arial is present on Windows and is the same metric fallback.
matplotlib.rcParams["font.family"] = ["DejaVu Sans", "Arial", "sans-serif"]
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 1e-4
RES = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")

MAIN = os.path.join(RES, "heis_tau_L18_dt0.05_T20_D128_mauto.csv")
PROF = os.path.join(RES, "heis_tau_prof_L18_dt0.05_T20_D128_mauto.csv")

# (csv scheme key, output stem, legend label, colour, marker, linestyle, zorder)
# ⛔ `tdvp_cbe1s` IS DASHED AND DRAWN ON TOP, AND THAT IS NOT DECORATION. At this tolerance the two
# TDVP arms agree to 0.03% (8.0158e-03 vs 8.0134e-03), so with both solid the green curve vanishes
# under the blue one and the panel shows TWO curves while the legend lists THREE -- which reads as a
# missing arm rather than as two arms that coincide. Dashed-over-solid states the coincidence.
ARMS = [("bug_interleaved", "bug",    "BUG",             "#c1440e", "D", "-",  3),
        ("tdvp_cbe1s",      "cbe1s",  "1-site TDVP-CBE", "#2a6558", "s", "--", 5),
        ("tdvp2",           "tdvp2",  "2-site TDVP",     "#33507a", "o", "-",  4)]

SZ_LIM = 0.5          # |<S^z>| <= 1/2 exactly, for any state: a physics bound, not a data range
V_FRONT = 1.0         # XX chain (delta=0), J=1: eps(k)=J cos k -> max group velocity |v|=J
WALL = 9.5            # domain wall of |up..up dn..dn> at L=18 sits between sites 9 and 10


def load_profiles():
    """-> {scheme: (ts, sites, Z[t, j])} for the selected tau."""
    rows = [r for r in csv.DictReader(open(PROF, encoding="utf-8"))
            if abs(float(r["tau_trunc"]) - TAU) < 1e-30]
    if not rows:
        raise SystemExit("no profile rows at tau=%g in %s" % (TAU, PROF))
    out = {}
    for sch in {r["scheme"] for r in rows}:
        d = {}
        for r in rows:
            if r["scheme"] == sch:
                d[(round(float(r["t"]), 6), int(r["site"]))] = float(r["sz_mps"])
        ts = sorted({t for t, _ in d})
        js = sorted({j for _, j in d})
        Z = np.array([[d[(t, j)] for j in js] for t in ts])
        out[sch] = (np.array(ts), np.array(js), Z)
    return out


def load_series():
    """-> {scheme: (ts, chi, err, secs)} for the selected tau.

    `secs` is the driver's accumulated per-step stepping time, so `secs[-1]` is the arm's total
    wall clock -- READ FROM THE CSV, never written into the label by hand, so the legend cannot
    drift away from the run it describes.
    """
    rows = [r for r in csv.DictReader(open(MAIN, encoding="utf-8"))
            if abs(float(r["tau_trunc"]) - TAU) < 1e-30]
    out = {}
    for sch in {r["scheme"] for r in rows}:
        rs = sorted((float(r["t"]), int(r["maxbond"]), float(r["err_prof"]),
                     float(r["seconds"])) for r in rows if r["scheme"] == sch)
        ts, chi, err, secs = (np.array(x) for x in zip(*rs))
        out[sch] = (ts, chi, err, secs)
    return out


def lightcone(sch, stem, profs):
    ts, js, Z = profs[sch]
    fig, ax = plt.subplots(figsize=(4.3, 3.5))
    # Cell EDGES, not centres: pcolormesh with centre coordinates silently drops the last row and
    # column, which on a 41x18 map is the whole final time slice.
    je = np.arange(js[0] - .5, js[-1] + 1.5)
    dt_ = ts[1] - ts[0] if len(ts) > 1 else 1.0
    te = np.concatenate([ts - dt_ / 2, [ts[-1] + dt_ / 2]])
    m = ax.pcolormesh(je, te, Z, cmap="RdBu_r", vmin=-SZ_LIM, vmax=SZ_LIM, shading="flat",
                      rasterized=True)
    # The free-fermion front, as the falsifiable prediction it is.
    tt = np.array([ts[0], ts[-1]])
    for s in (+1, -1):
        ax.plot(WALL + s * V_FRONT * tt, tt, "-", color="#111", lw=1.0, alpha=.55)
    ax.set_xlim(je[0], je[-1])
    ax.set_ylim(te[0], te[-1])
    ax.set_xlabel("site $j$")
    ax.set_ylabel("$t$")
    cb = fig.colorbar(m, ax=ax, pad=.03)
    cb.set_label(r"$\langle S^z_j\rangle$")
    cb.set_ticks([-0.5, -0.25, 0, 0.25, 0.5])
    fig.tight_layout()
    out = os.path.join(RES, "lightcone_%s.svg" % stem)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def growth(series, which):
    fig, ax = plt.subplots(figsize=(4.6, 3.5))
    for sch, stem, lab, col, mk, ls, z in ARMS:
        if sch not in series:
            continue
        ts, chi, err, secs = series[sch]
        y = chi if which == "bond" else err
        if which == "error":
            # t=0 is exact by construction (err = 0) and log(0) is not plottable; dropping it is
            # honest here because the panel is about GROWTH, and the first sampled point is kept.
            keep = y > 0
            ts, y = ts[keep], y[keep]
        # Offset markevery per arm so coincident curves do not stack their markers either.
        ax.plot(ts, y, ls, color=col, lw=2.0, marker=mk, ms=3.6, markevery=(z % 3, 4), zorder=z,
                label="%s  (%.0f s)" % (lab, secs[-1]))
    ax.set_xlabel("$t$")
    if which == "bond":
        ax.set_ylabel(r"max bond dimension $\chi$")
        ax.legend(fontsize=8.5, frameon=False, loc="upper left")
    else:
        ax.set_yscale("log")
        ax.set_ylabel(r"$\max_j\,|\langle S^z_j\rangle - \mathrm{exact}|$")
        ax.legend(fontsize=8.5, frameon=False, loc="lower right")
    ax.grid(True, which="both", lw=.4, alpha=.3)
    fig.tight_layout()
    out = os.path.join(RES, "%s_growth.svg" % which)
    fig.savefig(out, format="svg", bbox_inches="tight")
    plt.close(fig)
    print("wrote", out)


def main():
    profs, series = load_profiles(), load_series()
    for sch, stem, lab, col, mk, ls, z in ARMS:
        if sch in profs:
            lightcone(sch, stem, profs)
    growth(series, "bond")
    growth(series, "error")
    print("\nconfig: tau=%g, L=18, T=20, dt=0.05, cap 128, split=1e-14" % TAU)
    print("%-18s %-9s %-11s" % ("scheme", "final chi", "final err"))
    for sch, stem, lab, col, mk, ls, z in ARMS:
        if sch in series:
            ts, chi, err, secs = series[sch]
            print("%-18s %-9d %-11.4e" % (lab, chi[-1], err[-1]))


if __name__ == "__main__":
    main()
