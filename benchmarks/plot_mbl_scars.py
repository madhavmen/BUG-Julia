#!/usr/bin/env python3
"""Figures for the MBL / quantum-many-body-scars campaign, from mbl_scars_*.csv.

    mbl_imbalance.svg    I(t) vs t, one panel per disorder W, ED overlaid      <- the RMP observable
    mbl_entropy.svg      S(t) vs log t, one panel per W, ED overlaid           <- the log-growth test
    mbl_err_vs_chi.svg   worst-time |I - I_ED| vs rank cap, one line per arm   <- THE HEADLINE
    pxp_fidelity.svg     |<Z2|psi(t)>|^2 vs t, one panel per cap, ED overlaid  <- the revivals
    pxp_entropy.svg      S(t) vs t -- oscillating, not saturating
    pxp_err_vs_chi.svg   worst-time fidelity error vs rank cap

No titles: captions live in the thesis. Text stays text in the SVG so it can be restyled.

⛔ MEAN FOR THE OBSERVABLE, MEDIAN FOR THE ERROR, and the two are not interchangeable. The
disorder average of `I(t)` IS an arithmetic mean over realisations -- that is what the quantity
means, and what an experiment measures. The ERROR across realisations is heavy-tailed (a single
realisation with a near-degeneracy can be orders of magnitude worse than typical), so its mean is
set by the worst seed and reports the tail rather than the method; the median is the typical
failure and is what the ladder is read off. Both are computed from the same rows, so the choice is
visible here and not baked into the measurement.

⛔ THE ED CURVE IS PER-REALISATION TOO. `obs_exact` is the reference for THAT seed, so it is
averaged over exactly the same seed set as the arm it is being compared with -- never over all
seeds present in the directory. A reference averaged over a different set is not a reference.

usage:  python plot_mbl_scars.py [results_dir] [out_dir]
"""

import os
import sys
import csv
import glob
import math
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

plt.rcParams.update({
    "svg.fonttype": "none",
    "font.size": 11,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

# Same colours and markers as the rest of the campaign, so a reader who has learned the legend
# once never re-reads it. `plot_large_chi.py` is the source of these.
ARMS = [
    ("bug",   "BUG",             "#c1440e", "D", "-"),
    ("cbe1s", "1-site TDVP-CBE", "#2a6558", "s", "--"),
    ("tdvp2", "2-site TDVP",     "#33507a", "o", "-"),
]
ED_STYLE = dict(color="black", lw=2.0, ls=(0, (4, 2)), zorder=1, label="ED (exact)")

RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "results")
OUT = sys.argv[2] if len(sys.argv) > 2 else RES


def load(res_dir):
    """Every row of every CSV in `res_dir`, with the numeric columns converted."""
    rows = []
    files = sorted(glob.glob(os.path.join(res_dir, "mbl_scars_*.csv")))
    if not files:
        sys.exit("no mbl_scars_*.csv in %s" % res_dir)
    for path in files:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                try:
                    for k in ("L", "seed", "maxdim"):
                        r[k] = int(r[k])
                    for k in ("W", "delta", "dt", "t", "obs", "obs_exact", "obs_err", "entropy",
                              "entropy_exact", "entropy_err", "sz_err", "energy",
                              "energy_drift", "norm", "elapsed", "smin", "sgapmin"):
                        r[k] = float(r[k])
                    r["maxbond"] = int(r["maxbond"])
                except (ValueError, KeyError):
                    continue                      # a truncated final line from a killed run
                rows.append(r)
    print("loaded %d rows from %d file(s)" % (len(rows), len(files)))
    return rows


def mean(xs):
    xs = [x for x in xs if not math.isnan(x)]
    return sum(xs) / len(xs) if xs else float("nan")


def median(xs):
    xs = sorted(x for x in xs if not math.isnan(x))
    if not xs:
        return float("nan")
    m = len(xs) // 2
    return xs[m] if len(xs) % 2 else 0.5 * (xs[m - 1] + xs[m])


def by_time(rows, field):
    """{t: [values]} -- the sample grid is a STEP COUNT on both sides, so `t` matches exactly."""
    out = defaultdict(list)
    for r in rows:
        out[r["t"]].append(r[field])
    return out


def curve(rows, field, avg=mean):
    d = by_time(rows, field)
    ts = sorted(d)
    return ts, [avg(d[t]) for t in ts]


def sel(rows, **kw):
    return [r for r in rows if all(r[k] == v for k, v in kw.items())]


# ── MBL ──────────────────────────────────────────────────────────────────────────────────────

def fig_mbl_time(rows, field, exact_field, ylabel, fname, logx=True):
    """One FIGURE per Δ, one panel per W, at the largest cap: does the method land on ED?

    ⛔ Δ IS NEVER AVERAGED OVER. Δ=0 is Anderson (free fermions, entropy SATURATES) and Δ=1 is
    MBL (entropy grows like log t); pooling them would average two different phenomena into a
    curve that is neither, and on the entropy panel it would manufacture an apparent slope.
    """
    for delta in sorted({r["delta"] for r in rows if r["suite"] == "mbl"}):
        stem, ext = os.path.splitext(fname)
        _fig_mbl_time_one(rows, field, exact_field, ylabel,
                          "%s_D%g%s" % (stem, delta, ext), delta, logx)


def _fig_mbl_time_one(rows, field, exact_field, ylabel, fname, delta, logx):
    mbl = [r for r in rows if r["suite"] == "mbl" and r["delta"] == delta]
    if not mbl:
        return
    ws = sorted({r["W"] for r in mbl})
    chi = max(r["maxdim"] for r in mbl)
    fig, axes = plt.subplots(1, len(ws), figsize=(4.1 * len(ws), 3.5), sharey=True)
    axes = [axes] if len(ws) == 1 else list(axes)
    for ax, w in zip(axes, ws):
        cells = sel(mbl, W=w, maxdim=chi)
        if not cells:
            continue
        ts, ed = curve(cells, exact_field)
        ax.plot(ts, ed, **ED_STYLE)
        for key, label, colour, marker, ls in ARMS:
            arm = [r for r in cells if r["scheme"] == key]
            if not arm:
                continue
            ts, ys = curve(arm, field)
            ax.plot(ts, ys, color=colour, ls=ls, marker=marker, ms=3.5, lw=1.4, label=label)
        nseed = len({r["seed"] for r in cells})
        ax.set_xlabel("t")
        ax.set_title(r"$\Delta=%g$, $W=%g$, $\chi\leq%d$, %d seed(s)" % (delta, w, chi, nseed),
                     fontsize=10)
        if logx:
            ax.set_xscale("log")
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("wrote", fname)


def fig_mbl_err_vs_chi(rows):
    """THE HEADLINE FIGURE. Worst-time imbalance error against the rank cap, per arm and W.

    Worst-time rather than final-time: a method can be wrong for three decades and then recover
    onto a plateau that ED also reaches, and a final-time error would score that as success.
    """
    for delta in sorted({r["delta"] for r in rows if r["suite"] == "mbl"}):
        _fig_mbl_err_one(rows, delta)


def _fig_mbl_err_one(rows, delta):
    mbl = [r for r in rows if r["suite"] == "mbl" and r["delta"] == delta]
    if not mbl:
        return
    ws = sorted({r["W"] for r in mbl})
    fig, axes = plt.subplots(1, len(ws), figsize=(4.1 * len(ws), 3.5), sharey=True)
    axes = [axes] if len(ws) == 1 else list(axes)
    for ax, w in zip(axes, ws):
        for key, label, colour, marker, ls in ARMS:
            chis, errs = [], []
            for chi in sorted({r["maxdim"] for r in mbl}):
                cells = [r for r in sel(mbl, W=w, maxdim=chi) if r["scheme"] == key]
                if not cells:
                    continue
                # worst time PER SEED, then the median over seeds -- see the module docstring.
                per_seed = defaultdict(list)
                for r in cells:
                    per_seed[r["seed"]].append(r["obs_err"])
                chis.append(chi)
                errs.append(median([max(v) for v in per_seed.values()]))
            if chis:
                ax.plot(chis, errs, color=colour, ls=ls, marker=marker, ms=5, lw=1.5,
                        label=label)
        ax.set_xscale("log", base=2)
        ax.set_yscale("log")
        ax.set_xlabel(r"rank cap $\chi$")
        ax.set_title(r"$\Delta=%g$, $W=%g$" % (delta, w), fontsize=10)
    axes[0].set_ylabel(r"median worst-time $|I - I_{\rm ED}|$")
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    name = "mbl_err_vs_chi_D%g.svg" % delta
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


# ── PXP ──────────────────────────────────────────────────────────────────────────────────────

def fig_pxp_time(rows, field, exact_field, ylabel, fname):
    """One panel per cap: the revival is either reproduced at that rank or it is not."""
    pxp = [r for r in rows if r["suite"] == "pxp"]
    if not pxp:
        return
    chis = sorted({r["maxdim"] for r in pxp})
    fig, axes = plt.subplots(1, len(chis), figsize=(3.6 * len(chis), 3.3), sharey=True)
    axes = [axes] if len(chis) == 1 else list(axes)
    for ax, chi in zip(axes, chis):
        cells = sel(pxp, maxdim=chi)
        ts, ed = curve(cells, exact_field)
        ax.plot(ts, ed, **ED_STYLE)
        for key, label, colour, marker, ls in ARMS:
            arm = [r for r in cells if r["scheme"] == key]
            if not arm:
                continue
            ts, ys = curve(arm, field)
            ax.plot(ts, ys, color=colour, ls=ls, lw=1.4, label=label)
        ax.set_xlabel(r"$\Omega t$")
        ax.set_title(r"$\chi\leq%d$" % chi, fontsize=10)
    axes[0].set_ylabel(ylabel)
    axes[0].legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, fname))
    plt.close(fig)
    print("wrote", fname)


def fig_pxp_err_vs_chi(rows):
    pxp = [r for r in rows if r["suite"] == "pxp"]
    if not pxp:
        return
    fig, ax = plt.subplots(figsize=(4.4, 3.5))
    for key, label, colour, marker, ls in ARMS:
        chis, errs = [], []
        for chi in sorted({r["maxdim"] for r in pxp}):
            cells = [r for r in sel(pxp, maxdim=chi) if r["scheme"] == key]
            if cells:
                chis.append(chi)
                errs.append(max(r["obs_err"] for r in cells))
        if chis:
            ax.plot(chis, errs, color=colour, ls=ls, marker=marker, ms=5, lw=1.5, label=label)
    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel(r"rank cap $\chi$")
    ax.set_ylabel(r"worst-time $|F - F_{\rm ED}|$")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, "pxp_err_vs_chi.svg"))
    plt.close(fig)
    print("wrote pxp_err_vs_chi.svg")


def fig_err_vs_smin(rows, suite):
    """THE MECHANISM FIGURE: per-sample error against the smallest retained Schmidt value.

    The ranking figures say WHICH integrator is better. This one tests WHY, and it is the only
    figure here that can falsify the explanation while leaving the ranking intact.

    The claim is structural, not statistical: 1-site TDVP inverts the bond matrix (error ~ 1/smin)
    and 2-site TDVP carries a BACKWARD `exp(+i tau H_1)` substep, while the BUG has neither an
    inverse nor a backward step and a published bound independent of small singular values. If
    that is the operative mechanism, the TDVP arms' error must RISE as `smin` falls while the
    BUG's stays flat. If every arm is flat in `smin`, the ranking may still hold but this
    explanation for it is wrong -- and that is worth knowing before it goes in a paper.

    Log-log scatter, one colour per arm, every sample plotted (no averaging): the correlation is
    the content, so binning it away would hide exactly what is being tested.
    """
    sel_rows = [r for r in rows if r["suite"] == suite
                and not math.isnan(r["smin"]) and r["smin"] > 0 and r["obs_err"] > 0]
    if not sel_rows:
        return
    fig, ax = plt.subplots(figsize=(4.6, 3.6))
    for key, label, colour, marker, ls in ARMS:
        pts = [(r["smin"], r["obs_err"]) for r in sel_rows if r["scheme"] == key]
        if pts:
            ax.plot([p[0] for p in pts], [p[1] for p in pts], marker, color=colour, ms=3.0,
                    alpha=0.55, ls="none", label=label)
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.invert_xaxis()                      # worse conditioning to the RIGHT
    ax.set_xlabel(r"smallest retained Schmidt value $s_{\min}$  (worse $\rightarrow$)")
    ax.set_ylabel("per-sample error vs ED")
    ax.legend(frameon=False, fontsize=9)
    fig.tight_layout()
    name = "%s_err_vs_smin.svg" % suite
    fig.savefig(os.path.join(OUT, name))
    plt.close(fig)
    print("wrote", name)


def report_smoke(rows):
    """The gate, printed rather than plotted.

    ⛔ THE CRITERION IS THE dt RATIO, NOT AN ABSOLUTE THRESHOLD. A full-rank sweep is NOT
    exp(tau H): TDVP2 is a second-order splitting over bonds and carries an O(dt^2) error at any
    rank, so an absolute bound rejects a correct implementation (measured: every arm at ~1e-7 for
    dt=0.05 at L=8, with observables matching ED to 6 digits). Halving dt must cut the error by
    ~4x. Norm must be 1 to 1e-8 -- that is the separate, and sharper, check: it is what catches a
    non-Hermitian MPO, which no accuracy number distinguishes from a merely inaccurate step.
    """
    sm = [r for r in rows if r["suite"] == "smoke"]
    if not sm:
        print("no smoke rows -- the correctness gate has not been run on this data")
        return
    cells = defaultdict(dict)
    for r in sm:
        cells[(r["model"], r["sym"], r["delta"], r["scheme"])][r["dt"]] = r
    print("\ncorrectness gate: dt-ratio must be >= 3 (2nd order gives 4), norm must be 1:")
    for key in sorted(cells):
        by_dt = cells[key]
        dts = sorted(by_dt, reverse=True)          # coarse first
        coarse = by_dt[dts[0]]
        ratio = (coarse["sz_err"] / by_dt[dts[-1]]["sz_err"]
                 if len(dts) > 1 and by_dt[dts[-1]]["sz_err"] > 0 else float("nan"))
        badnorm = any(abs(r["norm"] - 1.0) > 1e-8 for r in by_dt.values())
        ok = (coarse["sz_err"] < 1e-11 or ratio >= 3.0) and not badnorm
        print("  %s %-4s %-5s D=%-4g %-6s err=%.3e ratio=%.2f %s" %
              ("OK  " if ok else "FAIL", key[0], key[1], key[2], key[3],
               coarse["sz_err"], ratio,
               "NORM=%.6f" % max(by_dt.values(), key=lambda r: abs(r["norm"] - 1))["norm"]
               if badnorm else ""))


def main():
    os.makedirs(OUT, exist_ok=True)
    rows = load(RES)
    report_smoke(rows)
    fig_mbl_time(rows, "obs", "obs_exact", r"imbalance $I(t)$", "mbl_imbalance.svg")
    fig_mbl_time(rows, "entropy", "entropy_exact", r"$S_{L/2}(t)$ [nats]", "mbl_entropy.svg")
    fig_mbl_err_vs_chi(rows)
    fig_pxp_time(rows, "obs", "obs_exact", r"$|\langle Z_2|\psi(t)\rangle|^2$",
                 "pxp_fidelity.svg")
    fig_pxp_time(rows, "entropy", "entropy_exact", r"$S_{L/2}(t)$ [nats]", "pxp_entropy.svg")
    fig_pxp_err_vs_chi(rows)
    fig_err_vs_smin(rows, "mbl")
    fig_err_vs_smin(rows, "pxp")


main()
