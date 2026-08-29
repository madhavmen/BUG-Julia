"""Plots for the L=16 / T=5 CBE-sweep campaign.

Reads the streamed CSV from `cbe_sweeps_l16.jl` and writes one figure per suite:

  dtscan   error vs dt, log-log, with a dt^2 guide. THE POINT OF THIS PANEL is that the curve is
           expected to have TWO regimes -- slope 2 where the time-discretisation error dominates,
           flattening to slope 0 at the rank floor -- and that the floor is the same for every
           integrator. A single-slope reading of it would be wrong.
  gse      energy vs sweep, |E - E_analytic| vs sweep (log), bond growth vs sweep
  ite      E(beta), |E - E_analytic| vs beta (log), bond growth
  rte      error vs exact (log), 1 - fidelity (log), bond growth

Errors are against ANALYTIC references (Bethe for the ring, the closed form for Haldane-Shastry)
or against exact sparse propagation in the Sz=0 sector -- never against another approximation.

usage:  python benchmarks/plot_cbe_sweeps_l16.py [results.csv] [outdir]
"""

import csv
import glob
import math
import os
import sys
import textwrap
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
# A GLOB, NOT ONE FILE. The campaign runs as a slurm array with one task per (phase, scheme) so
# that no two schemes time each other, and each task streams its OWN CSV. The figures are of the
# union. A single path still works if given explicitly.
#
# ⛔ The glob must NOT pin T. ITE is run to beta=20 while GSE/RTE run to T=5, so the filename
# carries a different T and a `T5.0*` glob drops the whole ITE panel SILENTLY -- an empty
# subplot reads as "the run failed", not as "the file was never opened".
CSV = sys.argv[1] if len(sys.argv) > 1 else os.path.join(
    HERE, "results", "cbe_sweeps_l16_T*.csv")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")

# One colour per integrator, one linestyle per maxdim, held fixed across every figure so the
# same scheme is the same colour everywhere.
COLOR = {
    "bug_decoupled":   "#1f77b4",   # RETIRED -- see LABEL note; kept so old CSVs still plot
    "bug_interleaved": "#d62728",
    # The KRYLOV-DEPTH arms of the SAME `bug_interleaved` sweep. Shades of its colour on purpose:
    # they are one method at different depths, and giving them unrelated colours would present a
    # tuning axis as four competing integrators.
    "bug_interleaved_m0":   "#f4a3a3",
    "bug_interleaved_m3":   "#e06666",
    "bug_interleaved_tol4": "#a83232",
    "bug_interleaved_tol6": "#7a1f1f",
    "tdvp_cbe1s":      "#2ca02c",
    "tdvp2":           "#7f4fc0",
    "dmrg_cbe1s":      "#1f77b4",
    "dmrg_cbe_rsvd":   "#1f77b4",
    "dmrg_cbe_exact":  "#d62728",
    "dmrg_cbe_rsvd_reuse": "#2ca02c",
    "jan_bug":         "#ff7f0e",
    "jan_bug_centre":  "#8c564b",
}
LABEL = {
    # `bug_interleaved` is labelled plainly "CBE-BUG" -- it is now the ONLY BUG scheme in the
    # study, used identically in the closed suite and every open-system driver.
    #
    # ⛔ `bug_decoupled` IS RETIRED, NOT MERELY UNPLOTTED. Its flag lived on the K-step path, which
    # was removed with `kaug`/`rexpand` on 2026-08-24; after that both branches called
    # `cbe_bug_step!` with IDENTICAL arguments. The old note here read "the two agree to 8 digits"
    # and treated that as a measurement -- it was not, they had become the same run. Kept in the
    # maps only so previously-collected CSVs still render.
    "bug_decoupled":   "CBE-BUG (retired: same run as interleaved)",
    "bug_interleaved": "CBE-BUG",
    # ⚠ SAME SWEEP, DIFFERENT KRYLOV DEPTH -- the labels say so, because a legend listing them
    # beside `2-site TDVP` would otherwise read as five competing methods rather than one method
    # and a tuning axis.
    "bug_interleaved_m0":   "CBE-BUG (m=0)",
    "bug_interleaved_m3":   "CBE-BUG (m=3)",
    "bug_interleaved_tol4": "CBE-BUG (adaptive 1e-4)",
    "bug_interleaved_tol6": "CBE-BUG (adaptive 1e-6)",
    "tdvp_cbe1s":      "1-site TDVP-CBE",
    "tdvp2":           "2-site TDVP",
    "dmrg_cbe1s":      "1-site DMRG-CBE",
    "dmrg_cbe_rsvd":   "DMRG-CBE (RSVD)",
    "dmrg_cbe_exact":  "DMRG-CBE (exact)",
    "dmrg_cbe_rsvd_reuse": "DMRG-CBE (RSVD, seeded)",
    "jan_bug":         "jan-BUG (end root)",
    "jan_bug_centre":  "jan-BUG (centre root)",
}
# Linestyle keys on the TOLERANCE, because that -- not a cap -- is what sets the rank.
DASH = {1e-8: "-", 1e-10: "--", 1e-12: ":"}

# Schemes measured but NOT PLOTTED. The CSV rows are untouched -- this is a presentation filter, so
# the data survives for anyone who wants it and re-including a scheme is a one-line edit here.
# `COLOR`/`LABEL` entries are deliberately left in place for the same reason.
#
# `bug_decoupled` is dropped too: it and `bug_interleaved` are the two CBE-BUG expansion orderings
# and they agree to 8 digits on these arms (identical Krylov counts, energies within 1e-9 at β=20),
# so plotting both spends a colour and a legend row on a duplicate. `bug_interleaved` is kept as
# the representative. Both remain in the CSVs.
DROP_SCHEMES = {"jan_bug", "jan_bug_centre", "bug_decoupled"}

# One marker per integrator. Colour alone cannot separate curves that COINCIDE -- 2-site TDVP and
# 1-site TDVP-CBE agree to every printed digit here -- and a thin line drawn over a thick one is
# still easy to miss. Markers at staggered offsets make each series individually readable even
# where the lines lie exactly on top of each other.
MARKER = {
    "bug_interleaved": "o",
    "bug_decoupled":   "s",
    "tdvp_cbe1s":      "^",
    "tdvp2":           "v",
    "dmrg_cbe_rsvd":   "o",
    "dmrg_cbe_exact":  "s",
    "dmrg_cbe_rsvd_reuse": "D",
    "dmrg_cbe1s":      "o",
}

# (suite, model) arms measured but ABANDONED, excluded from the figures for cause.
#
# Haldane-Shastry real time was STOPPED as degenerate, not merely unfinished: `S^z_{j0}|0>` at L=16
# starts with a centre bond of 247 against a 256 cap, so the state is essentially full rank before
# the first step, nothing is ever truncated, and the arm's only grade -- energy conservation -- is
# satisfied by construction (measured 2.66e-15). Its few surviving rows would draw an axis, a
# legend and no curve, which reads as a FAILED RUN rather than an abandoned one. HS real time is
# now reported by `hs_skw_*.png`, against the Li et al. structure factor, which does discriminate.
DROP_ARMS = {("rte", "hs")}
# Heisenberg is an OPEN CHAIN; only Haldane-Shastry is on a ring. "ring" is kept so the
# superseded capped-rank CSVs still plot with an honest label.
MODEL = {"chain": "Heisenberg chain (open)", "hs": "Haldane-Shastry (ring)",
         "xx": "XX chain (open)", "ring": "Heisenberg ring [SUPERSEDED]"}

# WHAT `err` MEANS DEPENDS ON THE ARM, because each is measured against the strongest ANALYTIC
# statement available for that model -- never against another computation. Labelling the axis
# generically would invite reading a conservation drift as a state error.
ERRLABEL = {
    ("ite", "chain"): r"$|E(\beta) - E_{\rm Bethe}|$",
    ("rte", "xx"):    r"$\max_j |\langle S^z_j\rangle - \langle S^z_j\rangle_{\rm free\ ferm.}|$",
    ("rte", "hs"):    r"$|E(t) - E(0)|$  (energy is exactly conserved)",
}


def load(paths):
    rows = []
    for path in paths:
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                for k in ("maxdim", "maxbond", "maxbond_states", "centrebond",
                          "peak", "krylov"):
                    r[k] = int(r[k])
                for k in ("dt", "cutoff", "x", "energy", "err", "fid", "err_fnl",
                          "discarded", "elapsed"):
                    r[k] = float(r[k])
                r["_src"] = os.path.basename(path)
                rows.append(r)
    return _dedupe_arms(rows)


def _dedupe_arms(rows):
    """Keep ONE series per arm when the same arm appears in more than one CSV.

    ⛔ WHY THIS EXISTS. The campaign runs as a slurm array and the figures are of the UNION of
    every matching CSV, so an arm that was run twice -- a killed job later restarted, a rerun of
    one scheme -- appears twice. Matplotlib then draws both, and a TRUNCATED series overlays a
    complete one in the same colour and linestyle: the short curve looks like the long one simply
    stopping early, which reads as an integrator that diverged or a run that failed. Measured
    case: `tdvp2` at cutoff 1e-5 was killed at step 17 of 80 and rerun into its own file.

    The longest series wins, because the only way an arm legitimately appears twice is that one
    attempt was cut short. Ties keep the first, which is stable under `sorted(paths)`. Anything
    dropped is REPORTED -- silently discarding measured rows is its own failure mode.
    """
    key = lambda r: (r["suite"], r["scheme"], r["model"], r["sym"], r["cutoff"], r["dt"])
    series = defaultdict(list)
    for r in rows:
        series[(key(r), r["_src"])].append(r)
    best, dropped = {}, []
    for (k, src), rs in series.items():
        if k not in best or len(rs) > len(best[k][1]):
            if k in best:
                dropped.append((k, best[k][0], len(best[k][1]), src, len(rs)))
            best[k] = (src, rs)
        else:
            dropped.append((k, src, len(rs), best[k][0], len(best[k][1])))
    for k, lost_src, lost_n, kept_src, kept_n in dropped:
        print(f"  DUPLICATE ARM {k[0]}/{k[1]} tol={k[4]:g}: using {kept_n} rows from "
              f"{kept_src}, ignoring {lost_n} rows from {lost_src}")
    return [r for _, rs in best.values() for r in rs]


def group(rows, *keys):
    out = defaultdict(list)
    for r in rows:
        out[tuple(r[k] for k in keys)].append(r)
    return out


def finite(vals):
    return [v for v in vals if v is not None and math.isfinite(v) and v > 0]


def _suptitle(fig, text, fontsize=10):
    """Wrap the caption to the FIGURE's own width and reserve room for it.

    A one-line `suptitle` is clipped at both ends whenever the figure is narrow -- which it is
    as soon as a phase has a single model (5.4 in), and that is the common case. Matplotlib does
    not wrap or shrink it, and `tight_layout` does not reserve space for it, so the caption is
    silently truncated in the saved PNG while looking fine in the code.
    """
    width_in = fig.get_size_inches()[0]
    # An average glyph in this font is ~0.55 em, so at `fontsize` pt it is 0.55*fontsize/72 in
    # and a line holds ~131/fontsize characters per inch. Measured, not guessed: 190 in place of
    # the 131 still clipped the first line at 5.4 in.
    ncols = max(24, int(width_in * 131 / fontsize))
    wrapped = textwrap.fill(text, ncols)
    nlines = wrapped.count("\n") + 1
    fig.suptitle(wrapped, fontsize=fontsize)
    # Leave the wrapped block room at the top rather than letting it overlap the first row. The
    # reserve scales with the LINE COUNT, since that is what the wrap just produced.
    height_in = fig.get_size_inches()[1]
    fig.tight_layout(rect=(0, 0, 1, 1 - (0.34 + 1.35 * nlines) * fontsize / 72 / height_in))


def _style(ax, xlabel, ylabel, title, logy=False):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    # Wrap, for the same reason `_suptitle` does: a subplot title is not shrunk or wrapped by
    # matplotlib, so a long one runs past the axes and is CLIPPED at the figure edge in the saved
    # PNG while looking fine in the source. Panels here are 5.4 in wide and the titles carry the
    # caveat that makes the panel readable ("drift is error"), which is exactly the half that
    # falls off.
    ax.set_title(textwrap.fill(title, 52), fontsize=10)
    if logy:
        ax.set_yscale("log")
    ax.grid(alpha=0.3, linewidth=0.5)


def plot_dtscan(rows, outdir):
    rows = [r for r in rows if r["suite"] == "dtscan"]
    if not rows:
        return
    maxdims = sorted({r["maxdim"] for r in rows})
    fig, axes = plt.subplots(1, len(maxdims), figsize=(5 * len(maxdims), 4.2), squeeze=False)
    for ax, md in zip(axes[0], maxdims):
        sub = [r for r in rows if r["maxdim"] == md]
        for scheme in sorted({r["scheme"] for r in sub}):
            pts = sorted((r["dt"], r["err"]) for r in sub if r["scheme"] == scheme)
            pts = [(d, e) for d, e in pts if math.isfinite(e) and e > 0]
            if not pts:
                continue
            ax.plot([p[0] for p in pts], [p[1] for p in pts], "o-", ms=3.5,
                    color=COLOR.get(scheme, "k"), label=LABEL.get(scheme, scheme))
        # dt^2 guide anchored on the largest dt of the first series
        allpts = sorted((r["dt"], r["err"]) for r in sub
                        if math.isfinite(r["err"]) and r["err"] > 0)
        if allpts:
            d0, e0 = allpts[-1]
            xs = [d0 / 2 ** k for k in range(0, 6)]
            ax.plot(xs, [e0 * (x / d0) ** 2 for x in xs], "k:", lw=1,
                    label=r"$\propto dt^2$")
        ax.set_xscale("log")
        _style(ax, "dt",
               r"$\max_j |\langle S^z_j\rangle - \langle S^z_j\rangle_{\rm free\ ferm.}|$",
               f"$\\chi$={md}", logy=True)
        ax.legend(fontsize=7)
    # NO SUPTITLE, by request; see the module docstring for the two-regime reading of this panel.
    fig.tight_layout()
    out = os.path.join(outdir, "cbe_l16_dtscan.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def plot_gse(rows, outdir):
    rows = [r for r in rows if r["suite"] == "gse"]
    if not rows:
        return
    models = sorted({r["model"] for r in rows})
    fig, axes = plt.subplots(3, len(models), figsize=(5.4 * len(models), 10.5), squeeze=False)
    for col, mdl in enumerate(models):
        sub = [r for r in rows if r["model"] == mdl]
        # Same nesting as `plot_evolve`, and for a sharper reason here: `dmrg_cbe_exact` reproduces
        # the RSVD arm ROW FOR ROW with identical Krylov counts, which is the central claim of this
        # panel -- that the randomised selection costs nothing in accuracy. Painted at equal width
        # the two are indistinguishable from one curve, so the very result being demonstrated is
        # the thing the figure hides.
        series = sorted(group(sub, "sym", "cutoff", "scheme").items())
        for i, ((sym, cut, scheme), g) in enumerate(series):
            g = sorted(g, key=lambda r: r["x"])
            lw = max(1.0, 3.4 - 0.6 * i)
            # BOND DIMENSION, NOT STATE COUNT, by request. `maxbond` is the SU(2) rank in
            # MULTIPLETS; `maxbond_states` expands each multiplet to its 2S+1 members. Both arms
            # here are SU(2) on the same model, so the two differ by a common factor and the
            # comparison is unaffected -- but a cross-SYMMETRY panel would not be, since a U(1)
            # multiplet is one state and an SU(2) one is not. Keep that in mind before putting a
            # U(1) arm in this figure.
            lab = f"{LABEL.get(scheme, scheme)}, {sym}, tol={cut:g}"
            sty = DASH.get(cut, "-")
            c = COLOR.get(scheme, "k")
            axes[0][col].plot([r["x"] for r in g], [r["energy"] for r in g], sty,
                              color=c, marker="o", ms=3, lw=lw, label=lab)
            errs = [(r["x"], r["err"]) for r in g if math.isfinite(r["err"]) and r["err"] > 0]
            if errs:
                axes[1][col].plot([e[0] for e in errs], [e[1] for e in errs], sty,
                                  color=c, marker="o", ms=3, lw=lw, label=lab)
            axes[2][col].plot([r["x"] for r in g], [r["maxbond"] for r in g], sty,
                              color=c, marker="o", ms=3, lw=lw, label=lab)
        _style(axes[0][col], "sweep", "energy", "energy")
        _style(axes[1][col], "sweep", r"$|E - E_{\rm analytic}|$", "error", logy=True)
        _style(axes[2][col], "sweep", "max bond dim", "bond growth")
        for r_ in range(3):
            axes[r_][col].legend(fontsize=7)
    # NO SUPTITLE, by request -- `tight_layout` still has to run or the panels keep the margin
    # the suptitle would have reserved and sit low in the frame.
    fig.tight_layout()
    out = os.path.join(outdir, "cbe_l16_gse.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def plot_evolve(rows, suite, outdir):
    rows = [r for r in rows if r["suite"] == suite]
    # ITE SHOWS ONLY 1e-8 AND 1e-10, by request. The bracketing ladder (1e-3/1e-4/1e-5) was run to
    # locate the crossover -- at beta=20 the error is beta-limited at 0.385*exp(-0.4384*beta), so
    # a cutoff below ~6e-5 never binds and the tight arms coincide by construction. Those rows are
    # NOT deleted; they stay in their own CSV and are reported separately, because they carry the
    # only matched-rank separation measured on this arm (cutoff 1e-3, all schemes at bd=8:
    # CBE-BUG 1.600e-03 vs both TDVP variants 2.830e-03).
    if suite == "ite":
        keep = {1e-8, 1e-10}
        dropped = {r["cutoff"] for r in rows} - keep
        rows = [r for r in rows if r["cutoff"] in keep]
        if dropped:
            print(f"  ite: showing tol {sorted(keep)}, holding back "
                  f"{sorted(dropped)} (see the bracketing figure)")
    if not rows:
        return
    # ONE COLUMN PER (model, symmetry), not per model. Imaginary time runs the SAME model twice
    # -- U(1) from Neel and SU(2) from the dimer -- because Neel has no SU(2) representation, so
    # keying on the model alone would silently overlay two different runs in one panel.
    keys = sorted({(r["model"], r["sym"]) for r in rows})
    fig, axes = plt.subplots(3, len(keys), figsize=(5.4 * len(keys), 10.5), squeeze=False)
    xlab = r"$\beta$" if suite == "ite" else "t"
    for col, (mdl, sym) in enumerate(keys):
        sub = [r for r in rows if r["model"] == mdl and r["sym"] == sym]
        # ⛔ CURVES HERE GENUINELY COINCIDE, AND A HIDDEN CURVE READS AS MISSING DATA. 2-site TDVP
        # and 1-site TDVP-CBE agree to EVERY PRINTED DIGIT on this arm (6.018194e-05 at chi=56,
        # beta=20), so drawing both at one width paints the second exactly over the first: the
        # legend lists four schemes, the panel shows three lines, and the honest result -- that two
        # different algorithms land on the same number -- looks instead like a failed run.
        # Decreasing the width with draw order nests them, so coincidence is VISIBLE as a thin line
        # inside a thick one. Order is `sorted(...)`, hence stable across regenerations.
        series = sorted(group(sub, "scheme", "cutoff").items())
        n_ser = max(1, len(series))
        for i, ((scheme, cut), g) in enumerate(series):
            g = sorted(g, key=lambda r: r["x"])
            lab = f"{LABEL.get(scheme, scheme)}, tol={cut:g}"
            sty = DASH.get(cut, "-")
            c = COLOR.get(scheme, "k")
            lw = max(1.0, 3.0 - 0.5 * i)
            # Markers STAGGERED per series: coincident curves would otherwise stamp their markers
            # on top of each other too, and the reader would be no better off than with the lines.
            kw = dict(color=c, lw=lw, marker=MARKER.get(scheme, "o"), ms=5,
                      markevery=(i * 3, max(6, len(g) // 7)), markerfacecolor="none",
                      markeredgewidth=1.3, label=lab)
            xs = [r["x"] for r in g]
            if suite == "rte":
                # ENERGY IS CONSERVED in real time, so the interesting quantity is the DRIFT, not
                # the value. Plotting the raw energy puts every scheme on top of a large constant
                # and shows 1e-13 wiggles against a -6.9 offset -- unreadable, and it hides how
                # many orders below the physics the drift actually sits. |E(t) - E(0)| on a log
                # axis is the same information, scaled to what varies.
                e0 = g[0]["energy"]
                dr = [(r["x"], abs(r["energy"] - e0)) for r in g
                      if math.isfinite(r["energy"]) and abs(r["energy"] - e0) > 0]
                if dr:
                    axes[0][col].plot([d[0] for d in dr], [d[1] for d in dr], sty, **kw)
            else:
                axes[0][col].plot(xs, [r["energy"] for r in g], sty, **kw)
            errs = [(r["x"], r["err"]) for r in g if math.isfinite(r["err"]) and r["err"] > 0]
            if errs:
                axes[1][col].plot([e[0] for e in errs], [e[1] for e in errs], sty, **kw)
            axes[2][col].plot(xs, [r["maxbond_states"] for r in g], sty, **kw)
        # ⛔ WHY THE TOLERANCE ARMS COINCIDE: SHOW WHAT IS ACTUALLY LIMITING THE ERROR.
        # Measured on this arm, the tail decay rate is 0.4384 for EVERY scheme at BOTH tolerances,
        # constant to four digits and not flattening -- so the error is pure residual
        # excited-state contamination from the Neel start, `A exp(-Delta_eff beta)`, and the
        # truncation cutoff is not limiting anything. At beta=20 the error is ~6e-5, i.e. nearly
        # 6000x ABOVE the 1e-08 cutoff; the fit says it would not reach 1e-08 until beta~40 and
        # 1e-10 until beta~50. Without these guides the panel invites the opposite reading -- that
        # tol=1e-08 is somehow "bad" -- when the run simply never enters the regime tol controls.
        if suite == "ite":
            tails = [sorted(g, key=lambda r: r["x"]) for (_, _), g in series]
            tails = [t for t in tails if len(t) >= 6
                     and all(math.isfinite(r["err"]) and r["err"] > 0 for r in t[-6:])]
            if tails:
                longest = max(tails, key=lambda t: t[-1]["x"])
                a, b = longest[-6], longest[-1]
                rate = -(math.log(b["err"]) - math.log(a["err"])) / (b["x"] - a["x"])
                amp = b["err"] * math.exp(rate * b["x"])
                xg = [x * 0.25 for x in range(4, int(4 * b["x"]) + 1)]
                axes[1][col].plot(xg, [amp * math.exp(-rate * x) for x in xg],
                                  color="0.25", lw=1.0, ls=(0, (1, 2)), zorder=0,
                                  label=rf"$\beta$-limited: $e^{{-{rate:.3f}\beta}}$")
                for cut in sorted({c_ for (_, c_) in group(sub, "scheme", "cutoff")}):
                    axes[1][col].axhline(cut, color="0.6", lw=0.8, ls=":", zorder=0)
                    axes[1][col].annotate(f"cutoff {cut:g}", (0.02, cut), xycoords=("axes fraction",
                                          "data"), fontsize=6, color="0.4", va="bottom")
        mid = ERRLABEL.get((suite, mdl), r"$|$error vs the analytic reference$|$")
        if suite == "rte":
            _style(axes[0][col], xlab, r"$|E(t) - E(0)|$", "energy drift", logy=True)
        else:
            _style(axes[0][col], xlab, "energy", "energy")
        _style(axes[1][col], xlab, mid, "error", logy=True)
        _style(axes[2][col], xlab, "max bond dim (states)", "bond growth")
        for r_ in range(3):
            axes[r_][col].legend(fontsize=7)
    # NO SUPTITLE, by request, on either evolution figure. `tight_layout` still has to be called
    # or the panels keep the margin `_suptitle` would have reserved and sit low in the frame.
    fig.tight_layout()
    out = os.path.join(outdir, f"cbe_l16_{suite}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    paths = sorted(glob.glob(CSV)) if any(c in CSV for c in "*?[") else (
        [CSV] if os.path.exists(CSV) else [])
    # SUPERSEDED runs are kept on disk for provenance and must never be plotted alongside live
    # ones -- they were measured at a pinned rank, which is a different experiment entirely.
    paths = [p for p in paths if "SUPERSEDED" not in os.path.basename(p)]
    if not paths:
        sys.exit(f"no CSV matching {CSV} -- run benchmarks/cbe_sweeps_l16.jl first")
    os.makedirs(OUTDIR, exist_ok=True)
    for p in paths:
        print("reading", os.path.basename(p))
    rows = load(paths)
    # jan-BUG is EXCLUDED FROM THE FIGURES by request. The rows stay in the CSVs -- this drops
    # them at the plotting boundary only, so nothing measured is lost and re-including them is a
    # one-line change. Report the count rather than dropping them silently: a scheme vanishing
    # from a figure with no trace in the log is indistinguishable from a run that never happened.
    dropped = [r for r in rows if r["scheme"] in DROP_SCHEMES]
    if dropped:
        by = sorted({(r["scheme"], r["suite"]) for r in dropped})
        print(f"excluding {len(dropped)} rows from the plots: "
              + ", ".join(f"{s}/{su}" for s, su in by))
    rows = [r for r in rows if r["scheme"] not in DROP_SCHEMES]
    ab = [r for r in rows if (r["suite"], r["model"]) in DROP_ARMS]
    if ab:
        print(f"excluding {len(ab)} rows from ABANDONED arms: "
              + ", ".join(f"{su}/{m}" for su, m in sorted({(r["suite"], r["model"])
                                                           for r in ab})))
    rows = [r for r in rows if (r["suite"], r["model"]) not in DROP_ARMS]
    print(f"{len(rows)} rows, suites: {sorted({r['suite'] for r in rows})}")
    plot_dtscan(rows, OUTDIR)
    plot_gse(rows, OUTDIR)
    plot_evolve(rows, "ite", OUTDIR)
    plot_evolve(rows, "rte", OUTDIR)


if __name__ == "__main__":
    main()
