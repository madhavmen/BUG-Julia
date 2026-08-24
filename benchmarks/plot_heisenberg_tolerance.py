"""Figures for the Heisenberg tolerance study (docs/PLAN-tolerance-control.md).

  heis_knee_L<n>.png     0a/0b  err and chi against tau_trunc -- IS THERE A KNEE, AND WHERE?
  heis_dtcal_L<n>.png    0c/0d  err against dt, one line per tau_trunc; the calibration curve
  heis_split_L<n>.png    6A     root truncation OFF, half-sweep cutoff scanned
  heis_grid_L<n>.png     6B     the 2x2: which truncation sets the error, which sets the rank

⛔ THE KNEE IS THE RESULT, AND A FLAT CURVE IS ALSO A RESULT. The question this study opens with
is why the error does not respond to tau_trunc. If 0a is flat across eight decades then the
truncation never binds at this dt and the error is time-integration error -- which is the answer,
not a failed measurement. What would make it a failed measurement is running it at a CAPPED
maxdim, where the cap sets the error floor and no tau_trunc can move it. Hence `maxdim` is
uncapped in the driver, and `chi` is plotted beside the error so a reader can see the rank was
free to grow.

⛔ chi IS PLOTTED BESIDE err, NEVER INSTEAD OF IT. Below the knee, a tighter tolerance buys rank
and buys nothing else -- that is the cost half of the argument and it is invisible on an error
axis alone. In 6A it is the primary diagnostic: RUNAWAY chi AT FLAT ERROR is the signature of a
cutoff that is not binding.

⛔ COST IS `krylov`, NOT `seconds`. Operator applications survive contention; local wall clock on
this machine spreads 2.6x on bit-identical work. `seconds` is carried in the CSV for orientation
and is deliberately not plotted.

usage:  python3 benchmarks/plot_heisenberg_tolerance.py [resultsdir]
"""

import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")

COLOR = {"cbe_bug": "#c1440e", "tdvp_cbe1s": "#2a6558", "tdvp2": "#33507a",
         "cbe_bug_cbe_only": "#2a6558", "cbe_bug_historic": "#7a3d8f"}
LABEL = {"cbe_bug": "CBE-BUG", "tdvp_cbe1s": "1-site TDVP-CBE", "tdvp2": "2-site TDVP",
         "cbe_bug_cbe_only": "CBE-BUG / cbe_only", "cbe_bug_historic": "CBE-BUG / historic"}
# Errors can legitimately reach machine precision at the tightest tolerances. Clamping keeps a
# log axis from collapsing onto a single point.
FLOOR = 1e-16

# ⛔ NESTED, NOT OVERPAINTED, AND THIS IS NOT COSMETIC. `tdvp2` and `tdvp_cbe1s` agree to every
# printed digit on this model -- MEASURED, 8.425e-07 for both at every tolerance below the knee.
# Drawn at equal width the second one painted ERASES the first, and the figure then shows two
# curves where three ran, which reads as a missing arm rather than as an agreement. The same
# defect hid `tdvp2` under `tdvp_cbe1s` in the L=16 campaign figures. Widths here are deliberate:
# the widest is drawn first and underneath, so a coincidence shows as a halo and a divergence
# shows as two lines.
WIDTH = {"tdvp2": 4.2, "tdvp_cbe1s": 2.0, "cbe_bug": 1.6}
ZORDER = {"tdvp2": 1, "tdvp_cbe1s": 2, "cbe_bug": 3}
MSIZE = {"tdvp2": 8.0, "tdvp_cbe1s": 5.0, "cbe_bug": 4.0}
DRAW_ORDER = ["tdvp2", "tdvp_cbe1s", "cbe_bug", "cbe_bug_cbe_only", "cbe_bug_historic",
              "cbe_bug_lubich"]


def order(schemes):
    "Widest first, so coincident curves nest instead of erasing each other."
    return sorted(schemes, key=lambda s: DRAW_ORDER.index(s) if s in DRAW_ORDER else 99)


def tighten(ax, values, pad=6.0):
    """Fit a log y-axis to the data.

    Left to matplotlib the axis spans the clamp floor, and a decade of empty space below the
    lowest point is a decade the reader has to discount. `pad` is a multiplicative margin.
    """
    vals = [v for v in values if v > 0]
    if vals:
        ax.set_ylim(min(vals) / pad, max(vals) * pad)

FLOAT_COLS = {"dt", "tau_trunc", "split_cutoff", "t", "stag", "stag_exact", "err_stag",
              "err_prof", "norm", "energy", "dE", "err_fnl", "discarded", "seconds"}
INT_COLS = {"L", "root_trunc", "close_trunc", "maxdim", "maxbond", "krylov"}


def load(pattern):
    rows = []
    for path in sorted(glob.glob(os.path.join(RES, pattern))):
        with open(path, newline="") as fh:
            for r in csv.DictReader(fh):
                for k in FLOAT_COLS:
                    if k in r and r[k] != "":
                        r[k] = float(r[k])
                for k in INT_COLS:
                    if k in r and r[k] != "":
                        r[k] = int(r[k])
                rows.append(r)
    return rows


def final(rows, key):
    """Last sample of each arm, keyed by `key(row)` -- the arm's converged error and rank.

    Keyed on max `t` rather than on file order: a run killed mid-arm still contributes its last
    complete sample instead of dropping out of the figure silently.
    """
    best = {}
    for r in rows:
        k = key(r)
        if k not in best or r["t"] > best[k]["t"]:
            best[k] = r
    return best


def style(ax, xlabel, ylabel, title=None):
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title, fontsize=10.5, pad=8)
    ax.grid(True, which="both", lw=.4, alpha=.35)
    ax.tick_params(labelsize=9)


def save(fig, name):
    fig.tight_layout()
    out = os.path.join(RES, name)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


# ── 0a / 0b: the knee ────────────────────────────────────────────────────────
def plot_knee():
    rows = load("heis_tau_*.csv")
    if not rows:
        return
    L, dt = rows[0]["L"], rows[0]["dt"]
    fin = final(rows, lambda r: (r["scheme"], r["tau_trunc"]))

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    allerr, knees = [], {}
    for sch in order({k[0] for k in fin}):
        pts = sorted((tau, fin[(s, tau)]) for (s, tau) in fin if s == sch)
        xs = [t for t, _ in pts]
        es = [max(r["err_prof"], FLOOR) for _, r in pts]
        allerr += es
        kw = dict(color=COLOR.get(sch, "#666"), label=LABEL.get(sch, sch),
                  lw=WIDTH.get(sch, 1.6), ms=MSIZE.get(sch, 4.5), zorder=ZORDER.get(sch, 5))
        a1.plot(xs, es, "o-", **kw)
        a2.plot(xs, [r["maxbond"] for _, r in pts], "o-", **kw)
        # The knee = the loosest tolerance already within 2x of this scheme's floor. Read off
        # the data rather than eyeballed, so the annotation cannot drift from the curve.
        floor = min(es)
        cands = [t for t, e in zip(xs, es) if e <= 2 * floor]
        cands and knees.__setitem__(sch, max(cands))
    for ax in (a1, a2):
        ax.set_xscale("log")
        ax.invert_xaxis()          # tighter tolerance to the RIGHT, i.e. more work rightwards
    a1.set_yscale("log")
    tighten(a1, allerr)
    for sch, k in knees.items():
        a1.axvline(k, color=COLOR.get(sch, "#666"), lw=.9, ls="--", alpha=.55, zorder=0)
    style(a1, r"$\tau_{\rm trunc}$", r"$L_\infty$ error in $\langle S^z_j\rangle$",
          "0a  Rising = truncation-limited. Flat = dt-limited.\n"
          "Dashed = the knee: tightening past it buys nothing.")
    style(a2, r"$\tau_{\rm trunc}$", r"final $\chi$",
          "0b  Rank bought. Right of the knee it is paid for nothing.")
    a1.legend(fontsize=9, frameon=False, loc="lower left")
    # Coincidence is a RESULT here, so it is stated rather than left for the reader to infer
    # from a line they cannot see.
    a1.text(.98, .04, "TDVP arms coincide to all printed digits\n(drawn nested: thick under thin)",
            transform=a1.transAxes, ha="right", va="bottom", fontsize=7.5, color="#555")
    fig.suptitle(f"Heisenberg XXZ $\\Delta$=1, Néel quench, L={L}, dt={dt}, uncapped maxdim",
                 fontsize=10, y=1.0)
    save(fig, f"heis_knee_L{L}.png")


# ── 0c / 0d: the dt calibration ──────────────────────────────────────────────
def plot_dtcal():
    rows = load("heis_dt_*.csv")
    if not rows:
        return
    L = rows[0]["L"]
    fin = final(rows, lambda r: (r["scheme"], r["tau_trunc"], r["dt"]))
    schemes = sorted({k[0] for k in fin})

    fig, axes = plt.subplots(1, len(schemes), figsize=(4.6 * len(schemes), 4.3), squeeze=False)
    for ax, sch in zip(axes[0], order(schemes)):
        taus = sorted({k[1] for k in fin if k[0] == sch}, reverse=True)
        for i, tau in enumerate(taus):
            pts = sorted((d, fin[(sch, tau, d)]) for (s, tt, d) in fin
                         if s == sch and tt == tau)
            ax.plot([d for d, _ in pts], [max(r["err_prof"], FLOOR) for _, r in pts], "o-",
                    ms=4.5, lw=1.6, color=plt.cm.viridis(i / max(len(taus) - 1, 1)),
                    label=rf"$\tau$={tau:.0e}")
        ax.set_xscale("log"); ax.set_yscale("log")
        style(ax, "dt", r"$L_\infty$ error", LABEL.get(sch, sch))
        ax.legend(fontsize=8, frameon=False)
    fig.suptitle("0c  Where a line leaves the common envelope, truncation has taken over "
                 f"from time integration  (L={L})", fontsize=10, y=1.0)
    save(fig, f"heis_dtcal_L{L}.png")


# ── 6A: root off, half-sweep cutoff scanned ──────────────────────────────────
def plot_split():
    rows = load("heis_split_*.csv")
    if not rows:
        return
    L, dt = rows[0]["L"], rows[0]["dt"]
    fin = final(rows, lambda r: r["split_cutoff"])
    xs = sorted(fin)

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(14.5, 4.3))
    a1.plot(xs, [max(fin[x]["err_prof"], FLOOR) for x in xs], "o-",
            color=COLOR["cbe_bug"], ms=5, lw=1.7)
    a1.set_xscale("log"); a1.set_yscale("log"); a1.invert_xaxis()
    style(a1, "half-sweep split_cutoff", r"$L_\infty$ error",
          "6a  Monotone = the half-sweeps govern accuracy.\nFlat = they do not.")

    for x in xs:
        ts = sorted([r for r in rows if r["split_cutoff"] == x], key=lambda r: r["t"])
        a2.plot([r["t"] for r in ts], [r["maxbond"] for r in ts], lw=1.5, label=f"{x:.0e}")
    style(a2, "t", r"$\chi$", "6b  No closing truncation: rank can only ratchet up.\n"
                              "Runaway $\\chi$ at flat error = a non-binding cutoff.")
    a2.legend(fontsize=8, frameon=False, ncol=2, title="split_cutoff", title_fontsize=8)

    a3.plot([fin[x]["maxbond"] for x in xs], [max(fin[x]["err_prof"], FLOOR) for x in xs],
            "o-", color=COLOR["cbe_bug"], ms=5, lw=1.7)
    for x in xs:
        a3.annotate(f"{x:.0e}", (fin[x]["maxbond"], max(fin[x]["err_prof"], FLOOR)),
                    fontsize=7, xytext=(4, 4), textcoords="offset points")
    a3.set_yscale("log")
    style(a3, r"final $\chi$", r"$L_\infty$ error",
          "6c  Pareto: is half-sweep truncation\nan efficient place to spend rank?")

    fig.suptitle("6A  CBE-BUG, closing truncation OFF and root split non-truncating — "
                 f"split_cutoff is the ONLY rank control  (L={L}, dt={dt})", fontsize=10, y=1.0)
    save(fig, f"heis_split_L{L}.png")


# ── 6B: the 2D scan, root-out x half-sweep ───────────────────────────────────
def plot_grid():
    """The matrix that separates the two remaining truncations.

    ⛔ THE HEATMAP IS THE ANSWER AND THE LINE FAMILY IS THE PROOF. A heatmap alone invites reading
    a colour gradient that is really one decade of noise; the line panel puts the numbers on a log
    axis where a flat line is unambiguously flat. Both are drawn, from the same data.

    ⛔ ROW/COLUMN STRUCTURE IS WHAT IS BEING READ, NOT THE OVERALL LEVEL. Variation down columns
    only means the root-out pass governs; across rows only means the half-sweeps do; along the
    anti-diagonal means the LOOSER of the two governs and they are redundant. The diagonal
    (both thresholds equal) is marked because it is the "one tolerance governs the step"
    configuration the study is aiming at.
    """
    allrows = load("heis_grid_*.csv")
    if not allrows:
        return
    # One matrix per (HALF-SWEEP STRUCTURE, L, x-axis). Three separate keys, all load-bearing:
    #
    #   scheme  -- a pooled heatmap would silently average three different algorithms.
    #   L       -- smoke-test CSVs at small L sit in the same directory and the same glob, and
    #              `final()` keys on (root, x) alone, so an L=8 row would OVERWRITE the L=12 row
    #              it collides with. That is a wrong number in a figure, not a missing one.
    #   x-axis  -- `split_cutoff` and the cbe cutoff are DIFFERENT truncations; the CSV carries
    #              both columns and only one of them varies per run.
    by_key = defaultdict(list)
    for r in allrows:
        xaxis = "cbe" if float(r.get("cbe_cutoff", 0) or 0) > 0 else "split"
        by_key[(r["scheme"], r["L"], xaxis)].append(r)
    for key in sorted(by_key, key=lambda k: (k[1], k[2],
                                             DRAW_ORDER.index(k[0]) if k[0] in DRAW_ORDER else 99)):
        _one_grid(key, by_key[key])
    for L in sorted({k[1] for k in by_key}):
        for xaxis in sorted({k[2] for k in by_key if k[1] == L}):
            sel = {k[0]: v for k, v in by_key.items() if k[1] == L and k[2] == xaxis}
            _grid_compare(sel, L, xaxis)


SWEEP_LABEL = {
    "cbe_bug": "default  (both frames, evolve, union)",
    "cbe_bug_cbe_only": "sweep 1  cbe_only  (both frames + split, NO evolve)",
    "cbe_bug_historic": "sweep 2  historic  (both frames, evolve, no union)",
    "cbe_bug_lubich": "lubich  (U_ex only — V_ex discarded)",
}
SWEEP_COLOR = {"cbe_bug": "#c1440e", "cbe_bug_cbe_only": "#2a6558",
               "cbe_bug_historic": "#7a3d8f", "cbe_bug_lubich": "#8a7a20"}


XLABEL = {"split": "half-sweep  split_cutoff  (SVD of the evolved 1-site tensor)",
          "cbe": "half-sweep  cbe cutoff  (stol_pre = stol_fnl, inside cbe_expand)"}


def _one_grid(key, rows):
    sch, L, xaxis = key
    dt = rows[0]["dt"]
    xcol = "cbe_cutoff" if xaxis == "cbe" else "split_cutoff"
    fin = final(rows, lambda r: (r["tau_trunc"], r[xcol]))
    roots = sorted({k[0] for k in fin}, reverse=True)     # loose -> tight, top to bottom
    splits = sorted({k[1] for k in fin}, reverse=True)    # loose -> tight, left to right

    def grid_of(field):
        return [[fin[(rt, sp)][field] if (rt, sp) in fin else float("nan")
                 for sp in splits] for rt in roots]

    import math
    err = grid_of("err_prof")
    chi = grid_of("maxbond")
    logerr = [[math.log10(max(v, FLOOR)) if v == v else float("nan") for v in row] for row in err]

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.0))
    ticks = [f"{v:.0e}" for v in splits]
    yticks = [f"{v:.0e}" for v in roots]

    for ax, data, cmap, name, fmt in (
            (axes[0], logerr, "viridis_r", r"$\log_{10}$  $L_\infty$ error", "{:.1f}"),
            (axes[1], chi, "magma_r", r"final $\chi$", "{:.0f}")):
        im = ax.imshow(data, cmap=cmap, aspect="auto")
        ax.set_xticks(range(len(splits)), ticks, fontsize=8, rotation=45, ha="right")
        ax.set_yticks(range(len(roots)), yticks, fontsize=8)
        for i in range(len(roots)):
            for j in range(len(splits)):
                v = data[i][j]
                if v == v:
                    ax.text(j, i, fmt.format(v), ha="center", va="center", fontsize=7,
                            color="#fff" if _dark(im, v) else "#111")
        # The diagonal: both thresholds equal, i.e. one tolerance for the whole step.
        for i, rt in enumerate(roots):
            if rt in splits:
                ax.add_patch(plt.Rectangle((splits.index(rt) - .5, i - .5), 1, 1,
                                           fill=False, edgecolor="#00b3b3", lw=1.8, zorder=4))
        ax.set_xlabel(XLABEL[xaxis], fontsize=8.5)
        ax.set_ylabel("root-out  trunc_thresh  (#3)")
        ax.set_title(name, fontsize=10.5, pad=8)
        fig.colorbar(im, ax=ax, fraction=.046, pad=.03)

    a3 = axes[2]
    allv = []
    for i, rt in enumerate(roots):
        ys = [max(fin[(rt, sp)]["err_prof"], FLOOR) for sp in splits if (rt, sp) in fin]
        xs = [sp for sp in splits if (rt, sp) in fin]
        allv += ys
        a3.plot(xs, ys, "o-", ms=4, lw=1.5,
                color=plt.cm.viridis(i / max(len(roots) - 1, 1)), label=f"{rt:.0e}")
    a3.set_xscale("log"); a3.set_yscale("log"); a3.invert_xaxis()
    tighten(a3, allv)
    style(a3, XLABEL[xaxis].split("  (")[0], r"$L_\infty$ error",
          "Flat lines = the half-sweeps are inert.\nSeparated lines = the root-out pass governs.")
    a3.legend(fontsize=7.5, frameon=False, ncol=2, title="root-out thresh", title_fontsize=8)

    fig.suptitle(f"6B  {SWEEP_LABEL.get(sch, sch)} — root-out threshold × {xaxis} cutoff, "
                 f"closing pass ON  (L={L}, dt={dt}).  Cyan = both equal.",
                 fontsize=10, y=1.0)
    save(fig, f"heis_grid_{xaxis}_{sch}_L{L}.png")


def _grid_compare(by_scheme, L, xaxis):
    """The three half-sweep structures on ONE pair of axes.

    ⛔ THIS IS THE PLOT THAT ANSWERS THE QUESTION, and the per-sweep heatmaps are the evidence
    behind it. Read alone, three heatmaps invite eyeballing colour against colour across figures
    with independently scaled colourbars -- which is not a comparison. Here every sweep is on one
    log axis at matched thresholds.

    The x-axis is the DIAGONAL of each matrix (both thresholds equal), because that is the
    configuration the study is aiming at: one tolerance governing the whole step.
    """
    xcol = "cbe_cutoff" if xaxis == "cbe" else "split_cutoff"
    diag = {}
    for sch, rows in by_scheme.items():
        fin = final(rows, lambda r: (r["tau_trunc"], r[xcol]))
        pts = sorted((rt, fin[(rt, sp)]) for (rt, sp) in fin if rt == sp)
        if pts:
            diag[sch] = pts
    if len(diag) < 2:
        return

    fig, (a1, a2) = plt.subplots(1, 2, figsize=(11.5, 4.5))
    allv = []
    for sch in order(diag):
        pts = diag[sch]
        xs = [t for t, _ in pts]
        es = [max(r["err_prof"], FLOOR) for _, r in pts]
        allv += es
        kw = dict(color=SWEEP_COLOR.get(sch, "#666"), label=SWEEP_LABEL.get(sch, sch),
                  ms=4.5, lw=1.8)
        a1.plot(xs, es, "o-", **kw)
        a2.plot([r["krylov"] for _, r in pts], es, "o-", **kw)
    a1.set_xscale("log"); a1.set_yscale("log"); a1.invert_xaxis()
    tighten(a1, allv)
    style(a1, r"$\tau$  (both thresholds equal)", r"$L_\infty$ error",
          "Accuracy along the diagonal")
    a2.set_xscale("log"); a2.set_yscale("log")
    tighten(a2, allv)
    # Operator applications, not seconds: contention-immune, and the only cost axis that means
    # the same thing across three sweeps doing structurally different work per bond.
    style(a2, "operator applications", r"$L_\infty$ error",
          "Cost of that accuracy — the comparison that decides it")
    a1.legend(fontsize=8, frameon=False, loc="lower left")
    fig.suptitle(f"Half-sweep structures on the diagonal, {xaxis} cutoff axis, "
                 f"same reference  (L={L})", fontsize=10.5, y=1.0)
    save(fig, f"heis_sweeps_{xaxis}_L{L}.png")


def _dark(im, v):
    "White text on the dark end of the colormap, so every cell label stays readable."
    lo, hi = im.get_clim()
    t = 0.0 if hi == lo else (v - lo) / (hi - lo)
    r, g, b, _ = im.get_cmap()(t)
    return (0.299 * r + 0.587 * g + 0.114 * b) < 0.5


for f in (plot_knee, plot_dtcal, plot_split, plot_grid):
    f()
