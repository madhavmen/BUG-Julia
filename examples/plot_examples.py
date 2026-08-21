"""Figures for every CSV in `examples/results/`.

One figure per run -- a run being one (model, symmetry, L, maxdim, dover, dt) -- with the three
panels the examples exist to produce:

  (a) the observable against time, with its EXACT value underneath
  (b) |observable - exact| against time, log scale        <- how the schemes are ranked
  (c) max bond dimension against time                     <- what the rank adaptivity did

Written to `examples/results/<model>_L<n>_<sym>_D<d>[_dover<k>].png`.

⛔ RANK THE SCHEMES BY PANEL (b), NEVER BY ENERGY DRIFT. `dE` is in the CSV as a conservation
check and it disagrees outright with accuracy on these models: a scheme can conserve energy to
machine precision and still be an order of magnitude further from the exact solution. It is not
plotted here on purpose.

usage:  python3 examples/plot_examples.py [glob-or-csv] [outdir]
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
PATTERN = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "*.csv")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")

# Same colour per integrator as benchmarks/plot_oat.py, so a scheme is the same colour in every
# figure in the repo.
COLOR = {"tdvp_cbe1s": "#2ca02c", "tdvp2": "#7f4fc0", "cbe_bug": "#d62728"}
LABEL = {"tdvp_cbe1s": "1-site TDVP-CBE", "tdvp2": "2-site TDVP", "cbe_bug": "CBE-BUG"}

# ⛔ A PER-SCHEME DASH PATTERN AND WIDTH, BECAUSE THE CURVES GENUINELY COINCIDE. At or above the
# exact rank the three schemes agree to ~1e-14, and that agreement is the result rather than an
# accident -- but drawn in one style they hide each other exactly, and a reader cannot tell
# "three curves agree" from "one curve was plotted". Widest first, narrowing dashes on top.
STYLE = {                     # (linestyle, linewidth, z-order)
    "tdvp2":      ("-",           3.2, 3),
    "tdvp_cbe1s": ((0, (6, 3)),   1.9, 4),
    "cbe_bug":    ((0, (1.5, 2)), 1.5, 5),
}
ORDER = ["tdvp2", "tdvp_cbe1s", "cbe_bug"]

# The floor below which "error" is roundoff rather than the integrator. Plotted as a guide so a
# curve sitting on it is not read as a measured accuracy.
FLOOR = 1e-15


def load(pattern):
    paths = sorted(glob.glob(pattern)) if any(c in pattern for c in "*?[") else [pattern]
    if not paths:
        sys.exit("no CSV matched %r -- run one of the example scripts first" % pattern)
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                r["_src"] = os.path.basename(p)
                rows.append(r)
    if not rows:
        sys.exit("matched %d file(s) but they contain no rows" % len(paths))
    return rows


def key(r):
    # `dover` post-dates the first CSVs; treat a file without the column as the default.
    return (r["model"], r["symmetry"], int(r["L"]), int(r["maxdim"]),
            int(r.get("dover", 0) or 0), float(r["dt"]))


def num(rows, col):
    return [float(r[col]) for r in rows]


def panel_observable(ax, byscheme, model):
    any_exact = False
    for name in ORDER:
        rows = byscheme.get(name)
        if not rows:
            continue
        ls, lw, z = STYLE[name]
        ax.plot(num(rows, "t"), num(rows, "obs"), color=COLOR[name], linestyle=ls,
                linewidth=lw, zorder=z, label=LABEL[name])
        if not any_exact:
            # One exact curve, drawn UNDER everything: it is identical in every scheme's rows.
            ax.plot(num(rows, "t"), num(rows, "obs_exact"), color="0.35", linewidth=1.0,
                    linestyle=(0, (1, 1.5)), zorder=1, label="exact")
            any_exact = True
    ax.set_xlabel("t")
    ax.set_ylabel(OBS_LABEL.get(model, "observable"))
    ax.set_title("(a) observable")
    ax.legend(fontsize=8, framealpha=0.9)
    ax.grid(alpha=0.25)


def panel_error(ax, byscheme):
    lo = FLOOR
    for name in ORDER:
        rows = byscheme.get(name)
        if not rows:
            continue
        ls, lw, z = STYLE[name]
        # Clamp at the roundoff floor so an exact-to-machine-precision arm still draws a line
        # instead of vanishing when `err` underflows to 0 and log scale drops the point.
        err = [max(e, FLOOR) for e in num(rows, "err")]
        ax.semilogy(num(rows, "t"), err, color=COLOR[name], linestyle=ls,
                    linewidth=lw, zorder=z, label=LABEL[name])
        lo = min(lo, min(err))
    ax.axhline(FLOOR, color="0.6", linewidth=0.8, linestyle=":", zorder=0)
    ax.text(0.02, FLOOR * 2.2, "roundoff floor", transform=ax.get_yaxis_transform(),
            fontsize=7, color="0.45")
    ax.set_xlabel("t")
    ax.set_ylabel(r"$|\mathrm{obs} - \mathrm{exact}|$")
    ax.set_title("(b) error")
    ax.grid(alpha=0.25, which="both")


def panel_bond(ax, byscheme, cap):
    for name in ORDER:
        rows = byscheme.get(name)
        if not rows:
            continue
        ls, lw, z = STYLE[name]
        ax.plot(num(rows, "t"), num(rows, "maxbond"), color=COLOR[name], linestyle=ls,
                linewidth=lw, zorder=z, label=LABEL[name])
    # The cap is the whole point of panel (c): a curve pinned to it is TRUNCATION-limited and
    # its error in (b) is the truncation's, not the time step's.
    ax.axhline(cap, color="0.4", linewidth=1.0, linestyle="--", zorder=1)
    ax.text(0.02, cap, " maxdim", transform=ax.get_yaxis_transform(),
            va="bottom", fontsize=7, color="0.35")
    ax.set_xlabel("t")
    ax.set_ylabel(r"max $\chi$")
    ax.set_title("(c) bond dimension")
    ax.set_ylim(bottom=0)
    ax.grid(alpha=0.25)


OBS_LABEL = {
    "OAT":  r"$S^x_{\mathrm{tot}}(t)$",
    "XX":   r"$\sum_{\ell \leq L/2} \langle S^z_\ell \rangle$",
    "TFIM": r"$\sum_{\ell \leq L/2} \langle X_\ell \rangle$",
}


def main():
    rows = load(PATTERN)
    groups = defaultdict(list)
    for r in rows:
        groups[key(r)].append(r)

    os.makedirs(OUTDIR, exist_ok=True)
    for (model, sym, L, cap, dover, dt), grp in sorted(groups.items()):
        byscheme = defaultdict(list)
        for r in grp:
            byscheme[r["scheme"]].append(r)
        for v in byscheme.values():
            v.sort(key=lambda r: float(r["t"]))

        fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
        panel_observable(axes[0], byscheme, model)
        panel_error(axes[1], byscheme)
        panel_bond(axes[2], byscheme, cap)
        fig.suptitle("%s   L = %d   symmetry = %s   maxdim = %d   dt = %g%s"
                     % (model, L, sym, cap, dt,
                        "" if dover == 0 else "   dover = %d" % dover),
                     fontsize=11)
        fig.tight_layout(rect=(0, 0, 1, 0.94))

        name = "%s_L%d_%s_D%d%s.png" % (model.lower(), L, sym, cap,
                                        "" if dover == 0 else "_dover%d" % dover)
        out = os.path.join(OUTDIR, name)
        fig.savefig(out, dpi=140)
        plt.close(fig)

        # The number the figure is for, printed so a run can be read without opening the PNG.
        finals = {n: v[-1] for n, v in byscheme.items() if v}
        summary = "  ".join("%s %.2e" % (n, float(finals[n]["err"]))
                            for n in ORDER if n in finals)
        print("wrote %s    final err:  %s" % (out, summary))


if __name__ == "__main__":
    main()
