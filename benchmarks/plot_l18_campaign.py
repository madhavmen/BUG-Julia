#!/usr/bin/env python3
"""Figures for the L=18 trajectory campaign: physics first, then the two cost axes.

    python3 benchmarks/plot_l18_campaign.py [results/l18_campaign_<tag>.csv ...]

One PNG per model, eight panels:

    (a) the PHYSICAL OBSERVABLE     staggered magnetisation, with the exact curve where one exists
    (b) the Sz PROFILE              snapshots in space -- this is where a light cone or a
                                    diffusive front is visible at all, and a scalar summary hides
                                    both
    (c) BOND DIMENSION vs t
    (d) OPERATOR ENTANGLEMENT       log_2, arXiv:2410.18468 Eq. (3) -- the open models' physics
                                    validation, and the axis their entanglement transition lives on
    (e) ERROR vs t                  log scale
    (f) KRYLOV APPLICATIONS vs t    cumulative
    (g) WALL CLOCK vs t             cumulative
    (h) THE EFFICIENCY FRONTIER     error vs cumulative wall clock -- error at EQUAL COST

--- READING RULES, WHICH ARE NOT COSMETIC -------------------------------------------------

* Panel (e) IS AN ERROR ONLY FOR THE MODELS IN `EXACT_MODELS`: `xx` (free-fermion
  Jordan-Wigner) and `xx_open` (the dephasing correlation-matrix ODE, `xx_dephasing_sz` in
  tests/common/free_fermion.jl). Everywhere else the column holds a DEVIATION FROM `tdvp2`, so
  `tdvp2` is identically zero there by construction and is drawn as a dashed line at the axis
  floor rather than as a curve that could be mistaken for an accuracy result. The panel title
  says which of the two it is, per model, rather than leaving it to the caption.

* Panel (h) IS THE ONLY PANEL THAT RANKS THE ARMS. (f) and (g) are cost at fixed accuracy-per-step
  and (e) is accuracy at fixed t; neither answers "which arm is better", because the arms do not
  cost the same. (h) puts error and wall clock on the same axes, so DOWN-AND-LEFT is better and a
  crossing is a genuine regime boundary. ⛔ It is meaningless unless the `secs` column was measured
  with ONE timing job on the machine -- two concurrent Julia processes oversubscribe the cores and
  shift every curve right by a factor that is not a property of the integrator.

* Panels (e) and (f) DISAGREE, AND BOTH ARE PLOTTED FOR THAT REASON. `krylov` counts operator
  applications, but a 2-site matvec (tdvp2), a 1-site matvec (tdvp_cbe1s) and a 0-site one
  (cbe_bug's root) are not the same unit of work -- and `cbe_bug`'s count EXCLUDES the CBE
  expansion's `H*Theta`, which is invisible to `krylov_dims`. So (e) is the structural axis and
  (f) is the honest one. Quoting (e) alone overstates `cbe_bug`; on an 8x8 cylinder it used 19x
  fewer applications and took 1.91x LONGER.
"""
import csv
import sys
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

# ⛔ THE LINE WIDTHS AND DASHES ARE LOAD-BEARING, NOT DECORATION. `tdvp2` and `tdvp_cbe1s` agree
# to six digits on these models -- measured at L=16 they matched to EVERY printed digit -- so drawn
# as two solid lines of equal width the second one vanishes under the first and the figure reads as
# a missing arm rather than as agreement. `tdvp2` is therefore wide, pale-ish and drawn FIRST
# (lowest zorder); `tdvp_cbe1s` is dashed on top of it, so coincidence shows as a dashed line
# inside a broad band and divergence shows as two separate curves.
ARM_STYLE = {
    "tdvp2":      dict(color="#5b9bd5", marker="o", ls="-",  lw=4.0, ms=4.5, z=1,
                       label="2-site TDVP"),
    "tdvp_cbe1s": dict(color="#8a5a00", marker="s", ls="--", lw=1.6, ms=3.5, z=3,
                       label="1-site CBE-TDVP"),
    "cbe_bug":    dict(color="#a03030", marker="^", ls="-",  lw=1.8, ms=4.0, z=4,
                       label="CBE-BUG (rSVD, parallel)"),
}
# The models whose `err` column is a true error rather than a mutual deviation.
# ⛔ `xx_open` BELONGS HERE. Onsite dephasing closes the single-particle correlation matrix, so
# `dC/dt = -i[h,C] - gamma*(1-delta_ij)*C_ij` is an exact L x L ODE at any L and any gamma
# (`xx_dephasing_sz`). It is the ONE open model scored against a number rather than a published
# trend, which is what lets "TDVP degraded" be told apart from "BUG drifted".
EXACT_MODELS = {"xx", "xx_open"}


def load(paths):
    rows = []
    for p in paths:
        with open(p, newline="") as fh:
            for r in csv.DictReader(fh):
                rows.append(r)
    return rows


def load_profiles(tag):
    """(model, arm) -> {t: [values by site]}, or an empty dict when the file is absent."""
    path = RESULTS / f"l18_profile_{tag}.csv"
    out = defaultdict(lambda: defaultdict(list))
    if not path.exists():
        return out
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            out[(r["model"], r["arm"])][float(r["t"])].append(
                (int(r["site"]), float(r["value"])))
    for key in out:
        for t in out[key]:
            out[key][t].sort()
    return out


def f(row, key):
    v = row[key]
    if v in ("", "NaN", "nan", "Inf", "-Inf"):
        return float("nan")
    return float(v)


def plot_model(model, rows, profiles, tag):
    arms = [a for a in ARM_STYLE if any(r["arm"] == a for r in rows)]
    if not arms:
        return None
    L = int(rows[0]["L"])
    cap = int(rows[0]["cap"])
    exact = model in EXACT_MODELS

    gamma = f(rows[0], "gamma")
    d = int(rows[0]["d"])
    fig, axes = plt.subplots(2, 4, figsize=(21.5, 8.8))
    fig.suptitle(f"{model}   L = {L}, cap $\\chi$ = {cap}, "
                 + (f"$\\gamma$ = {gamma:g}, fused $d$ = {d}   " if d == 4 else "")
                 + "—   2-site TDVP vs 1-site CBE-TDVP vs CBE-BUG",
                 fontsize=13, y=0.98)
    (ax_obs, ax_prof, ax_chi, ax_sop), (ax_err, ax_kry, ax_wall, ax_front) = axes

    for arm in arms:
        rs = sorted([r for r in rows if r["arm"] == arm], key=lambda r: f(r, "t"))
        st = ARM_STYLE[arm]
        kw = dict(color=st["color"], marker=st["marker"], ls=st["ls"],
                  label=st["label"], ms=st["ms"], lw=st["lw"], zorder=st["z"])
        t = [f(r, "t") for r in rs]

        ax_obs.plot(t, [f(r, "mstag") for r in rs], **kw)
        ax_chi.plot(t, [f(r, "maxbond") for r in rs], **kw)
        ax_sop.plot(t, [f(r, "sop") for r in rs], **kw)
        ax_kry.plot(t, [f(r, "krylov") for r in rs], **kw)
        ax_wall.plot(t, [f(r, "secs") for r in rs], **kw)

        err = [f(r, "err") for r in rs]
        if arm == "tdvp2" and not exact:
            # Zero by construction -- see the module docstring.
            ax_err.axhline(float("nan"))
            ax_err.plot([], [], color=st["color"], ls="--",
                        label="2-site TDVP (the reference: 0 by construction)")
        else:
            pos = [(ti, ei) for ti, ei in zip(t, err)
                   if ei == ei and ei > 0]
            if pos:
                ax_err.semilogy([p[0] for p in pos], [p[1] for p in pos], **kw)
            # (h) THE FRONTIER. Same error values, but against the wall clock that bought them.
            # Only drawable where `err` is a true error -- a deviation-from-tdvp2 has no meaning
            # on this axis, since the reference arm would sit at zero cost and zero error.
            if exact:
                fr = [(f(r, "secs"), f(r, "err")) for r in rs
                      if f(r, "err") == f(r, "err") and f(r, "err") > 0 and f(r, "secs") > 0]
                if fr:
                    ax_front.loglog([p[0] for p in fr], [p[1] for p in fr], **kw)

        # Spatial snapshots: first, middle and last sample, for this arm only when it is the
        # BUG arm (three arms x three times in one panel is unreadable).
        prof = profiles.get((model, arm), {})
        if prof and arm == "cbe_bug":
            times = sorted(prof)
            for ti, alpha in zip([times[0], times[len(times) // 2], times[-1]],
                                 [0.35, 0.65, 1.0]):
                vals = [v for _, v in prof[ti]]
                ax_prof.plot(range(1, len(vals) + 1), vals, marker="o", ms=3,
                             lw=1.3, color=st["color"], alpha=alpha,
                             label=f"$t$ = {ti:g}")

    ax_obs.set(xlabel="$t$", ylabel=r"$M_{\rm stag} = \frac{1}{L}\sum_j (-1)^{j+1}\langle S^z_j\rangle$",
               title="(a) physical observable")
    ax_obs.legend(fontsize=8)
    ax_obs.grid(alpha=0.3)

    ax_prof.set(xlabel="site $j$", ylabel=r"$\langle S^z_j\rangle$",
                title="(b) spatial profile (CBE-BUG)")
    ax_prof.legend(fontsize=8)
    ax_prof.grid(alpha=0.3)

    ax_chi.set(xlabel="$t$", ylabel=r"max bond dimension $\chi$",
               title="(c) rank growth")
    ax_chi.axhline(cap, color="k", ls=":", lw=1,
                   label=f"cap = {cap}" + (" (SATURATED: rank-limited, not\nintegrator-limited)"
                                          if any(f(r, "maxbond") >= cap for r in rows) else ""))
    ax_chi.legend(fontsize=8)
    ax_chi.grid(alpha=0.3)

    # (d) THE PAPER'S AXIS. arXiv:2410.18468 Eq. (3) quotes the operator entanglement in log_2,
    # not nats -- `operator_entanglement(...; base = 2)` -- so a curve plotted from a nats column
    # is off by ln 2 = 1.44x and would read as a different transition.
    ax_sop.set(xlabel="$t$", ylabel=r"$S_{\rm op}$  [$\log_2$]",
               title="(d) operator entanglement — arXiv:2410.18468 Eq. (3)"
                     if d == 4 else "(d) entanglement entropy [$\\log_2$]")
    ax_sop.legend(fontsize=8)
    ax_sop.grid(alpha=0.3)

    ax_err.set(xlabel="$t$",
               ylabel=r"max$_j |\langle S^z_j\rangle - $ref$|$",
               title="(e) ERROR vs the exact solution" if exact
                     else "(e) DEVIATION from 2-site TDVP — not an error")
    ax_err.legend(fontsize=8)
    ax_err.grid(alpha=0.3, which="both")

    ax_kry.set(xlabel="$t$", ylabel="cumulative operator applications",
               title="(f) STRUCTURAL cost — mixed units, see caption")
    ax_kry.legend(fontsize=8)
    ax_kry.grid(alpha=0.3)

    ax_wall.set(xlabel="$t$", ylabel="cumulative step time [s]",
                title="(g) WALL CLOCK — the honest cost axis")
    ax_wall.legend(fontsize=8)
    ax_wall.grid(alpha=0.3)

    if exact:
        ax_front.set(xlabel="cumulative step time [s]",
                     ylabel=r"max$_j |\langle S^z_j\rangle - $exact$|$",
                     title="(h) EFFICIENCY FRONTIER — down-and-left is better")
        ax_front.legend(fontsize=8)
        ax_front.grid(alpha=0.3, which="both")
    else:
        ax_front.set(title="(h) EFFICIENCY FRONTIER — needs an exact reference")
        ax_front.text(0.5, 0.5,
                      f"no closed form for {model},\nso there is no error to put on this axis.\n\n"
                      "A deviation-from-tdvp2 cannot go here:\nthe reference arm would sit at zero "
                      "error\nand nonzero cost, which ranks it first\nby construction.",
                      ha="center", va="center", fontsize=9, color="#666666",
                      transform=ax_front.transAxes)
        ax_front.set_xticks([])
        ax_front.set_yticks([])

    fig.text(0.5, 0.005,
             "(f) mixes units: 2-site (tdvp2), 1-site (tdvp_cbe1s) and 0-site (cbe_bug root) "
             "matvecs are not the same work, and cbe_bug's count EXCLUDES the CBE expansion's "
             "H·Theta. Read (f) with (g), never alone; (h) is the only panel that ranks the arms.",
             ha="center", fontsize=8, style="italic", color="#444444")
    fig.tight_layout(rect=(0, 0.025, 1, 0.96))
    out = RESULTS / f"l18_{model}_{tag}.png"
    fig.savefig(out, dpi=145)
    plt.close(fig)
    return out


def main():
    args = sys.argv[1:]
    paths = [Path(a) for a in args] if args else sorted(RESULTS.glob("l18_campaign_*.csv"))
    if not paths:
        sys.exit("no l18_campaign_*.csv in benchmarks/results")
    for path in paths:
        tag = path.stem.replace("l18_campaign_", "")
        rows = load([path])
        profiles = load_profiles(tag)
        by_model = defaultdict(list)
        for r in rows:
            by_model[r["model"]].append(r)
        for model, rs in by_model.items():
            out = plot_model(model, rs, profiles, tag)
            if out:
                print("wrote", out.relative_to(HERE.parent))


if __name__ == "__main__":
    main()
