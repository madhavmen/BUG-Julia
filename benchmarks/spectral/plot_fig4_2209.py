#!/usr/bin/env python3
"""Fig. 4 of arXiv:2209.00739 -- S(q,omega) of the square-lattice Heisenberg model.

Panels follow the paper: the BZ path, the spectrum along it, the dispersion near M, the
dispersion against linear spin-wave theory, and frequency cuts at high-symmetry momenta. A sixth
panel carries the integrator comparison, which is this repository's question rather than theirs.

eps(q) is Eq. (19), argmax_w S(q,w) -- the paper's preferred extractor ("using Eq. (19) yields a
more accurate method of obtaining the dispersion relation"). <w>(q) is Eq. (20), the first moment,
shown for contrast.

WHAT IS AND IS NOT A REPRODUCTION. The paper runs C=6, L=36 -> 216 sites. At the sizes here
S(q,w) is a few delta functions smeared by the Gaussian, only q_y = 2*pi*n/Ly are exact quantum
numbers, and the resolution along the open direction is 2*pi/Lx. The pipeline, the sum rules, the
qualitative magnon branch and the RELATIVE COST OF THE INTEGRATORS are meaningful; the dispersion
values are not their Fig. 4 d).

Usage:  python3 benchmarks/spectral/plot_fig4_2209.py [results_dir] [LxxLy]
"""
import csv
import glob
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


def lswt_square(qx, qy, J=1.0):
    """Linear spin-wave magnon energy of the square-lattice Heisenberg AFM.

    w_q = z J S sqrt(1 - gamma_q^2) with z = 4, S = 1/2, gamma_q = (cos qx + cos qy)/2,
    i.e. 2 J sqrt(1 - gamma^2). At q = (pi, 0) this is 2J, the textbook LSWT value that quantum
    corrections renormalise upward. The paper rescales SWT "by a common factor"; it is drawn
    UNSCALED here so the raw comparison is visible.
    """
    g = (np.cos(qx) + np.cos(qy)) / 2.0
    return 2.0 * J * np.sqrt(np.maximum(0.0, 1.0 - g ** 2))


def load_sqw(path):
    rows = list(csv.DictReader(open(path)))
    iqs = sorted({int(r["iq"]) for r in rows})
    ws = sorted({float(r["w"]) for r in rows})
    wi = {w: i for i, w in enumerate(ws)}
    S = np.zeros((len(iqs), len(ws)))
    meta = {}
    for r in rows:
        S[int(r["iq"]) - 1, wi[float(r["w"])]] = float(r["S"])
        meta[int(r["iq"])] = (float(r["qx"]), float(r["qy"]), r["label"],
                              float(r["dist"]), int(r["exact_qy"]))
    return np.array(ws), S, meta


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else os.path.join(os.path.dirname(__file__), "..", "results")
    files = sorted(glob.glob(os.path.join(d, "fig4_Sqw_*.csv")))
    if not files:
        sys.exit("no fig4_Sqw_*.csv in %s -- run benchmarks/spectral/fig4_2209.jl first" % d)
    if len(sys.argv) > 2:
        files = [f for f in files if sys.argv[2] in f]

    arms = {}
    for f in files:
        base = os.path.basename(f)[len("fig4_Sqw_"):-len(".csv")]
        geo, arm = base.rsplit("_", 1)
        arms[arm] = load_sqw(f)
    geo = base.rsplit("_", 1)[0]

    ref = arms.get("bug") or list(arms.values())[0]
    ws, S, meta = ref
    nq = S.shape[0]
    dist = np.array([meta[i + 1][3] for i in range(nq)])
    qx = np.array([meta[i + 1][0] for i in range(nq)])
    qy = np.array([meta[i + 1][1] for i in range(nq)])
    exact = np.array([meta[i + 1][4] for i in range(nq)], bool)
    ticks = [(meta[i + 1][3], meta[i + 1][2]) for i in range(nq) if meta[i + 1][2]]

    # Eq. (19) and Eq. (20). Negative weight is a finite-record artefact, not physics, so the
    # moment is taken over the positive part -- and the fraction clipped is reported, because a
    # large one means the record or the broadening is not adequate.
    Spos = np.clip(S, 0.0, None)
    eps19 = ws[np.argmax(Spos, axis=1)]
    # ⛔ argmax OF NOTHING IS NOT A DISPERSION POINT. At q = Gamma the weight is ~1e-18 (S^z_tot
    # commutes with H, so it vanishes identically) and `argmax` then returns wherever the
    # roundoff happens to peak. The paper says the same of its own figure -- "the value at
    # q = Gamma is not accurate, as there is essentially zero [weight]". Mask any q carrying less
    # than 1e-6 of the largest weight on the path, so those points are ABSENT rather than drawn
    # as if they were measurements.
    peak = Spos.max(axis=1)
    weak = peak < 1e-6 * peak.max()
    eps19 = np.where(weak, np.nan, eps19)
    denom = Spos.sum(axis=1)
    w20 = np.where(denom > 0, (Spos * ws[None, :]).sum(axis=1) / np.maximum(denom, 1e-300), np.nan)
    # Eq. (20) needs the SAME mask as Eq. (19): it is a ratio whose denominator is the total
    # weight, so at q = Gamma it divides ~1e-18 by ~1e-18 and returns a confident-looking number
    # made entirely of roundoff. Masking one extractor and not the other would have left that
    # point on the figure through the orange curve alone.
    w20 = np.where(weak, np.nan, w20)
    negfrac = np.abs(np.clip(S, None, 0.0)).sum() / max(np.abs(S).sum(), 1e-300)

    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 3, hspace=0.32, wspace=0.28)

    # (a) the path through the BZ
    ax = fig.add_subplot(gs[0, 0])
    ax.plot(qx, qy, "-", color="0.6", lw=1)
    ax.scatter(qx[exact], qy[exact], s=26, c="tab:blue", label=r"$q_y$ allowed", zorder=3)
    ax.scatter(qx[~exact], qy[~exact], s=18, facecolors="none", edgecolors="tab:red",
               label=r"$q_y$ interpolated", zorder=3)
    for xx, lab in [(0, r"$\Gamma$"), (np.pi, "X"), (np.pi, "M")]:
        pass
    ax.annotate(r"$\Gamma$", (0, 0), fontsize=13, xytext=(4, 4), textcoords="offset points")
    ax.annotate("X", (np.pi, 0), fontsize=13, xytext=(4, 4), textcoords="offset points")
    ax.annotate("M", (np.pi, np.pi), fontsize=13, xytext=(4, 4), textcoords="offset points")
    ax.set_xlabel(r"$q_x$"); ax.set_ylabel(r"$q_y$")
    ax.set_title("a) path in the Brillouin zone")
    ax.legend(fontsize=8); ax.set_aspect("equal"); ax.grid(alpha=0.3)

    # (b) the spectrum along the path
    ax = fig.add_subplot(gs[0, 1])
    im = ax.pcolormesh(dist, ws, Spos.T, shading="auto", cmap="magma")
    ax.plot(dist, eps19, "o-", ms=3, lw=1.2, color="cyan", label=r"$\epsilon(q)$ Eq. (19)")
    ax.plot(dist, lswt_square(qx, qy), "--", lw=1.4, color="w", label="LSWT (unscaled)")
    for x, lab in ticks:
        ax.axvline(x, color="w", lw=0.6, alpha=0.5)
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([r"$\Gamma$" if t[1] == "G" else t[1] for t in ticks])
    ax.set_ylabel(r"$\omega / J$")
    ax.set_title("b) $S(q,\\omega)$  [%s]" % (("bug" if "bug" in arms else list(arms)[0])))
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, pad=0.02)

    # (c) dispersion near M, fitted to v q + Delta as in the paper
    ax = fig.add_subplot(gs[0, 2])
    iM = int(np.argmin(np.abs(dist - [t[0] for t in ticks][2]))) if len(ticks) > 2 else nq // 2
    sel = slice(max(0, iM - 5), min(nq, iM + 1))
    dq = dist[sel] - dist[iM]
    ev = eps19[sel]
    ax.plot(dq, ev, "o", color="tab:blue", label=r"$\epsilon(q)$")
    # ⛔ ONLY FIT A VELOCITY IF THE MOMENTUM GRID CAN RESOLVE ONE, and refuse loudly otherwise.
    # The velocity is the SLOPE of eps(q) near M, so it needs several DISTINCT eps values inside
    # a window that is actually near M. On a 3-column cylinder the allowed q_x are spaced 2*pi/3
    # and eps(q) is flat across the whole selection -- fitting that returned "v = 0.00", which is
    # not a small velocity, it is no measurement at all. A number like that on a figure outlives
    # every caveat in the text.
    m = (dq != 0) & np.isfinite(ev)
    spread = np.ptp(ev[m]) if m.sum() > 1 else 0.0
    if m.sum() > 2 and spread > 1e-3 and abs(dq[m]).max() < 1.0:
        p = np.polyfit(dq[m], ev[m], 1)
        ax.plot(dq, np.polyval(p, dq), "-", color="tab:blue",
                label=r"$vq+\Delta$: $v$=%.2f $\Delta$=%.2f" % (abs(p[0]), p[1]))
        p0 = np.sum(dq[m] * ev[m]) / max(np.sum(dq[m] ** 2), 1e-30)
        ax.plot(dq, p0 * dq, "--", color="tab:red",
                label=r"$\Delta\equiv0$: $v$=%.2f" % abs(p0))
    else:
        ax.text(0.5, 0.45,
                "no velocity fit:\n%d usable $q$ points, spread %.1e over $|q-q_M|\\leq$%.2f\n"
                "the grid ($2\\pi/L_x$) cannot resolve a slope at this size" %
                (int(m.sum()), spread, abs(dq).max() if len(dq) else 0.0),
                ha="center", va="center", transform=ax.transAxes, fontsize=8.5,
                bbox=dict(boxstyle="round", fc="mistyrose", ec="crimson"))
    ax.set_xlabel(r"$q - q_M$"); ax.set_ylabel(r"$\omega / J$")
    ax.set_title("c) dispersion near M")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (d) dispersion vs LSWT
    ax = fig.add_subplot(gs[1, 0])
    ax.plot(dist, eps19, "o-", ms=4, label=r"$\epsilon(q)$ Eq. (19)")
    ax.plot(dist, w20, "s--", ms=3, alpha=0.7, label=r"$\langle\omega\rangle(q)$ Eq. (20)")
    ax.plot(dist, lswt_square(qx, qy), "-", color="k", lw=1.4, label="LSWT (unscaled)")
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([r"$\Gamma$" if t[1] == "G" else t[1] for t in ticks])
    ax.set_ylabel(r"$\omega / J$")
    ax.set_title("d) magnon dispersion")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) frequency cuts at the high-symmetry momenta
    ax = fig.add_subplot(gs[1, 1])
    for x, lab in ticks:
        i = int(np.argmin(np.abs(dist - x)))
        nm = r"$\Gamma$" if lab == "G" else lab
        ax.plot(ws, Spos[i], lw=1.4, label="%s  ($S_{max}$=%.1e)" % (nm, Spos[i].max()))
    ax.set_xlabel(r"$\omega / J$"); ax.set_ylabel(r"$S(q,\omega)$")
    ax.set_title("e) cuts at high-symmetry $q$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (f) the integrator comparison -- this repository's question
    ax = fig.add_subplot(gs[1, 2])
    cost = os.path.join(d, "fig4_cost_%s.csv" % geo)
    if os.path.exists(cost) and len(arms) > 1:
        rows = list(csv.DictReader(open(cost)))
        names = [r["arm"] for r in rows]
        mv = np.array([float(r["matvec"]) for r in rows])
        base = mv[names.index("tdvp2")] if "tdvp2" in names else mv[0]
        xs = np.arange(len(names))
        sec = np.array([float(r["seconds"]) for r in rows])
        sbase = sec[names.index("tdvp2")] if "tdvp2" in names else sec[0]
        w = 0.38
        ax.bar(xs - w / 2, mv / base, w, color="0.55", label="operator applications")
        ax.bar(xs + w / 2, sec / sbase, w, color="tab:green", label="wall clock")
        for i in range(len(names)):
            ax.text(xs[i] - w / 2, mv[i] / base, "%d" % mv[i], ha="center", va="bottom",
                    fontsize=7.5)
            ax.text(xs[i] + w / 2, sec[i] / sbase, "%.0fs" % sec[i], ha="center", va="bottom",
                    fontsize=7.5)
        ax.set_xticks(xs); ax.set_xticklabels(names)
        ax.axhline(1.0, color="k", ls=":", lw=1)
        ax.set_ylim(0, 1.35)
        ax.set_ylabel("relative to tdvp2")
        ax.set_title("f) cost on the same correlator")
        ax.legend(fontsize=7.5, loc="upper right")
        # ⛔ THE TWO BARS DO NOT MEASURE THE SAME THING, AND THE GAP IS NOT A BUG.
        # `krylov_dims` counts `apply_one_site` calls only; BUG's expansion work lives inside
        # `cbe_expand`/`sketch_h_*` and is INVISIBLE to it, so the operator-application bar
        # UNDERSTATES BUG. Wall clock is the complete measure and is the one to quote.
        ax.text(0.02, 0.97,
                "matvec undercounts BUG:\ncbe_expand work is not in krylov_dims\n-> quote wall clock",
                transform=ax.transAxes, fontsize=7, va="top",
                bbox=dict(boxstyle="round", fc="lightyellow", ec="0.6"))
        # Agreement between arms: same physics, or a speedup that bought a different answer.
        if "tdvp2" in arms:
            devs = []
            for a in arms:
                if a == "tdvp2":
                    continue
                # `dev`, not `d`: `d` is the results DIRECTORY, and shadowing it here turned the
                # output path into a float at the very last line of the script.
                dev = np.abs(arms[a][1] - arms["tdvp2"][1]).max() / \
                    max(np.abs(arms["tdvp2"][1]).max(), 1e-300)
                devs.append("%s %.1e" % (a, dev))
            ax.text(0.02, 0.60, "max|dS| vs tdvp2:\n" + "\n".join(devs),
                    transform=ax.transAxes, fontsize=7.5, va="top",
                    bbox=dict(boxstyle="round", fc="honeydew", ec="0.6"))
    else:
        ax.text(0.5, 0.5, "run more than one arm\nfor the cost comparison",
                ha="center", va="center", transform=ax.transAxes)
        ax.set_axis_off()

    fig.suptitle("arXiv:2209.00739 Fig. 4 -- square-lattice $S(q,\\omega)$   [%s;  "
                 "negative-weight fraction %.1e]" % (geo, negfrac), fontsize=12)
    out = os.path.join(d, "fig4_2209_%s.png" % geo)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    print("wrote", out)
    print("negative-weight fraction %.3e (finite record; large => widen eta or lengthen T)" % negfrac)


if __name__ == "__main__":
    main()
