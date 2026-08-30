#!/usr/bin/env python3
"""Fig. 4 of arXiv:2209.00739 -- S(q,omega) of the square-lattice Heisenberg model.

Panels follow the paper: the BZ path, the spectrum along it, the dispersion against linear
spin-wave theory, and frequency cuts at high-symmetry momenta. A fifth panel carries the
integrator comparison, which is this repository's question rather than theirs.

⛔ THE "DISPERSION NEAR M" VELOCITY PANEL IS GONE. The velocity is the SLOPE of eps(q) near M, and
on a 3-column cylinder the allowed q_x are spaced 2*pi/3 with eps(q) flat across the whole
selection -- the fit refused itself ("5 usable q points, spread 0.0e+00") and the panel rendered as
an empty box holding a red error message. A panel that can never carry a measurement at these sizes
is not a caveat worth a sixth of the figure. The velocity belongs to the L=20 chain, where there is
an EXACT number (v = pi/2) to score against; see `plot_fig45_2209.py` panel e).

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
    corrections renormalise upward. The paper rescales SWT "by a common factor" and so does this
    figure -- see `lswt_scale`.
    """
    g = (np.cos(qx) + np.cos(qy)) / 2.0
    return 2.0 * J * np.sqrt(np.maximum(0.0, 1.0 - g ** 2))


# Literature quantum renormalisation of the LSWT magnon energy for the S=1/2 square-lattice
# Heisenberg AFM. Series expansion and QMC both put it near 1.18 (Singh & Gelfand PRB 52 R15695;
# Sandvik & Singh PRL 86 528; Ronnow et al. PRL 87 037202). It is the number the FITTED scale below
# is checked against -- NOT a factor applied to the curve, which would assume the answer.
ZC_LIT = {"square": 1.18}


def lswt_scale(eps, wl):
    """Single common factor Z minimising ||Z*w_LSWT - eps||, and the mask it was fitted on.

    ⛔ LSWT AT S=1/2 IS NOT SUPPOSED TO MATCH IN MAGNITUDE, so an unscaled curve tests nothing that
    a reader can act on -- it sits uniformly below the data and every panel looks equally wrong.
    What LSWT can certify is the SHAPE of the branch, and that is only visible once the one free
    overall factor is removed. The paper does exactly this ("rescaled by a common factor").

    ⛔ THE FACTOR IS FITTED, NOT SET TO THE LITERATURE Z_c. Fitting leaves Z as a MEASUREMENT that
    can be scored against 1.18; hard-coding 1.18 would draw a curve that agrees by construction and
    could never disagree. Z is reported on the figure for that reason.

    Fitted only where LSWT is nonzero: at Gamma and M the magnon energy vanishes, those points
    carry no scale information, and including them would just add zeros to both sums.
    """
    m = np.isfinite(eps) & (wl > 1e-9)
    if m.sum() < 2:
        return None, m
    return float(np.sum(wl[m] * eps[m]) / np.sum(wl[m] ** 2)), m


def shape_rank(Spos, idx, tol=1e-3):
    """How many INDEPENDENT spectral shapes a set of q actually carries.

    ⛔ A DISPERSION CAN ONLY BE EXTRACTED WHERE S(q,w) CHANGES SHAPE WITH q. On a short cylinder it
    often does not, and then `argmax_w` returns the same number at every q -- a flat line that
    looks like a measured dispersion and is not one.

    MEASURED on the square 3x4 (Lx = 3 columns), along the whole Gamma->X segment:

        S(q,w) = (1 - cos q_x) * f(w)        to 3.8e-6

    which is FORCED, not accidental. With the source on the centre column the displacements are
    dx in {-1,0,+1}; inversion about the centre gives g(-1,t) = g(+1,t); and because S^z_tot
    commutes with H the column sums obey g(0,t) + 2 g(1,t) = 0 at EVERY t. Hence
    F(q,t) = g(0,t) + 2 cos(q_x) g(1,t) = g(0,t) (1 - cos q_x): one time-dependence, one scalar
    prefactor, zero q information. Lx columns leave Lx//2 independent channels after that
    constraint -- 1 at Lx=3, 2 at Lx=5, 18 at the paper's Lx=36.

    So this returns the numerical rank of the L2-NORMALISED spectra (normalised so that the
    amplitude prefactor, which does vary, cannot masquerade as shape variation). Rank 1 means the
    segment carries no dispersion at all and must not be drawn as one.

    ⛔ ROUNDOFF q MUST BE DROPPED BEFORE THE RANK IS TAKEN. At q = Gamma the weight is ~1e-17
    (S^z_tot commutes with H), and its NORMALISED shape is whatever the roundoff happened to be --
    a direction unrelated to every other q. Including it adds a spurious second singular value, so
    a segment that genuinely carries one shape reports rank 2 and the mask silently never fires.
    """
    if len(idx) < 2:
        return 0, np.array([])
    M = Spos[idx]
    nrm = np.linalg.norm(M, axis=1)
    keep = nrm > 1e-6 * np.linalg.norm(Spos, axis=1).max()
    if keep.sum() < 2:
        return 0, np.array([])
    M = M[keep] / nrm[keep][:, None]
    sv = np.linalg.svd(M, compute_uv=False)
    return int((sv > tol * sv[0]).sum()), sv


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
    # Number of COLUMNS along the open direction. It is what bounds how many independent spectral
    # shapes the path can carry (see `shape_rank`), so it is quoted on the figure rather than left
    # for the reader to infer from a filename.
    try:
        LX_COLS = int(geo.split("x")[0])
    except (ValueError, IndexError):
        LX_COLS = 0

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

    # ⛔ A SEGMENT WHOSE SPECTRA ALL HAVE THE SAME SHAPE CARRIES NO DISPERSION, AND DRAWING ONE
    # THERE INVENTS A MEASUREMENT. See `shape_rank`: at Lx=3 the Gamma->X segment is exactly
    # (1 - cos q_x) * f(w), so every point returns the same argmax and the curve comes out flat at
    # a value that has nothing to do with the magnon. Masked, and named on the figure, so the
    # absence is visible rather than silently plotted.
    labs = [meta[i + 1][2] for i in range(nq)]
    bounds = [i for i in range(nq) if labs[i]]
    flat_seg = []
    for a, b in zip(bounds[:-1], bounds[1:]):
        idx = list(range(a, b + 1))
        rk, _ = shape_rank(Spos, idx)
        if rk <= 1:
            nm_a = "Γ" if labs[a] == "G" else labs[a]
            nm_b = "Γ" if labs[b] == "G" else labs[b]
            flat_seg.append(f"{nm_a}→{nm_b}")
            eps19[idx] = np.nan
            w20[idx] = np.nan

    negfrac = np.abs(np.clip(S, None, 0.0)).sum() / max(np.abs(S).sum(), 1e-300)

    # ONE common factor, fitted once and used by BOTH LSWT panels, so the two cannot disagree.
    wl = lswt_square(qx, qy)
    Zfit, zmask = lswt_scale(eps19, wl)
    if Zfit is None:
        wl_s, zlab = wl, "LSWT (unscaled -- too few points to fit a scale)"
    else:
        wl_s, zlab = Zfit * wl, r"LSWT $\times\,Z$=%.2f" % Zfit

    # 5 panels, not 6: the bottom row is INSET BY ONE COLUMN so the two rows stay centred on each
    # other rather than leaving a hole where the deleted velocity panel used to be.
    fig = plt.figure(figsize=(15, 9))
    gs = fig.add_gridspec(2, 6, hspace=0.32, wspace=0.85)

    # (a) the path through the BZ
    ax = fig.add_subplot(gs[0, 0:2])
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
    ax = fig.add_subplot(gs[0, 2:4])
    im = ax.pcolormesh(dist, ws, Spos.T, shading="auto", cmap="magma")
    ax.plot(dist, eps19, "o-", ms=3, lw=1.2, color="cyan", label=r"$\epsilon(q)$ Eq. (19)")
    ax.plot(dist, wl_s, "--", lw=1.4, color="w", label=zlab)
    for x, lab in ticks:
        ax.axvline(x, color="w", lw=0.6, alpha=0.5)
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([r"$\Gamma$" if t[1] == "G" else t[1] for t in ticks])
    ax.set_ylabel(r"$\omega / J$")
    ax.set_title("b) $S(q,\\omega)$  [%s]" % (("bug" if "bug" in arms else list(arms)[0])))
    ax.legend(fontsize=8, loc="upper right")
    fig.colorbar(im, ax=ax, pad=0.02)

    # (c) dispersion vs LSWT, rescaled by the SAME fitted factor panel b) uses
    ax = fig.add_subplot(gs[0, 4:6])
    ax.plot(dist, eps19, "o-", ms=4, label=r"$\epsilon(q)$ Eq. (19)")
    ax.plot(dist, w20, "s--", ms=3, alpha=0.7, label=r"$\langle\omega\rangle(q)$ Eq. (20)")
    ax.plot(dist, wl_s, "-", color="k", lw=1.4, label=zlab)
    ax.set_xticks([t[0] for t in ticks])
    ax.set_xticklabels([r"$\Gamma$" if t[1] == "G" else t[1] for t in ticks])
    ax.set_ylabel(r"$\omega / J$")
    ax.set_title("c) magnon dispersion")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)
    # The fitted factor is the quantitative content of this panel, so it is REPORTED and scored
    # against the literature value rather than left for the reader to judge by eye. A residual is
    # given too: rescaling can only fix the magnitude, and if the SHAPE disagrees the residual says
    # so where an overlaid curve would just look approximately right.
    if Zfit is not None:
        zc = ZC_LIT.get("square")
        resid = float(np.sqrt(np.mean((eps19[zmask] - wl_s[zmask]) ** 2)))
        msg = (r"fitted $Z$ = %.3f  (%d $q$ points)" "\n"
               r"literature $Z_c \approx$ %.2f  $\rightarrow$ %+.0f%%" "\n"
               r"rms residual after scaling: %.3f $J$") % (
                   Zfit, int(zmask.sum()), zc, 100.0 * (Zfit / zc - 1.0), resid)
        ok = abs(Zfit / zc - 1.0) < 0.25
        ax.text(0.02, 0.03, msg, transform=ax.transAxes, fontsize=7.5, va="bottom",
                bbox=dict(boxstyle="round", fc="honeydew" if ok else "mistyrose",
                          ec="0.6" if ok else "crimson"))
    if flat_seg:
        ax.text(0.98, 0.97,
                "NO DISPERSION on " + ", ".join(flat_seg) + "\n"
                "$S(q,\\omega)$ has ONE shape there, so\n"
                "$\\arg\\max_\\omega$ is constant by construction\n"
                "($L_x{=}%d$ leaves $%d$ independent channel%s)" %
                (LX_COLS, max(LX_COLS // 2, 0), "" if LX_COLS // 2 == 1 else "s"),
                transform=ax.transAxes, ha="right", va="top", fontsize=7.5,
                bbox=dict(boxstyle="round", fc="mistyrose", ec="crimson"))

    # (d) frequency cuts at the high-symmetry momenta
    ax = fig.add_subplot(gs[1, 1:3])
    # ⛔ NORMALISED BY EACH CUT'S OWN MAXIMUM, as the paper does: "We divide the values by the
    # maximum intensity S_max to view all three points on the same axis." Without it the M cut
    # (weight 1.0) flattens every other curve onto the axis -- at 3x4 the Gamma cut is ~1e-17 and
    # simply cannot be seen beside it. S_max is kept in the legend so the normalisation never
    # hides HOW MUCH weight a momentum actually carries.
    # ⛔ AND A ROUNDOFF CUT IS NEVER NORMALISED. Dividing the Gamma cut (S_max ~ 1e-17, pure
    # roundoff because S^z_tot commutes with H) by its own maximum rescales noise to O(1) and puts
    # a full-height fake peak on the panel -- the normalisation MANUFACTURES the signal it is
    # supposed to reveal. Such cuts are named as carrying no weight instead of being drawn.
    floor = 1e-6 * Spos.max()
    dead = []
    for x, lab in ticks:
        i = int(np.argmin(np.abs(dist - x)))
        nm = r"$\Gamma$" if lab == "G" else lab
        smax = Spos[i].max()
        if smax <= floor:
            dead.append("%s ($S_{max}$=%.0e)" % (nm, smax))
            continue
        ax.plot(ws, Spos[i] / smax, lw=1.4, label="%s  ($S_{max}$=%.1e)" % (nm, smax))
    if dead:
        ax.text(0.5, 0.14, "no weight, not drawn:  " + ",  ".join(dead),
                transform=ax.transAxes, ha="center", va="center", fontsize=8,
                bbox=dict(boxstyle="round", fc="mistyrose", ec="crimson"))
    ax.set_xlabel(r"$\omega / J$"); ax.set_ylabel(r"$S(q,\omega)\,/\,S_{max}$")
    ax.set_title("d) cuts at high-symmetry $q$, each $/S_{max}$")
    ax.legend(fontsize=8); ax.grid(alpha=0.3)

    # (e) the integrator comparison -- this repository's question
    ax = fig.add_subplot(gs[1, 3:5])
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
        ax.set_title("e) cost on the same correlator")
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
