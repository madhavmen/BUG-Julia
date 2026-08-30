#!/usr/bin/env python3
"""Figs. 4 (square) and 5 (triangular) of arXiv:2209.00739, from `fig45_2209.jl`.

  python3 plot_fig45_2209.py <square|triangle> [LxxLy]

Panels follow the paper: the BZ path, the spectrum along it, the STATIC structure factor (their
Fig. 5 a, which is where the 120-degree order shows as a peak at K), the dispersion against linear
spin-wave theory, frequency cuts at the high-symmetry momenta, and -- this repository's question
rather than theirs -- the cost of the three integrators on the same correlator.

eps(q) is Eq. (19), argmax_w S(q,w). <w>(q) is Eq. (20), the first moment.

FOUR RULES THIS FILE ENFORCES, EACH BECAUSE THE FIRST VERSION OF THE SQUARE PLOTTER BROKE IT:

1. eps(q) AND <w>(q) ARE MASKED WHERE THERE IS NO WEIGHT. At Gamma the total weight is ~1e-18
   (S^z_tot commutes with H, so it must vanish) and `argmax` over roundoff returns a uniformly
   distributed frequency. Drawing it produces a confident-looking dispersion made of noise.
   Eq. (20) needs the same mask for the same reason -- its denominator divides ~1e-18 by ~1e-18.

2. A VELOCITY FIT IS REFUSED WHEN THE GRID CANNOT RESOLVE A SLOPE. With a handful of allowed
   momenta the fitted v came out 0.00, which is not a small velocity, it is no measurement. The
   panel says so instead of drawing a flat line through it.

3. THE COST PANEL SHOWS WALL CLOCK AS THE CLAIM. `matvec` counts `apply_one_site` and is blind to
   `cbe_expand`'s sketch work, so it UNDERSTATES every BUG row; it is drawn beside wall clock with
   that caveat written into the panel, never alone.

4. POINTS THAT ARE NOT EXACT QUANTUM NUMBERS ARE MARKED. Only q with q.T in 2*pi*Z are momenta of
   the cylinder, where T is the wrapping's identification vector. The triangle uses the paper's
   XC wrapping (Sec. IIB), T = (0, sqrt3*C/2), so the condition is Eq. (8), q_y = 4*pi*n/(sqrt3*C),
   and q_x is free -- which is why K = (4pi/3, 0) and the whole K -> Gamma segment are exact at
   every even C. See `lattice2d.jl` and the 30-check `xc_gate.jl`.
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

sys.stdout.reconfigure(errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")

SQRT3 = np.sqrt(3.0)

# Weight below this fraction of the path-maximum is roundoff, not signal (rule 1).
WEIGHT_FLOOR = 1e-6


def lswt_square(qx, qy, J=1.0):
    """w_q = 2 J sqrt(1 - gamma^2), gamma = (cos qx + cos qy)/2  (z=4, S=1/2).

    Drawn UNSCALED. The paper rescales SWT "by a common factor analogous to what is done in
    Ref. 127"; leaving it raw keeps the comparison honest about what was and was not adjusted.
    """
    g = (np.cos(qx) + np.cos(qy)) / 2.0
    return 2.0 * J * np.sqrt(np.maximum(0.0, 1.0 - g ** 2))


def lswt_triangular(qx, qy, J=1.0):
    """w_q = 3 J S sqrt((1 - g)(1 + 2g)),  g = (1/3) sum_delta cos(q.delta),  S = 1/2.

    The three NN vectors of the 120-degree lattice are a1, a2 and a1+a2 (Chernyshev &
    Zhitomirsky, PRB 79 144416 -- the paper's Ref. 150). Zeros at Gamma AND at K, which is the
    signature of the 120-degree ordered state: K is the ordering wavevector, so the magnon is
    gapless there too.
    """
    a1, a2 = np.array([1.0, 0.0]), np.array([-0.5, SQRT3 / 2])
    q = np.stack([np.atleast_1d(qx), np.atleast_1d(qy)], axis=-1)
    g = sum(np.cos(q @ d) for d in (a1, a2, a1 + a2)) / 3.0
    return 1.5 * J * np.sqrt(np.maximum(0.0, (1.0 - g) * (1.0 + 2.0 * g)))


LSWT = {"square": lswt_square, "triangle": lswt_triangular}


def load_sqw(path):
    q, lab, dist, allowed, sq, w, S = [], [], [], [], [], [], []
    with open(path, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            q.append((float(r["qx"]), float(r["qy"])))
            lab.append(r["label"])
            dist.append(float(r["dist"]))
            allowed.append(int(r["allowed"]))
            sq.append(float(r["Sq"]))
            w.append(float(r["w"]))
            S.append(float(r["S"]))
    nq = len(set(dist))
    ws = sorted(set(w))
    nw = len(ws)
    grid = np.array(S).reshape(nq, nw)
    idx = list(range(0, len(dist), nw))
    return dict(qs=[q[i] for i in idx], labels=[lab[i] for i in idx],
                dists=[dist[i] for i in idx], allowed=[allowed[i] for i in idx],
                Sq=[sq[i] for i in idx], ws=np.array(ws), S=grid)


def masked_dispersion(d):
    """eps(q) [Eq. 19] and <w>(q) [Eq. 20], both NaN where the weight is roundoff (rule 1)."""
    S = np.clip(d["S"], 0.0, None)
    tot = S.sum(axis=1)
    live = tot > WEIGHT_FLOOR * max(tot.max(), 1e-300)
    eps = np.where(live, d["ws"][np.argmax(S, axis=1)], np.nan)
    with np.errstate(invalid="ignore", divide="ignore"):
        mom = np.where(live, (S * d["ws"]).sum(axis=1) / np.where(tot > 0, tot, np.nan), np.nan)
    return eps, mom, live


def ticks(d):
    pos = [(x, l) for x, l in zip(d["dists"], d["labels"]) if l]
    return [p[0] for p in pos], [p[1] for p in pos]


# The paper's velocity panel (Fig. 4 c, Fig. 5 c): fit eps(q) to a line near the gapless point,
# along the direction with the MOST allowed momenta.
#   square    near M = (pi,pi), on the path towards X = (0,pi)   ("we take the path from M -> X
#             as we have the most allowed q values in that direction")
#   triangle  near K = (4pi/3,0), on the path towards Gamma      ("we take the direction from
#             K -> Gamma, as this has the most allowed q values in the linear region of SWT")
# ⛔ THE ANCHOR ITSELF IS EXCLUDED. "We omit the value of eps(q = M) in the fits" -- it carries the
# largest finite-size effect, and on a finite system the spectrum is never gapless there.
VELOCITY = {"square": ("M", "before"), "triangle": ("K", "after")}


def velocity_fit(d, eps, lat, npts=6):
    """-> (x, y, anchor_label, {'free': (v, delta), 'zero': (v, 0)}) or None."""
    spec = VELOCITY.get(lat)
    if spec is None:
        return None
    anchor, side = spec
    idx = [i for i, l in enumerate(d["labels"]) if l == anchor]
    if not idx:
        return None
    i0 = idx[0]
    take = range(i0 - 1, max(i0 - 1 - npts, -1), -1) if side == "before" \
        else range(i0 + 1, min(i0 + 1 + npts, len(d["labels"])))
    q0 = np.array(d["qs"][i0])
    pts = [(float(np.hypot(*(np.array(d["qs"][i]) - q0))), float(eps[i]))
           for i in take if np.isfinite(eps[i])]
    if len(pts) < 3:
        return None
    x = np.array([p[0] for p in pts]); y = np.array([p[1] for p in pts])
    v_free, delta = np.polyfit(x, y, 1)                      # y = v*x + delta
    v_zero = float(x @ y / (x @ x))                          # y = v*x, delta forced to 0
    return x, y, anchor, {"free": (float(v_free), float(delta)), "zero": (v_zero, 0.0)}


# ── THE SPIN-WAVE CHECK ───────────────────────────────────────────────────────────────────────
#
# "if it does not match SWT then something is wrong" is right about the SHAPE and wrong about the
# SCALE, and conflating the two would flag correct physics as a bug.
#
# ⛔ LINEAR SWT AT S = 1/2 IS NOT SUPPOSED TO MATCH IN MAGNITUDE. Quantum corrections renormalise
# the whole dispersion by a roughly q-independent factor: UPWARD on the square lattice
# (Z_c ~ 1.18, which is why the paper says the SWT curve is "adjusted by a common factor
# analogous to what is done in Ref. 127") and DOWNWARD on the frustrated triangular lattice, where
# LSWT additionally MISSES the roton-like minimum near M entirely -- a real feature the paper
# reports seeing and attributes to physics beyond SWT.
#
# So the test is: fit ONE scale factor Z by least squares, then ask whether the SHAPE agrees after
# scaling. A shape that tracks LSWT with a sane Z is a pass. What is genuinely a bug:
#   * the dispersion not vanishing where LSWT vanishes (Gamma on both; ALSO K on the triangle --
#     the 120-degree ordering wavevector is gapless, and a gap there means the state is not the
#     120-degree ordered one)
#   * Z far from 1 (a factor of 2, a factor of 3): wrong J convention, wrong Eq.-(5) factor of 3,
#     or a time axis off by a constant
#   * correlation with LSWT near zero: the peak-tracking is following noise, not a magnon
SWT_ZERO_POINTS = {"square": ["Γ"], "triangle": ["Γ", "K"]}


def swt_report(d, eps, lswt, lat):
    """-> dict with the fitted scale, the post-scale shape agreement, and the gap at each node."""
    m = np.isfinite(eps) & np.isfinite(lswt) & (lswt > 1e-9)
    if m.sum() < 4:
        return None
    Z = float((eps[m] @ lswt[m]) / (lswt[m] @ lswt[m]))       # least squares eps ~ Z * lswt
    resid = float(np.sqrt(np.mean((eps[m] - Z * lswt[m]) ** 2)))
    scale = float(np.mean(np.abs(eps[m])))
    corr = float(np.corrcoef(eps[m], lswt[m])[0, 1]) if m.sum() > 2 else float("nan")
    gaps = {}
    for lbl in SWT_ZERO_POINTS.get(lat, []):
        idx = [i for i, l in enumerate(d["labels"]) if l == lbl]
        # Gamma carries no S^z weight at all (S^z_tot commutes with H), so eps is masked there by
        # construction; report it as such rather than as a measured gap of zero.
        gaps[lbl] = float(eps[idx[0]]) if idx and np.isfinite(eps[idx[0]]) else float("nan")
    return dict(Z=Z, resid=resid, rel=resid / max(scale, 1e-12), corr=corr, gaps=gaps,
                n=int(m.sum()))


def main():
    lat = sys.argv[1] if len(sys.argv) > 1 else "square"
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    pat = os.path.join(RES, f"fig45_Sqw_{lat}_{tag or '*'}_*.csv")
    files = sorted(glob.glob(pat))
    if not files:
        print("no data:", pat)
        return
    arms = {}
    for f in files:
        arm = os.path.basename(f)[:-4].split("_")[-1]
        arms[arm] = load_sqw(f)
    if tag is None:
        # basename = fig45_Sqw_<lat>_<Lx>x<Ly>_D<cap>_<arm>.csv -- the tag is everything between
        # the lattice name and the trailing arm, so it must be rebuilt, not indexed at a fixed
        # position (adding the _D<cap> field is exactly what breaks a hard-coded index).
        parts = os.path.basename(files[0])[:-4].split("_")
        tag = "_".join(parts[3:-1])

    cost = []
    cf = os.path.join(RES, f"fig45_cost_{lat}_{tag}.csv")
    if os.path.exists(cf):
        with open(cf, encoding="utf-8") as f:
            cost = list(csv.DictReader(f))

    main_arm = "bug" if "bug" in arms else sorted(arms)[0]
    d = arms[main_arm]
    xt, xl = ticks(d)
    eps, mom, live = masked_dispersion(d)
    lswt = LSWT[lat](np.array([q[0] for q in d["qs"]]), np.array([q[1] for q in d["qs"]]))

    fig, ax = plt.subplots(2, 4, figsize=(23.0, 10.2))

    # a) the path, with the allowed momenta marked
    a = ax[0][0]
    qs = np.array(d["qs"])
    al = np.array(d["allowed"], dtype=bool)
    a.plot(qs[:, 0], qs[:, 1], "-", color="0.6", lw=1, zorder=1)
    a.scatter(qs[al, 0], qs[al, 1], s=34, color="#1f77b4", zorder=3,
              label=r"$q$ allowed (exact)")
    a.scatter(qs[~al, 0], qs[~al, 1], s=28, facecolors="none", edgecolors="#d62728", zorder=3,
              label=r"$q$ interpolated")
    for x, l in zip(d["dists"], d["labels"]):
        if l:
            i = d["dists"].index(x)
            a.annotate(l, d["qs"][i], fontsize=13, fontweight="bold",
                       xytext=(5, 5), textcoords="offset points")
    a.set_xlabel(r"$q_x$"); a.set_ylabel(r"$q_y$")
    a.set_title(f"a) path in the Brillouin zone  [{lat}]")
    a.legend(fontsize=8); a.set_aspect("equal"); a.grid(alpha=0.25)

    # b) the spectrum
    a = ax[0][1]
    im = a.pcolormesh(d["dists"], d["ws"], np.clip(d["S"], 0, None).T,
                      shading="auto", cmap="magma")
    fig.colorbar(im, ax=a)
    a.plot(d["dists"], eps, "-o", ms=3, color="cyan", lw=1.2, label=r"$\epsilon(q)$ Eq. (19)")
    a.plot(d["dists"], lswt, "--", color="white", lw=1.4, label="LSWT (unscaled)")
    a.set_xticks(xt); a.set_xticklabels(xl)
    for x in xt:
        a.axvline(x, color="w", lw=0.6, alpha=0.5)
    a.set_ylabel(r"$\omega/J$"); a.set_ylim(0, d["ws"].max())
    a.set_title(f"b) $S(q,\\omega)$  [{main_arm}]")
    a.legend(fontsize=8, loc="upper right")

    # c) the STATIC structure factor -- the paper's Fig. 5 a)
    a = ax[0][2]
    a.plot(d["dists"], d["Sq"], "-o", ms=4, color="#2ca02c")
    for x, ok in zip(d["dists"], d["allowed"]):
        if not ok:
            a.axvspan(x, x, color="#d62728", alpha=0.0)
    a.scatter([x for x, ok in zip(d["dists"], d["allowed"]) if ok],
              [s for s, ok in zip(d["Sq"], d["allowed"]) if ok],
              s=48, color="#2ca02c", zorder=4, label="exact momenta")
    a.set_xticks(xt); a.set_xticklabels(xl)
    a.set_ylabel(r"$S(q)$  (equal time)")
    a.set_title("c) static structure factor\n"
                "the ordering peak lives here", fontsize=11)
    a.grid(alpha=0.3); a.legend(fontsize=8)
    if lat == "triangle":
        nex = sum(1 for ok in d["allowed"] if ok)
        a.text(0.5, 0.955, f"XC wrapping: {nex}/{len(d['allowed'])} path points are exact "
                           f"momenta.\n$K=(4\\pi/3,0)$ has $q_y=0$, so it is exact at every "
                           f"even $C$.",
               transform=a.transAxes, ha="center", va="top", fontsize=8,
               bbox=dict(fc="#eef7ee", ec="#2ca02c"))

    # d) dispersion against LSWT
    a = ax[0][3]
    a.plot(d["dists"], eps, "-o", ms=4, label=r"$\epsilon(q)$ Eq. (19)")
    a.plot(d["dists"], mom, "--s", ms=4, alpha=0.8, label=r"$\langle\omega\rangle(q)$ Eq. (20)")
    a.plot(d["dists"], lswt, "-", color="k", lw=1.6, label="LSWT (unscaled)")
    sw = swt_report(d, eps, lswt, lat)
    if sw:
        a.plot(d["dists"], sw["Z"] * lswt, ":", color="#d62728", lw=1.8,
               label=rf"LSWT $\times Z$, $Z={sw['Z']:.3f}$")
    a.set_xticks(xt); a.set_xticklabels(xl)
    a.set_ylabel(r"$\omega/J$")
    a.set_title("d) magnon dispersion")
    a.legend(fontsize=8); a.grid(alpha=0.3)
    if sw:
        gp = "  ".join(f"$\\epsilon({k})$={v:.3f}" if np.isfinite(v) else f"$\\epsilon({k})$=n/a"
                       for k, v in sw["gaps"].items())
        verdict = ("shape OK" if sw["corr"] > 0.9 and sw["rel"] < 0.25 else
                   "⚠ SHAPE DISAGREES")
        a.text(0.02, 0.97,
               f"vs LSWT: $Z$={sw['Z']:.3f}, corr={sw['corr']:.3f}, "
               f"rel.resid={sw['rel']:.3f}\n{gp}   [{verdict}]\n"
               f"$Z\\neq1$ is EXPECTED ($Z_c$>1 square, <1 triangle)",
               transform=a.transAxes, va="top", fontsize=7,
               bbox=dict(fc="#eef3fb" if "OK" in verdict else "#ffecec", ec="#999"))
    if (~live).any():
        a.text(0.02, 0.02, f"{int((~live).sum())} of {len(live)} q masked:\n"
                           "spectral weight below 1e-6 of peak",
               transform=a.transAxes, fontsize=7.5, va="bottom",
               bbox=dict(fc="#fffbe6", ec="#999"))

    # e) the magnon velocity -- the paper's Fig. 4 c) / Fig. 5 c)
    a = ax[1][0]
    vf = velocity_fit(d, eps, lat)
    if vf is None:
        a.axis("off")
        a.text(0.5, 0.5, "velocity fit needs >= 3 finite\n$\\epsilon(q)$ points near the anchor",
               ha="center", va="center", fontsize=9)
    else:
        x, y, anchor, fits = vf
        a.scatter(x, y, s=48, color="#1f77b4", zorder=4, label=r"$\epsilon(q)$ Eq. (19)")
        xs = np.linspace(0, x.max() * 1.05, 50)
        v, dl = fits["free"]
        a.plot(xs, v * xs + dl, "-", color="#1f77b4", lw=1.6,
               label=rf"$vq+\Delta$:  $v={v:.3f}$, $\Delta={dl:.3f}$")
        vz = fits["zero"][0]
        a.plot(xs, vz * xs, "-", color="#d62728", lw=1.6,
               label=rf"$\Delta\equiv 0$:  $v={vz:.3f}$")
        a.set_xlabel(rf"$|q-{anchor}|$"); a.set_ylabel(r"$\omega/J$")
        a.set_title(f"e) magnon velocity near ${anchor}$\n"
                    f"(${anchor}$ itself omitted, as in the paper)", fontsize=11)
        a.legend(fontsize=8); a.grid(alpha=0.3)
        # ⛔ NO QMC REFERENCE LINE IS DRAWN. The paper compares the square-lattice velocity against
        # QMC (its Fig. 4 c, green) but quotes the number only as a line in the figure; the value
        # lives in its Ref. 148 (Sen, Suwa & Sandvik, PRB 92, 195145). Drawing a remembered
        # constant as a reference line is exactly the failure the external-anchor rule exists to
        # prevent -- add it here once it has been read out of that paper, not before.

    # f) frequency cuts at the labelled momenta
    a = ax[1][1]
    for x, l in zip(d["dists"], d["labels"]):
        if not l:
            continue
        i = d["dists"].index(x)
        cut = np.clip(d["S"][i], 0, None)
        a.plot(d["ws"], cut, lw=1.4, label=f"{l}  ($S_{{max}}$={cut.max():.1e})")
    a.set_xlabel(r"$\omega/J$"); a.set_ylabel(r"$S(q,\omega)$")
    a.set_title("f) cuts at high-symmetry $q$")
    a.legend(fontsize=8); a.grid(alpha=0.3)

    # g) the integrator comparison
    a = ax[1][2]
    if cost:
        names = [c["arm"] for c in cost]
        secs = np.array([float(c["seconds"]) for c in cost])
        mvs = np.array([float(c["matvec"]) for c in cost])
        ref = next((i for i, n in enumerate(names) if n == "tdvp2"), 0)
        x = np.arange(len(names))
        a.bar(x - 0.2, mvs / mvs[ref], 0.4, label="operator applications", color="0.55")
        a.bar(x + 0.2, secs / secs[ref], 0.4, label="wall clock", color="#2ca02c")
        for i, (m, s) in enumerate(zip(mvs, secs)):
            a.text(i - 0.2, m / mvs[ref], f"{int(m)}", ha="center", va="bottom", fontsize=7)
            a.text(i + 0.2, s / secs[ref], f"{s:.0f}s", ha="center", va="bottom", fontsize=7)
        a.set_xticks(x); a.set_xticklabels(names)
        a.axhline(1.0, ls=":", color="k")
        a.set_ylabel(f"relative to {names[ref]}")
        a.set_title("g) cost on the same correlator")
        a.legend(fontsize=8)
        a.text(0.02, 0.97, "matvec UNDERCOUNTS bug:\ncbe_expand work is not in\nkrylov_dims "
                           "-> quote wall clock",
               transform=a.transAxes, va="top", fontsize=7.5,
               bbox=dict(fc="#fffbe6", ec="#999"))
        # Agreement between arms, on the spectrum itself.
        if len(arms) > 1:
            base = arms.get("tdvp2", arms[main_arm])["S"]
            txt = "max|dS| vs tdvp2:\n" + "\n".join(
                f"{k} {np.abs(v['S'] - base).max():.1e}" for k, v in sorted(arms.items())
                if k != "tdvp2")
            a.text(0.02, 0.55, txt, transform=a.transAxes, va="top", fontsize=7.5,
                   bbox=dict(fc="#eefaf0", ec="#999"))
    else:
        a.axis("off")
        a.text(0.5, 0.5, "no cost file", ha="center")

    neg = float((d["S"] < 0).sum()) / d["S"].size
    fig.suptitle(f"arXiv:2209.00739 Fig.{'4' if lat == 'square' else '5'} -- "
                 f"{lat} lattice $S(q,\\omega)$   [{tag}; negative-weight fraction {neg:.3e}]",
                 fontsize=14)
    fig.tight_layout(rect=(0, 0, 1, 0.96))
    out = os.path.join(RES, f"fig45_{lat}_{tag}.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)
    print(f"negative-weight fraction {neg:.3e}")
    if sw:
        print(f"SWT check [{lat} {tag}]: Z={sw['Z']:.4f}  corr={sw['corr']:.4f}  "
              f"rel.resid={sw['rel']:.4f}  over {sw['n']} q-points")
        for k, v in sw["gaps"].items():
            print(f"  eps({k}) = {v:.4f}" if np.isfinite(v)
                  else f"  eps({k}) = n/a (masked: no spectral weight there)")
        ok = sw["corr"] > 0.9 and sw["rel"] < 0.25
        print("  VERDICT: shape tracks LSWT" if ok else
              "  ⚠ VERDICT: shape does NOT track LSWT -- investigate before quoting the spectrum")
        print("  (Z != 1 is expected: quantum corrections renormalise LSWT up on the square, "
              "down on the triangle.)")


if __name__ == "__main__":
    main()
