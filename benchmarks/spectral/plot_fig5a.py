#!/usr/bin/env python3
"""Fig. 5 a) of arXiv:2209.00739 -- the STATIC structure factor, with the zone and its
high-symmetry points drawn.

  python3 plot_fig5a.py <triangle|square> [LxxLy]

Two panels:
  a)  S(q) over the plane, with the Brillouin-zone boundary, the high-symmetry points, the LINES
      of allowed momenta, and the measured peak marked.
  b)  S(q) along the Gamma -> M -> K -> Gamma path (Gamma -> X -> M -> Gamma on the square), both
      the translation-averaged double sum and the centre-referenced form the paper's G(x,t=0)
      gives.

WHAT THE FIGURE IS FOR. The triangular Heisenberg ground state is 120-degree ordered, so S(q)
MUST peak at K = (4pi/3, 0); the square's Neel state must peak at M = (pi,pi). A peak anywhere
else means the geometry is wrong -- sheared positions, a mis-ordered snake, the wrong wrap
direction -- and every dynamical figure built on the same geometry would be wrong in the same way
while still looking like a spectrum.

⛔ THE ALLOWED-MOMENTUM LINES ARE THE POINT OF PANEL a). A cylinder identifies r with r + T, so
only q with q.T in 2*pi*Z are exact quantum numbers -- a family of parallel lines with normal T,
NOT a square grid. The triangle here uses the paper's XC wrapping (Sec. IIB), T = (0, sqrt3*C/2),
so the lines are HORIZONTAL, q_y = 4*pi*n/(sqrt3*C) -- the paper's Eq. (8) -- and q_x is free.
That is why K = (4pi/3, 0) is exact at every even C: it sits on the n = 0 line, and so does the
entire K -> Gamma segment. Under YC the same family is TILTED at 120 degrees and K needs 3 | C.

⚠ THREE KINDS OF POINT, AND THE FIGURE MUST NOT MIX THEM. Blue lines are exact momenta. Orange
lines are their C6 images: the paper restores the six-fold symmetry (Eq. 10, "O(Rq) = O(q)"), so
S(q) there is a value, not an interpolation -- Fig. 2's caption calls these out explicitly.
Everything else is smooth continuation off the allowed set and is NOT a spectrum of the cylinder.
Panel b) marks each path point with which of the three it is.
"""
import csv
import os
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon

sys.stdout.reconfigure(errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
SQRT3 = np.sqrt(3.0)

# THE IDENTIFICATION VECTOR T, per unit circumference. Everything the wrappings disagree about
# follows from this -- keep it in step with `triangular_wrap_vector` in
# src/RSVDCBEBondUpdate/models/triangular_cylinder.jl.
#   square       T = C*(0,1)
#   triangle XC  T = C*a1 - (C/2)*a2 = (0, sqrt3*C/2)   <- the paper's choice, Sec. IIB
#   triangle YC  T = C*(-1/2, sqrt3/2)
T_PER_C = {
    "square": np.array([0.0, 1.0]),
    "triangle": np.array([0.0, SQRT3 / 2]),
    "triangle_yc": np.array([-0.5, SQRT3 / 2]),
}


def zone(lat):
    """(boundary vertices, {label: [images]}) for the first Brillouin zone."""
    if lat.startswith("triangle"):
        K = 4 * np.pi / 3
        M = 2 * np.pi / SQRT3
        hexv = [(K * np.cos(t), K * np.sin(t)) for t in np.arange(6) * (np.pi / 3)]
        return hexv, {
            "K": [(K * np.cos(t), K * np.sin(t)) for t in np.arange(6) * (np.pi / 3)],
            "M": [(M * np.cos(t), M * np.sin(t))
                  for t in np.pi / 6 + np.arange(6) * (np.pi / 3)],
            "Γ": [(0.0, 0.0)],
        }
    sq = [(np.pi, np.pi), (-np.pi, np.pi), (-np.pi, -np.pi), (np.pi, -np.pi)]
    return sq, {
        "M": [(np.pi, np.pi), (-np.pi, np.pi), (np.pi, -np.pi), (-np.pi, -np.pi)],
        "X": [(0.0, np.pi), (np.pi, 0.0), (0.0, -np.pi), (-np.pi, 0.0)],
        "Γ": [(0.0, 0.0)],
    }


def allowed_lines(lat, Ly, qmax, rot=0.0):
    """The lines q.T = 2*pi*n, as endpoint pairs clipped to the plotted square.

    `rot` rotates the whole family, which is how the C6 images (the paper's orange set, Eq. 10)
    are drawn: S(Rq) = S(q), so a rotation of an exact line is a line of equally valid values.
    """
    T1 = T_PER_C[lat]
    c, s = np.cos(rot), np.sin(rot)
    T1 = np.array([c * T1[0] - s * T1[1], s * T1[0] + c * T1[1]])
    nrm = T1 / np.linalg.norm(T1)                 # unit normal of the family
    spacing = 2 * np.pi / (Ly * np.linalg.norm(T1))
    t = np.array([-nrm[1], nrm[0]])               # along the line
    out = []
    nmax = int(np.ceil(2 * qmax / spacing)) + 1
    for n in range(-nmax, nmax + 1):
        p = n * spacing * nrm
        out.append((p - 3 * qmax * t, p + 3 * qmax * t))
    return out


def main():
    lat = sys.argv[1] if len(sys.argv) > 1 else "triangle"
    tag = sys.argv[2] if len(sys.argv) > 2 else None
    if tag is None:
        cands = [f for f in os.listdir(RES)
                 if f.startswith(f"fig5a_{lat}_") and f.endswith(".csv")]
        if not cands:
            print("no fig5a data for", lat)
            return
        tag = cands[0][len(f"fig5a_{lat}_"):-4]
    Lx, Ly = (int(v) for v in tag.split("x"))

    gp = os.path.join(RES, f"fig5a_{lat}_{tag}.csv")
    pp = os.path.join(RES, f"fig5a_path_{lat}_{tag}.csv")
    if not os.path.exists(gp):
        print("missing", gp)
        return

    qx, qy, Sf, Sc = [], [], [], []
    with open(gp, encoding="utf-8") as f:
        for r in csv.DictReader(f):
            qx.append(float(r["qx"])); qy.append(float(r["qy"]))
            Sf.append(float(r["S_full"])); Sc.append(float(r["S_centred"]))
    ux, uy = np.unique(qx), np.unique(qy)
    G = np.array(Sf).reshape(len(ux), len(uy)).T          # rows = qy, cols = qx
    qmax = float(np.max(ux))

    path = []
    if os.path.exists(pp):
        with open(pp, encoding="utf-8") as f:
            path = list(csv.DictReader(f))

    fig, ax = plt.subplots(1, 2, figsize=(14.4, 6.3),
                           gridspec_kw={"width_ratios": [1.05, 1]})

    # ── a) the zone map ─────────────────────────────────────────────────────────────────────
    a = ax[0]
    im = a.pcolormesh(ux, uy, G, shading="auto", cmap="magma")
    fig.colorbar(im, ax=a, label=r"$S(q)$")

    # C6 images first (the paper's orange set, Eq. 10) so the exact family draws on top.
    nrot = 6 if lat.startswith("triangle") else 4
    for k in range(1, nrot):
        for p, q in allowed_lines(lat, Ly, qmax, rot=2 * np.pi * k / nrot):
            a.plot([p[0], q[0]], [p[1], q[1]], color="#ff9f0a", lw=0.6, alpha=0.35, zorder=2)
    a.plot([], [], color="#ff9f0a", lw=1.2, alpha=0.7,
           label=rf"$C_{{{nrot}}}$ images (Eq. 10)")
    for p, q in allowed_lines(lat, Ly, qmax):
        a.plot([p[0], q[0]], [p[1], q[1]], color="#5ac8fa", lw=1.0, alpha=0.9, zorder=3)
    a.plot([], [], color="#5ac8fa", lw=1.4, label=rf"exact $q$  ($C={Ly}$)")

    verts, pts = zone(lat)
    a.add_patch(Polygon(verts, closed=True, fill=False, ec="white", lw=1.8, zorder=3))
    marks = {"Γ": ("o", "white"), "K": ("*", "#39ff14"), "M": ("s", "#ff2d55"),
             "X": ("^", "#ffd60a")}
    for lbl, images in pts.items():
        mk, col = marks[lbl]
        xs = [p[0] for p in images]; ys = [p[1] for p in images]
        a.scatter(xs, ys, marker=mk, s=190 if mk == "*" else 70, c=col,
                  edgecolors="black", linewidths=0.6, zorder=5, label=lbl)
        for x, y in images:
            a.annotate(lbl, (x, y), color="white", fontsize=11, fontweight="bold",
                       xytext=(6, 5), textcoords="offset points", zorder=6)

    iy, ix = np.unravel_index(np.argmax(G), G.shape)
    a.scatter([ux[ix]], [uy[iy]], marker="x", s=150, c="cyan", linewidths=2.4, zorder=7,
              label=f"peak ({ux[ix]:.2f}, {uy[iy]:.2f})")
    a.set_xlabel(r"$q_x$"); a.set_ylabel(r"$q_y$")
    # ⛔ CLAMP TO THE DATA. The allowed-momentum lines are drawn 3*qmax long so they always cross
    # the frame, and letting matplotlib autoscale to THEM shrank the S(q) map to a stamp in the
    # middle of a mostly-empty +-20 axis.
    a.set_xlim(ux.min(), ux.max()); a.set_ylim(uy.min(), uy.max())
    a.set_aspect("equal")
    wrapname = {"square": "square", "triangle": "XC", "triangle_yc": "YC"}[lat]
    a.set_title(f"a) static structure factor $S(q)$\n{lat} cylinder, {wrapname} wrapping  "
                f"$L={Lx}$, $C={Ly}$, $N={Lx*Ly}$")
    a.legend(loc="upper right", fontsize=8, framealpha=0.85)

    # ── b) the path cut ─────────────────────────────────────────────────────────────────────
    b = ax[1]
    if path:
        d = [float(r["dist"]) for r in path]
        sf = [float(r["S_full"]) for r in path]
        sc = [float(r["S_centred"]) for r in path]
        # `status` falls back to `allowed` for CSVs written before the XC switch.
        st = [r.get("status") or ("exact" if int(r["allowed"]) else "interp") for r in path]
        b.plot(d, sf, "-", lw=1.6, color="#2ca02c", label=r"$S(q)$  full double sum")
        b.plot(d, sc, "--", lw=1.4, color="#9467bd",
               label=r"$S(q)$  centred (= $G(x,t{=}0)$)")
        nrot_b = 6 if lat.startswith("triangle") else 4
        for kind, mk, col, sz, lab in (("exact", "o", "#2ca02c", 54, "exact momenta"),
                                       ("image", "D", "#ff9f0a", 30,
                                        f"$C_{{{nrot_b}}}$ image (Eq. 10)")):
            xs = [x for x, k in zip(d, st) if k == kind]
            ys = [s for s, k in zip(sf, st) if k == kind]
            if xs:
                b.scatter(xs, ys, marker=mk, s=sz, color=col, zorder=4,
                          edgecolors="black", linewidths=0.4, label=lab)
        nex, nim = st.count("exact"), st.count("image")
        b.text(0.02, 0.96, f"{nex} exact · {nim} $C_{{{nrot_b}}}$ image · {len(st)-nex-nim} interpolated",
               transform=b.transAxes, fontsize=8, va="top", color="0.35")
        ticks = [(float(r["dist"]), r["label"]) for r in path if r["label"]]
        b.set_xticks([t[0] for t in ticks])
        b.set_xticklabels([t[1] for t in ticks], fontsize=12, fontweight="bold")
        for t, _ in ticks:
            b.axvline(t, color="0.75", lw=0.9, zorder=1)
        pk = max(range(len(sf)), key=lambda i: sf[i])
        b.annotate(f"peak at {path[pk]['label'] or f'{d[pk]:.2f}'}", (d[pk], sf[pk]),
                   xytext=(8, -14), textcoords="offset points", fontsize=9,
                   arrowprops=dict(arrowstyle="->", lw=1))
    b.set_ylabel(r"$S(q)$"); b.grid(alpha=0.3)
    b.set_title("b) along the high-symmetry path\n"
                "the ordering peak is the pass/fail")
    b.legend(fontsize=8)

    fig.suptitle(f"arXiv:2209.00739 Fig. 5a -- static structure factor, {lat} "
                 f"cylinder $L={Lx}$, $C={Ly}$ ({wrapname})", fontsize=13)
    fig.tight_layout(rect=(0, 0, 1, 0.94))
    out = os.path.join(RES, f"fig5a_{lat}_{tag}.png")
    fig.savefig(out, dpi=150)
    print("wrote", out)
    print(f"grid peak at ({ux[ix]:.4f}, {uy[iy]:.4f})  S = {G[iy, ix]:.6f}")


if __name__ == "__main__":
    main()
