"""Plot S(k, omega) for the Haldane-Shastry ring against the two-spinon continuum.

    python3 benchmarks/plot_hs_skw.py [results/hs_skw_*.csv] [outdir]

WHAT THIS FIGURE IS FOR, and what it is NOT for. The reference is the behaviour reported in
Li et al. (arXiv:2208.10972) Fig. 3, which is a THERMODYNAMIC-LIMIT object; this is a finite ring
whose spectrum is a comb of delta peaks smeared by the finite time record. So the claim being
checked is structural, not pointwise:

  * the weight lies INSIDE the two-spinon band  w_lower(k) <= w <= w_upper(k),
  * it piles up at the LOWER edge (the square-root divergence),
  * it vanishes above the upper edge and below w = 0.

The band is drawn on top so those three can be read off directly. Anyone reading a pointwise
value off this plot and comparing it to the paper is making a category error, which is why the
band is drawn and no numerical residual is quoted.
"""

import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
PAT = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results", "hs_skw_*.csv")
OUTDIR = sys.argv[2] if len(sys.argv) > 2 else os.path.join(HERE, "results")


def load(path):
    ks, ws = [], []
    grid = defaultdict(dict)
    band = {}
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            k, w = float(r["k"]), float(r["omega"])
            grid[k][w] = float(r["S"])
            band[k] = (float(r["w_lower"]), float(r["w_upper"]))
    ks = sorted(grid)
    ws = sorted(grid[ks[0]])
    S = np.array([[grid[k][w] for w in ws] for k in ks])
    return np.array(ks), np.array(ws), S, band


def plot(path, outdir):
    ks, ws, S, band = load(path)
    tag = os.path.splitext(os.path.basename(path))[0]

    fig, ax = plt.subplots(figsize=(7.2, 5.0))
    # The spectrum spans orders of magnitude; a linear map shows only the lower edge. Clip at a
    # small positive floor so the log scale does not choke on the (physical) exact zeros outside
    # the band -- those zeros are the POINT, so they must render as the floor colour, not as gaps.
    floor = max(S.max(), 1e-300) * 1e-4
    Sp = np.clip(S, floor, None)
    mesh = ax.pcolormesh(ks, ws, Sp.T, shading="auto",
                         norm=matplotlib.colors.LogNorm(vmin=floor, vmax=Sp.max()),
                         cmap="magma")
    fig.colorbar(mesh, ax=ax, label=r"$S(k,\omega)$  (log)")

    kk = np.array(sorted(band))
    lo = np.array([band[k][0] for k in kk])
    hi = np.array([band[k][1] for k in kk])
    # Densify the analytic curves: the data lives on L ring momenta, which is far too coarse to
    # draw a smooth boundary and would make the band look like it misses the weight.
    kd = np.linspace(0, 2 * np.pi, 400)
    kf = np.where(kd > np.pi, 2 * np.pi - kd, kd)
    # Both curves bound the SAME two-spinon continuum: the lower edge is one spinon carrying all
    # the momentum, the upper edge is two sharing it equally. Labelling them "one spinon" and "two
    # spinons" would suggest two different excitation branches, which is not what they are.
    ax.plot(kd, kf * (np.pi - kf) / 2, "c--", lw=1.4,
            label=r"$\omega_-$ lower edge (one spinon carries $k$)")
    ax.plot(kd, kf * (2 * np.pi - kf) / 4, "w--", lw=1.4,
            label=r"$\omega_+$ upper edge (two share $k$)")

    ax.set_xlabel("$k$")
    ax.set_ylabel(r"$\omega$")
    ax.set_xlim(ks.min(), ks.max())
    # Crop to the band plus headroom instead of showing the whole computed range. `wmax` is 6 by
    # default while the continuum tops out at `w_+(pi) = pi^2/4 = 2.47`, so more than half the
    # frame was empty black and the band was squashed into the bottom third. The negative-omega
    # strip is KEPT -- its emptiness is one of the things the figure is asserting.
    band_top = float(hi.max()) if hi.size else float(ws.max())
    ax.set_ylim(ws.min(), min(ws.max(), band_top * 1.45))
    # k in units of pi: the momenta are 2*pi*m/L, so raw radians put the ticks at 0, 0.785, 1.571
    # and nothing lines up with the zone boundary the eye is looking for.
    ax.xaxis.set_major_locator(matplotlib.ticker.MultipleLocator(np.pi / 2))
    ax.xaxis.set_major_formatter(matplotlib.ticker.FuncFormatter(
        lambda v, _: {0: "0", 1: r"$\pi/2$", 2: r"$\pi$", 3: r"$3\pi/2$", 4: r"$2\pi$"}
        .get(int(round(v / (np.pi / 2))), "")))
    ax.set_title(f"Haldane-Shastry dynamical structure factor — {tag}", fontsize=10)
    ax.legend(fontsize=8, loc="upper right")
    fig.tight_layout()
    out = os.path.join(outdir, f"{tag}.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)

    # A cut at the zone boundary, where the continuum is widest and the edge easiest to read.
    ik = int(np.argmin(np.abs(ks - np.pi)))
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(ws, S[ik], lw=1.2)
    l, h = band[ks[ik]]
    ax.axvline(l, color="c", ls="--", lw=1.2, label=r"$\omega_-$")
    ax.axvline(h, color="k", ls="--", lw=1.2, label=r"$\omega_+$")
    ax.axvline(0.0, color="r", ls=":", lw=1.0, label=r"$\omega=0$")
    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$S(k=\pi,\omega)$")
    # Same crop as the map: past `w_+` there is nothing to see, and showing it to `wmax` spends
    # most of the frame proving a point the axis limit already makes.
    ax.set_xlim(ws.min(), min(ws.max(), h * 1.45))
    ax.set_title(r"cut at $k \simeq \pi$: weight must sit between the dashed lines", fontsize=10)
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3, lw=0.5)
    fig.tight_layout()
    out = os.path.join(outdir, f"{tag}_cut.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    paths = sorted(glob.glob(PAT)) if any(c in PAT for c in "*?[") else (
        [PAT] if os.path.exists(PAT) else [])
    if not paths:
        sys.exit(f"no CSV matching {PAT} -- run benchmarks/run_hs_structure_factor.jl first")
    os.makedirs(OUTDIR, exist_ok=True)
    for p in paths:
        plot(p, OUTDIR)


if __name__ == "__main__":
    main()
