#!/usr/bin/env python3
"""WHY THE OPEN MODEL CANNOT USE A LOGARITHMIC GRID -- the one figure that carries the claim.

Two panels, SAME physical chain, two generators:

  LEFT   -H, the IMAGINARY-TIME generator of models 1 and 2 (Heisenberg chain, cylinder).
         H is Hermitian, so every eigenvalue is REAL and exp(-H tau) is a sum of MONOTONE decays.
         Nothing oscillates, so nothing can alias, and a log grid's exponentially growing steps
         cost nothing. This is why the log grid WORKS there.

  RIGHT  L, the LINDBLADIAN of model 3. Non-Hermitian, so eigenvalues are COMPLEX and exp(Lt) is a
         sum of decaying OSCILLATIONS at angular frequencies |Im lambda|. A step larger than
         pi/max|Im lambda| cannot resolve the fastest one. This is why the log grid CANNOT work.

⛔ THE POINT IS THE CONTRAST, NOT EITHER PANEL ALONE. A reader shown only the right panel could
reasonably ask why a log grid was ever used; shown only the left, why it was ever abandoned. The
grid choice is a property of WHERE THE SPECTRUM LIVES, and that is what these two panels show.

⚠ THE BOUND DEPENDS ON THE OBSERVABLE AND BOTH ARE DRAWN. A LINEAR observable (a magnetisation)
carries exp(lambda t), so its fastest angular frequency is max|Im lambda| and the Nyquist step is
pi/max|Im|. A QUADRATIC one (a correlator, a current) beats PAIRS of eigenvalues together, doubling
the frequency and HALVING the step. Quoting the linear bound for a quadratic observable would place
the limit at twice the step it belongs at and show aliasing runs as resolving.

Run:  python3 benchmarks/imaginary_time/plot_spectrum.py [csv]
"""
import csv
import math
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
DEFAULT = os.path.join(HERE, "..", "results", "dissip_heis_spectrum.csv")


def load(path):
    rows = defaultdict(list)
    with open(path, newline="") as fh:
        for r in csv.DictReader(fh):
            rows[(r["generator"], int(r["L"]))].append((float(r["re"]), float(r["im"])))
    return rows


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else DEFAULT
    if not os.path.exists(path):
        sys.exit(f"no spectrum CSV at {path} -- run: "
                 "julia --project=. benchmarks/dissipative_heisenberg.jl spectrum")
    rows = load(path)
    Ls = sorted({L for (_, L) in rows})
    if not Ls:
        sys.exit("no rows")
    L = max(Ls)                      # the largest available; the structure is L-independent
    ham = rows.get(("hamiltonian", L), [])
    lin = rows.get(("lindblad", L), [])
    if not ham or not lin:
        sys.exit(f"need both generators at L={L}")

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.4))

    ax = axes[0]
    ax.scatter([re for re, _ in ham], [im for _, im in ham], s=16, alpha=0.75,
               color="tab:blue", label="eigenvalues of $-H$")
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
    ax.set_ylabel(r"$\mathrm{Im}\,\lambda$")
    ax.set_title(f"IMAGINARY TIME  (models 1-2, L={L})\n"
                 r"$H$ Hermitian $\Rightarrow$ spectrum REAL $\Rightarrow$ monotone decay"
                 "\nlog grid WORKS: nothing to alias", fontsize=9, loc="left")
    # a real spectrum is a line; give it visible vertical extent so it does not read as an error
    ax.set_ylim(-1.0, 1.0)
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8, loc="upper right")

    ax = axes[1]
    ax.scatter([re for re, _ in lin], [im for _, im in lin], s=16, alpha=0.75,
               color="tab:red", label=r"eigenvalues of $\mathcal{L}$")
    ax.axhline(0.0, color="0.6", lw=0.8, ls=":")
    mim = max(abs(im) for _, im in lin)
    for s, lab in ((+1, None), (-1, None)):
        ax.axhline(s * mim, color="k", ls="--", lw=1.0)
    ax.annotate(rf"$\max|\mathrm{{Im}}\,\lambda| = {mim:.3f}$" "\n"
                rf"linear obs:    $\Delta t \leq \pi/\max|\mathrm{{Im}}| = {math.pi/mim:.3f}$" "\n"
                rf"quadratic obs: $\Delta t \leq {math.pi/(2*mim):.3f}$",
                xy=(0.03, 0.03), xycoords="axes fraction", fontsize=8,
                bbox=dict(boxstyle="round,pad=0.35", fc="white", ec="0.7", alpha=0.9))
    ax.set_xlabel(r"$\mathrm{Re}\,\lambda$")
    ax.set_title(f"REAL TIME, LINDBLAD  (model 3, L={L})\n"
                 r"$\mathcal{L}$ non-Hermitian $\Rightarrow$ spectrum COMPLEX "
                 r"$\Rightarrow$ decaying oscillation"
                 "\nlog grid CANNOT work: its late steps alias", fontsize=9, loc="left")
    ax.grid(alpha=0.3, lw=0.4)
    ax.legend(fontsize=8, loc="upper right")

    fig.tight_layout()
    out = os.path.join(HERE, "..", "results", "spectrum_why_no_log_grid.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


if __name__ == "__main__":
    main()
