#!/usr/bin/env python3
"""Three models, three integrators: observables in time, and the BUG's three-knob matrix.

    A  heisenberg_su2_chain_L16     Heisenberg chain, SU(2), dimer start
    B  square_su2_4x4_L16           Heisenberg on a 4x4 square cylinder, SU(2), dimer start
    C  kagome_dipolarXY_2x3_L18     dipolar XY on a breathing kagome (arXiv:2603.21147), U(1)

Every model has an EXACT reference at these sizes, so the error panels are real errors against
the truth, not spreads between schemes.

⛔ TWO COST AXES, PLOTTED SEPARATELY AND ON PURPOSE. Operator applications and wall clock DISAGREE
at scale -- measured on an 8x8 cylinder, `cbe_bug` used 19x fewer applications and took 1.91x
LONGER. `krylov` cannot see the CBE expansion's SVD/QR work, so a cost claim that quotes only one
axis is not a cost claim.

Run:  python3 benchmarks/plot_three_models.py [results_dir]
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LogNorm

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")

TITLE = {
    "heisenberg_su2_chain_L16": "A · Heisenberg chain, SU(2), L=16",
    "square_su2_4x4_L16":       "B · Heisenberg square cylinder 4×4, SU(2), 16 sites",
    "kagome_dipolarXY_2x3_L18": "C · dipolar XY, breathing kagome 2×3, U(1), 18 sites",
}
ORDER = ["heisenberg_su2_chain_L16", "square_su2_4x4_L16", "kagome_dipolarXY_2x3_L18"]
COL = {"tdvp2": "#c0392b", "tdvp_cbe1s": "#e67e22", "cbe_bug": "#2471a3"}
LAB = {"tdvp2": "2-site TDVP", "tdvp_cbe1s": "1-site CBE-TDVP", "cbe_bug": "CBE-BUG"}


def load_evolve():
    d = defaultdict(lambda: defaultdict(list))
    p = os.path.join(RES, "three_models_evolve.csv")
    if not os.path.exists(p):
        return {}
    for r in csv.DictReader(open(p)):
        d[r["model"]][r["scheme"]].append(r)
    for m in d:
        for s in d[m]:
            d[m][s].sort(key=lambda r: float(r["t"]))
    return d


def plot_evolve(data):
    models = [m for m in ORDER if m in data]
    if not models:
        return
    fig, axes = plt.subplots(3, len(models), figsize=(5.0 * len(models), 10.5), squeeze=False)
    for c, m in enumerate(models):
        # ── row 0: the OBSERVABLE in time, with the exact curve underneath ──────────────
        ax = axes[0][c]
        any_s = next(iter(data[m]))
        t = [float(r["t"]) for r in data[m][any_s]]
        ex = [float(r["obs_exact"]) for r in data[m][any_s]]
        # exact drawn THICK and pale underneath so a scheme lying on top is visible as
        # coincidence rather than hidden by overdraw
        ax.plot(t, ex, lw=6, color="0.75", zorder=1, label="exact")
        for s in ("tdvp2", "tdvp_cbe1s", "cbe_bug"):
            if s not in data[m]:
                continue
            ax.plot(t, [float(r["obs"]) for r in data[m][s]], lw=1.6, color=COL[s],
                    label=LAB[s], zorder=3)
        ax.set_title(TITLE.get(m, m), fontsize=10)
        ax.set_ylabel("mean bond energy  ⟨Sᵢ·Sⱼ⟩" if c == 0 else "")
        ax.set_xlabel("t")
        ax.grid(alpha=.25, lw=.5)
        if c == 0:
            ax.legend(fontsize=8)

        # ── row 1: the ERROR in time (log) ─────────────────────────────────────────────
        ax = axes[1][c]
        for s in ("tdvp2", "tdvp_cbe1s", "cbe_bug"):
            if s not in data[m]:
                continue
            e = [max(float(r["err"]), 1e-16) for r in data[m][s]]
            ax.semilogy(t, e, lw=1.6, color=COL[s], label=LAB[s])
        ax.set_ylabel("max |Δ⟨SᵢSⱼ⟩| vs exact" if c == 0 else "")
        ax.set_xlabel("t")
        ax.grid(alpha=.25, which="both", lw=.5)

        # ── row 2: the two COST axes, which disagree ────────────────────────────────────
        ax = axes[2][c]
        w = 0.35
        xs = np.arange(3)
        kry = [float(data[m][s][-1]["krylov"]) if s in data[m] else 0
               for s in ("tdvp2", "tdvp_cbe1s", "cbe_bug")]
        sec = [float(data[m][s][-1]["secs"]) if s in data[m] else 0
               for s in ("tdvp2", "tdvp_cbe1s", "cbe_bug")]
        base = max(kry) or 1.0
        ax.bar(xs - w / 2, [k / base for k in kry], w, label="operator applications",
               color="#5d6d7e")
        bsec = max(sec) or 1.0
        ax.bar(xs + w / 2, [s / bsec for s in sec], w, label="wall clock", color="#af7ac5")
        ax.set_xticks(xs)
        ax.set_xticklabels([LAB[s] for s in ("tdvp2", "tdvp_cbe1s", "cbe_bug")],
                           fontsize=8, rotation=12)
        ax.set_ylabel("fraction of the worst" if c == 0 else "")
        ax.grid(alpha=.25, axis="y", lw=.5)
        if c == 0:
            ax.legend(fontsize=8)
        for x, (k, s_) in enumerate(zip(kry, sec)):
            ax.text(x - w / 2, k / base + .02, f"{int(k)}", ha="center", fontsize=7)
            ax.text(x + w / 2, s_ / bsec + .02, f"{s_:.0f}s", ha="center", fontsize=7)

    fig.suptitle("Three models · three integrators · exact reference in every panel", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.975])
    out = os.path.join(RES, "three_models_evolve.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(out))


def plot_matrix():
    p = os.path.join(RES, "three_models_matrix.csv")
    if not os.path.exists(p):
        print("no matrix csv yet")
        return
    rows = list(csv.DictReader(open(p)))
    models = [m for m in ORDER if any(r["model"] == m for r in rows)]
    ms = sorted({int(r["m"]) for r in rows})
    for field, fname, log, fmt in (("err", "three_models_matrix_err.png", True, "{:.1e}"),
                                   ("states", "three_models_matrix_chi.png", False, "{:.0f}")):
        fig, axes = plt.subplots(len(models), len(ms),
                                 figsize=(3.7 * len(ms), 3.5 * len(models)), squeeze=False)
        vals = [float(r[field]) for r in rows if float(r[field]) > 0]
        norm = LogNorm(vmin=min(vals), vmax=max(vals)) if log and vals else None
        for i, m in enumerate(models):
            for j, mm in enumerate(ms):
                sub = [r for r in rows if r["model"] == m and int(r["m"]) == mm]
                ys = sorted({float(r["tau_trunc"]) for r in sub}, reverse=True)
                xs = sorted({float(r["split_cutoff"]) for r in sub}, reverse=True)
                g = np.full((len(ys), len(xs)), np.nan)
                for r in sub:
                    g[ys.index(float(r["tau_trunc"])), xs.index(float(r["split_cutoff"]))] = \
                        float(r[field])
                ax = axes[i][j]
                im = ax.imshow(g, cmap="viridis_r" if log else "viridis", norm=norm, aspect="auto")
                ax.set_xticks(range(len(xs)))
                ax.set_xticklabels([f"{x:.0e}" for x in xs], rotation=45, fontsize=7)
                ax.set_yticks(range(len(ys)))
                ax.set_yticklabels([f"{y:.0e}" for y in ys], fontsize=7)
                if i == 0:
                    ax.set_title(f"m = {mm}" + (" (to the cap)" if mm == 30 else ""), fontsize=10)
                if j == 0:
                    ax.set_ylabel(TITLE.get(m, m).split("·")[0].strip() + "\nroot-out",
                                  fontsize=8)
                if i == len(models) - 1:
                    ax.set_xlabel("half-sweep `split_cutoff`", fontsize=8)
                for a in range(len(ys)):
                    for b in range(len(xs)):
                        if np.isfinite(g[a, b]):
                            ax.text(b, a, fmt.format(g[a, b]), ha="center", va="center",
                                    fontsize=6, color="white")
        fig.suptitle(("CBE-BUG error" if field == "err" else "CBE-BUG rank (states)") +
                     " · three-knob matrix · rows = model, cols = Krylov depth", fontsize=11)
        fig.tight_layout(rect=[0, 0, 1, 0.96])
        fig.savefig(os.path.join(RES, fname), dpi=150, bbox_inches="tight")
        plt.close(fig)
        print("wrote", fname)


def main():
    d = load_evolve()
    if d:
        plot_evolve(d)
    else:
        print("no evolve csv yet")
    plot_matrix()
    return 0


if __name__ == "__main__":
    sys.exit(main())
