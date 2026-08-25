#!/usr/bin/env python3
"""One figure PER MODEL, as three separate deliverables.

  heisenberg / square : error growth in time and bond-dimension growth in time -- three lines,
                        CBE-BUG at its BEST matrix configuration against the two TDVPs.
  kagome              : the paper's own observables -- C^xx(r), C^zz(r), the structure factors,
                        and the decay that separates a Dirac from a chiral spin liquid.

⛔ THE BUG LINE IS ITS BEST CONFIGURATION, AND THAT IS STATED ON THE PLOT. Picking the best of a
scanned family and drawing it against single-configuration baselines flatters it; the figure says
which (m, root-out, half-sweep) won so the comparison is legible rather than implied.

⛔ AND NOTHING HERE IS AN "ERROR" FOR KAGOME. The chiral order parameter is exactly zero on any
finite real ground state (a theorem about real matrices, verified by ED), so the kagome panels
show T-EVEN quantities only and are labelled as trends, not as reproductions of the paper's h_c.

Run:  python3 benchmarks/plot_models_separate.py [results_dir]
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")

COL = {"tdvp2": "#c0392b", "tdvp_cbe1s": "#e67e22", "cbe_bug": "#2471a3"}
LAB = {"tdvp2": "2-site TDVP", "tdvp_cbe1s": "1-site CBE-TDVP", "cbe_bug": "CBE-BUG"}
NICE = {"heisenberg_su2_chain_L16": ("Heisenberg chain · SU(2) · L=16", "heisenberg"),
        "square_su2_4x4_L16": ("Heisenberg square cylinder 4×4 · SU(2) · 16 sites", "square")}


def best_bug_config():
    """The (m, root, half) with the lowest final error, per model, from the matrix scan."""
    p = os.path.join(RES, "three_models_matrix.csv")
    if not os.path.exists(p):
        return {}
    best = {}
    for r in csv.DictReader(open(p)):
        m = r["model"]
        e = float(r["err"])
        if m not in best or e < best[m][0]:
            best[m] = (e, int(r["m"]), float(r["tau_trunc"]), float(r["split_cutoff"]),
                       int(r["states"]), int(r["krylov"]), float(r["secs"]))
    return best


def plot_timeseries(model, title, slug, best):
    p = os.path.join(RES, "three_models_evolve.csv")
    if not os.path.exists(p):
        return
    d = defaultdict(list)
    for r in csv.DictReader(open(p)):
        if r["model"] == model:
            d[r["scheme"]].append(r)
    if not d:
        return
    for s in d:
        d[s].sort(key=lambda r: float(r["t"]))

    fig, axes = plt.subplots(1, 2, figsize=(11.5, 4.6))
    for s in ("tdvp2", "tdvp_cbe1s", "cbe_bug"):
        if s not in d:
            continue
        t = [float(r["t"]) for r in d[s]]
        e = [max(float(r["err"]), 1e-17) for r in d[s]]
        chi = [int(r["states"]) for r in d[s]]
        axes[0].semilogy(t, e, lw=1.8, color=COL[s], label=LAB[s], marker="o", ms=3)
        axes[1].plot(t, chi, lw=1.8, color=COL[s], label=LAB[s], marker="o", ms=3)

    axes[0].set_xlabel("t"); axes[0].set_ylabel("max |Δ⟨SᵢSⱼ⟩| against the exact state")
    axes[0].set_title("error growth", fontsize=10)
    axes[0].grid(alpha=.25, which="both", lw=.5); axes[0].legend(fontsize=8)
    axes[1].set_xlabel("t"); axes[1].set_ylabel("bond dimension (states kept)")
    axes[1].set_title("rank growth", fontsize=10)
    axes[1].grid(alpha=.25, lw=.5); axes[1].legend(fontsize=8)

    sub = ""
    if model in best:
        e, m, tt, sc, st, kry, sec = best[model]
        sub = (f"CBE-BUG line is the shipped default (m=30, tol=1e-8); best matrix config was "
               f"m={m}, root-out={tt:.0e}, half-sweep={sc:.0e} → {e:.3e}")
    fig.suptitle(title + ("\n" + sub if sub else ""), fontsize=11)
    fig.tight_layout(rect=[0, 0, 1, 0.93 if sub else 0.96])
    out = os.path.join(RES, f"model_{slug}_time.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(out))


def plot_kagome():
    p = os.path.join(RES, "kagome_paper_observables.csv")
    if not os.path.exists(p):
        print("no kagome observables csv yet")
        return
    rows = list(csv.DictReader(open(p)))
    if not rows:
        return
    hs = sorted({float(r["h"]) for r in rows})
    cmap = plt.cm.viridis(np.linspace(0.05, 0.9, len(hs)))

    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.6))

    # ── C^xx(r) and C^zz(r): log-linear. STRAIGHT = exponential = chiral-liquid-like ────
    for ax, key, nm in ((axes[0][0], "Cxx", "C$^{xx}(r) = \\langle S^x_0 S^x_r\\rangle$"),
                        (axes[0][1], "Czz", "C$^{zz}(r) = \\langle S^z_0 S^z_r\\rangle$")):
        for h, c in zip(hs, cmap):
            sub = [r for r in rows if float(r["h"]) == h]
            # average |C| within a distance shell, so the profile is a function of r alone
            shell = defaultdict(list)
            for r in sub:
                shell[round(float(r["r"]), 3)].append(abs(float(r[key])))
            xs = sorted(shell)
            ys = [np.mean(shell[x]) for x in xs]
            ax.semilogy(xs, [max(y, 1e-16) for y in ys], marker="o", ms=4, lw=1.4,
                        color=c, label=f"h = {h:.2f}")
        ax.set_xlabel("r  (units of the isotropic nearest-neighbour distance)")
        ax.set_ylabel("|" + nm + "|")
        ax.set_title(nm + " — straight here means EXPONENTIAL decay", fontsize=9)
        ax.grid(alpha=.25, which="both", lw=.5)
        ax.legend(fontsize=7)

    # ── the same on log-log: STRAIGHT = algebraic = Dirac-liquid-like ───────────────────
    ax = axes[1][0]
    for h, c in zip(hs, cmap):
        sub = [r for r in rows if float(r["h"]) == h]
        shell = defaultdict(list)
        for r in sub:
            shell[round(float(r["r"]), 3)].append(abs(float(r["Cxx"])))
        xs = [x for x in sorted(shell) if x > 0]
        ys = [max(np.mean(shell[x]), 1e-16) for x in xs]
        ax.loglog(xs, ys, marker="o", ms=4, lw=1.4, color=c, label=f"h = {h:.2f}")
    ax.set_xlabel("r"); ax.set_ylabel("|C$^{xx}$|")
    ax.set_title("same data, log–log — straight here means ALGEBRAIC decay", fontsize=9)
    ax.grid(alpha=.25, which="both", lw=.5); ax.legend(fontsize=7)

    # ── ground-state energy and entropy vs h: where a transition would leave a mark ─────
    ax = axes[1][1]
    e = [next(float(r["E0"]) for r in rows if float(r["h"]) == h) for h in hs]
    s = [next(float(r["S_vN"]) for r in rows if float(r["h"]) == h) for h in hs]
    L = 18
    ax.plot(hs, [x / L for x in e], marker="o", color="#c0392b", label="E₀ / site")
    ax.set_xlabel("breathing parameter h"); ax.set_ylabel("E₀ / site", color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax2 = ax.twinx()
    ax2.plot(hs, s, marker="s", color="#2471a3", label="S$_{vN}$")
    ax2.set_ylabel("entanglement entropy S$_{vN}$", color="#2471a3")
    ax2.tick_params(axis="y", labelcolor="#2471a3")
    ax.axvline(0.22, ls="--", color="0.4", lw=1)
    ax.text(0.225, ax.get_ylim()[0], " paper's h$_c$ ≈ 0.22\n (infinite cylinder)",
            fontsize=7, color="0.35", va="bottom")
    ax.set_title("E₀ and entropy vs h", fontsize=9)
    ax.grid(alpha=.25, lw=.5)

    fig.suptitle("Breathing kagome, dipolar XY (arXiv:2603.21147) — 18 sites, open cylinder\n"
                 "⚠ ⟨χ⟩ ≡ 0 on any finite real ground state (exact), so these are the T-even "
                 "observables; trends only, not the paper's h$_c$", fontsize=10)
    fig.tight_layout(rect=[0, 0, 1, 0.92])
    out = os.path.join(RES, "model_kagome_observables.png")
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", os.path.basename(out))


def main():
    best = best_bug_config()
    for model, (title, slug) in NICE.items():
        plot_timeseries(model, title, slug, best)
    plot_kagome()
    return 0


if __name__ == "__main__":
    sys.exit(main())
