#!/usr/bin/env python3
"""Plots for benchmarks/l18_matrix.jl.

    python benchmarks/plot_l18.py [results.json] [outdir]

Three figures, which are the three questions the benchmark exists to answer:

  light cone   <S_j.S_{j+1}>(j, t) as a heatmap, arms beside the EXACT reference. This is the
               physics: a dimer quench spreads correlation at a finite velocity, and an arm that
               has run out of bond dimension stops reproducing the cone rather than blurring
               uniformly. The reference panel is what makes this readable -- without it a wrong
               cone and a right one look equally plausible.

  error        ||<S.S>_arm - <S.S>_exact||(t). Genuine ACCURACY, not mutual consistency,
               because the reference is exact (sparse Krylov, not a better-converged MPS).
               Expect it to grow and then saturate: at late times no chi<=128 state can track
               a volume-law entangled state, so this measures WHEN each arm gives up.

  bond growth  max_b chi_b(t), i.e. how fast each arm spends its budget. Read together with the
               error panel: an arm that is at the cap early and still accurate is doing well;
               one at the cap early and inaccurate is wasting the budget on the wrong directions.

Symmetry is the cross-cutting comparison: :none and :SU2 run the same state, the same
Hamiltonian and the same reference, so their curves should LIE ON TOP OF EACH OTHER. Divergence
is a bug, not a finding. Note chi counts multiplets under SU(2) and states under :none, so the
bond-growth panel is NOT comparable across symmetries -- only the error panel is.
"""
import json
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SRC = sys.argv[1] if len(sys.argv) > 1 else "benchmarks/l18_results.json"
OUT = sys.argv[2] if len(sys.argv) > 2 else "benchmarks/figs"

import os
os.makedirs(OUT, exist_ok=True)

with open(SRC) as fh:
    data = json.load(fh)

# The reference is keyed by INIT (`__reference__dimer` / `__reference__domainwall`) so a dimer
# and a domain-wall run can share one file. Match the PREFIX: looking for the bare
# `__reference__` finds nothing, and because the reference dict has no "arm" key it is filtered
# out of `runs` silently -- so the failure mode is not a crash but light-cone figures that
# quietly lose the EXACT panel, which is the panel that makes the others interpretable.
refs = {k: v for k, v in data.items() if k.startswith("__reference__")}
for k in refs:
    data.pop(k)
ref = next(iter(refs.values()), None)
if ref is None:
    print("WARNING: no __reference__* entry found -- light cones will have no exact panel")
runs = [v for v in data.values() if isinstance(v, dict) and "arm" in v]
if not runs:
    sys.exit("no runs in %s" % SRC)

L = ref["L"] if ref else (len(runs[0]["obs"][0]) + 1)

# Label from the DATA, not from a hardcoded guess. A dimer run measures <S_j.S_{j+1}> per BOND
# and a domain-wall run measures <Sz_j> per SITE; printing the wrong one turns a correct figure
# into a misleading one, which is worse than a missing label.
OBS = (ref or {}).get("obs_name") or runs[0].get("obs_name") or r"observable"
INIT = (ref or {}).get("init") or runs[0].get("init") or "dimer"
XLAB = "bond $j$" if INIT == "dimer" else "site $j$"
CBLAB = (r"$\langle S_j \cdot S_{j+1} \rangle$" if INIT == "dimer"
         else r"$\langle S^z_j \rangle$")


def groups(mode):
    """(sym, maxdim, cutoff) -> [run], for one mode."""
    out = defaultdict(list)
    for r in runs:
        if r["mode"] == mode and "obs" in r:
            out[(r["sym"], r["maxdim"], r["cutoff"])].append(r)
    return out


# ── 1. light cone ────────────────────────────────────────────────────────────────────
def light_cone(mode, ref_key):
    g = groups(mode)
    if not g:
        return
    for (sym, md, ct), rs in sorted(g.items()):
        rs = sorted(rs, key=lambda r: r["arm"])
        n = len(rs) + (1 if ref else 0)
        fig, axes = plt.subplots(1, n, figsize=(3.1 * n, 3.4), sharey=True,
                                 constrained_layout=True)
        axes = np.atleast_1d(axes)
        panels = []
        if ref and ref.get(ref_key):
            panels.append(("EXACT", np.array(ref["t"]), np.array(ref[ref_key])))
        panels += [(r["arm"], np.array(r["t"]), np.array(r["obs"])) for r in rs]
        vmin = min(p[2].min() for p in panels)
        vmax = max(p[2].max() for p in panels)
        for ax, (name, t, obs) in zip(axes, panels):
            im = ax.pcolormesh(np.arange(1, obs.shape[1] + 1), t, obs,
                               shading="nearest", vmin=vmin, vmax=vmax, cmap="magma")
            ax.set_title(name, fontsize=8)
            ax.set_xlabel(XLAB)
        axes[0].set_ylabel(r"$\beta$" if mode == "imag" else "$t$")
        fig.colorbar(im, ax=axes[-1], label=CBLAB)
        fig.suptitle(f"{mode}  {INIT}  sym={sym}  $\\chi$={md}  cutoff={ct:g}  (L={L})", fontsize=10)
        fig.savefig(f"{OUT}/lightcone_{mode}_{sym}_chi{md}_ct{ct:g}.png", dpi=140)
        plt.close(fig)


# ── 2. error growth, 3. bond growth ──────────────────────────────────────────────────
def curves(mode, field, ylabel, fname, logy=True):
    g = groups(mode)
    if not g:
        return
    syms = sorted({k[0] for k in g})
    mds = sorted({k[1] for k in g})
    cts = sorted({k[2] for k in g})
    fig, axes = plt.subplots(len(cts), len(mds),
                             figsize=(4.0 * len(mds), 3.2 * len(cts)),
                             squeeze=False, sharex=True, sharey=True,
                             constrained_layout=True)
    for i, ct in enumerate(cts):
        for j, md in enumerate(mds):
            ax = axes[i][j]
            for sym in syms:
                # SU(2) dashed, :none solid -- the two should coincide.
                style = "--" if sym == "SU2" else "-"
                for r in sorted(g.get((sym, md, ct), []), key=lambda r: r["arm"]):
                    y = np.array(r[field], dtype=float)
                    ax.plot(r["t"], y, style, lw=1.2,
                            label=f"{r['arm']} [{sym}]")
            if logy:
                ax.set_yscale("log")
            ax.set_title(f"$\\chi$={md}  cutoff={ct:g}", fontsize=9)
            ax.grid(alpha=0.3)
    axes[-1][0].set_xlabel(r"$\beta$" if mode == "imag" else "$t$")
    axes[0][0].set_ylabel(ylabel)
    h, l = axes[0][0].get_legend_handles_labels()
    fig.legend(h, l, loc="center right", fontsize=6, framealpha=0.9)
    fig.suptitle(f"{mode}: {ylabel}  (L={L})", fontsize=11)
    fig.savefig(f"{OUT}/{fname}_{mode}.png", dpi=140, bbox_inches="tight")
    plt.close(fig)


for mode, rk in (("real", "obs_real"), ("imag", "obs_imag")):
    light_cone(mode, rk)
    # Label from the data too -- this panel is the error in whichever observable was measured.
    curves(mode, "err", r"$\|$obs $-$ exact$\|$", "error", logy=True)
    curves(mode, "maxbd", r"max bond dim", "bonddim", logy=False)

# ── ground state ─────────────────────────────────────────────────────────────────────
gr = [r for r in runs if r["mode"] == "ground"]
if gr:
    fig, ax = plt.subplots(figsize=(6, 4), constrained_layout=True)
    for r in sorted(gr, key=lambda r: (r["arm"], r["sym"], r["maxdim"])):
        e = np.array(r["energies"], dtype=float) - r["E0"]
        ax.semilogy(np.arange(1, len(e) + 1), np.maximum(e, 1e-16), "--" if r["sym"] == "SU2" else "-",
                    lw=1.2, label=f"{r['arm']} [{r['sym']}] $\\chi$={r['maxdim']} ct={r['cutoff']:g}")
    ax.set_xlabel("sweep")
    ax.set_ylabel("$E-E_0$")
    ax.set_title(f"ground state convergence (L={L})")
    ax.grid(alpha=0.3)
    ax.legend(fontsize=6)
    fig.savefig(f"{OUT}/ground.png", dpi=140, bbox_inches="tight")
    plt.close(fig)

print(f"wrote figures to {OUT}/")
