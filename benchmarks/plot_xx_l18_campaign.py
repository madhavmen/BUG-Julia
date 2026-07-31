#!/usr/bin/env python
"""Plots for the L=18 XX domain-wall campaign: CBE-BUG against 2-site TDVP.

    ~/miniconda3/envs/cfd_env/bin/python benchmarks/plot_xx_l18_campaign.py

Reads results/xx_l18_campaign.csv (written by benchmarks/xx_l18_campaign.jl main).

The four panels answer four separate questions, and it matters that they are separate:

  (a) accuracy       err(t) against the FREE-FERMION EXACT profile, not a finer run
  (b) bond growth    max_i chi_i (t) -- how fast each method's manifold has to widen
  (c) working set    stored elements the step must hold at once -- the MEASURED
                     `peak_elems` (`state + transients alive`, sampled inside the step at
                     a definition both integrators share), not `state + theta`. The proxy
                     flattered CBE-BUG twice over: its `theta_elems` is 0 by construction,
                     and its basis sweep is transiently WIDER than the state that survives
                     truncation, so the state alone understates it. This is the second
                     design goal made quantitative, and it has to be made on the peak.
  (d) efficiency     accuracy against working set -- the Pareto view. A method is more
                     resource efficient only if it sits DOWN AND LEFT of the other; a
                     lower error bought with a wider bond is not an efficiency claim.
"""
import os
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
CSV = os.path.join(HERE, "..", "results", "xx_l18_campaign.csv")
OUT = os.path.join(HERE, "..", "results")

rows = np.genfromtxt(CSV, delimiter=",", names=True, dtype=None, encoding="utf-8")
if rows.ndim == 0:
    rows = rows.reshape(1)


def label(scheme, maxdim, dex):
    if scheme == "tdvp2":
        return f"2-site TDVP, $\\chi\\!\\le\\!{maxdim}$"
    d = "r" if dex == 0 else str(dex)
    return f"CBE-BUG $D_{{ex}}\\!=\\!{d}$, $\\chi\\!\\le\\!{maxdim}$"


# One curve per (scheme, maxdim, dex). Colour carries the METHOD, linestyle the maxdim,
# so the eye groups by method first -- that is the comparison being made.
COL = {("tdvp2", 0): "#c1272d", ("cbe", 2): "#0057a4", ("cbe", 0): "#1b7f3b"}
LS = {32: "--", 64: "-"}
MK = {32: "s", 64: "o"}

series = []
for scheme in ("tdvp2", "cbe"):
    for maxdim in sorted(set(rows["maxdim"][rows["scheme"] == scheme].tolist())):
        for dex in sorted(set(rows["dex"][(rows["scheme"] == scheme)
                                          & (rows["maxdim"] == maxdim)].tolist())):
            m = ((rows["scheme"] == scheme) & (rows["maxdim"] == maxdim)
                 & (rows["dex"] == dex))
            r = rows[m]
            r = r[np.argsort(r["t"])]
            series.append(dict(
                scheme=scheme, maxdim=int(maxdim), dex=int(dex), r=r,
                lbl=label(scheme, int(maxdim), int(dex)),
                c=COL[(scheme, int(dex))], ls=LS.get(int(maxdim), "-"),
                mk=MK.get(int(maxdim), "o")))

fig, ax = plt.subplots(2, 2, figsize=(12.4, 9.0))
(a, b), (c, d) = ax

for s in series:
    r, kw = s["r"], dict(color=s["c"], ls=s["ls"], lw=1.9, label=s["lbl"])
    a.semilogy(r["t"], np.maximum(r["err"], 1e-16), **kw)
    b.plot(r["t"], r["maxbond"], **kw)
    # working set: MEASURED inside the step (state + transients alive), same definition
    # for both integrators. See the module docstring on why the state+theta proxy is out.
    c.plot(r["t"], r["peak_elems"] / 1e3, **kw)

a.set(xlabel="time $t$", ylabel=r"$\max_j\,|\langle S^z_j\rangle - \mathrm{exact}|$",
      title="(a) accuracy vs the exact free-fermion profile")
a.grid(alpha=.3, which="both")
a.legend(fontsize=8, loc="lower right")

b.set(xlabel="time $t$", ylabel=r"$\max_i \chi_i$",
      title="(b) bond-dimension growth")
b.grid(alpha=.3)

c.set(xlabel="time $t$", ylabel="peak working set  [$10^3$ stored elements]",
      title="(c) memory the step must hold (measured peak: state + live transients)")
c.grid(alpha=.3)

# (d) The Pareto view. Each method contributes one point per maxdim: its WORST error
# over the second half of the run (the hard part, once the wall has melted) against its
# PEAK working set. Worst-case error, not final -- a curve that spikes and recovers is
# not accurate.
for s in series:
    r = s["r"]
    late = r[r["t"] >= 0.5 * r["t"].max()]
    err = late["err"].max() if len(late) else r["err"][-1]
    peak = r["peak_elems"].max()
    d.loglog([peak / 1e3], [max(err, 1e-16)], s["mk"], color=s["c"], ms=11,
             mfc=s["c"] if s["maxdim"] == 64 else "white", mew=1.8, label=s["lbl"])
    d.annotate(f"  {s['lbl'].split(',')[0]}", (peak / 1e3, max(err, 1e-16)),
               fontsize=7, va="center")

d.set(xlabel="peak working set  [$10^3$ stored elements]",
      ylabel="worst error, $t \\geq t_{max}/2$",
      title="(d) resource efficiency: down and left is better")
d.grid(alpha=.3, which="both")

fig.suptitle("XX chain, $L=18$, U(1), domain wall, $t \\leq 15$ "
             "— CBE-BUG vs 2-site TDVP", fontsize=13)
fig.tight_layout(rect=(0, 0, 1, 0.97))
p = os.path.join(OUT, "xx_l18_campaign.png")
fig.savefig(p, dpi=150)
print("wrote", p)

# ── the numbers behind panel (d), printed so the claim is checkable ──────────────
print(f"\n{'method':34s} {'peak ws':>10s} {'worst err':>11s} {'peak chi':>9s} "
      f"{'wall s':>8s}")
print("-" * 78)
for s in sorted(series, key=lambda s: (s["maxdim"], s["scheme"], s["dex"])):
    r = s["r"]
    late = r[r["t"] >= 0.5 * r["t"].max()]
    err = late["err"].max() if len(late) else r["err"][-1]
    ws = r["peak_elems"].max()
    print(f"{s['lbl']:34s} {ws:10d} {err:11.3e} {r['maxbond'].max():9d} "
          f"{r['elapsed'][-1]:8.1f}")
