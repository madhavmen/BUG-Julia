#!/usr/bin/env python3
"""Figures for the two targeted L=18 studies.

    python3 benchmarks/plot_l18_special.py

  l18_square_open_protocol.png   the arXiv:2410.18468 protocol on `square_open`
  l18_backward_step.png          the backward-substep stability experiment

--- WHAT THE PROTOCOL FIGURE MAY AND MAY NOT BE READ AS ------------------------------------

Panel (a) plots `S_op` against `log_2(tJ)` because that is the axis on which arXiv:2410.18468's
late-time law `S_op = eta log_2(tJ) + S_0` is a STRAIGHT LINE. A straight segment here is the
signature; curvature means the window has not reached the asymptotic regime.

  ⛔ AND ON 18 SITES IT IS NOT EXPECTED TO. The paper is an INFINITE chain at chi up to 50000 out
     to tJ ~ 250; a finite open patch saturates instead. The fitted slope is printed so it can be
     compared BETWEEN gammas, never quoted as the paper's asymptotic eta.
  ⛔ AND THE DISSIPATOR IS NOT THE PAPER'S. The paper's jump operator is the two-site SU(2)
     exchange P_{i,i+1}; ours is single-site dephasing S^z_j, which the paper treats as the
     CONTRASTING case. Panel (a) is therefore the contrast, not a reproduction.

Panel (b) is this model's own published target -- ZENO SLOWING (arXiv:2408.10304): the
magnetisation relaxes MORE SLOWLY as dephasing rises, so the curve must go UP with gamma. The
decay itself is not the signature: for pure dephasing the dissipator contributes exactly zero to
d<S^z>/dt, so a decaying magnetisation is the Hamiltonian's doing, not the bath's.
"""
import csv
import math
from collections import defaultdict
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
RESULTS = HERE / "results"

ARM_STYLE = {
    "tdvp2":      dict(color="#1f4e79", marker="o", label="2-site TDVP"),
    "tdvp_cbe1s": dict(color="#b8860b", marker="s", label="1-site CBE-TDVP"),
    "cbe_bug":    dict(color="#a03030", marker="^", label="CBE-BUG"),
}


def rows(path):
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def f(r, k):
    v = r[k]
    return float("nan") if v in ("", "NaN", "nan") else float(v)


def fit_log2(ts, ys, frac=0.5):
    pts = [(math.log2(t), y) for t, y in zip(ts, ys) if t > 0 and t >= frac * max(ts)]
    if len(pts) < 3:
        return float("nan"), float("nan")
    n = len(pts)
    xm = sum(p[0] for p in pts) / n
    ym = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - xm) ** 2 for p in pts)
    if sxx <= 0:
        return float("nan"), float("nan")
    eta = sum((p[0] - xm) * (p[1] - ym) for p in pts) / sxx
    return eta, ym - eta * xm


def fit_lin(ts, ys, frac=0.5):
    pts = [(t, y) for t, y in zip(ts, ys) if t >= frac * max(ts)]
    if len(pts) < 3:
        return float("nan"), float("nan")
    n = len(pts)
    xm = sum(p[0] for p in pts) / n
    ym = sum(p[1] for p in pts) / n
    sxx = sum((p[0] - xm) ** 2 for p in pts)
    if sxx <= 0:
        return float("nan"), float("nan")
    a = sum((p[0] - xm) * (p[1] - ym) for p in pts) / sxx
    b = ym - a * xm
    res = max(abs(y - (a * x + b)) for x, y in pts)
    return a, res


def resid_log2(ts, ys, frac=0.5):
    eta, s0 = fit_log2(ts, ys, frac)
    if eta != eta:
        return float("nan")
    pts = [(math.log2(t), y) for t, y in zip(ts, ys) if t > 0 and t >= frac * max(ts)]
    return max(abs(y - (eta * x + s0)) for x, y in pts) if pts else float("nan")


def protocol_figure():
    path = RESULTS / "l18_square_open_protocol.csv"
    if not path.exists():
        print("skip: no l18_square_open_protocol.csv")
        return
    rs = rows(path)
    L = int(rs[0]["L"])
    cap = int(rs[0]["cap"])

    fig, axes = plt.subplots(2, 3, figsize=(16.8, 9.2))
    fig.suptitle(f"square_open, L = {L} (6x3 open patch), Neel start   —   is the "
                 f"dissipative operator-entanglement TRANSITION of arXiv:2410.18468 present?\n"
                 f"(guide, NOT a reproduction: the paper's dissipator is the two-site SU(2) "
                 f"exchange $P_{{i,i+1}}$ on an infinite chain at $\\chi$ up to 50000; "
                 f"ours is single-site dephasing $S^z_j$)", fontsize=11.5)
    (ax_lin, ax_log, ax_chi), (ax_supp, ax_zeno, ax_arms) = axes

    bug = [r for r in rs if r["arm"] == "cbe_bug"]
    by_g = defaultdict(list)
    for r in bug:
        by_g[f(r, "gamma")].append(r)
    gammas = sorted(by_g)
    cmap = plt.get_cmap("viridis")
    zx, zy, sup_g, sup_s, sup_ratio, sup_chi = [], [], [], [], [], []

    for i, g in enumerate(gammas):
        rr = sorted(by_g[g], key=lambda r: f(r, "t"))
        t = [f(r, "t") for r in rr]
        s = [f(r, "sop_log2") for r in rr]
        m = [f(r, "mstag") for r in rr]
        chi = [f(r, "maxbond") for r in rr]
        # gamma = 0 is the CONTROL and is drawn black and dashed so it never reads as one more
        # member of the dissipative family.
        ctrl = g == 0.0
        col = "k" if ctrl else cmap(i / max(1, len(gammas) - 1) * 0.85)
        kw = dict(marker="o", ms=3, lw=2.0 if ctrl else 1.4, color=col,
                  ls="--" if ctrl else "-")
        lab = r"$\gamma$ = 0  (UNITARY CONTROL)" if ctrl else rf"$\gamma$ = {g:g}"

        ax_lin.plot(t, s, label=lab, **kw)
        ax_log.plot([ti for ti in t if ti > 0], [si for ti, si in zip(t, s) if ti > 0],
                    label=lab, **kw)
        ax_chi.plot(t, chi, label=lab, **kw)

        rlog = resid_log2(t, s)
        _, rlin = fit_lin(t, s)
        sup_g.append(g)
        sup_s.append(s[-1])
        sup_ratio.append(rlin / rlog if rlog and rlog == rlog and rlog > 0 else float("nan"))
        sup_chi.append(chi[-1])
        if m and abs(m[0]) > 1e-12:
            zx.append(g)
            zy.append(m[-1] / m[0])

    ax_lin.set(xlabel="$tJ$", ylabel=r"$S_{\rm op}$ [bits]",
               title="(a) growth law: ballistic at $\\gamma$=0, suppressed once damped")
    ax_lin.legend(fontsize=7.5)
    ax_lin.grid(alpha=0.3)

    ax_log.set_xscale("log", base=2)
    ax_log.set(xlabel=r"$tJ$ (log$_2$)", ylabel=r"$S_{\rm op}$ [bits]",
               title="(b) same data on log$_2 t$: the paper's law is a STRAIGHT LINE here")
    ax_log.grid(alpha=0.3, which="both")

    ax_chi.axhline(cap, color="r", ls=":", lw=1.2, label=f"cap = {cap} (rank-limited above)")
    ax_chi.set(xlabel="$tJ$", ylabel=r"max bond dimension $\chi$",
               title="(c) INDEPENDENT confirmation: rank tracks $S_{\\rm op}$")
    ax_chi.legend(fontsize=7.5)
    ax_chi.grid(alpha=0.3)

    # (d) the quantitative transition: suppression + which fit wins, both against gamma.
    ax_supp.plot(sup_g, sup_s, marker="o", color="#1f4e79", lw=1.7, label=r"$S_{\rm op}(T)$")
    ax_supp.set(xlabel=r"$\gamma/J$", ylabel=r"$S_{\rm op}(T)$ [bits]",
                title="(d) suppression + fit preference vs $\\gamma$")
    ax_supp.grid(alpha=0.3)
    ax2 = ax_supp.twinx()
    ax2.plot(sup_g, sup_ratio, marker="s", color="#a03030", lw=1.7, ls="--",
             label=r"res$_{\rm lin}$/res$_{\rm log}$")
    ax2.axhline(1.0, color="#a03030", ls=":", lw=1)
    ax2.set_ylabel(r"res$_{\rm lin}$/res$_{\rm log}$   ($>1$: log fits better)",
                   color="#a03030")
    h1, l1 = ax_supp.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    ax_supp.legend(h1 + h2, l1 + l2, fontsize=8, loc="center right")
    falling = len(sup_s) >= 2 and all(b <= a + 1e-9 for a, b in zip(sup_s, sup_s[1:]))
    ax_supp.text(0.03, 0.06,
                 "S_op falls monotonically with gamma" if falling
                 else "S_op NOT monotonic in gamma — transition NOT clean at this L/window",
                 transform=ax_supp.transAxes, fontsize=8.5,
                 color="#206020" if falling else "#a03030", weight="bold")

    ax_zeno.plot(zx, zy, marker="o", color="#a03030", lw=1.6)
    ax_zeno.set(xlabel=r"dephasing rate $\gamma/J$",
                ylabel=r"$M_{\rm stag}(T)\,/\,M_{\rm stag}(0)$",
                title="(e) ZENO check (arXiv:2408.10304): must RISE with $\\gamma$")
    ax_zeno.grid(alpha=0.3)
    if len(zy) >= 2:
        # gamma = 0 is not part of the Zeno statement (no bath), so judge from the damped rows.
        damped = [y for g, y in zip(zx, zy) if g > 0]
        rising = len(damped) >= 2 and all(b >= a - 1e-9 for a, b in zip(damped, damped[1:]))
        ax_zeno.text(0.03, 0.06,
                     "rising across gamma > 0: ZENO TREND PRESENT" if rising
                     else "NOT monotonic — trend not resolved at this L/window",
                     transform=ax_zeno.transAxes, fontsize=8.5,
                     color="#206020" if rising else "#a03030", weight="bold")

    gref = min([g for g in gammas if g > 0], key=lambda g: abs(g - 0.10)) if any(
        g > 0 for g in gammas) else (gammas[0] if gammas else 0.0)
    for arm, st in ARM_STYLE.items():
        rr = sorted([r for r in rs if r["arm"] == arm and f(r, "gamma") == gref],
                    key=lambda r: f(r, "t"))
        if not rr:
            continue
        ax_arms.plot([f(r, "t") for r in rr], [f(r, "sop_log2") for r in rr],
                     marker=st["marker"], ms=3.5,
                     lw=4.0 if arm == "tdvp2" else 1.6,
                     ls="--" if arm == "tdvp_cbe1s" else "-",
                     color=st["color"], label=st["label"],
                     zorder=1 if arm == "tdvp2" else 3)
    ax_arms.set(xlabel="$tJ$", ylabel=r"$S_{\rm op}$ [bits]",
                title=f"(f) integrator agreement at $\\gamma$ = {gref:g}\n"
                      "(a)-(e) are PHYSICS only if these coincide")
    ax_arms.legend(fontsize=8)
    ax_arms.grid(alpha=0.3)

    fig.text(0.5, 0.005,
             "A good log_2(t) fit alone proves nothing — over a short window a straight line fits "
             "a logarithm well. The claim in (b)/(d) is COMPARATIVE: the log fit must beat the "
             "linear one and the margin must grow with gamma, while the gamma=0 control stays "
             "linear. Any chi at the cap makes that row's S_op a LOWER BOUND.",
             ha="center", fontsize=8.2, style="italic", color="#444444")
    fig.tight_layout(rect=(0, 0.022, 1, 0.9))
    out = RESULTS / "l18_square_open_protocol.png"
    fig.savefig(out, dpi=145)
    plt.close(fig)
    print("wrote", out.relative_to(HERE.parent))


def backward_figure():
    path = RESULTS / "l18_backward_step.csv"
    if not path.exists():
        print("skip: no l18_backward_step.csv")
        return
    rs = rows(path)
    models = []
    for r in rs:
        if r["model"] not in models:
            models.append(r["model"])

    fig, axes = plt.subplots(3, len(models), figsize=(5.4 * len(models), 11.5), squeeze=False)
    fig.suptitle("The backward substep: both TDVP variants evolve backwards in time "
                 "(tdvp2 one-site, tdvp_cbe1s zero-site);\nCBE-BUG has exactly one forward "
                 "evolution and no backward one. On a dissipative generator that is "
                 "anti-dissipative.", fontsize=12)

    for c, model in enumerate(models):
        mr = [r for r in rs if r["model"] == model]
        d = int(mr[0]["d"])
        closed = d == 2
        # Hoisted out of the arm loop: it is read after the loop, and an arm with no rows
        # `continue`s before it would be assigned.
        akey = "amp_phys" if "amp_phys" in mr[0] else "amp_max"
        for arm, st in ARM_STYLE.items():
            rr = sorted([r for r in mr if r["arm"] == arm], key=lambda r: f(r, "dt"))
            if not rr:
                continue
            dts = [f(r, "dt") for r in rr]
            kw = dict(marker=st["marker"], ms=5, lw=1.5, color=st["color"], label=st["label"])
            # ⛔ PLOT `amp_phys`, NEVER `amp_max`. `fused_lindblad` factors the zero-body term out
            # of the MPO, so the RAW ratio `amp_max` is inflated by `exp(+gamma*L*dt/4)` on EVERY
            # arm by construction and all three curves land on top of that shift instead of on the
            # physics -- measured, they agreed to SIX DIGITS including cbe_bug, which has no
            # backward step at all. `amp_phys` has the shift divided out and is the purity witness.
            axes[0][c].plot(dts, [f(r, akey) for r in rr], **kw)
            tr = [max(f(r, "trace_err"), 1e-18) for r in rr]
            axes[1][c].loglog(dts, tr, **kw)
            axes[2][c].loglog(dts, [f(r, "secs") for r in rr], **kw)
            for r in rr:
                if int(r["died_at"]) > 0:
                    axes[0][c].annotate("DIED", (f(r, "dt"), f(r, akey)),
                                        fontsize=8, color=st["color"], weight="bold")
        # The raw column, drawn once as a grey reference so the reader can see how far the shift
        # alone displaces it -- this is the curve an earlier version of this figure mistook for
        # the effect.
        if akey == "amp_phys":
            r2 = sorted([r for r in mr if r["arm"] == "cbe_bug"], key=lambda r: f(r, "dt"))
            if r2:
                axes[0][c].plot([f(r, "dt") for r in r2], [f(r, "amp_max") for r in r2],
                                color="#999999", ls=":", lw=1.4, marker="x", ms=4, zorder=0,
                                label="raw ratio (= the zero-body\nshift, NOT the physics)")

        title = f"{model}" + ("   (CLOSED CONTROL)" if closed else "   (dissipative, d = 4)")
        axes[0][c].axhline(1.0, color="k", ls="--", lw=1.2)
        axes[0][c].set(xlabel="$dt$",
                       ylabel=r"max$_k\ \|\psi_{k+1}\|/\|\psi_k\| \times e^{dt\,\rm shift}$",
                       title=f"{title}\npurity witness (shift divided out) — dashed line is the "
                             "EXACT bound")
        axes[0][c].legend(fontsize=8)
        axes[0][c].grid(alpha=0.3)
        axes[1][c].set(xlabel="$dt$", ylabel=r"max$_k\ |{\rm Tr}\,\rho - 1|$",
                       title="conservation-law violation" if not closed
                             else "(no trace law for a closed state)")
        axes[1][c].legend(fontsize=8)
        axes[1][c].grid(alpha=0.3, which="both")
        axes[2][c].set(xlabel="$dt$", ylabel="total step time [s]",
                       title="cost of the same physical time")
        axes[2][c].legend(fontsize=8)
        axes[2][c].grid(alpha=0.3, which="both")

    fig.text(0.5, 0.005,
             "Amplification above 1 is a PROVABLE violation, not a tolerance question: dephasing "
             "is a unital channel, so Tr(rho^2) = |||rho>>||^2 cannot increase — but only once the "
             "factored-out zero-body shift exp(+gamma*L*dt/4) is divided out (grey dotted = the raw "
             "ratio, which is that shift and nothing else). The closed control column is the "
             "discriminator: the mechanism predicts NO effect there, because for tau = -i*dt the "
             "backward step is unitary. MEASURED at gamma=0.1, L=12 the corrected witness is a NULL "
             "RESULT — no arm violates the bound, so the separation there is in the trace row, not "
             "this one.",
             ha="center", fontsize=8.5, style="italic", color="#444444")
    fig.tight_layout(rect=(0, 0.02, 1, 0.94))
    out = RESULTS / "l18_backward_step.png"
    fig.savefig(out, dpi=145)
    plt.close(fig)
    print("wrote", out.relative_to(HERE.parent))


if __name__ == "__main__":
    protocol_figure()
    backward_figure()
