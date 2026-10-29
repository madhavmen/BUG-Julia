"""benchmarks/xx_efficiency/plot_figs.py -- plan Figs. 1-4 from the drivers' raw JSON.

usage: python plot_figs.py RUNDIR OUTDIR
RUNDIR holds fig1_dd.json / fig2_krylov.json / fig3_recycle.json / fig4_endtoend.json (any subset).
"""
import json
import os
import statistics as st
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

# reference categorical palette (dataviz references/palette.md), fixed order
C1, C2, C3, C4, C5 = "#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e4e3df"
plt.rcParams.update({"font.size": 9, "axes.edgecolor": MUTED, "axes.labelcolor": INK,
                     "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True,
                     "grid.color": GRID, "grid.linewidth": 0.6, "axes.spines.top": False,
                     "axes.spines.right": False, "legend.frameon": False,
                     "lines.linewidth": 2, "lines.markersize": 6})

RUN, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)


def load(name):
    p = os.path.join(RUN, name)
    return json.load(open(p)) if os.path.exists(p) else None


def save(fig, name):
    fig.tight_layout()
    for ext in ("pdf", "png"):
        fig.savefig(os.path.join(OUT, f"{name}.{ext}"), dpi=200)
    plt.close(fig)
    print("wrote", name)


def med(xs):
    return st.median(xs) if xs else float("nan")


# ── Fig. 1 ─────────────────────────────────────────────────────────────────────
f1 = load("fig1_dd.json")
if f1:
    rows = f1["rows"]
    cases = [("t4.0_r16_b8", "active: J t = 4, rank 16, bond 8"),
             ("t4.0_r8_b5", "quiet: J t = 4, rank 8, bond 5")]
    tols = (1e-3, 1e-5, 1e-7)
    arms = [("adaptive", "adaptive (production w0)", C1, "o"),
            ("adaptive_b4", "adaptive (w0 = 4, blocks of 4)", C3, "s"),
            ("fixed_p4", "fixed  k + 4", C2, "v"),
            ("fixed_p8", "fixed  k + 8", C4, "^"),
            ("fixed_p16", "fixed  k + 16", C5, "D")]
    fig, axs = plt.subplots(2, 3, figsize=(10, 6), sharey="row")
    for r, (case, title) in enumerate(cases):
        for c, tol in enumerate(tols):
            ax = axs[r][c]
            sel = [x for x in rows if x["figure"] == "1A" and x["case"] == case and x["tol"] == tol]
            if not sel:
                continue
            for arm, lab, col, mk in arms:
                pts = [x for x in sel if x["arm"] == arm]
                ax.scatter([x["work"] for x in pts], [max(x["error"], 1e-17) for x in pts],
                           marker=mk, s=28, label=lab, edgecolors=col, linewidths=1.2,
                           facecolors=[col if x["pass"] else "none" for x in pts])
            tgt = med([x["target"] for x in sel])
            ax.axhline(tgt, color=MUTED, lw=1, ls="--")
            ax.text(0.02, tgt, " target", color=MUTED, fontsize=7,
                    va="bottom", transform=ax.get_yaxis_transform())
            ax.set_yscale("log")
            ax.set_title(f"{title}\ntol_basis = {tol:.0e}", fontsize=8)
            if r == 1:
                ax.set_xlabel("streamed MPO-channel columns (incl. holdout)")
            if c == 0:
                ax.set_ylabel(r"audited $\rho_{DD}$")
    axs[0][0].legend(fontsize=7, loc="lower left")
    fig.suptitle("Fig. 1A  adaptive stopping vs one-shot fixed oversampling (same DD + QR); "
                 "hollow = a seed/side missed the target", fontsize=9)
    save(fig, "figS1a_adaptive_vs_fixed")

    fig, axs = plt.subplots(1, 3, figsize=(10, 3.4), sharey=True)
    for ax, (case, title) in zip(axs, [("t4.0_r16_b8", "J t = 4, rank 16, bond 8"),
                                       ("t4.0_r8_b8", "J t = 4, rank 8, bond 8"),
                                       ("t4.0_r8_b5", "J t = 4, rank 8, bond 5")]):
        for arm, lab, col, mk in (("DD", r"DD  $P_L^D X P_R^D$", C1, "o"),
                                  ("one_sided", r"one-sided  $P_L^D X$", C2, "s")):
            g = defaultdict(list)
            for x in rows:
                if x["figure"] == "1B" and x["case"] == case and x["arm"] == arm:
                    g[x["param"]].append(x)
            ws = sorted(g, key=lambda p: int(p.split("=")[1]))
            ax.plot([med([x["columns"] for x in g[w]]) for w in ws],
                    [max(med([x["error"] for x in g[w]]), 1e-16) for w in ws],
                    color=col, marker=mk, label=lab)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=8)
        ax.set_xlabel("selected new bond columns (median)")
    axs[0].set_ylabel(r"audited $\rho_{DD}$ (median)")
    axs[0].legend(fontsize=7)
    fig.suptitle("Fig. 1B  DD vs one-sided projection at equal Gaussian width (2, 4, 8, 16), same QR",
                 fontsize=9)
    save(fig, "figS1b_dd_vs_one_sided")

    # plan Fig. 1 (two panels): the active bond at the one tolerance that binds there; the
    # looser targets lie above rho_DD = 1 and the quiet bonds are trivial (supplementary S1a/S1b)
    case, tol = "t4.0_r16_b8", 1e-7
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    sel = [x for x in rows if x["figure"] == "1A" and x["case"] == case and x["tol"] == tol]
    for arm, lab, col, mk in arms:
        pts = [x for x in sel if x["arm"] == arm]
        a.scatter([x["work"] for x in pts], [max(x["error"], 1e-17) for x in pts],
                  marker=mk, s=28, label=lab, edgecolors=col, linewidths=1.2,
                  facecolors=[col if x["pass"] else "none" for x in pts])
    tgt = med([x["target"] for x in sel])
    a.axhline(tgt, color=MUTED, lw=1, ls="--")
    a.text(0.02, tgt, " target", color=MUTED, fontsize=7, va="bottom",
           transform=a.get_yaxis_transform())
    a.set_yscale("log")
    a.set_xlabel("streamed MPO-channel columns (incl. holdout)")
    a.set_ylabel(r"audited $\rho_{DD}$")
    a.set_title(f"A  adaptive vs fixed oversampling, tol_basis = {tol:.0e}; hollow = missed",
                fontsize=8)
    a.legend(fontsize=7, loc="center left")
    for arm, lab, col, mk in (("DD", r"DD  $P_L^D X P_R^D$", C1, "o"),
                              ("one_sided", r"one-sided  $P_L^D X$", C2, "s")):
        g = defaultdict(list)
        for x in rows:
            if x["figure"] == "1B" and x["case"] == case and x["arm"] == arm:
                g[x["param"]].append(x)
        ws = sorted(g, key=lambda p: int(p.split("=")[1]))
        b.plot([med([x["columns"] for x in g[w]]) for w in ws],
               [max(med([x["error"] for x in g[w]]), 1e-16) for w in ws],
               color=col, marker=mk, label=lab)
    b.set_yscale("log")
    b.set_xlabel("selected new bond columns (median)")
    b.set_ylabel(r"audited $\rho_{DD}$ (median)")
    b.set_title("B  DD vs one-sided at equal Gaussian width, same QR", fontsize=8)
    b.legend(fontsize=7)
    fig.suptitle("Fig. 1  active XX bond (J t = 4, rank 16, bond 8)", fontsize=9)
    save(fig, "fig1_rsvd_dd")

# ── Fig. 2 ─────────────────────────────────────────────────────────────────────
f2 = load("fig2_krylov.json")
if f2:
    rows = f2["rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for arm, lab, col, mk in (("adaptive", "time-weighted stop", C1, "o"),
                              ("fixed_m8", "fixed m = 8", C2, "v"),
                              ("fixed_m16", "fixed m = 16", C4, "^"),
                              ("fixed_m32", "fixed m = 32", C5, "D")):
        pts = [x for x in rows if x["arm"] == arm]
        a.scatter([x["work"] for x in pts], [max(x["error"] / x["target"], 1e-14) for x in pts],
                  color=col, marker=mk, s=24, label=lab, alpha=0.85)
    a.axhline(1.0, color=MUTED, lw=1, ls="--")
    a.text(0.98, 1.0, "allowance ", color=MUTED, fontsize=7, ha="right", va="bottom",
           transform=a.get_yaxis_transform())
    a.set_xscale("log", base=2)
    a.set_yscale("log")
    a.set_xlabel("new one-site H applications")
    a.set_ylabel(r"independent $\epsilon_K$ / allowance")
    a.set_title("A  every (snapshot, side, h, tol_basis) on a frozen CBE basis", fontsize=8)
    a.legend(fontsize=7)
    hs = sorted({x["h"] for x in rows})
    for tol, col, mk in ((1e-3, C1, "o"), (1e-5, C3, "s"), (1e-7, C2, "^")):
        ys = [med([x["work"] for x in rows if x["arm"] == "adaptive" and x["h"] == h
                   and x["tol"] == tol]) for h in hs]
        b.plot(hs, ys, color=col, marker=mk, label=f"adaptive, tol_basis {tol:.0e}")
    for m in (8, 16, 32):
        npass = sum(x["pass"] for x in rows if x["arm"] == f"fixed_m{m}")
        ntot = sum(1 for x in rows if x["arm"] == f"fixed_m{m}")
        b.axhline(m, color=MUTED, lw=1, ls=":")
        b.text(hs[-1], m, f" m = {m} (passes {npass}/{ntot})", color=MUTED, fontsize=7,
               va="bottom", ha="right")
    b.set_yscale("log", base=2)
    b.set_xlabel("step h")
    b.set_ylabel("H applications (median over cases, sides)")
    b.set_title("B  applications selected by the stop vs fixed depths", fontsize=8)
    b.legend(fontsize=7, loc="center left")
    fig.suptitle("Fig. 2  time-weighted Krylov stopping vs fixed depth (no recycling)", fontsize=9)
    save(fig, "fig2_krylov_stopping")

# ── Fig. 3 ─────────────────────────────────────────────────────────────────────
f3 = load("fig3_recycle.json")
if f3:
    rows = f3["rows"]
    cases = sorted({x["case"] for x in rows})
    fig, axs = plt.subplots(1, 2, figsize=(10, 3.6))
    wbar = 0.38
    a, b = axs
    labels, F, R = [], [], []
    for case in cases:
        for rd in (1, 2, 3):
            labels.append(f"{case.replace('t4.0_', '')}\nround {rd}")
            for arm, acc in (("F", F), ("R", R)):
                acc.append(sum(x["work"] for x in rows if x["case"] == case and x["arm"] == arm
                               and x["param"].endswith(f"round={rd}")))
    xs = range(len(labels))
    a.bar([x - wbar / 2 for x in xs], F, wbar, color=C1, label="F: fresh Krylov")
    a.bar([x + wbar / 2 for x in xs], R, wbar, color=C2, label="R: recycled (refreshed images)")
    a.set_xticks(list(xs))
    a.set_xticklabels(labels, fontsize=7)
    a.set_ylabel("new one-site H applications (3 steps)")
    a.set_title("A  H applications per order-3 round", fontsize=8)
    a.legend(fontsize=7, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2)
    for case, ls in zip(cases, ("-", "--")):
        for arm, col in (("F", C1), ("R", C2)):
            ts = []
            for rd in (1, 2, 3):
                ts.append(sum(float(x["time_s"]) for x in rows if x["case"] == case
                              and x["arm"] == arm and x["param"].endswith(f"round={rd}")
                              and not x["param"].startswith("step=1,")))
            cum = [sum(ts[:i + 1]) for i in range(3)]
            b.plot([1, 2, 3], cum, color=col, ls=ls, marker="o",
                   label=f"{arm}, {case.replace('t4.0_', '')}")
    b.set_xticks([1, 2, 3])
    b.set_xlabel("round")
    b.set_ylabel("cumulative local-enrichment s (steps 2-3)")
    b.set_title("B  cumulative time -- PROVISIONAL (not the timing job; step 1 = JIT, dropped)",
                fontsize=8)
    b.legend(fontsize=7)
    fig.suptitle("Fig. 3  Krylov recycling (R) vs fresh (F), same adaptive DD-CBE and stop rule",
                 fontsize=9)
    save(fig, "fig3_krylov_recycling")

# ── Fig. 4 ─────────────────────────────────────────────────────────────────────
f4 = load("fig4_endtoend.json")
if f4:
    rows = f4["rows"]
    fig, (a, b) = plt.subplots(1, 2, figsize=(10, 3.8))
    for arm, lab, col, mk in (("full", "full", C1, "o"), ("norecycle", "no recycling", C3, "s"),
                              ("conventional", "conventional controls", C2, "^")):
        pts = sorted([x for x in rows if x["arm"] == arm], key=lambda x: x["dt"])
        a.plot([x["wall_s"] for x in pts], [x["eps_z"] for x in pts], color=col, marker=mk,
               label=lab)
        for x in pts:
            if arm == "full":
                a.annotate(f"dt {x['dt']}", (x["wall_s"], x["eps_z"]), fontsize=6, color=MUTED,
                           xytext=(4, 2), textcoords="offset points")
    a.set_xscale("log")
    a.set_yscale("log")
    a.set_xlabel("total wall time [s] (median of warmed repetitions)")
    a.set_ylabel(r"$\epsilon_z(T)$, exact RMS magnetisation error")
    a.set_title("A  L = 16 domain wall to J T = 2", fontsize=8)
    a.legend(fontsize=7)
    full = sorted([x for x in rows if x["arm"] == "full"], key=lambda x: x["dt"])
    dts = [x["dt"] for x in full]
    errs = [x["err_phase"] if x["err_phase"] > 0 else x["err"] for x in full]
    b.plot(dts, errs, color=C1, marker="o", label="full: exact state error (phase-aligned)")
    for p, ls in ((3, ":"), (4, "--")):
        ref = [errs[-1] * (d / dts[-1]) ** p for d in dts]
        b.plot(dts, ref, color=MUTED, lw=1, ls=ls, label=f"$h^{p}$ reference")
    b.set_xscale("log")
    b.set_yscale("log")
    # label only the dt values run; log minor labels collide
    b.set_xticks(dts, [f"{d:g}" for d in dts])
    b.xaxis.set_minor_formatter(plt.NullFormatter())
    b.set_xlabel("dt")
    b.set_ylabel(r"$\|\psi - \psi_{ex}(T)\|$")
    b.set_title("B  dt convergence of the full method", fontsize=8)
    b.legend(fontsize=7)
    fig.suptitle("Fig. 4  complete order-3 BUG solver, three settings", fontsize=9)
    save(fig, "fig4_total_solver")
