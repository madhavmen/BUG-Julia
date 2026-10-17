"""Figures for BUG-Julia/AGENTS_HANDOVER_5.md, one or more per completed run.

Usage: python make_h5_figs.py DATA_DIR OUT_DIR
DATA_DIR holds the cluster CSVs/logs (same relative paths as under $SCRATCH/rsvd_adaptive).
"""
import csv
import math
import os
import re
import sys
from collections import defaultdict

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

DATA, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)

# Reference palette (dataviz skill, references/palette.md), light mode, fixed slot order.
S = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
SURF, INK, INK2, MUTED, GRID, AXIS = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e1e0d9", "#c3c2b7"
plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "font.family": ["Segoe UI", "DejaVu Sans", "sans-serif"], "font.size": 10,
    "axes.edgecolor": AXIS, "axes.labelcolor": INK2, "axes.titlecolor": INK,
    "axes.titlesize": 11, "axes.titleweight": "semibold", "axes.titlelocation": "left",
    "xtick.color": MUTED, "ytick.color": MUTED, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.6, "axes.axisbelow": True, "axes.spines.top": False,
    "axes.spines.right": False, "lines.linewidth": 2, "lines.markersize": 6,
    "legend.frameon": False, "legend.labelcolor": INK2,
})


def p(*a):
    return os.path.join(DATA, *a)


def rows(path):
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except (TypeError, ValueError):
        return math.nan


def label(r):
    if r["arm"].startswith("tdvp"):
        return r["arm"]
    tb = float(r["tol_basis"])
    return f"bug2 {tb:.0e}".replace("e-0", "e-") + (" +fold" if r["fold"] == "true" else "")


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print("wrote", name)


def steps_by_label(path):
    out = defaultdict(list)
    for r in rows(path):
        if r["step"] == "0":
            continue
        out[label(r)].append((int(r["step"]), fnum(r["wall"]), r))
    return out


# ── fig 1: kscreen, mean s/step per variant at D = 512 ───────────────────────────────────
k1 = steps_by_label(p("2026-10-16-kscr", "bp_cost_10x4.csv"))
order = ["tdvp2e", "bug2 1e-3", "bug2 1e-3 +fold", "bug2 3e-3", "bug2 3e-3 +fold",
         "bug2 1e-2", "bug2 1e-2 +fold"]
means = {k: sum(w for _, w, _ in v) / len(v) for k, v in k1.items()}
fig, ax = plt.subplots(figsize=(7, 3.6))
y = list(range(len(order)))[::-1]
for yi, k in zip(y, order):
    c = S[1] if k == "tdvp2e" else S[0]
    ax.barh(yi, means[k], height=0.55, color=c)
    ratio = "" if k == "tdvp2e" else f"  {means['tdvp2e'] / means[k]:.2f}x"
    ax.text(means[k] + 1.5, yi, f"{means[k]:.1f} s{ratio}", va="center", color=INK2, fontsize=9,
            bbox=dict(facecolor=SURF, edgecolor="none", pad=1), zorder=4)
ax.axvline(means["tdvp2e"], color=S[1], lw=1, ls="--")
ax.set_yticks(y, order)
ax.set_xlabel("mean wall per step (s), 3 steps")
ax.set_xlim(0, max(means.values()) * 1.25)
ax.grid(axis="y", visible=False)
ax.set_title("kscreen 17295644: bug2 knobs vs tdvp2e, 10x4 D=512, dt 0.1 (ratio = tdvp2e/bug2)")
save(fig, "fig1_kscreen_d512.png")

# ── fig 2: kscreen2, per-step wall vs step ───────────────────────────────────────────────
k2path = p("2026-10-16-kscr2", "bp_cost_10x4.csv")
k2 = steps_by_label(k2path)
o2 = ["tdvp2e", "bug2 1e-2 +fold", "bug2 2e-2 +fold", "bug2 3e-2 +fold", "bug2 5e-2 +fold"]
fig, ax = plt.subplots(figsize=(7, 3.8))
for i, k in enumerate(o2):
    st = sorted(k2[k])
    ax.plot([s for s, _, _ in st], [w for _, w, _ in st], color=S[i], marker="o", label=k)
ax.set_xlabel("step (dt 0.1)")
ax.set_ylabel("wall per step (s)")
ax.set_xlim(0.5, 10.5)
ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), fontsize=8, ncol=3)
ax.set_title("kscreen2 17299494: per-step wall, 10x4 D=512 (bug2 grows as the state entangles)")
save(fig, "fig2_kscreen2_walls.png")


# ── fig 3/4: kscreen2 G(x,t) error vs the D = 1024 reference ─────────────────────────────
def gload(path):
    g = defaultdict(dict)
    for r in rows(path):
        g[round(float(r["t"]), 4)][int(r["site"])] = complex(float(r["ReG"]), float(r["ImG"]))
    return g


ra = gload(p("2026-10-16-ref1024", "tdvp2e", "bp_G_10x4_tdvp2e_D1024.csv"))
rb = gload(p("2026-10-16-ref1024", "bug2", "bp_G_10x4_bug2_thin_tb1e-03_D1024.csv"))
tref = sorted(set(ra) & set(rb))
ref = {t: {x: (ra[t][x] + rb[t][x]) / 2 for x in ra[t]} for t in tref}
unc = {t: max(abs(ra[t][x] - rb[t][x]) for x in ra[t]) / 2 for t in tref}
gfiles = {"tdvp2e": "bp_G_10x4_tdvp2e_D512.csv"}
for tb in ["1e-02", "2e-02", "3e-02", "5e-02"]:
    gfiles[f"bug2 {tb.replace('e-0', 'e-')} +fold"] = f"bp_G_10x4_bug2_thin_tb{tb}_fold_D512.csv"
errs = {}
for k, f in gfiles.items():
    g = gload(p("2026-10-16-kscr2", f))
    ts = [t for t in sorted(g) if t in ref and t > 0]
    errs[k] = [(t, max(abs(g[t][x] - ref[t][x]) for x in g[t])) for t in ts]
fig, ax = plt.subplots(figsize=(7, 3.8))
tu = [t for t in tref if 0 < t <= 1.0001]
ax.fill_between(tu, [1e-9] * len(tu), [3 * unc[t] for t in tu], color=GRID, lw=0,
                label="3x reference uncertainty (unresolved below)")
for i, k in enumerate(o2):
    ax.plot([t for t, _ in errs[k]], [e for _, e in errs[k]], color=S[i], marker="o", label=k)
ax.set_yscale("log")
ax.set_ylim(3e-7, 2e-3)
ax.set_xlabel("t")
ax.set_ylabel("max_x |G - G_ref|  (G(c,0) = 1/4)")
ax.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=3)
ax.set_title("kscreen2 graded: error vs D=1024 reference (mean of ref1024t, ref1024b)")
save(fig, "fig3_kscreen2_error_vs_ref.png")

cost = {k: sum(w for _, w, _ in v) / (len(v) * 0.1) for k, v in k2.items()}
fig, ax = plt.subplots(figsize=(6.2, 3.8))
for i, k in enumerate(o2):
    e1 = dict(errs[k])[1.0]
    ax.scatter(e1, cost[k], s=60, color=S[1] if k == "tdvp2e" else S[0], edgecolor=SURF,
               linewidth=2, zorder=3)
    off = {"bug2 1e-2 +fold": (-10, -14), "bug2 3e-2 +fold": (-10, 8),
           "bug2 5e-2 +fold": (-10, -14)}.get(k, (6, 4))
    ax.annotate(k, (e1, cost[k]), xytext=off, textcoords="offset points", fontsize=8,
                color=INK2, ha="right" if off[0] < 0 else "left")
ax.axvline(3 * unc[1.0], color=MUTED, lw=1, ls=":")
ax.text(3 * unc[1.0] * 1.1, min(cost.values()) * 0.98, "3x ref. unc.", color=MUTED, fontsize=8)
ax.set_xscale("log")
ax.set_xlabel("error at t = 1 (max_x |G - G_ref|)")
ax.set_ylabel("wall per unit time (s)")
ax.set_title("kscreen2: cost vs error at t=1 -- bug2 speedups are not at matched error")
save(fig, "fig4_kscreen2_cost_vs_error.png")

# ── fig 5: closure growth per step (kscreen2), two panels ────────────────────────────────
raug = defaultdict(list)
cur = None
with open(p("2026-10-16-kscr2", "kscreen2_17299494.out")) as f:
    for line in f:
        m = re.match(r"\s+(bug2/thin/tb(\S+?)/fold) D= 512 step (\d+)", line)
        if m:
            cur = f"bug2 {float(m.group(2)):.0e}".replace("e-0", "e-") + " +fold"
            continue
        m = re.search(r"closure: order 2\s+passes (\d+).*r_aug (\d+)", line)
        if m and cur:
            raug[cur].append(int(m.group(2)))
            cur = None
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10, 3.6))
for i, k in enumerate(o2[1:], start=1):
    st = sorted(k2[k])
    a1.plot([s for s, _, _ in st], [fnum(r["t_closure"]) for _, _, r in st], color=S[i],
            marker="o", label=k)
    a2.plot(range(1, len(raug[k]) + 1), raug[k], color=S[i], marker="o", label=k)
a1.set_xlabel("step")
a1.set_ylabel("closure time per step (s, CPU-phase clock)")
a1.set_title("a) closure time per step")
a2.axhline(512, color=MUTED, lw=1, ls=":")
a2.text(10, 516, "chi = 512", color=MUTED, fontsize=8, ha="right")
a2.set_xlabel("step")
a2.set_ylabel("frame width after closure (r_aug)")
a2.set_title("b) closure frame width")
a2.legend(fontsize=8)
fig.suptitle("kscreen2: bug2 order-2 closure, 10x4 D=512 -- j=1 fails its share at pass 0 "
             "every step", x=0.01, y=1.04, ha="left", fontsize=11, color=INK)
save(fig, "fig5_closure_growth.png")

# ── fig 6: abbd A/B ──────────────────────────────────────────────────────────────────────
ab = {}
for proc, tag in [("p0", "baseline"), ("p1", "bond tasks on"), ("p2", "heap hint 14G")]:
    for k, v in steps_by_label(p("2026-10-16-abbd", proc, "bp_cost_10x4.csv")).items():
        ab[(k.split()[0], tag)] = (sum(w for _, w, _ in v) / len(v),
                                   sum(fnum(r["gc_s"]) for _, _, r in v) / sum(w for _, w, _ in v))
fig, ax = plt.subplots(figsize=(7, 3.4))
keys = [("tdvp2e", "baseline"), ("tdvp2e", "heap hint 14G"), ("bug2", "baseline"),
        ("bug2", "bond tasks on"), ("bug2", "heap hint 14G")]
for i, kk in enumerate(keys):
    w, gc = ab[kk]
    ax.bar(i, w, width=0.6, color=S[1] if kk[0] == "tdvp2e" else S[0])
    ax.text(i, w + 2, f"{w:.1f} s\nGC {100 * gc:.0f}%", ha="center", color=INK2, fontsize=8)
ax.set_xticks(range(len(keys)), [f"{a}\n{b}" for a, b in keys], fontsize=8)
ax.set_ylabel("mean wall per step (s)")
ax.set_ylim(0, 215)
ax.grid(axis="x", visible=False)
ax.set_title("abbd 17287169: parallel plan (b) bond tasks, (d) heap hint -- no gain (D=512, 4 CPU)")
save(fig, "fig6_abbd_ab.png")

# ── fig 7: reference runs and the cancelled t10pair ──────────────────────────────────────
rt = [r for r in rows(p("2026-10-16-ref1024", "tdvp2e", "bp_cost_10x4.csv")) if r["step"] != "0"]
rbq = [r for r in rows(p("2026-10-16-ref1024", "bug2", "bp_cost_10x4.csv")) if r["step"] != "0"]
t10 = [r for r in rows(p("2026-10-16-t10", "bp_cost_10x4.csv")) if r["step"] != "0"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
tt = lambda rs: [int(r["step"]) * float(r["dt"]) for r in rs]  # noqa: E731
a1.plot(tt(rt), [fnum(r["wall"]) for r in rt], color=S[1], label="ref1024t tdvp2e D=1024")
a1.plot(tt(rbq), [fnum(r["wall"]) for r in rbq], color=S[0], marker="o",
        label="ref1024b bug2 1e-3 D=1024 (OOM at t=1.7)")
a1.plot(tt(rbq), [fnum(r["t_closure"]) for r in rbq], color=S[0], ls="--",
        label="  of which closure")
a1.set_xlabel("t")
a1.set_ylabel("wall per step (s)")
a1.set_title("a) D=1024 reference cost per step")
a1.legend(fontsize=8, loc="center right")
a2.plot(tt(rt), [fnum(r["discarded"]) for r in rt], color=S[1], label="tdvp2e D=1024 (ref1024t)")
a2.plot(tt(t10), [fnum(r["discarded"]) for r in t10], color=S[2],
        label="tdvp2e D=256 (t10pair, cancelled)")
a2.set_yscale("log")
a2.set_xlabel("t")
a2.set_ylabel("discarded norm per step")
a2.set_title("b) truncation per step grows with t")
a2.legend(fontsize=8, loc="lower right")
save(fig, "fig7_references.png")

# ── fig 8: validation vs exact (24-site profiles + 4x4 smoke), dot plot on log x ─────────
sets = [("6x4 square", p("2026-10-16-val24", "prof", "square", "fig45_cost_square_6x4_D512.csv")),
        ("6x4 J1-J2 0.5", p("2026-10-16-val24", "prof", "j1j2",
                            "fig45_cost_square_6x4_J20p500_D512.csv")),
        ("kagome 2x4 (24)", p("2026-10-16-val24", "prof", "kagome",
                              "fig45_cost_kagome_2x4_D512.csv"))]
arms = ["tdvp2", "cbe1s", "bug1", "bugmid", "bug2", "bug4"]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8), gridspec_kw={"width_ratios": [3, 2]})
for i, (name, path) in enumerate(sets):
    d = {r["arm"]: fnum(r["gerr_max"]) for r in rows(path)}
    ys = [len(arms) - j + (0.18 * (1 - i)) for j in range(len(arms))]
    a1.scatter([d[a] for a in arms], ys, s=40, color=S[i], edgecolor=SURF, linewidth=1.5,
               label=name, zorder=3)
a1.set_yticks([len(arms) - j for j in range(len(arms))], arms)
a1.set_xscale("log")
a1.set_xlabel("max |G - G_exact| to T=0.4  (D=512, old tol 1e-5/1e-4/1e-4)")
a1.set_title("a) 24-site validation profiles vs exact")
a1.legend(fontsize=8, loc="upper center", bbox_to_anchor=(0.5, -0.2), ncol=3)
sm = {r["arm"]: fnum(r["gerr_max"]) for r in rows(p("2026-10-16-s22", "sq",
                                                     "fig45_cost_square_4x4_D32.csv"))}
sa = ["tdvp2", "tdvp2e", "cbe1s", "cbe1se", "bug2"]
a2.scatter([sm[a] for a in sa], range(len(sa), 0, -1), s=40, color=S[0], edgecolor=SURF,
           linewidth=1.5, zorder=3)
a2.set_yticks(range(len(sa), 0, -1), sa)
a2.set_xscale("log")
a2.set_xlabel("max |G - G_exact| to T=0.5 (4x4, D=32)")
a2.set_title("b) s22 smoke (driver changes)")
save(fig, "fig8_validation_vs_exact.png")

# ── fig 9: order ladders vs exact ────────────────────────────────────────────────────────
fig, (a1, a2) = plt.subplots(1, 2, figsize=(10.5, 3.8))
for ax, path, title in [
        (a1, p("2026-10-16-order", "order_ladder_6x4.csv"),
         "a) 6x4, D=512 (binding cap) -- bug2/bug4 rungs running"),
        (a2, p("2026-10-16-order", "smoke", "order_ladder_4x3.csv"), "b) 4x3 smoke, D=128 (full rank)")]:
    by = defaultdict(list)
    for r in rows(path):
        by[r["arm"]].append((float(r["dt"]), float(r["err_sz"])))
    for i, a in enumerate(["tdvp2", "bug1", "bugmid", "bug2", "bug4"]):
        if a in by:
            v = sorted(by[a])
            ax.plot([x for x, _ in v], [e for _, e in v], color=S[i], marker="o", label=a)
    ax.set_xscale("log")
    ax.set_yscale("log")
    dts = sorted({x for v in by.values() for x, _ in v})
    ax.set_xticks(dts, [f"{x:g}" for x in dts])
    ax.minorticks_off()
    ax.set_xlabel("dt")
    ax.set_ylabel("max_j |<S^z_j>(T) - exact|")
    ax.set_title(title)
    ax.legend(fontsize=8)
save(fig, "fig9_order_ladders.png")
