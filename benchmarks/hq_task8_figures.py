"""HQ plan Task 8: figures P01--P13 from `benchmarks/hq_bench.jl` runs.

usage:  python benchmarks/hq_task8_figures.py RUNDIR[,RUNDIR...] OUTDIR [S3REF.json] [--figs=P12,...]

RUNDIR holds the `hq_bench.jl` JSON files (any number, any settings); `--figs` draws only the named
figures, so short twin scans can feed P12 without entering P01-P04's finest-run choice. S3REF is the spin-3 reference
band from `hq_xx_reference_band.jl` (keys t, m, j, band_m, band_j). A figure whose runs are absent
is skipped with a line saying which settings it needs, never drawn from a subset that looks whole.

Errors against the reference are taken only at times the reference holds. For s = 1/2 they are the
run's own `err_exact_*` (closed form, `tests/common/free_fermion.jl`); for s = 3 they are measured
against S3REF, and the band is the error FLOOR: a spin-3 error below it is not a measurement.

Each run's error floor (P08, P09) is the sum over its steps of what the step itself says it did not
resolve: `tol_basis` + root `expv_err` + compression `discarded` + `proj_err`, plus the reference
band. An order is read off `log2(e(h)/e(h/2))` only where both errors are 10x above their floors.
"""

import glob
import json
import os
import sys
from collections import defaultdict

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

MARGIN = 10.0


def load_runs(rundir):
    runs = []
    for p in sorted(glob.glob(os.path.join(rundir, "*.json"))):
        with open(p) as f:
            d = json.load(f)
        if "steps" not in d or "records" not in d:
            continue
        d["path"] = os.path.basename(p)
        runs.append(d)
    return runs


def spin(run):
    return run["config"]["twoS"] / 2


def key(run, *names):
    return tuple(run["config"][n] for n in names)


def exact_sz(L, t):
    """s = 1/2: `<S^z_i(t)>` from plan T7.1's closed-form open-chain propagator, sites 1..L/2 filled."""
    k = np.pi * np.arange(1, L + 1) / (L + 1)
    S = np.sqrt(2 / (L + 1)) * np.sin(np.outer(np.arange(1, L + 1), k))
    u = S @ np.diag(np.exp(-1j * t * np.cos(k))) @ S.T
    C = u @ np.diag([1.0 if i < L // 2 else 0.0 for i in range(L)]) @ u.conj().T
    return np.real(np.diag(C)) - 0.5


def ref_at(ref, t):
    if ref is None:
        return None
    for k, tr in enumerate(ref["t"]):
        if abs(tr - t) < 1e-9:
            return np.array(ref["m"][k]), ref["j"][k], ref["band_m"][k], ref["band_j"][k]
    return None


def errors(run, ref):
    """Per record: (t, max|dm|, L2 dm, |dj|, ref floor) against the run's reference."""
    out = []
    for r in run["records"]:
        if run["config"]["twoS"] == 1:
            out.append((r["t"], r["err_exact_m"], r["err_exact_l2"], r["err_exact_j"], 0.0))
            continue
        x = ref_at(ref, r["t"])
        if x is None:
            continue
        mref, jref, bm, bj = x
        dm = np.array(r["m"]) - mref
        out.append((r["t"], np.max(np.abs(dm)), np.linalg.norm(dm), abs(r["j"] - jref), max(bm, bj)))
    return np.array(out) if out else np.zeros((0, 5))


def step_floor(run, upto):
    f = 0.0
    for s in run["steps"]:
        if s["t"] > upto + 1e-9:
            break
        lg = s.get("ledger", {})
        f += run["config"]["tol_basis"] + lg.get("expv_err", 0) + lg.get("discarded", 0) + lg.get("proj_err", 0)
    return f


def save(fig, outdir, name):
    p = os.path.join(outdir, name)
    fig.savefig(p, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print("HQFIG", p)


def skip(name, why):
    print(f"HQFIG-SKIP {name}: {why}")


def standard(run):
    c = run["config"]
    return (c["policy"] == "dd_qr_streamed" and c["kls"] == "none" and c["parallel"] == "serial"
            and c.get("krylov_reuse", True))


def finest(runs):
    return min(runs, key=lambda r: (r["config"]["h"], r["config"]["tol_basis"], -r["config"]["order"]))


# ── P01 physical dynamics ──────────────────────────────────────────────────────────────────────
def p01(runs, ref, outdir):
    for twoS in (1, 6):
        rs = [r for r in runs if r["config"]["twoS"] == twoS and standard(r)]
        if not rs:
            skip(f"P01 s={twoS / 2}", "no standard run")
            continue
        r = finest(rs)
        t = np.array([x["t"] for x in r["records"]])
        m = np.array([x["m"] for x in r["records"]])
        fig, ax = plt.subplots(1, 3, figsize=(15, 4))
        im = ax[0].imshow(m, aspect="auto", origin="lower", cmap="RdBu_r",
                          extent=[0.5, m.shape[1] + 0.5, t[0], t[-1]], vmin=-twoS / 2, vmax=twoS / 2)
        fig.colorbar(im, ax=ax[0], label=r"$m_i(t)$")
        ax[0].set_xlabel("site i"); ax[0].set_ylabel("J t")
        ax[0].set_title(f"s = {twoS / 2}: {r['path']}")
        for tc in (1.0, 2.0, 4.0, 8.0):
            k = np.argmin(np.abs(t - tc))
            if abs(t[k] - tc) > 1e-9:
                continue
            line, = ax[1].plot(range(1, m.shape[1] + 1), m[k], "o", ms=3, label=f"t={tc:g}")
            if twoS == 1:
                ax[1].plot(range(1, m.shape[1] + 1), exact_sz(m.shape[1], tc), "-", color=line.get_color(), lw=0.8)
            else:
                x = ref_at(ref, tc)
                if x is not None:
                    ax[1].plot(range(1, m.shape[1] + 1), x[0], "-", color=line.get_color(), lw=0.8)
        ax[1].set_xlabel("site i"); ax[1].set_ylabel(r"$m_i$")
        ax[1].set_title("cuts: markers = BUG, lines = " + ("exact" if twoS == 1 else "converged reference"))
        ax[1].legend(fontsize=7)
        j = [x["j"] for x in r["records"]]
        ax[2].plot(t, j, label="BUG")
        if twoS == 6 and ref is not None:
            ax[2].plot(ref["t"], ref["j"], "--", label="reference")
        ax[2].set_xlabel("J t"); ax[2].set_ylabel(r"$j_{L/2}$"); ax[2].legend()
        save(fig, outdir, f"P01_dynamics_s{twoS}.png")


# ── P02 observable error ───────────────────────────────────────────────────────────────────────
def p02(runs, ref, outdir):
    for twoS in (1, 6):
        rs = [r for r in runs if r["config"]["twoS"] == twoS and standard(r)]
        orders = sorted({r["config"]["order"] for r in rs})
        if not rs:
            skip(f"P02 s={twoS / 2}", "no standard run")
            continue
        fig, ax = plt.subplots(3, len(orders), figsize=(4 * len(orders), 9), squeeze=False, sharex=True)
        tols = sorted({r["config"]["tol_basis"] for r in rs})
        cmap = plt.get_cmap("viridis", max(len(tols), 2))
        for c, p in enumerate(orders):
            for r in rs:
                if r["config"]["order"] != p:
                    continue
                e = errors(r, ref)
                if len(e) == 0:
                    continue
                col = cmap(tols.index(r["config"]["tol_basis"]))
                lab = f"h={r['config']['h']:g} tb={r['config']['tol_basis']:.0e}"
                for row, idx in enumerate((1, 2, 3)):
                    ax[row, c].semilogy(e[:, 0], np.maximum(e[:, idx], 1e-17), color=col, lw=0.8, label=lab)
                if twoS == 6:
                    for row in range(3):
                        ax[row, c].semilogy(e[:, 0], np.maximum(e[:, 4], 1e-17), "k:", lw=0.6)
            ax[0, c].set_title(f"s = {twoS / 2}, p = {p}")
            ax[2, c].set_xlabel("J t")
        for row, lab in enumerate((r"$\max_i|\Delta m_i|$", r"$\|\Delta m\|_2$", r"$|\Delta j_{L/2}|$")):
            ax[row, 0].set_ylabel(lab)
        ax[0, 0].legend(fontsize=6)
        save(fig, outdir, f"P02_error_s{twoS}.png")


# ── P03 norm / energy / symmetry ───────────────────────────────────────────────────────────────
def p03(runs, ref, outdir):
    rs = [r for r in runs if standard(r)]
    if not rs:
        return skip("P03", "no standard run")
    fig, ax = plt.subplots(1, 4, figsize=(17, 4))
    for r in rs:
        t = [x["t"] for x in r["records"]]
        s = spin(r)
        lab = f"s={s:g} p={r['config']['order']} h={r['config']['h']:g} tb={r['config']['tol_basis']:.0e}"
        ax[0].semilogy(t, [max(abs(x["norm"] - 1), 1e-17) for x in r["records"]], lw=0.7, label=lab)
        ax[1].semilogy(t, [max(abs(x["E"]), 1e-17) for x in r["records"]], lw=0.7)   # E(0) = 0: absolute
        ax[2].semilogy(t, [max(abs(x["sz"]), 1e-17) for x in r["records"]], lw=0.7)
        ax[3].semilogy(t, [max(abs(x["var"] - s ** 2), 1e-17) for x in r["records"]], lw=0.7)
    for a, lab in zip(ax, (r"$|\|\psi\|^2-1|$", r"$|E(t)-E(0)|$", r"$|S^z_{tot}|$", r"$|\mathrm{Var}H - J^2s^2|$")):
        a.set_ylabel(lab); a.set_xlabel("J t")
    ax[0].legend(fontsize=5)
    save(fig, outdir, "P03_conservation.png")


# ── P04 bond growth ────────────────────────────────────────────────────────────────────────────
def bond_stage_ranks(step, L):
    """Per bond: (D_old, D_CBE, D_joint, D_working, D_final)."""
    old, fin, work = step["D_old"], step["D_final"], step["D_working"]
    cbe, joint = list(old), list(old)
    for b in step.get("bond_log", []):
        i = b["bond"] - 1
        joint[i] = max(joint[i], b["width"])
        cbe[i] = max(cbe[i], b["width"] - b["n_krylov"])
    return np.array([old, cbe, joint, work, fin])


def p04(runs, ref, outdir):
    for twoS in (1, 6):
        rs = [r for r in runs if r["config"]["twoS"] == twoS and standard(r) and r["steps"]
              and "bond_log" in r["steps"][0]]
        tols = sorted({r["config"]["tol_basis"] for r in rs})
        if len(tols) < 3:
            skip(f"P04 s={twoS / 2}", f"needs three tol_basis values, have {tols}")
        if not rs:
            continue
        pick = [min((r for r in rs if r["config"]["tol_basis"] == tb), key=lambda r: r["config"]["h"]) for tb in tols]
        fig, ax = plt.subplots(2, len(pick), figsize=(5 * len(pick), 8), squeeze=False)
        for c, r in enumerate(pick):
            L = r["config"]["L"]
            t = np.array([s["t"] for s in r["steps"]])
            st = np.array([bond_stage_ranks(s, L) for s in r["steps"]])          # step, stage, bond
            for k, lab in enumerate(("D_old", "D_CBE", "D_joint Krylov", "D_working", "D_final")):
                ax[0, c].plot(t, st[:, k, :].max(axis=1), label=lab)
            ax[0, c].set_title(f"s={twoS / 2} p={r['config']['order']} h={r['config']['h']:g} tb={r['config']['tol_basis']:.0e}")
            ax[0, c].set_xlabel("J t"); ax[0, c].set_ylabel("max over bonds")
            ax[0, c].legend(fontsize=7)
            im = ax[1, c].imshow(st[:, 4, :], aspect="auto", origin="lower",
                                 extent=[0.5, st.shape[2] + 0.5, t[0], t[-1]])
            fig.colorbar(im, ax=ax[1, c], label=r"$D_i(t)$ final")
            ax[1, c].set_xlabel("bond i"); ax[1, c].set_ylabel("J t")
        save(fig, outdir, f"P04_bonds_s{twoS}.png")


# ── P05 DD-QR sketch efficiency ────────────────────────────────────────────────────────────────
def sketch_stats(run):
    """Per step: (t, retained, probes, holdout, sample indicator, failed checks, stream cols / useful)."""
    out = []
    for s in run["steps"]:
        lg, kc = s.get("ledger", {}), s["record"]["counters"]
        useful = max(lg.get("n_selected", 0), 1)
        nfail = sum(1 for b in s.get("bond_log", []) if not b["met"])
        out.append((s["t"], lg.get("n_selected", 0), lg.get("n_probes", 0), lg.get("n_holdout", 0),
                    lg.get("max_sample", 0), nfail, kc["n_stream_cols"] / useful,
                    lg.get("n_krylov", 0), s["record"]["root_iters"], s["record"]["D_final"]))
    return np.array(out)


def p05(runs, ref, outdir):
    pols = defaultdict(list)
    for r in runs:
        c = r["config"]
        if c["kls"] == "none" and c.get("krylov_reuse", True):
            pols[c["policy"]].append(r)
    if len(pols) < 2:
        skip("P05", f"needs dd_qr_streamed and one_sided_qr_streamed (and svd_select), have {sorted(pols)}")
    if not pols:
        return
    fig, ax = plt.subplots(2, 3, figsize=(15, 8))
    for pol, rs in sorted(pols.items()):
        for r in rs:
            x = sketch_stats(r)
            lab = f"{pol} s={spin(r):g} p={r['config']['order']} h={r['config']['h']:g}"
            ax[0, 0].plot(x[:, 0], x[:, 1], lw=0.7, label=lab)
            ax[0, 1].plot(x[:, 0], x[:, 2] / np.maximum(x[:, 9], 1), lw=0.7)
            ax[0, 2].semilogy(x[:, 0], np.maximum(x[:, 4], 1e-17), lw=0.7)
            ax[1, 0].plot(x[:, 0], x[:, 5], lw=0.7)
            ax[1, 1].plot(x[:, 0], x[:, 6], lw=0.7)
            ax[1, 2].plot(x[:, 0], x[:, 7], lw=0.7)
    for a, lab in zip(ax.flat, ("CBE directions retained", r"probes $\ell$ / $D$", "Test B indicator",
                                "bonds failing a check", "streamed columns / useful direction",
                                "Krylov directions added")):
        a.set_ylabel(lab); a.set_xlabel("J t")
    ax[0, 0].legend(fontsize=5)
    save(fig, outdir, "P05_sketch.png")


# ── P06 DD-QR kernel scaling (hq_dd_scaling.jl) ────────────────────────────────────────────────
KERNELS = ("input_dd", "stream", "output_dd", "qr", "sectors", "env",
           "onesided_draw", "onesided_stream", "onesided_output_dd", "onesided_qr")


def p06(rundir, outdir):
    files = []
    for p in sorted(glob.glob(os.path.join(rundir, "*.json"))):
        with open(p) as f:
            d = json.load(f)
        if "rows" in d:
            files.append((p, d))
    if not files:
        return skip("P06", "no hq_dd_scaling.jl JSON in the run directory")
    for path, d in files:
        rows = d["rows"]
        ws = sorted({r["w"] for r in rows})
        fig, ax = plt.subplots(2, len(ws) * 2, figsize=(6 * len(ws) * 2, 9), squeeze=False)
        for c, (w, side) in enumerate((w, s) for w in ws for s in ("left", "right")):
            rs = sorted((r for r in rows if r["w"] == w and r["side"] == side), key=lambda r: r["D_actual"])
            D = np.array([r["D_actual"] for r in rs], dtype=float)
            for k in KERNELS:
                t = np.array([r[k][0] for r in rs])
                b = np.array([r[k][1] for r in rs], dtype=float)
                g = np.log2(t[-1] / t[-2]) / np.log2(D[-1] / D[-2]) if len(D) > 1 else np.nan
                style = "--" if k.startswith("onesided") else "-"
                ax[0, c].loglog(D, t, style + "o", ms=3, label=f"{k} (gamma_last={g:.2f})")
                if np.all(b > 0):
                    ax[1, c].loglog(D, b, style + "o", ms=3, label=k)
                print(f"HQ-P06 w={w} {side} {k} gamma=" +
                      " ".join(f"{np.log2(t[n + 1] / t[n]) / np.log2(D[n + 1] / D[n]):.2f}" for n in range(len(D) - 1)))
            t0 = rs[0]["stream"][0]                                  # guides through the first stream point
            for e, lab in ((2, "D^2"), (3, "D^3")):
                ax[0, c].loglog(D, t0 * (D / D[0]) ** e, "k:", lw=0.6)
                ax[0, c].annotate(lab, (D[-1], t0 * (D[-1] / D[0]) ** e), fontsize=7)
            ax[0, c].set_title(f"w = {w}, {side}: cols {[r['cols'] for r in rs]}")
            ax[0, c].set_xlabel("D"); ax[0, c].set_ylabel("median seconds per call")
            ax[1, c].set_xlabel("D"); ax[1, c].set_ylabel("allocated bytes per call")
            ax[0, c].legend(fontsize=5)
        save(fig, outdir, "P06_dd_scaling.png")


# ── P07 Krylov reuse ───────────────────────────────────────────────────────────────────────────
def p07(runs, ref, outdir):
    rs = [r for r in runs if standard(r)]
    off = [r for r in runs if not r["config"].get("krylov_reuse", True)]
    if not rs:
        return skip("P07", "no standard run")
    if not off:
        skip("P07 reuse on/off", "no HQB_REUSE=0 run; only reuse-on counts are drawn")
    fig, ax = plt.subplots(1, 4, figsize=(20, 4))
    for r in rs + off:
        t = [s["t"] for s in r["steps"]]
        on = r["config"].get("krylov_reuse", True)
        lab = f"s={spin(r):g} p={r['config']['order']} h={r['config']['h']:g} reuse={'on' if on else 'off'}"
        st = "-" if on else "--"
        ax[0].plot(t, [s.get("ledger", {}).get("n_krylov", 0) for s in r["steps"]], st, lw=0.7, label=lab)
        ax[1].plot(t, [s["record"]["counters"]["n_one_site"] for s in r["steps"]], st, lw=0.7)
        ax[2].plot(t, [s["record"]["root_iters"] for s in r["steps"]], st, lw=0.7)
        e = errors(r, ref)
        if len(e):
            ax[3].semilogy(e[:, 0], np.maximum(e[:, 1], 1e-17), st, lw=0.7)
    # matched error: each reuse-off run against the reuse-on run of the same setting
    same = ("twoS", "order", "h", "tol_basis", "seed", "policy")
    for r in off:
        twin = [q for q in rs if key(q, *same) == key(r, *same)]
        if not twin:
            continue
        q = twin[0]
        tot = lambda x, f: sum(f(s) for s in x["steps"])
        ek, eq = final_error(r, ref), final_error(q, ref)
        print(f"HQ-P07 s={spin(r):g} p={r['config']['order']} h={r['config']['h']:g} "
              f"krylov_dirs on/off={tot(q, lambda s: s.get('ledger', {}).get('n_krylov', 0))}/"
              f"{tot(r, lambda s: s.get('ledger', {}).get('n_krylov', 0))} "
              f"h1 on/off={tot(q, lambda s: s['record']['counters']['n_one_site'])}/"
              f"{tot(r, lambda s: s['record']['counters']['n_one_site'])} "
              f"err_tf on/off={eq and eq[0]}/{ek and ek[0]}")
    for a, lab in zip(ax, ("new Krylov directions", "one-site H actions", "root iterations",
                           "max_i |m_i - m_i^ref|")):
        a.set_ylabel(lab); a.set_xlabel("J t")
    ax[0].legend(fontsize=5)
    save(fig, outdir, "P07_krylov.png")


# ── P08 / P09 order and control interactions ───────────────────────────────────────────────────
def final_error(run, ref):
    e = errors(run, ref)
    if len(e) == 0 or abs(e[-1, 0] - run["config"]["tf"]) > 1e-9:
        return None
    return e[-1, 1], e[-1, 3], step_floor(run, run["config"]["tf"]) + e[-1, 4]


def fit_range(h, e):
    """The asymptotic range from the largest h down, until halving h stops gaining a factor 1.5
    (a basis/root/reference plateau); the least-squares slope over it. `h` ascending."""
    k = len(h) - 1
    while k > 0 and e[k - 1] < e[k] / 1.5:
        k -= 1
    if len(h) - k < 2:
        return None
    return np.polyfit(np.log(h[k:]), np.log(e[k:]), 1)[0], k


P08_METRICS = (("obs", r"$\max_i|m_i - m_i^{\rm ref}|$ at $t_f$"),
               ("state", r"$e_p = \|\psi(t_f) - \psi_{\rm exact}(t_f)\|$"))


def p08_points(rs, metric, ref):
    """(h, error) at t_f, one per h, seed 1."""
    pts = {}
    for r in rs:
        last = r["records"][-1] if r["records"] else {}
        if abs(last.get("t", -1) - r["config"]["tf"]) > 1e-9:
            continue
        if metric == "state":
            if "err_state" in last:
                pts[r["config"]["h"]] = last["err_state"]
        else:
            fe = final_error(r, ref)
            if fe is not None:
                pts[r["config"]["h"]] = fe[0]
    return [np.array(x) for x in zip(*sorted(pts.items()))] if len(pts) >= 2 else None


def state_floor(groups, twoS, tb, ref):
    """The tol_basis floor of the state error: at an h where the two most accurate orders agree to
    3x both sit on it. It moves slowly with h (hq38: 1.7-3.7e-8 over h = 0.2-0.05), so the smallest
    such value stands for every h -- an order still above it at that h has not shown it (0: none)."""
    byh = defaultdict(list)
    for p in range(1, 6):
        xy = p08_points(groups.get((twoS, tb, p), []), "state", ref)
        if xy is not None:
            for h, e in zip(*xy):
                byh[h].append(e)
    hits = [sorted(es)[0] for es in byh.values() if len(es) >= 2 and sorted(es)[1] < 3 * sorted(es)[0]]
    return min(hits) if hits else 0.0


def p08(runs, ref, outdir):
    groups = defaultdict(list)
    for r in runs:
        if standard(r) and r["config"]["seed"] == 1:        # the seed spread is P13's
            groups[key(r, "twoS", "tol_basis", "order")].append(r)
    if not groups:
        return skip("P08", "no standard runs")
    # one row per (metric, spin) with data, one column per tol_basis
    rows = [(m, s) for m, _ in P08_METRICS for s in (1, 6)
            if any(p08_points(rs, m, ref) is not None for (tw, _, _), rs in groups.items() if tw == s)]
    tbs = sorted({tb for (_, tb, _) in groups})
    fig, ax = plt.subplots(len(rows), len(tbs), figsize=(5.2 * len(tbs), 4.4 * len(rows)),
                           squeeze=False, sharex=True)
    colors = {p: plt.cm.viridis((p - 1) / 4.2) for p in range(1, 6)}
    for i, (metric, twoS) in enumerate(rows):
        for j, tb in enumerate(tbs):
            a = ax[i, j]
            if all(p08_points(groups.get((twoS, tb, p), []), metric, ref) is None for p in range(1, 6)):
                a.axis("off")
                a.text(0.5, 0.5, f"no {metric} runs at s = {twoS / 2:g}, tol_basis = {tb:.0e}",
                       ha="center", va="center", transform=a.transAxes, fontsize=9, color="grey")
                continue
            for p in range(1, 6):
                xy = p08_points(groups.get((twoS, tb, p), []), metric, ref)
                if xy is None:
                    continue
                h, e = xy
                k0 = 0
                if metric == "state":                        # drop the floor points (h ascending)
                    fl = state_floor(groups, twoS, tb, ref)
                    ok = e > MARGIN * fl
                    k0 = len(ok) - int(np.argmin(ok[::-1])) if not ok.all() else 0
                    a.loglog(h[:k0], e[:k0], "o", color=colors[p], ms=5, mfc="none")
                f = fit_range(h[k0:], e[k0:])
                f = f and (f[0], f[1] + k0)
                lab = f"p={p}" + (f": slope {f[0]:.2f} on h={h[f[1]]:g}-{h[-1]:g}" if f else "")
                a.loglog(h, e, "-", color=colors[p], lw=0.8)
                a.loglog(h[k0:], e[k0:], "o-", color=colors[p], ms=4, label=lab)
                k = f[1] if f else len(h) - 1                # guide through the fit's smallest h
                a.loglog(h, e[k] * (h / h[k]) ** p, "--", color=colors[p], lw=0.8, alpha=0.6)
                print(f"HQ-P08 {metric} s={twoS / 2:g} tb={tb:.0e} p={p} h={[float(x) for x in h]} "
                      f"err={[f'{x:.2e}' for x in e]} fit={'none' if f is None else round(float(f[0]), 2)}")
            a.set_title(f"s = {twoS / 2:g}, tol_basis = {tb:.0e}", fontsize=10)
            hs = sorted({r["config"]["h"] for (tw, t, _), rs in groups.items() if tw == twoS for r in rs})
            a.set_xticks(hs, [f"{x:g}" for x in hs])
            a.minorticks_off()
            a.set_xlabel("J h")
            j == 0 and a.set_ylabel(dict(P08_METRICS)[metric])
            a.grid(True, which="major", alpha=0.3)
            a.legend(fontsize=7, loc="lower right")
    fig.suptitle(r"P08 time-step order at $Jt_f = 8$, L = 16 (dashed: $h^p$ through each fit's smallest h;"
                 r" open: within 10x of the tol_basis floor, not fitted)")
    fig.tight_layout()
    save(fig, outdir, "P08_order.png")


def p09(runs, ref, outdir):
    for twoS in (1, 6):
        for p in sorted({r["config"]["order"] for r in runs}):
            rs = [r for r in runs if standard(r) and r["config"]["twoS"] == twoS and r["config"]["order"] == p]
            hs = sorted({r["config"]["h"] for r in rs})
            tbs = sorted({r["config"]["tol_basis"] for r in rs})
            if len(hs) < 2 or len(tbs) < 2:
                continue
            E, D = np.full((len(tbs), len(hs)), np.nan), np.full((len(tbs), len(hs)), np.nan)
            for r in rs:
                fe = final_error(r, ref)
                i, j = tbs.index(r["config"]["tol_basis"]), hs.index(r["config"]["h"])
                if fe is not None:
                    E[i, j] = np.log10(max(fe[0], 1e-17))
                D[i, j] = max(s["record"]["D_final"] for s in r["steps"])
            fig, ax = plt.subplots(1, 2, figsize=(11, 4))
            for a, M, lab in ((ax[0], E, r"$\log_{10} e_p$"), (ax[1], D, r"$D_{max}$")):
                im = a.imshow(M, origin="lower", aspect="auto")
                a.set_xticks(range(len(hs)), [f"{h:g}" for h in hs]); a.set_yticks(range(len(tbs)), [f"{t:.0e}" for t in tbs])
                a.set_xlabel("J h"); a.set_ylabel("tol_basis"); fig.colorbar(im, ax=a, label=lab)
            ax[0].set_title(f"s = {twoS / 2}, p = {p}")
            save(fig, outdir, f"P09_controls_s{twoS}_p{p}.png")


# ── P10 time to accuracy ───────────────────────────────────────────────────────────────────────
def step_time(run):
    """The integrator's seconds, steps 2..n: step 1 of every process carries the JIT compilation
    (hq43t: 145-165 s against < 1 s of physics at D = 1), which would swamp every comparison."""
    return sum(s["t_step"] for s in run["steps"][1:])


def finished(run):
    return len(run["steps"]) > 1 and run["records"] and abs(run["records"][-1]["t"] - run["config"]["tf"]) < 1e-9


def p10(runs, ref, outdir):
    rs = [r for r in runs if standard(r) and final_error(r, ref) is not None]
    if not rs:
        return skip("P10", "no completed standard runs")
    fig, ax = plt.subplots(1, 4, figsize=(19, 4.5))
    for p in sorted({r["config"]["order"] for r in rs}):
        for twoS in (1, 6):
            sel = sorted((r for r in rs if r["config"]["order"] == p and r["config"]["twoS"] == twoS),
                         key=lambda r: r["config"]["h"])
            if not sel:
                continue
            em = [final_error(r, ref)[0] for r in sel]
            wall = [step_time(r) for r in sel]           # the integrator, not the observables or the JIT
            apps = [sum(s["krylov_dims"] for s in r["steps"]) for r in sel]
            rss = [max(s["record"]["peak_rss"] for s in r["steps"]) / 2 ** 30 for r in sel]
            mk = "o-" if twoS == 1 else "s--"
            ax[0].loglog(em, wall, mk, label=f"s={twoS / 2} p={p}")
            ax[1].loglog(em, apps, mk)
            ax[2].semilogx(em, rss, mk)
            hs = [r["config"]["h"] for r in sel]
            line, = ax[3].loglog(hs, [max(abs(r["records"][-1]["norm"] - 1), 1e-17) for r in sel], mk)
            # E(0) = 0 on the domain wall, so the drift is |E(t_f)| (plan P03/P10: absolute)
            ax[3].loglog(hs, [max(abs(r["records"][-1]["E"]), 1e-17) for r in sel], "x:", color=line.get_color())
    for a, lab in zip(ax, ("integrator time, steps 2..n [s]", "operator applications", "peak RSS [GiB]",
                           r"$|\Delta\|\psi\|^2|$ (o) and $|\Delta E|$ (x) at $t_f$")):
        a.set_ylabel(lab)
    for a in ax[:3]:
        a.set_xlabel(r"$\max_i|\Delta m_i(t_f)|$")
    hall = sorted({r["config"]["h"] for r in rs})
    ax[3].set_xticks(hall, [f"{h:g}" for h in hall]); ax[3].minorticks_off()
    ax[3].set_xlabel("J h")
    ax[0].legend(fontsize=6)
    save(fig, outdir, "P10_time_to_accuracy.png")


# ── P11 / P12 parallel ─────────────────────────────────────────────────────────────────────────
P11_SCHEDULES = ("sequential", "kl_then_s", "concurrent")


def p11(runs, ref, outdir):
    """Per (s, p, h, tf, tol_basis): integrator time vs workers for the three K/L/S schedules on the
    same scheme (`parallel = :full`); speedup and efficiency against sequential at one worker."""
    cells = defaultdict(dict)
    for r in runs:
        c = r["config"]
        if c["policy"] != "dd_qr_streamed" or not finished(r):
            continue
        if c["parallel"] == "full" and c["kls"] in P11_SCHEDULES:
            cells[key(r, "twoS", "order", "h", "tf", "tol_basis")][(c["kls"], c["threads"])] = r
        elif c["kls"] == "none" and c.get("krylov_reuse", True) and c["parallel"] in ("serial", "half_sweeps"):
            # the standard step at the same setting (hq45t): serial w1, half-sweeps w2/w3
            cells[key(r, "twoS", "order", "h", "tf", "tol_basis")][("standard", c["threads"])] = r
    cells = {k: v for k, v in cells.items() if set(P11_SCHEDULES) <= {s for s, _ in v}}
    if not cells:
        return skip("P11", "needs the three schedules on one scheme for one setting")
    print("HQ-P11 schedule  s  p  h  threads/blas  t(steps 2..n)  sum t_K  sum t_L  sum t_root  peak_rss  "
          "speedup(vs sequential w1)  vs standard w1")
    fig, ax = plt.subplots(1, 4, figsize=(20, 4.6))
    colors = {p: plt.cm.viridis((p - 1) / 4.2) for p in range(1, 6)}
    styles = {"sequential": ("o", "-"), "kl_then_s": ("^", "--"), "concurrent": ("s", ":"),
              "standard": ("x", "-.")}
    for (twoS, p, h, tf, tb), byk in sorted(cells.items()):
        base = byk.get(("sequential", 1))
        t1 = step_time(base) if base else np.nan
        std1 = byk.get(("standard", 1))
        ts1 = step_time(std1) if std1 else np.nan
        for sch in P11_SCHEDULES + ("standard",):
            if not any(s == sch for s, _ in byk):
                continue
            ws = sorted(w for s, w in byk if s == sch)
            T = np.array([step_time(byk[(sch, w)]) for w in ws])
            rss = [max(s["record"]["peak_rss"] for s in byk[(sch, w)]["steps"]) / 2 ** 30 for w in ws]
            mk, ls = styles[sch]
            ax[0].plot(ws, T, mk + ls, color=colors[p])
            ax[1].plot(ws, t1 / T, mk + ls, color=colors[p])
            ax[2].plot(ws, t1 / T / np.array(ws), mk + ls, color=colors[p])
            ax[3].plot(ws, rss, mk + ls, color=colors[p])
            for w, x in zip(ws, T):
                st = byk[(sch, w)]["steps"][1:]
                print(f"HQ-P11 {sch} {twoS / 2:g} {p} {h:g} {w}/{byk[(sch, w)]['config']['blas_threads']} {x:.1f} "
                      f"{sum((s.get('kls') or {}).get('t_K', 0) for s in st):.1f} "
                      f"{sum((s.get('kls') or {}).get('t_L', 0) for s in st):.1f} "
                      f"{sum(s['t_root'] for s in st):.1f} {rss[ws.index(w)]:.2f}G {t1 / x:.2f} {ts1 / x:.2f}")
    from matplotlib.lines import Line2D
    ax[0].legend(handles=[Line2D([], [], color=colors[p], lw=2, label=f"p={p}") for p in range(1, 6)]
                 + [Line2D([], [], color="k", marker=styles[s][0], ls=styles[s][1], label=s)
                    for s in P11_SCHEDULES + ("standard",)],
                 fontsize=7)
    for a, lab in zip(ax, ("integrator time, steps 2..n [s]", "speedup vs sequential, 1 worker",
                           "parallel efficiency (speedup / workers)", "peak RSS [GiB]")):
        a.set_xlabel("workers (julia -t; BLAS 1 thread)"); a.set_ylabel(lab); a.set_xticks([1, 2, 3])
        a.grid(True, alpha=0.3)
    ax[1].axhline(1, color="grey", lw=0.8)
    (twoS, _, _, tf, tb) = next(iter(cells))
    hs = sorted({(k[1], k[2]) for k in cells})
    hlab = (f"h = {hs[0][1]:g}" if len({h for _, h in hs}) == 1 else
            "h = " + ", ".join(f"{h:g} (p{p})" for p, h in hs))
    fig.suptitle(f"P11 K/L/S schedules on parallel = :full, s = {twoS / 2:g}, L = 16, {hlab}, "
                 f"Jt = {tf:g}, tol_basis = {tb:.0e} (one job, one node)")
    fig.tight_layout()
    save(fig, outdir, "P11_schedules.png")


P12_T = 0.4      # every twin scan (s = 3 stops there) holds this record
# hq39: both arms flatten at 3-5e-9 while max|dm| is 1e-14 -- the ED-overlap measurement, not the
# method -- so state errors below this never enter a fit
STATE_FLOOR = 1e-8
DIST_FLOOR = 3e-8    # dist_twin = sqrt(<a|a> + <b|b> - 2 Re<a|b>): sqrt-roundoff 2.1e-8 for equal states
NORM_FLOOR = 1e-12   # accumulated roundoff of <psi|psi> over the steps


def _record_at(run, t):
    return next((x for x in run["records"] if abs(x["t"] - t) < 1e-9), None)


def p12(runs, ref, outdir):
    """Versus h, per (s, order, tol_basis): (a) each arm's OWN state error against the exact state
    (s = 1/2, last state time), so the K/L/S-minus-standard distance is attributed to an arm;
    (b) that distance at `P12_T`; (c) the invariants' drift up to `P12_T` -- the only grade at s = 3."""
    groups = defaultdict(dict)
    for r in runs:
        if r["config"]["twin"] and _record_at(r, P12_T) is not None:
            groups[key(r, "twoS", "order", "tol_basis")][r["config"]["h"]] = r
    if not groups:
        return skip("P12", f"no HQB_TWIN = 1 run reaching t = {P12_T}")
    from matplotlib.lines import Line2D
    fig, ax = plt.subplots(1, 3, figsize=(18, 5.2))
    colors = {p: plt.cm.viridis((p - 1) / 4.2) for p in range(1, 6)}
    masked = lambda v, lo: np.where(np.asarray(v) > lo, v, np.nan)   # below a floor = not a measurement
    slopes = defaultdict(dict)
    for (twoS, p, tb), byh in sorted(groups.items()):
        hs = np.array(sorted(byh))
        mk = "o" if twoS == 1 else "s"
        ls_tb = "-" if (twoS == 6 or tb == 1e-7) else "--"
        lab = f"s={twoS / 2:g} p={p} tb={tb:.0e}"
        st = [next((x for x in reversed(byh[h]["records"]) if "twin_err_state" in x), None) for h in hs]
        if all(x is not None for x in st) and len(hs) >= 2:
            for arm, ls, name in (("err_state", "-", "K/L/S"), ("twin_err_state", "--", "std")):
                e = np.array([x[arm] for x in st])
                k0 = int(np.sum(e < STATE_FLOOR))                    # h ascending: floor points first
                f = fit_range(hs[k0:], e[k0:])
                slopes[p][name] = f[0] if f else np.nan
                ax[0].loglog(hs, e, mk + ls, color=colors[p], ms=4, mfc="none" if name == "std" else None)
                print(f"HQ-P12 state {lab} {name} t={st[0]['t']:g} h={list(map(float, hs))} "
                      f"err={[f'{x:.2e}' for x in e]} fit={'none' if f is None else round(float(f[0]), 2)}")
        rec = [_record_at(byh[h], P12_T) for h in hs]
        d = np.array([x["dist_twin"] for x in rec])
        ax[1].loglog(hs, masked(d, DIST_FLOOR), mk + ls_tb, color=colors[p], ms=4)
        dn = np.array([max(abs(x["norm"] - 1) for x in byh[h]["records"] if x["t"] <= P12_T + 1e-9) for h in hs])
        dv = np.array([abs(x["var"] - (twoS / 2) ** 2) for x in rec])
        ax[2].loglog(hs, masked(dn, NORM_FLOOR), mk + ls_tb, color=colors[p], ms=4)
        if twoS == 6:                     # at s = 1/2 Var H drift is each arm's own error (panel a)
            ax[2].loglog(hs, dv, mk + ":", color=colors[p], ms=4, mfc="none")
        if all("twin_var" in x for x in rec):
            ax[2].loglog(hs, [abs(x["twin_var"] - (twoS / 2) ** 2) for x in rec], "x-.", color=colors[p])
        print(f"HQ-P12 kls {lab} t={P12_T} h={list(map(float, hs))} dist={[f'{x:.2e}' for x in d]} "
              f"dnorm={[f'{x:.2e}' for x in dn]} dvar={[f'{x:.2e}' for x in dv]}")
    pkey = [Line2D([], [], color=colors[p], lw=2,
                   label=f"p={p}" + (f"  slope K/L/S {slopes[p]['K/L/S']:.2f}, std {slopes[p]['std']:.2f}"
                                     if p in slopes else "")) for p in range(1, 6)]
    ax[0].axhline(STATE_FLOOR, color="grey", lw=0.8, ls=":")
    ax[0].legend(handles=pkey + [Line2D([], [], color="k", marker="o", label="K/L/S"),
                                 Line2D([], [], color="k", marker="o", mfc="none", ls="--", label="standard")],
                 fontsize=7, loc="lower right")
    ax[0].set_title(r"(a) $s=1/2$, tol_basis 1e-7: each arm vs exact, $Jt=0.8$", fontsize=9)
    ax[0].set_ylabel(r"$\|\psi-\psi_{\rm exact}\|$")
    ax[1].axhline(DIST_FLOOR, color="grey", lw=0.8, ls=":")
    ax[1].set_title(rf"(b) $\|\psi_{{par}}-\psi_{{std}}\|$ at $Jt={P12_T:g}$ (missing = below floor)", fontsize=9)
    sk = [Line2D([], [], color="k", marker="s", label="s=3, tb 1e-5"),
          Line2D([], [], color="k", marker="o", label="s=1/2, tb 1e-7"),
          Line2D([], [], color="k", marker="o", ls="--", label="s=1/2, tb 1e-5")]
    ax[1].legend(handles=[Line2D([], [], color=colors[p], lw=2, label=f"p={p}") for p in range(1, 6)] + sk,
                 fontsize=7, loc="lower right")
    ax[2].set_title(rf"(c) K/L/S invariant drift up to $Jt={P12_T:g}$ (Var H exact $=s^2$)", fontsize=9)
    ax[2].legend(handles=[Line2D([], [], color="k", marker="s", label=r"$|\|\psi\|^2-1|$ (solid)"),
                          Line2D([], [], color="k", marker="s", mfc="none", ls=":",
                                 label=r"$|{\rm Var}H - s^2|$, s=3"),
                          Line2D([], [], color="k", marker="x", ls="-.", label="standard Var H (if recorded)")],
                 fontsize=7, loc="lower right")
    hall = sorted({h for byh in groups.values() for h in byh})
    for a in ax:
        a.set_xticks(hall, [f"{h:g}" for h in hall]); a.minorticks_off()
        a.set_xlabel("J h"); a.grid(True, which="major", alpha=0.3)
    fig.suptitle("P12 parallel K/L/S vs standard BUG, L = 16 (circles s = 1/2, squares s = 3)")
    fig.tight_layout()
    save(fig, outdir, "P12_parallel_accuracy.png")


# ── P13 failure modes ──────────────────────────────────────────────────────────────────────────
P13_EVENTS = ("n_targeted", "n_d_after_c", "n_passes")


def p13(runs, ref, outdir):
    print("HQ-P13 run  steps  unmet  limits  seeds")
    # per (tol_basis, order) on the standard step: steps, unmet steps, limit hits, ledger events
    ev = defaultdict(lambda: defaultdict(int))
    for r in runs:
        lim = defaultdict(int)
        nun = 0
        for s in r["steps"]:
            nun += s.get("status") == "budget_unmet"
            for x in s["record"]["limits"]:
                lim[x] += 1
        print(f"HQ-P13 {r['path']} {len(r['steps'])} {nun} {dict(lim)} seed={r['config']['seed']}")
        if standard(r):
            e = ev[key(r, "tol_basis", "order")]
            e["steps"] += len(r["steps"])
            e["unmet"] += nun
            e["limits"] += sum(lim.values())
            for s in r["steps"]:
                for k in P13_EVENTS:
                    e[k] += s["ledger"].get(k, 0)
    # `krylov_reuse` is in the key: a reuse-off rerun at seed 1 is not a second seed
    seeds = defaultdict(list)
    for r in runs:
        fe = final_error(r, ref)
        if fe is not None:
            seeds[key(r, "twoS", "order", "h", "tol_basis", "policy", "kls") +
                  (r["config"].get("krylov_reuse", True),)].append(
                (r["config"]["seed"], fe[0], max(r["steps"][-1]["D_final"])))
    spread = {}
    for k, v in sorted(seeds.items()):
        if len(v) >= 2:
            e = [x[1] for x in v]
            d = [x[2] for x in v]
            spread[k] = (e, d)
            print(f"HQ-P13-SEEDS {k} n={len(v)} err min={min(e):.2e} max={max(e):.2e} D {min(d)}-{max(d)}")
    if not ev or not spread:
        return skip("P13", "needs standard-step runs and >= 2 seeds of one setting")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4.2))
    cols = plt.cm.viridis(np.linspace(0, 0.9, 5))
    for (twoS, p, h, tb, pol, kls, _), (e, d) in spread.items():
        ax[0].errorbar(h, np.median(e), yerr=[[np.median(e) - min(e)], [max(e) - np.median(e)]],
                       fmt="o", color=cols[p - 1], capsize=4, label=f"p = {p}")
        ax[1].plot(p + (h - 0.15), (max(e) - min(e)) / np.median(e), "o" if h < 0.15 else "s",
                   color=cols[p - 1])
        ax[1].plot(p + (h - 0.15), (max(d) - min(d)) / np.median(d), "x", color=cols[p - 1])
    hd, lb = ax[0].get_legend_handles_labels()
    u = dict(zip(lb, hd))
    ax[0].legend(u.values(), u.keys(), fontsize=8)
    hs = sorted({k[2] for k in spread})
    ax[0].set_xscale("log"); ax[0].set_yscale("log")
    ax[0].set_xticks(hs, [f"{h:g}" for h in hs]); ax[0].minorticks_off()
    ax[0].set_xlabel("J h"); ax[0].set_ylabel("max|Δm(t_f)|, median and seed range")
    tbs = sorted({k[3] for k in spread})
    ax[0].set_title(f"seed spread of the error (tol_basis {', '.join(f'{t:g}' for t in tbs)})")
    ax[1].set_yscale("log"); ax[1].set_xticks(range(1, 6))
    ax[1].set_xlabel("order p (offset: h = 0.1 left, 0.2 right)")
    ax[1].set_ylabel("(max − min) / median over seeds")
    ax[1].set_title("relative seed spread: error (o h = 0.1, □ 0.2), D_max (×)", fontsize=10)
    for i, tb in enumerate(sorted({k[0] for k in ev})):
        ps = sorted(p for (t, p) in ev if t == tb)
        ax[2].plot(ps, [ev[(tb, p)]["n_d_after_c"] / ev[(tb, p)]["steps"] for p in ps], "o-",
                   label=f"tol_basis {tb:g}")
    tot = defaultdict(int)
    for e in ev.values():
        for k, v in e.items():
            tot[k] += v
    ax[2].set_xlabel("order p"); ax[2].set_ylabel("Test D grew the frame after Test C passed, per step")
    ax[2].set_xticks(range(1, 6)); ax[2].legend(fontsize=8)
    ax[2].set_title(f"{tot['steps']} steps: unmet {tot['unmet']}, limits {tot['limits']}, "
                    f"holdout widening {tot['n_targeted']}", fontsize=9)
    for a in ax:
        a.grid(True, alpha=0.3)
    fig.suptitle(f"P13 failure modes, L = 16, s = {', '.join(sorted({f'{k[0] / 2:g}' for k in spread}))}")
    fig.tight_layout()
    save(fig, outdir, "P13_failure_modes.png")


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--figs=")]
    only = [a.split("=", 1)[1].split(",") for a in sys.argv[1:] if a.startswith("--figs=")]
    if len(args) < 2:
        sys.exit(__doc__)
    rundirs = args[0].split(",")
    runs = [r for d in rundirs for r in load_runs(d)]
    outdir = args[1]
    os.makedirs(outdir, exist_ok=True)
    ref = None
    if len(args) > 2:
        with open(args[2]) as f:
            ref = json.load(f)
    print(f"HQFIG-RUNS {len(runs)}")
    want = (lambda name: name in only[0]) if only else (lambda name: True)
    for fn in (p01, p02, p03, p04, p05, p07, p08, p09, p10, p11, p12, p13):
        want(fn.__name__.upper()) and fn(runs, ref, outdir)
    if want("P06"):
        for d in rundirs:
            p06(d, outdir)


if __name__ == "__main__":
    main()
