"""HQ plan Task 8: figures P01--P13 from `benchmarks/hq_bench.jl` runs.

usage:  python benchmarks/hq_task8_figures.py RUNDIR OUTDIR [S3REF.json]

RUNDIR holds the `hq_bench.jl` JSON files (any number, any settings); S3REF is the spin-3 reference
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
    return c["policy"] == "dd_qr_streamed" and c["kls"] == "none" and c["parallel"] == "serial"


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
        if c["kls"] == "none":
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
    if not rs:
        return skip("P07", "no standard run")
    skip("P07 reuse on/off", "the step has no reuse-off ablation; only reuse-on counts are drawn")
    fig, ax = plt.subplots(1, 3, figsize=(15, 4))
    for r in rs:
        t = [s["t"] for s in r["steps"]]
        lab = f"s={spin(r):g} p={r['config']['order']} h={r['config']['h']:g}"
        ax[0].plot(t, [s.get("ledger", {}).get("n_krylov", 0) for s in r["steps"]], lw=0.7, label=lab)
        ax[1].plot(t, [s["record"]["counters"]["n_one_site"] for s in r["steps"]], lw=0.7)
        ax[2].plot(t, [s["record"]["root_iters"] for s in r["steps"]], lw=0.7)
    for a, lab in zip(ax, ("new Krylov directions", "one-site H actions", "root iterations")):
        a.set_ylabel(lab); a.set_xlabel("J t")
    ax[0].legend(fontsize=5)
    save(fig, outdir, "P07_krylov.png")


# ── P08 / P09 order and control interactions ───────────────────────────────────────────────────
def final_error(run, ref):
    e = errors(run, ref)
    if len(e) == 0 or abs(e[-1, 0] - run["config"]["tf"]) > 1e-9:
        return None
    return e[-1, 1], e[-1, 3], step_floor(run, run["config"]["tf"]) + e[-1, 4]


def p08(runs, ref, outdir):
    groups = defaultdict(list)
    for r in runs:
        if standard(r):
            groups[key(r, "twoS", "order", "tol_basis", "tol_root", "tol_trunc")].append(r)
    if not groups:
        return skip("P08", "no standard runs")
    fig, ax = plt.subplots(1, 2, figsize=(13, 5))
    for (twoS, p, tb, tr, tt), rs in sorted(groups.items()):
        pts = sorted((r["config"]["h"],) + final_error(r, ref) for r in rs if final_error(r, ref) is not None)
        if len(pts) < 2:
            continue
        h, em, ej, fl = map(np.array, zip(*pts))
        line, = ax[twoS == 6].loglog(h, em, "o-", label=f"p={p} tb={tb:.0e}")
        ax[twoS == 6].loglog(h, fl, ":", color=line.get_color(), lw=0.7)
        ok = (em > MARGIN * fl)
        slopes = [np.log2(em[k + 1] / em[k]) / np.log2(h[k + 1] / h[k])
                  for k in range(len(h) - 1) if ok[k] and ok[k + 1] and abs(h[k + 1] / h[k] - 2) < 1e-9]
        print(f"HQ-P08 s={twoS / 2} p={p} tb={tb:.0e} tr={tr:.0e} tt={tt:.0e} h={list(h)} "
              f"err={[f'{x:.2e}' for x in em]} floor={[f'{x:.2e}' for x in fl]} slopes={[round(s, 2) for s in slopes]}")
        h0, e0 = h[-1], em[-1]
        ax[twoS == 6].loglog(h, e0 * (h / h0) ** p, "--", color=line.get_color(), lw=0.5)
    for a, s in zip(ax, ("1/2", "3")):
        a.set_xlabel("J h"); a.set_ylabel(r"$\max_i|\Delta m_i(t_f)|$")
        a.set_title(f"s = {s}: dashed = $h^p$, dotted = the run's floor")
        a.legend(fontsize=6)
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
            wall = [r["steps"][-1]["wall"] for r in sel]
            apps = [sum(s["krylov_dims"] for s in r["steps"]) for r in sel]
            rss = [max(s["record"]["peak_rss"] for s in r["steps"]) / 2 ** 30 for r in sel]
            mk = "o-" if twoS == 1 else "s--"
            ax[0].loglog(em, wall, mk, label=f"s={twoS / 2} p={p}")
            ax[1].loglog(em, apps, mk)
            ax[2].semilogx(em, rss, mk)
            ax[3].loglog([r["config"]["h"] for r in sel],
                         [max(abs(r["records"][-1]["norm"] - 1), 1e-17) for r in sel], mk)
    for a, lab in zip(ax, ("wall [s]", "operator applications", "peak RSS [GiB]", r"$|\Delta\|\psi\|^2|$ at $t_f$")):
        a.set_ylabel(lab)
    for a in ax[:3]:
        a.set_xlabel(r"$\max_i|\Delta m_i(t_f)|$")
    ax[3].set_xlabel("J h")
    ax[0].legend(fontsize=6)
    save(fig, outdir, "P10_time_to_accuracy.png")


# ── P11 / P12 parallel ─────────────────────────────────────────────────────────────────────────
def p11(runs, ref, outdir):
    sch = defaultdict(list)
    for r in runs:
        c = r["config"]
        if c["policy"] == "dd_qr_streamed":
            sch[(c["parallel"], c["kls"])].append(r)
    if len(sch) < 3:
        return skip("P11", f"needs the three schedules on one scheme, have {sorted(sch)}")
    print("HQ-P11 schedule  s  p  h  threads  wall  sum t_K  sum t_L  sum t_root  peak_rss")
    for (par, kls), rs in sorted(sch.items()):
        for r in rs:
            st = r["steps"]
            tK = sum((s.get("kls") or {}).get("t_K", 0) for s in st)
            tL = sum((s.get("kls") or {}).get("t_L", 0) for s in st)
            print(f"HQ-P11 {par}/{kls} {spin(r):g} {r['config']['order']} {r['config']['h']:g} "
                  f"{r['config']['threads']}/{r['config']['blas_threads']} {st[-1]['wall']:.1f} "
                  f"{tK:.1f} {tL:.1f} {sum(s['t_root'] for s in st):.1f} "
                  f"{max(s['record']['peak_rss'] for s in st) / 2 ** 30:.2f}G")


def p12(runs, ref, outdir):
    rs = [r for r in runs if r["config"]["twin"]]
    if not rs:
        return skip("P12", "no HQB_TWIN = 1 run")
    fig, ax = plt.subplots(1, 2, figsize=(12, 4))
    for r in rs:
        t = [x["t"] for x in r["records"]]
        lab = f"s={spin(r):g} p={r['config']['order']} h={r['config']['h']:g} {r['config']['kls']}"
        ax[0].semilogy(t, [max(x["dist_twin"], 1e-17) for x in r["records"]], label=lab)
        ax[1].semilogy(t, [max(x.get("err_exact_m", np.nan), 1e-17) for x in r["records"]], label=lab + " K/L/S")
        if r["config"]["twoS"] == 1:
            ex = [abs(np.max(np.abs(np.array(x["twin_m"]) - (np.array(x["m"]))))) for x in r["records"]]
            ax[1].semilogy(t, np.maximum(ex, 1e-17), ":", label=lab + " |m_par - m_std|")
    ax[0].set_ylabel(r"$\|\psi_{par}-\psi_{std}\|$"); ax[1].set_ylabel("error")
    for a in ax:
        a.set_xlabel("J t"); a.legend(fontsize=6)
    save(fig, outdir, "P12_parallel_accuracy.png")


# ── P13 failure modes ──────────────────────────────────────────────────────────────────────────
def p13(runs, ref, outdir):
    print("HQ-P13 run  steps  unmet  limits  seeds")
    for r in runs:
        lim = defaultdict(int)
        nun = 0
        for s in r["steps"]:
            nun += s.get("status") == "budget_unmet"
            for x in s["record"]["limits"]:
                lim[x] += 1
        print(f"HQ-P13 {r['path']} {len(r['steps'])} {nun} {dict(lim)} seed={r['config']['seed']}")
    seeds = defaultdict(list)
    for r in runs:
        fe = final_error(r, ref)
        if fe is not None:
            seeds[key(r, "twoS", "order", "h", "tol_basis", "policy", "kls")].append((r["config"]["seed"], fe[0]))
    for k, v in sorted(seeds.items()):
        if len(v) >= 2:
            e = [x[1] for x in v]
            print(f"HQ-P13-SEEDS {k} n={len(v)} err min={min(e):.2e} max={max(e):.2e}")


def main():
    if len(sys.argv) < 3:
        sys.exit(__doc__)
    runs = load_runs(sys.argv[1])
    outdir = sys.argv[2]
    os.makedirs(outdir, exist_ok=True)
    ref = None
    if len(sys.argv) > 3:
        with open(sys.argv[3]) as f:
            ref = json.load(f)
    print(f"HQFIG-RUNS {len(runs)}")
    for fn in (p01, p02, p03, p04, p05, p07, p08, p09, p10, p11, p12, p13):
        fn(runs, ref, outdir)
    p06(sys.argv[1], outdir)


if __name__ == "__main__":
    main()
