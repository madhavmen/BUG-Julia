"""Tables for the adaptive CBE-DMRG tolerance study (adaptive_dmrg_tolerance.jl).

usage: python analyze_dmrg_tolerance.py DIR  -> markdown on stdout
Reads DIR/dmrgtol_<model>.csv. Per model: each study's table, the loosest setting reaching each
energy target (dE/N), and the cost relative to the legacy fixed-knob run (same process).
"""
import csv
import glob
import os
import sys

TARGETS = (1e-4, 1e-6, 1e-8, 1e-10)


def load(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.DictReader(f))


def fnum(x):
    try:
        return float(x)
    except ValueError:
        return float("nan")


def table(rows):
    hdr = ["tol_basis", "tol_trunc", "exact", "dE/N", "variance", "sweeps", "chi", "r_aug",
           "matvec", "probes", "wall s", "status", "unmet"]
    out = ["| " + " | ".join(hdr) + " |", "|" + "---|" * len(hdr)]
    for r in rows:
        out.append("| " + " | ".join([
            r["tol_basis"], r["tol_trunc"], r["exact"], f"{fnum(r['dE_site']):.2e}",
            f"{fnum(r['variance']):.2e}", r["sweeps"], r["chi"], r["r_aug_max"], r["matvec"],
            r["probes"], f"{fnum(r['wall']):.1f}", r["status_last"], r["unmet_last"] or "-"]) + " |")
    return "\n".join(out)


def report(path):
    rows = load(path)
    if not rows:
        return ""
    model = rows[0]["model"]
    mode = " -- derived stopping threshold (dmrg_etol)" if path.endswith("_autoetol.csv") else ""
    md = [f"## {model} (maxdim {rows[0]['maxdim']}){mode}", ""]
    leg = [r for r in rows if r["study"] == "legacy"]
    for study, title in (("legacy", "Legacy fixed-knob DMRG (cost baseline)"),
                         ("basis", "tol_basis sweep (tol_trunc = 1e-8)"),
                         ("trunc", "tol_trunc sweep (tol_basis = 1e-5)"),
                         ("combo", "Both loose together"),
                         ("exact", "exact = true (full local basis instead of the RSVD sketch)")):
        sel = [r for r in rows if r["study"] == study]
        if sel:
            md += [f"### {title}", "", table(sel), ""]
    # HOW LOOSE: per target, the loosest tol_basis (basis sweep) and tol_trunc (trunc sweep)
    # whose run still reaches it, with the rank and cost that buys.
    md += ["### Loosest tolerance reaching each energy target", "",
           "| target dE/N | loosest tol_basis | dE/N | chi | wall s | loosest tol_trunc | dE/N | chi | wall s |",
           "|---|---|---|---|---|---|---|---|---|"]
    for t in TARGETS:
        cells = []
        for study, key in (("basis", "tol_basis"), ("trunc", "tol_trunc")):
            ok = [r for r in rows if r["study"] == study and abs(fnum(r["dE_site"])) <= t]
            if ok:
                b = max(ok, key=lambda r: fnum(r[key]))
                cells += [f"{fnum(b[key]):.0e}", f"{fnum(b['dE_site']):.1e}", b["chi"],
                          f"{fnum(b['wall']):.1f}"]
            else:
                cells += ["none", "", "", ""]
        md.append(f"| {t:.0e} | " + " | ".join(cells) + " |")
    md.append("")
    # Loosest setting per target: among adaptive runs reaching |dE/N| <= target, the cheapest
    # by wall time (ties by matvec), reported with its cost relative to legacy.
    ad = [r for r in rows if r["study"] != "legacy" and r["exact"] == "false"]
    md += ["### Cheapest adaptive setting reaching each energy target", "",
           "| target dE/N | tol_basis | tol_trunc | dE/N | wall s | x legacy wall | matvec | x legacy matvec |",
           "|---|---|---|---|---|---|---|---|"]
    lw = fnum(leg[0]["wall"]) if leg else float("nan")
    lm = fnum(leg[0]["matvec"]) if leg else float("nan")
    for t in TARGETS:
        ok = [r for r in ad if abs(fnum(r["dE_site"])) <= t]
        if not ok:
            md.append(f"| {t:.0e} | none | | | | | | |")
            continue
        b = min(ok, key=lambda r: (fnum(r["wall"]), fnum(r["matvec"])))
        md.append(f"| {t:.0e} | {b['tol_basis']} | {b['tol_trunc']} | {fnum(b['dE_site']):.2e} | "
                  f"{fnum(b['wall']):.1f} | {fnum(b['wall']) / lw:.2f} | {b['matvec']} | "
                  f"{fnum(b['matvec']) / lm:.2f} |")
    if leg:
        md += ["", f"Legacy reference: dE/N {fnum(leg[0]['dE_site']):.2e}, "
                   f"{fnum(leg[0]['wall']):.1f} s, {leg[0]['matvec']} matvec, chi {leg[0]['chi']}."]
    return "\n".join(md) + "\n"


def main():
    d = sys.argv[1] if len(sys.argv) > 1 else "."
    files = sorted(glob.glob(os.path.join(d, "dmrgtol_*.csv")))
    print("# Adaptive CBE-DMRG tolerance study\n")
    for p in files:
        print(report(p))


if __name__ == "__main__":
    main()
