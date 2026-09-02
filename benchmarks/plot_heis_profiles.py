"""Observable figures from the per-site profile CSVs written by heisenberg_tolerance.jl.

  heis_prof_<phase><sweep>_L<n>_D<cap>.png
      A  <Sz_j>(t) against site -- MPS markers over the EXACT ANALYTIC curve
      B  per-site |mps - exact| at the same times
      C  max-site error against t, one line per arm

WHY THIS IS A SEPARATE SCRIPT. The scalar CSV carries one number per sample (`err_prof`, the
max over sites). That number cannot say WHERE the error lives, and for a domain wall the answer
is not uniform: the front is the only place anything moves, so a max-norm that is flat in t can
still be a front that is drifting. Panels A/B are the only view that separates "the profile is
right" from "the scalar happens to agree".

⛔ THE REFERENCE IS EXACT, NOT A FINER GRID. `sz_exact` is the closed-form free-fermion profile
   <Sz_j(t)> = sum_{k occ} |[exp(-i h t)]_{jk}|^2 - 1/2 at Delta=0, valid at ANY L -- which is
   why this same script serves L=50, where no diagonalisation exists. Panel B is therefore a
   true error, not a difference between two approximations.

⛔ ONLY ARMS THAT REACHED t_max ARE RANKED. Error grows with t, so an arm interrupted at t=17
   scores better than every complete arm at t=20 and would be picked as "best". That exact defect
   put a killed configuration in the grid figure's best-config panel on 2026-09-01. Arms short of
   t_max are dropped from the ranking, counted, and reported -- never silently kept.

usage:  python3 benchmarks/plot_heis_profiles.py [resultsdir]
"""

import csv
import glob
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
RES = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "results")

FLOOR = 1e-16          # keeps a log axis from collapsing when an arm hits machine precision
NTIMES = 5             # snapshots drawn in panels A/B


def load(path):
    """-> {(tau, split): {t: {site: (mps, exact)}}}, plus L and the header fields."""
    arms = defaultdict(lambda: defaultdict(dict))
    meta = {}
    with open(path) as f:
        for r in csv.DictReader(f):
            key = (float(r["tau_trunc"]), float(r["split_cutoff"]))
            arms[key][float(r["t"])][int(r["site"])] = (float(r["sz_mps"]),
                                                        float(r["sz_exact"]))
            meta.setdefault("scheme", r["scheme"])
            meta.setdefault("phase", r["phase"])
            meta.setdefault("maxdim", r["maxdim"])
    return arms, meta


def max_err(sites):
    return max(abs(m - e) for m, e in sites.values())


def rank(arms):
    """Best/worst by max-site error at t_max -- COMPLETE ARMS ONLY (see module docstring)."""
    tmax = max(t for a in arms.values() for t in a)
    scored, dropped = {}, 0
    for key, samples in arms.items():
        if any(abs(t - tmax) < 1e-9 for t in samples):
            t = max(samples)
            scored[key] = max_err(samples[t])
        else:
            dropped += 1
    if dropped:
        print("  %d of %d arms dropped: did not reach t=%g" % (dropped, len(arms), tmax))
    if not scored:
        return None, None, tmax
    order = sorted(scored, key=lambda k: scored[k])
    return order[0], order[-1], tmax


def label(key):
    return "root=%.0e split=%.0e" % key


def figure(path):
    arms, meta = load(path)
    if not arms:
        return
    best, worst, tmax = rank(arms)
    if best is None:
        print("  no complete arm in", os.path.basename(path))
        return

    samples = arms[best]
    ts = sorted(samples)
    # evenly spaced snapshots INCLUDING both endpoints: t=0 shows the initial wall, t_max the
    # spread front. Fewer than NTIMES samples simply draws all of them.
    picks = ([ts[round(i * (len(ts) - 1) / (NTIMES - 1))] for i in range(NTIMES)]
             if len(ts) >= NTIMES else ts)
    sites = sorted(samples[ts[0]])

    fig, (a1, a2, a3) = plt.subplots(1, 3, figsize=(16.5, 4.8))
    cmap = plt.get_cmap("viridis")

    for i, t in enumerate(picks):
        c = cmap(i / max(1, len(picks) - 1))
        mps = [samples[t][j][0] for j in sites]
        exa = [samples[t][j][1] for j in sites]
        a1.plot(sites, exa, "-", color=c, lw=2.6, alpha=.55)
        a1.plot(sites, mps, "o", color=c, ms=4.2, mfc="none", mew=1.3,
                label="t=%g" % t)
        a2.semilogy(sites, [max(abs(m - e), FLOOR) for m, e in
                            ((samples[t][j][0], samples[t][j][1]) for j in sites)],
                    "o-", color=c, ms=3.4, lw=1.2, label="t=%g" % t)

    a1.set_title("A  $\\langle S^z_j\\rangle$   line = exact, markers = MPS\n%s"
                 % label(best), fontsize=10)
    a1.set_xlabel("site $j$")
    a1.set_ylabel("$\\langle S^z_j\\rangle$")
    a1.legend(fontsize=8, ncol=2)

    a2.set_title("B  per-site error  |MPS $-$ exact|", fontsize=10)
    a2.set_xlabel("site $j$")
    a2.set_ylabel("$|\\langle S^z_j\\rangle - $ exact$|$")
    a2.legend(fontsize=8, ncol=2)

    for key, samples in sorted(arms.items()):
        tt = sorted(samples)
        ee = [max(max_err(samples[t]), FLOOR) for t in tt]
        hot = key in (best, worst)
        a3.semilogy(tt, ee, "-", lw=2.2 if hot else .8,
                    color=("#c1440e" if key == best else
                           "#33507a" if key == worst else "#b8b8b8"),
                    alpha=1.0 if hot else .55, zorder=3 if hot else 1,
                    label=("best  " + label(key)) if key == best else
                          ("worst " + label(key)) if key == worst else None)
    a3.set_title("C  max-site error against t   (%d arm%s)"
                 % (len(arms), "" if len(arms) == 1 else "s"), fontsize=10)
    a3.set_xlabel("t")
    a3.set_ylabel("$\\max_j |\\langle S^z_j\\rangle - $ exact$|$")
    a3.legend(fontsize=8)

    for ax in (a1, a2, a3):
        ax.grid(True, which="both", lw=.4, alpha=.35)
        ax.tick_params(labelsize=9)

    base = os.path.basename(path).replace("heis_", "").replace("_prof", "").replace(".csv", "")
    # ⚠ `t_max` IS THE DEEPEST TIME IN THE FILE, NOT THE RUN'S TARGET T. This figure is
    # regenerated while the campaign is still running, so labelling it "T=" would assert a
    # finished run every time it is refreshed mid-arm.
    fig.suptitle("%s   %s   maxdim=%s   t_max in file = %g"
                 % (meta["phase"], meta["scheme"], meta["maxdim"], tmax), fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .95))
    out = os.path.join(RES, "heis_prof_%s.png" % base)
    fig.savefig(out, dpi=160)
    plt.close(fig)
    print("wrote", out)


def main():
    paths = sorted(glob.glob(os.path.join(RES, "heis_*_prof_*.csv")))
    if not paths:
        print("no profile CSVs in", RES)
        return
    for p in paths:
        print(os.path.basename(p))
        figure(p)


if __name__ == "__main__":
    main()
