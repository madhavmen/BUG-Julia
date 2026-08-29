#!/usr/bin/env python3
"""The wall-time and bond-dimension-growth figures of arXiv:2606.02930, for BUG vs 2-site TDVP.

FIGURE 1 (`logtime_traj_<model>.png`) -- three panels ALONG the imaginary-time trajectory, which
is the form the paper's own figures take and the reason a summary table will not do: the argument
for a logarithmic grid is entirely about how the LATE, HUGE steps behave, and an end-of-run number
cannot show whether the rank exploded at step 3 and was truncated back afterwards.

  LEFT    cumulative wall time vs tau
  MIDDLE  max bond dimension vs tau   <- the "bond dimension growth" plot
  RIGHT   energy error vs tau, against the analytic Bethe reference

FIGURE 2 (`krylov_depth.png`) -- how deep a Krylov basis one step needs, as a function of dt.

  ⛔ THIS PANEL EXISTS TO SETTLE A THEORETICAL OBJECTION, and it is worth stating plainly because
  I got the reading wrong first. BUG's defining property (Ceruti-Lubich) is an error bound
  INDEPENDENT of the smallest singular value, where projector-splitting TDVP carries a
  `1/sigma_min` -- so BUG should NOT fail where 2-site TDVP fails. Yet on the log grid at L=20
  both broke at `dt ~ 46-54`. I proposed a robustness failure, then overflow. The third
  explanation is the one the code points at: `krylov_basis = 30` with `krylov_tol = 1e-6`, against
  `tau*||H|| ~ 1000` at `dt = 50`. If the LEFT panel shows the error falling to machine precision
  once `m` is large enough, the ceiling was a PARAMETER CHOICE and the robustness theory stands.

  LEFT    infidelity vs m, one curve per dt
  RIGHT   the m needed to reach 1e-10, vs dt -- the design curve a log grid actually needs, since
          a single `krylov_basis` has to cover three orders of magnitude of dt within one run.

Run:  python3 benchmarks/imaginary_time/plot_logtime_traj.py
"""
import csv
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = os.path.dirname(os.path.abspath(__file__))
OUTDIR = os.path.join(HERE, "..", "results")
TRAJ = os.path.join(OUTDIR, "logtime_traj.csv")
KDEPTH = os.path.join(OUTDIR, "krylov_depth.csv")

SCHEME_COLOUR = {"bug": "tab:blue", "tdvp2": "tab:red"}
# the step count is the STRUCTURAL variable within a model, so it gets the linestyle
NSTEP_STYLE = {8: ("-", "o"), 12: ("--", "s"), 24: (":", "^")}
FLOOR = 1e-16          # a zero error would vanish on a log axis; floor at double-precision noise


def read(path):
    if not os.path.exists(path):
        return None
    with open(path, newline="") as fh:
        return list(csv.DictReader(fh))


def _f(row, key):
    """Read a float, mapping the NaN a failed arm writes onto None so it breaks the line."""
    try:
        v = float(row[key])
    except (TypeError, ValueError):
        return None
    return v if v == v else None       # NaN != NaN


def group(rows, key):
    out = defaultdict(list)
    for r in rows:
        out[key(r)].append(r)
    return out


def traj_panel(ax, rows, ykey, ylabel, logy, floor=None):
    for (scheme, n), pts in sorted(group(rows, lambda r: (r["scheme"], int(r["nsteps"]))).items()):
        pts.sort(key=lambda r: int(r["step"]))
        xs, ys = [], []
        for r in pts:
            y = _f(r, ykey)
            if y is None:
                continue                      # a broken step contributes no point, not a zero
            xs.append(_f(r, "tau"))
            ys.append(max(y, floor) if floor else y)
        if not xs:
            continue
        ls, mk = NSTEP_STYLE.get(n, ("-", "o"))
        ax.plot(xs, ys, ls, marker=mk, ms=4, color=SCHEME_COLOUR.get(scheme, "tab:grey"),
                label=f"{scheme}, n={n}")
    ax.set_xscale("log")
    if logy:
        ax.set_yscale("log")
    ax.set_xlabel(r"imaginary time $\tau$")
    ax.set_ylabel(ylabel)
    ax.grid(alpha=0.3, which="both", lw=0.4)


def figure_trajectory(rows):
    for model, mrows in group(rows, lambda r: r["model"]).items():
        fig, axes = plt.subplots(1, 3, figsize=(15, 4.4))
        traj_panel(axes[0], mrows, "seconds", "cumulative wall time [s]", True)
        traj_panel(axes[1], mrows, "chi", r"max bond dimension $\chi$", False)
        traj_panel(axes[2], mrows, "err", "|E - E_Bethe|", True, floor=FLOOR)
        axes[0].set_title(f"{model}: cost", fontsize=10, loc="left")
        axes[1].set_title("bond-dimension growth", fontsize=10, loc="left")
        axes[2].set_title("accuracy vs analytic Bethe", fontsize=10, loc="left")
        axes[0].legend(fontsize=8, framealpha=0.9)
        fig.tight_layout()
        out = os.path.join(OUTDIR, f"logtime_traj_{model}.png")
        fig.savefig(out, dpi=150)
        plt.close(fig)
        print("wrote", out)


def figure_depth(rows):
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.4))
    by_dt = group(rows, lambda r: float(r["dt"]))
    needed = []
    cmap = plt.get_cmap("viridis")
    dts = sorted(by_dt)
    for i, dt in enumerate(dts):
        pts = sorted(by_dt[dt], key=lambda r: int(r["m"]))
        ms = [int(r["m"]) for r in pts]
        es = [_f(r, "err_vs_deep") for r in pts]
        col = cmap(i / max(len(dts) - 1, 1))
        xs = [m for m, e in zip(ms, es) if e is not None]
        ys = [max(e, FLOOR) for e in es if e is not None]
        axes[0].plot(xs, ys, "-o", ms=4, color=col, label=f"dt = {dt:g}")
        # the smallest m that got within 1e-10 of the deep reference: the design number
        hit = next((m for m, e in zip(ms, es) if e is not None and e < 1e-10), None)
        if hit is not None:
            needed.append((dt, hit))
    axes[0].axhline(1e-10, color="k", lw=0.8, ls=":")
    axes[0].set_xscale("log")
    axes[0].set_yscale("log")
    axes[0].set_xlabel("Krylov vectors per bond, m")
    axes[0].set_ylabel("infidelity vs deep reference")
    axes[0].set_title("one step: convergence in m", fontsize=10, loc="left")
    axes[0].legend(fontsize=8)
    axes[0].grid(alpha=0.3, which="both", lw=0.4)

    if needed:
        axes[1].plot([d for d, _ in needed], [m for _, m in needed], "-o", color="tab:blue")
    axes[1].set_xscale("log")
    axes[1].set_xlabel("step size dt")
    axes[1].set_ylabel("m needed for 1e-10")
    axes[1].set_title("the design curve a log grid needs", fontsize=10, loc="left")
    axes[1].grid(alpha=0.3, which="both", lw=0.4)

    fig.tight_layout()
    out = os.path.join(OUTDIR, "krylov_depth.png")
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print("wrote", out)


def main():
    traj, kd = read(TRAJ), read(KDEPTH)
    if traj is None and kd is None:
        sys.exit("no CSVs -- run benchmarks/imaginary_time/logtime_trajectory.jl and "
                 "benchmarks/imaginary_time/krylov_depth_scan.jl first")
    if traj:
        figure_trajectory(traj)
    else:
        print("skipping trajectory figure:", TRAJ, "missing")
    if kd:
        figure_depth(kd)
    else:
        print("skipping depth figure:", KDEPTH, "missing")


if __name__ == "__main__":
    main()
