#!/usr/bin/env python3
"""Pick the production BUG settings from the chi=128 knob scans, and write them to disk.

  python3 select_knobs.py [--target N] [geom ...]

Reads  results/knobs_l*_A_*.csv and results/knobs_l*_B_*.csv
Writes results/bug_optimal_<geom>_D<cap>.json  <- read back by `bug_knobs.jl`

WHY THIS IS A SCRIPT AND NOT A JUDGEMENT CALL IN A COMMIT MESSAGE. The campaign's plan is
"tune at chi=128, then run chi=512 with those settings". If the chosen settings are retyped by
hand into the production driver, the chi=512 run is no longer traceable to the measurement that
justified it -- and a transcription slip in one tolerance is invisible in the output. The winner
is selected here, written to a file, and read by the driver.

⛔ "OPTIMAL" NEEDS A DEFINITION OR IT MEANS "THE ONE I LIKED". Two obvious rules are both wrong:
picking the LOWEST error always selects the most expensive arm, and picking the FASTEST always
selects the least accurate. The rule used here is a Pareto choice with an explicit accuracy
target:

    among rows whose err <= TARGET * (best err over all BUG rows), take the SMALLEST seconds

with TARGET = 10 by default. That reads as "the cheapest configuration that stays within a factor
of ten of the best accuracy anyone achieved", which is a statement a reader can disagree with --
which is the point. It is printed with every selection.

⛔ AND THE SELECTION IS AN EXTRAPOLATION ACROSS BOND DIMENSION. The scan runs at chi=128; the
production run is chi=512. The optimal Krylov depth or preselection tolerance at one rank is not
guaranteed optimal at another -- deeper Krylov matters more when the basis is richer, and the
sketch's payoff grows with chi because it shrinks work that scales with chi. The JSON records the
chi it was measured at so the driver can say so out loud rather than implying the settings were
measured where they are being used.

⚠ ROWS AT THE REFERENCE FLOOR ARE NOT SEPARABLE. `exact_correlator` stops each expv at 1e-12, so
several arms can report ~1e-12 and be indistinguishable. When the best error is at the floor the
tie is broken by cost alone, and the JSON says `at_floor: true`.
"""
import csv
import glob
import json
import os
import sys
from collections import defaultdict

sys.stdout.reconfigure(errors="replace")

HERE = os.path.dirname(os.path.abspath(__file__))
RES = os.path.join(HERE, "..", "results")
REF_FLOOR = 1e-12

# Which knob each scanned `knob` column maps onto, and how to parse its value.
KNOB_CAST = {
    "m": int,
    "stol_pre": float,
    "split_cutoff": float,
    "root_cutoff": float,
    "dex": int,
    "dover": int,
    "comp_ratio": float,
    "growth": float,
}


def load(study):
    rows = []
    for fn in glob.glob(os.path.join(RES, f"knobs_l*_{study}_*.csv")):
        with open(fn, encoding="utf-8") as f:
            for r in csv.DictReader(f):
                if r["study"] != study:
                    continue
                r["err"] = float(r["err"])
                r["seconds"] = float(r["seconds"])
                r["D"] = int(r["D"])
                rows.append(r)
    return rows


def pick(rows, target):
    """The Pareto choice described in the module docstring, over BUG rows only."""
    bug = [r for r in rows if r["arm"].startswith("bug")]
    if not bug:
        return None
    best = min(r["err"] for r in bug)
    thresh = max(best * target, REF_FLOOR)
    ok = [r for r in bug if r["err"] <= thresh]
    win = min(ok, key=lambda r: r["seconds"])
    return win, best, thresh, best <= REF_FLOOR


def main():
    target = 10.0
    args = sys.argv[1:]
    if "--target" in args:
        i = args.index("--target")
        target = float(args[i + 1])
        del args[i:i + 2]
    want = set(args) if args else None

    A, B = load("A"), load("B")
    if not A and not B:
        print("no knob CSVs -- run knobs_l20.jl A and B first")
        return

    # ⛔ GROUP BY (GEOMETRY, CAP), NOT BY GEOMETRY. `pick` ranks on `seconds` among rows within a
    # factor of the best error, and a chi=64 row is ALWAYS faster than a chi=128 one -- so pooling
    # the caps makes the cap win every time and reports it as a knob choice. The two caps also have
    # genuinely different winners by construction: `dex` is an absolute budget, `growth` a
    # proportional one.
    keys = sorted({(r["geom"], r["D"]) for r in A + B
                   if want is None or r["geom"] in want})
    for g, D in keys:
        out = {"geom": g, "D": D, "target_factor": target}
        note = []
        for study, rows in (("A", A), ("B", B)):
            gr = [r for r in rows if r["geom"] == g and r["D"] == D]
            if not gr:
                continue
            res = pick(gr, target)
            if res is None:
                continue
            win, best, thresh, at_floor = res
            out[f"study_{study}"] = {
                "arm": win["arm"], "knob": win["knob"], "value": win["value"],
                "err": win["err"], "seconds": win["seconds"], "chi_measured": win["D"],
                "best_err": best, "threshold": thresh, "at_floor": at_floor,
                "N": int(win["N"]),
            }
            # Translate the winning row back into the keyword it sets.
            k = win["knob"]
            if k in KNOB_CAST:
                try:
                    out.setdefault("knobs", {})[k] = KNOB_CAST[k](win["value"])
                except ValueError:
                    pass
            elif k == "krylov_tol(beta)":
                out.setdefault("knobs", {})["m"] = 0        # the adaptive rule
            elif k in ("baseline", "parallel", "exact"):
                note.append(f"study {study} winner is the {k} row -- baseline settings stand")
            out[f"study_{study}"]["note"] = note
            print(f"[{g} D={D}] study {study}: winner {win['arm']:<22} "
                  f"err={win['err']:.3e}  {win['seconds']:.2f}s  "
                  f"(best err {best:.3e}, threshold {thresh:.3e}"
                  f"{', AT REFERENCE FLOOR' if at_floor else ''})")

        if "knobs" not in out:
            out["knobs"] = {}
            print(f"[{g} D={D}] no knob overrode the baseline -- production uses BUG_TOL/RSVD_KNOBS")
        p = os.path.join(RES, f"bug_optimal_{g}_D{D}.json")
        with open(p, "w") as f:
            json.dump(out, f, indent=2)
        print(f"[{g}] wrote {p}")
        print(f"[{g}] ⚠ measured at chi={out.get('study_A', out.get('study_B', {})).get('chi_measured', '?')}; "
              "using it at chi=512 is an extrapolation across bond dimension")


if __name__ == "__main__":
    main()
