#!/usr/bin/env bash
# THE arXiv:2209.00739 CAMPAIGN, IN THE ORDER IT HAS TO RUN.
#
# Stage 4 depends on stage 3 depends on stage 2 depends on stage 1, and the dependencies are real
# rather than tidy:
#
#   1 GATE       does the ground state converge to the EXACT energy, and at which chi?
#                Everything downstream starts from |Om>. An unconverged ground state does not give
#                a slightly-wrong spectrum, it gives the spectrum of a different state.
#   2 KNOBS      at chi=128, every setting scored against the EXACT correlator.
#   3 SELECT     the Pareto winner per geometry -> results/bug_optimal_<geom>.json
#   4 PRODUCE    Figs. 4 and 5 at the production cap (512 square / 256 triangle -- half of full
#                rank in each case), BUG running the settings stage 3 chose.
#
# ⛔ ONE TIMING JOB AT A TIME. Every stage past the gate reports wall clock as a headline number,
# and two Julia processes on one box spread those by ~2.6-2.8x on bit-identical work. The stages
# below are deliberately SEQUENTIAL, not backgrounded.
#
# ⛔ STAGE 4 IS CLUSTER-SCALE AND IS NOT RUN BY DEFAULT. At chi=512 with Tmax=40 (400 steps) a
# single arm is hours; three arms on three geometries is one to two days. `STAGE=4` runs it; the
# default stops after stage 3 and prints what stage 4 would cost.
#
# ⛔⛔ TWO TIERS, BECAUSE `L >> C` AND "SCORED AGAINST EXACT TRUTH" CANNOT BOTH HOLD.
#
# The paper takes L >> C, specifically L = C^2 -- "Due to computational limitations, C is quite
# small, and so we take L >> C" (Sec. IIB) and "a cylindrical geometry with circumference C and
# length L = C^2" (Fig. 3 caption); Sec. IIE runs C=6, L=36, chi=512. That is N=64 at C=4 and
# N=216 at C=6. The exact S^z=0 reference dies long before: the sector is binomial(N, N/2), so
# N=20 -> 184,756 states (2.8 MB/vector), N=24 -> 2.7 M (41 MB), N=26 -> 10.4 M (159 MB),
# N=28 -> 40 M (612 MB), and N=64 -> 1.8e18. There is no size that is both.
#
#   TIER A -- CERTIFIED (stages 1-3). Exact ED exists, so every integrator and every knob row is
#   scored against TRUTH. L/C is 1.25-1.5, so these are CORRECTNESS and COST geometries and NOT
#   Fig. 4/5 physics.
#     chain      N=20  L20        Bethe AND sparse ED (they certify each other)
#     square     N=20  L5  C4     M=(pi,pi), X=(0,pi) exact -> Fig. 4 path in full
#     triangle   N=20  L5  C4     XC: K AND M both exact (M needs 4|C, so C=4 beats C=6 here)
#
#   TIER B -- THE PAPER'S GEOMETRY (stage 4). No ED; scored by chi-convergence, the variational
#   bound, and the paper's own Fig. 3 curve.
#     triangle   N=64  L16 C4     the paper's OWN C=4 point, L = C^2 exactly. L/C = 4.
#                                 K and M both exact. 16 distinct q_x on Gamma->K.
#     triangle   N=216 L36 C6     the paper's headline (Sec. IIE), chi=512. M is a C6 image only.
#
# ⚠ `L = 3, C = 6` (N=18) IS NOT IN EITHER TIER. L/C = 0.5 -- the aspect ratio is INVERTED, not
# merely small. It runs in `STAGE=first` as a PIPELINE check (chi=512 is full rank there, so it is
# essentially exact) and must never be captioned as reproducing Fig. 5.
#
# ⛔ THE TRIANGLE IS `XC`, WHICH REMOVED THE OLD `Lx6 Ly3` ROW. arXiv:2209.00739 Sec. IIB uses XC
# throughout ("we use the XC geometry throughout this work"); this repo previously built YC while
# labelling it XC. They are different Hamiltonians -- MEASURED at 3x6, E0/N = -0.505379340823 (XC)
# vs -0.500802978159 (YC). XC identifies top and bottom with T = C*a1 - (C/2)*a2, a vertical
# ZIGZAG ring, so **C must be EVEN** and `C = 3` is not a cylinder XC can build at all.
#
# ⛔ AND THE OLD "3 | C" RULE WAS A YC ARTIFACT, NOT LATTICE PHYSICS. Under XC, `K = (4pi/3,0)` has
# q_y = 0 and satisfies Eq. (8) with n = 0 at EVERY even C -- MEASURED: the whole K->Gamma segment
# is 21/21 exact under XC and 3/21 under YC at C=6. That is the paper's stated reason for the
# choice, and it is what makes C=4 usable for Fig. 5 where YC forbade it.
#
# Gate before trusting any of this:  julia --project=. benchmarks/spectral/xc_gate.jl
#
# ⛔ FULL RANK AT THE CENTRE BOND IS 2^(N/2): 1024 at N=20, 512 at N=18, and unreachable in tier B.
# A cap at or above full rank truncates NOTHING, so such a row measures the solver's floor rather
# than the integrator under a binding cap. In tier A that makes chi=512 BINDING at N=20 and EXACT
# at N=18 -- both useful, and the two must not be read as one convergence study.
#
# Usage:  bash benchmarks/run_2209_campaign.sh [STAGE]
#         STAGE = first | knobs | cube | 1 | 2 | 3 | 4 | all   (default: 1-3)
#                 first = the certified Fig.4/5 set at chi_evo 64 and 128
#                 knobs = studies A (BUG) and B (rSVD sketch), per geometry, both caps
#                 cube  = split_cutoff x root_cutoff x depth heatmaps, same caps

set -u
cd "$(dirname "$0")/.."
JL="julia -t 2 --project=. --startup-file=no"
STAGE="${1:-123}"

# TIER A, used by stages 1-3. LX LY models...
# ⛔ `6 3 triangle` IS GONE: XC cannot close a zigzag ring on an odd circumference.
# ⛔ EVERY ROW HERE MUST STAY <= ED_MAX_N (26) or stages 2-3 lose their exact scoring.
GEOMS=(
  "5 4 chain square"
  "5 4 triangle"
)

banner() { printf '\n\033[1m=== %s ===\033[0m\n' "$*"; }

if [[ "$STAGE" == first ]]; then
  banner "FIRST IMPRESSION -- Figs. 4 and 5 at CERTIFIABLE sizes, XC wrapping"
  # ⛔ THESE ARE NOT THE PAPER'S ASPECT RATIO AND MUST NOT BE CAPTIONED AS IF THEY WERE. The paper
  # takes L = C^2; here L/C is 1.25 (5x4) and 0.50 (3x6). What these DO give is the full Fig. 4 /
  # Fig. 5 panel set on the paper's own XC geometry, at sizes where the ground state is exactly
  # known -- so a wrong panel is attributable to the pipeline rather than to the cylinder.
  #
  # ⛔ TWO DIFFERENT CAPS, AND CONFLATING THEM WOULD DESTROY THE ONE CHECK THAT MATTERS.
  #   F_D   = 64, 128   the rank the REAL-TIME EVOLUTION is held to. This is the axis the
  #                     integrator comparison is about, and both values BIND at N=18 (full rank
  #                     512) and N=20 (1024), so every row is a genuine truncation measurement.
  #   F_DGS = 512       the DMRG ground-state cap. Held CONVERGED and independent of F_D.
  # If the ground state were also capped at 64 the arms would each be evolving a different,
  # non-ground state, and the spin-wave comparison downstream could not distinguish "the
  # integrator truncated" from "we started somewhere else". MEASURED at N=18: a 256 cap with 30
  # sweeps still gave dev=+9.9e-05 with converged=false.
  for D in 64 128; do
    for spec in "square 5 4" "triangle 5 4" "triangle 3 6"; do
      set -- $spec
      echo "-- Fig4/5  $1  L=$2 C=$3  N=$(($2*$3))  chi_evo=$D  chi_gs=512"
      F_D=$D F_DGS=512 F_TMAX=40 F_DT=0.1 \
        $JL benchmarks/spectral/fig45_2209.jl "$1" "$2" "$3" || exit 1
      python3 benchmarks/spectral/plot_fig45_2209.py "$1" "$2x$3_D$D"
    done
  done
  exit 0
fi

if [[ "$STAGE" == knobs ]]; then
  banner "KNOB SCANS -- studies A (BUG) and B (rSVD sketch) per geometry, at BOTH caps"
  # ⛔ BOTH CAPS, AND THEY ARE NOT REDUNDANT. `dex`/`dover` are ABSOLUTE per-side budgets while
  # `growth` is PROPORTIONAL to the neighbourhood rank, so a winner at chi=64 has no claim at
  # chi=128 unless it is one of the proportional knobs. `select_knobs.py` groups on (geom, D) and
  # files `bug_optimal_<geom>_D<cap>.json` separately for exactly this reason -- pooling them lets
  # the cheaper CAP win on wall clock and reports it as a knob.
  for D in 64 128; do
    for spec in "square 5 4" "triangle 5 4" "triangle 3 6"; do
      set -- $spec
      for study in A B; do
        echo "-- knobs $study  $1  L=$2 C=$3  N=$(($2*$3))  chi_evo=$D"
        K20_LX=$2 K20_LY=$3 K20_D=$D K20_DGS=512 \
          $JL benchmarks/spectral/knobs_l20.jl "$study" "$1" || exit 1
      done
    done
  done
  python3 benchmarks/spectral/plot_knobs_l20.py A
  python3 benchmarks/spectral/plot_knobs_l20.py B
  python3 benchmarks/spectral/select_knobs.py
  exit 0
fi

if [[ "$STAGE" == cube ]]; then
  banner "KNOB CUBE -- split_cutoff x root_cutoff x Krylov depth, per geometry and cap"
  # The 2D analogue of `benchmarks/su2_knob_cube.jl`, scored against the EXACT G(x,t) instead of a
  # bond-energy profile. 4 split x 4 root x 4 depths = 64 runs per (geometry, cap).
  #
  # ⚠ TMAX=4 HERE, NOT THE 8 STUDY A/B USE, AND IT IS A DELIBERATE TRADE. 64 runs at 80 steps
  # each is 5120 steps per panel-set; at 40 steps it is 2560. The cube's job is to RANK the three
  # knobs against each other, and truncation error accumulates monotonically, so the ranking
  # survives the shorter window. What the shorter window CANNOT see is a knob whose error only
  # turns on late -- so the winner it picks is still confirmed at full Tmax by the figure run.
  for D in 64 128; do
    for spec in "square 5 4" "triangle 5 4" "triangle 3 6"; do
      set -- $spec
      echo "-- cube $1  L=$2 C=$3  N=$(($2*$3))  chi_evo=$D"
      K20_LX=$2 K20_LY=$3 K20_D=$D K20_DGS=512 K20_TMAX=4.0 \
        $JL benchmarks/spectral/knobs_l20.jl C "$1" || exit 1
      pre=$( [ "$1" = square ] && echo "sq$2x$3D$D" || echo "tri$2x$3D$D" )
      python3 benchmarks/plot_knob_cube.py benchmarks/results "$(($2*$3))" "$pre"
    done
  done
  exit 0
fi

if [[ "$STAGE" == *1* || "$STAGE" == all ]]; then
  banner "STAGE 1 -- ground-state gate (chi ladder, vs exact)"
  for g in "${GEOMS[@]}"; do
    set -- $g; lx=$1; ly=$2; shift 2
    for m in "$@"; do
      echo "-- gate $m  Lx=$lx Ly=$ly  N=$((lx*ly))"
      G20_LX=$lx G20_LY=$ly $JL benchmarks/ground_state/gate_l20.jl "$m" || exit 1
    done
  done
fi

if [[ "$STAGE" == *2* || "$STAGE" == all ]]; then
  banner "STAGE 2 -- knob scans at chi=128, scored against the exact correlator"
  for g in "${GEOMS[@]}"; do
    set -- $g; lx=$1; ly=$2; shift 2
    for m in "$@"; do
      for study in A B; do
        echo "-- knobs $study  $m  Lx=$lx Ly=$ly"
        K20_LX=$lx K20_LY=$ly K20_D=128 K20_DGS=512 \
          $JL benchmarks/spectral/knobs_l20.jl "$study" "$m" || exit 1
      done
    done
  done
  python3 benchmarks/spectral/plot_knobs_l20.py A
  python3 benchmarks/spectral/plot_knobs_l20.py B
fi

if [[ "$STAGE" == *3* || "$STAGE" == all ]]; then
  banner "STAGE 3 -- select the production settings from the chi=128 scans"
  python3 benchmarks/spectral/select_knobs.py || exit 1
fi

if [[ "$STAGE" == *4* || "$STAGE" == all ]]; then
  banner "STAGE 4 -- Figs. 4 and 5 at the production cap, with the tuned settings"
  # ⛔ `F_TUNED=1` (the default) makes the driver LOAD stage 3's choice and PRINT its provenance.
  # The settings were measured at chi=128 and are used here: an extrapolation across bond
  # dimension, which the driver states rather than implying.
  #
  # ⛔ THE TRIANGLES RUN AT 256, NOT 512, AND THAT IS NOT A BUDGET CUT. Full rank at N=18 is
  # 2^9 = 512, so a 512 cap there truncates nothing -- the run would be exact, which measures the
  # solver's floor instead of the integrator under a binding cap, and would make the triangle
  # columns incomparable with the square's. 256 is half of full rank on the triangles exactly as
  # 512 is half of full rank on the square.
  # ⛔ THE TRIANGLE RUNS AT THE PAPER'S OWN GEOMETRY, L = C^2, NOT AT A CERTIFIED SIZE. There is
  # no ED at N=64; the claim these rows support is chi-convergence plus agreement with Fig. 3,
  # and the driver prints which one it is using.
  F_D=512 F_TMAX=40 F_DT=0.1 $JL benchmarks/spectral/fig45_2209.jl square   5  4 || exit 1
  F_D=512 F_TMAX=40 F_DT=0.1 $JL benchmarks/spectral/fig45_2209.jl triangle 16 4 || exit 1
  python3 benchmarks/spectral/plot_fig45_2209.py square   5x4
  python3 benchmarks/spectral/plot_fig45_2209.py triangle 16x4
else
  banner "STAGE 4 NOT RUN"
  cat <<'EOF'
  chi=512 (square) / 256 (triangles), Tmax=40 (400 steps), three arms, three geometries.
  Rough local cost: HOURS per arm, one to two DAYS for the set. Run it with

      bash benchmarks/run_2209_campaign.sh 4

  or dispatch the three fig45_2209.jl invocations separately -- they are independent, but each
  one internally times three integrators, so a single invocation must own the box.
EOF
fi
