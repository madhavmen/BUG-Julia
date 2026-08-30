# GROUND STATES FOR arXiv:2209.00739, AND THE PUBLISHED NUMBERS THEY ARE SCORED AGAINST.
#
# Sherman, Dupont & Moore, "Spectral function of the J1-J2 Heisenberg model on the triangular
# lattice". Their Fig. 3 IS a ground-state accuracy figure -- E0/N against J2 for two cylinder
# circumferences, with variational-QMC stars for the infinite system -- so it is the right target
# for the ground-state half of the campaign, before any S(q,omega) machinery exists.
#
# WHAT THE PAPER ACTUALLY SPECIFIES, quoted rather than remembered:
#
#   Eq. (1)   H = J1 sum_<ij> S_i.S_j + J2 sum_<<ij>> S_i.S_j,  hbar = J1 = 1
#   Fig. 3    "cylindrical geometry with circumference C and length L = C^2", chi = 512
#   Sec. IIE  "a cylinder of circumference C = 6 and length L = 36", chi = 512, DMRG in the
#             zero-magnetization sector with the magnetization conserved
#   Sec. IIB  "we use the XC geometry throughout this work"; XC identifies TOP AND BOTTOM
#             ("the YC boundary condition would be if we identified the left and right edges")
#
# ⛔ `L` IS THE LENGTH IN COLUMNS, NOT THE SITE COUNT. `L = C^2` with `C = 6` gives `L = 36` and
# `N = C*L = 216` sites; `C = 4` gives `L = 16` and `N = 64`. A "16" in that paper is a COLUMN
# COUNT. Our own 4x4 = 16-SITE smoke is a smaller member of the same family and is NOT the paper's
# `C = 4` point -- it may not be quoted as reproducing it.
#
# ⛔ THE PAPER'S chi = 512 IS IN STATES. Under SU(2) `maxdim` is Telum's `Nkeep`, which counts
# MULTIPLETS, and the two differ by roughly the mean multiplet size (a factor of ~3 at these sizes).
# Both are recorded per row -- `chi_mult` and `chi_state` -- and only `chi_state` is comparable to
# the paper. Setting `maxdim = 512` under SU(2) asks for far MORE than the paper kept, not less.
#
# ── WHY THE ANCHORS ARE THE ONES BELOW ───────────────────────────────────────────────────────
#
# The Fig. 3 stars are "the best estimate of the infinite system ground states, using variational
# QMC, reported in Ref. 21" = Iqbal, Hu, Thomale, Poilblanc & Becca, PRB 93, 144411 (2016).
# Those are THERMODYNAMIC-LIMIT numbers and no finite cylinder reproduces them; they bound the
# extrapolation, not any single run.
#
# ⛔ SO THE ONLY EXTERNAL ANCHOR A SINGLE RUN CAN BE SCORED AGAINST IS THE 6x6 TORUS ED VALUE,
# and that is why `torus` is a mode here and not an afterthought. Iqbal et al. Tables I and IV give
# exact diagonalisation on the 36-site torus, which is a CLOSED geometry we can build exactly
# (`periodic_x = true`) and must hit to the digits printed. A cylinder run agreeing with a
# thermodynamic-limit star to 1e-2 would be a coincidence of finite-size error, not a validation --
# the torus is where "compare the accuracy of our result to the paper" has a truth value.
#
# Run:  julia -t 2 --project=. benchmarks/ground_state/paper_2209.jl [MODE] [square]
#   MODE = torus  (36 sites, EXACT anchor -- run this first, it is the certification)
#          smoke  (4x4 = 16 sites, machinery + J2 axis only, NO external anchor)
#          c4     (4 x 16 = 64 sites, the paper's C = 4 curve)
#          c6     (6 x 36 = 216 sites, the paper's C = 6 curve -- CLUSTER)
# Out:  benchmarks/results/paper2209_gs_<mode>.csv

using LinearAlgebra, Printf, Random
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate.GroundState

const MODE   = let a = filter(x -> x in ("torus", "smoke", "c4", "c6"), ARGS)
    isempty(a) ? "torus" : first(a)
end
const SQUARE = "square" in ARGS
const NSWEEPS = parse(Int, get(ENV, "P22_NSWEEPS", "40"))
const ETOL    = parse(Float64, get(ENV, "P22_ETOL", "1e-10"))
const GROW    = parse(Int, get(ENV, "P22_GROW", "1"))
const OUT     = joinpath(@__DIR__, "..", "results")

"""
Published ground-state energies per site, in units of `J1`, for the triangular `J1-J2` model.

⛔ EVERY NUMBER HERE IS QUOTED FROM A PAPER AND CARRIES ITS SOURCE. Nothing in this table was
produced by this repository, and nothing produced by this repository may be added to it -- the
moment a self-computed value sits in the anchor table, every row scored against it becomes
circular. Internal cross-checks belong in `knob_study.jl`'s `converged_reference`, which is
labelled `self:` for exactly that reason.

`ed_torus36` is the only entry a SINGLE FINITE RUN can be compared with: it is exact
diagonalisation on the same closed 36-site cluster this driver can build. `vmc_inf` and `ed_inf`
are thermodynamic-limit extrapolations -- they are the Fig. 3 stars, and a finite cylinder is not
supposed to reproduce them.
"""
const ANCHORS = Dict(
    # J2  =>  (ED on the 6x6 TORUS,  VMC infinite-2D,  source)
    0.0   => (ed_torus36 = -0.5603734, vmc_inf = -0.545321, vmc_err = 7e-6,
              src = "Iqbal PRB93 144411 (2016) Tab.I; ED col = Ref.77 therein"),
    0.125 => (ed_torus36 = -0.515564,  vmc_inf = -0.51235,  vmc_err = 2.0e-4,
              src = "Iqbal PRB93 144411 (2016) Tab.IV (VMC zero-variance extrapolation)"),
)

"The J2 values Fig. 3 sweeps. The QSL window quoted in the paper is 0.08 <~ J2 <~ 0.16."
const J2S = let s = get(ENV, "P22_J2", "")
    isempty(s) ? [0.0, 0.05, 0.1, 0.125, 0.15, 0.2, 0.3, 0.4, 0.5] :
                 [parse(Float64, x) for x in split(s, ',')]
end

const COLS = ["mode", "lattice", "sym", "arm", "J2", "Lx", "Ly", "N", "torus", "D",
              "bound", "E_site", "anchor", "anchor_kind", "dev", "chi_mult", "chi_state",
              "matvec", "sweeps", "grow", "seconds", "src"]

"""
    geometry(mode) -> (Lx, Ly, torus, D)

`Lx` is the LENGTH in columns and `Ly` the CIRCUMFERENCE, so `N = Lx*Ly`. See the header on why
the paper's `L` is `Lx` here and not `N`.
"""
function geometry(mode::String)
    # `P22_LX`/`P22_LY` override the mode's shape, which is how the Fig. 3 sweep runs several
    # circumferences at a FIXED site count from one driver instead of a second copy of the
    # anchor table. `P22_TORUS=1` closes it. The mode still supplies the default `D`.
    lx = tryparse(Int, get(ENV, "P22_LX", ""))
    ly = tryparse(Int, get(ENV, "P22_LY", ""))
    tor = get(ENV, "P22_TORUS", "") == "1"
    base = mode == "torus" ? (6, 6, true,  300) :
           mode == "smoke" ? (4, 4, false, 64)  :
           mode == "c4"    ? (16, 4, false, 200) :
           mode == "c6"    ? (36, 6, false, 512) :
           error("unknown mode $mode")
    D = parse(Int, get(ENV, "P22_D", string(base[4])))
    return (lx === nothing ? base[1] : lx,
            ly === nothing ? base[2] : ly,
            (lx === nothing && ly === nothing) ? base[3] : tor,
            D)
end

"""
    run_arm(arm, psi0, W, D, N)
        -> (E_site, chi_mult, chi_state, matvec, sweeps, grow, seconds, psi)

One ground-state solve through the BUG update flow. `arm` is `:full` (the COMPLETE complement --
`cbe_dmrg!`, `exact = true`) or `:rsvd` (the sketched selection -- `rsvd_cbe_dmrg!`).

⛔ THE TWO ARMS DIFFER ONLY INSIDE `cbe_expand`. Same local eigensolve, same growth passes, same
truncation, so `matvec` CANNOT separate them -- it counts `apply_*` calls and the expansion issues
none. A difference between the arms is wall clock and selected directions, and the `matvec` column
is here to show it is NOT the explanation.
"""
function run_arm(arm::Symbol, psi0, W, D::Int, N::Int)
    psi = copy(psi0)
    chi0 = maximum(bond_dims(psi))
    opts = CBEBugOptions(; maxdim = D, trunc_thresh = 1e-12, exact = (arm === :full))
    t0 = time()
    info = (arm === :full ? cbe_dmrg! : rsvd_cbe_dmrg!)(psi, W; opts = opts,
                                                        n_sweeps = NSWEEPS, etol = ETOL,
                                                        grow_iters = GROW)
    sec = time() - t0
    # ⛔ THE FREEZE DETECTOR, AND IT EXISTS BECAUSE A FROZEN SOLVE REPORTS `converged = true`.
    # MEASURED on the C=3 (4x3) cylinder at J2 = 0.5: both selections returned exactly -0.375 per
    # site -- the DIMER PRODUCT energy -- at bond dimension 1 after two sweeps, while exact
    # diagonalisation of the same cluster gives -0.4670250922. The state never left its start.
    #
    # ⚠ AND IT IS NOT THE "EXPANSION CORRECTLY FOUND NOTHING" CASE. That would need the start to
    # be an eigenstate, so that P_perp(H*Theta) vanishes; MEASURED, the dimer state there has
    # residual ||Hv - <H>v|| = 6.8e-01, and the IDENTICAL residual at J2 = 0.3 where the solver
    # converges perfectly. There was a direction to find and the expansion did not find it, so
    # this is a defect and must never be quoted as a data point.
    #
    # ⛔ THE TEST MUST ALSO REQUIRE THAT THE START WAS A PRODUCT STATE (`chi0 <= 2`), and leaving
    # that out made this fire on EVERY ROW once continuation was switched on. Under continuation
    # the start is the previous parameter's CONVERGED state, so "rank did not grow and it stopped
    # in two sweeps" is exactly what SUCCESS looks like -- the detector was written against a cold
    # chi = 1 start and the start changed under it. A warning that fires on every row is worse
    # than none: it trains the reader to ignore the one row that matters.
    #
    # So the signature is: started at product rank, STAYED at product rank, stopped almost at
    # once. A genuinely rank-<= 2 ground state would also trip it, which is why the message says
    # to check against a reference rather than asserting the number is wrong.
    chi1 = maximum(bond_dims(psi))
    if chi0 <= 2 && chi1 <= chi0 && length(info.energies) <= 3
        @printf("  ⛔ FROZEN: %s never left its product start (chi %d -> %d in %d sweeps) -- check against the reference before using this row.\n",
                String(arm), chi0, chi1, length(info.energies))
    end
    # Re-measured on the returned state rather than taken from the sweep's best Ritz value: the
    # sweep value is a bound in the space that solve used, and `solve!` may truncate afterwards.
    E = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    return (E / N, maximum(bond_dims(psi)), maximum(state_bond_dims(psi)),
            sum(info.krylov_dims), length(info.energies), sum(info.grow_passes), sec, psi)
end

function main()
    Lx, Ly, torus, D = geometry(MODE)
    N = Lx * Ly
    mkpath(OUT)
    # ⛔ THE GEOMETRY IS IN THE FILENAME. With `P22_LX`/`P22_LY` two circumferences at the same
    # site count are two runs of the same MODE, and a mode-only name would have the second
    # silently overwrite the first -- leaving one curve on disk labelled as if it were both.
    out = joinpath(OUT, "paper2209_gs_$(MODE)_$(Lx)x$(Ly)$(torus ? "_torus" : "")" *
                        "$(SQUARE ? "_square" : "").csv")

    set_symmetry!(:SU2)
    lattice = SQUARE ? "square" : "triangular"
    @printf("\n%s\n%s  Lx(length)=%d  Ly(circumference)=%d  N=%d  %s  D=%d (MULTIPLETS)\n",
            "="^108, lattice, Lx, Ly, N, torus ? "TORUS" : "cylinder (XC)", D)
    @printf("SU(2); the paper uses U(1) zero-magnetisation -- this is the symmetry extension.\n%s\n",
            "="^108)

    # ⚠ A DISCARDED WARM-UP: the first solve in a session pays JIT for the whole sweep stack, so
    # an un-warmed `seconds` column ranks compilation rather than method.
    run_arm(:rsvd, dimer_state(8), heisenberg_su2_mpo(8), 8, 8)
    println("(warm-up done)\n")

    # Converged state per arm, carried across the J2 sweep -- see the note at the call site.
    carried = Dict{Symbol, Any}()
    open(out, "w") do io
        println(io, join(COLS, ","))
        # ⛔ THE SQUARE LATTICE HAS NO `J2` IN THIS PAPER -- Sec. II calls it "the same Hamiltonian
        # on the square lattice with only nearest-neighbor interactions (J2 = 0)", where it is the
        # BENCHMARK against QMC, not part of the phase diagram. Sweeping `J2` here would rebuild an
        # identical operator nine times and file the identical energy under nine different `J2`
        # labels -- rows that look like a flat J2 dependence and are actually one number repeated.
        for J2 in (SQUARE ? [0.0] : J2S)
            W = SQUARE ? square_cylinder_mpo(Lx, Ly) :
                triangular_cylinder_mpo(Lx, Ly; J2 = J2, geometry = :xc,
                                        periodic_y = true, periodic_x = torus)
            psi0 = dimer_state(N)            # the S = 0 sector, which is where the ground state is
            a = get(ANCHORS, J2, nothing)
            # Only the TORUS run has an anchor a single number can be compared with; see the
            # docstring on ANCHORS.
            anchor, kind, src = if a === nothing
                (NaN, "none", "-")
            elseif torus && !SQUARE
                (a.ed_torus36, "ed_torus36(exact)", a.src)
            else
                # ⚠ NO COMMAS IN ANY CSV FIELD WRITTEN AS `%s`. This column and `src` are free
                # text and land unquoted, so a comma here silently shifts every later column.
                (a.vmc_inf, "vmc_inf(2D-limit; NOT a finite-size target)", a.src)
            end

            for arm in (:full, :rsvd)
                # ⛔ ADIABATIC CONTINUATION: each J2 starts from the PREVIOUS J2's converged
                # state, not from a fresh dimer. This is what escapes the freeze documented on
                # `run_arm`, and it is also cheaper. MEASURED on the C=3 cylinder, cold against
                # continued, every point scored against exact diagonalisation:
                #
                #   J2 = 0.5   cold -0.3750000000 (FROZEN)   continued -0.4670250922 (exact)
                #   all nine   continued exact to <= 4.3e-11, in 2-3 sweeps against 6-7 cold
                #
                # ⚠ CONTINUATION CAN PROPAGATE A BAD STATE: a frozen point would seed the next
                # and the error would run silently down the sweep. That is what the freeze
                # detector in `run_arm` and the ED column are for -- do not run a continued sweep
                # without a reference, and do not reorder `J2S` so the sweep starts at a
                # degenerate point.
                start = get(carried, arm, psi0)
                E, chim, chis, mv, swp, gr, sec, psiE = run_arm(arm, start, W, D, N)
                carried[arm] = psiE
                dev = isnan(anchor) ? NaN : E - anchor
                bound = chim >= D ? 1 : 0
                @printf(io, "%s,%s,SU2,%s,%.4f,%d,%d,%d,%d,%d,%d,%.10f,%.10f,%s,%.3e,%d,%d,%d,%d,%d,%.2f,%s\n",
                        MODE, lattice, String(arm), J2, Lx, Ly, N, torus ? 1 : 0, D,
                        bound, E, anchor, kind, dev, chim, chis, mv, swp, gr, sec, src)
                @printf("  J2=%-5.3f %-5s E/N=%+.8f  %s  chi=%d mult/%d state %s  mv=%-6d swp=%-2d %6.1fs\n",
                        J2, String(arm), E,
                        isnan(dev) ? "dev=   --    " : @sprintf("dev=%+.2e", dev),
                        chim, chis, bound == 1 ? "BOUND " : "loose⚠", mv, swp, sec)
                flush(io); flush(stdout)

                # ⛔ A VARIATIONAL METHOD CANNOT BEAT AN EXACT DIAGONALISATION OF THE SAME CLUSTER.
                # Below the ED anchor by more than roundoff means the operator is not the one the
                # anchor describes (wrong geometry, doubled bond, wrong J2 shell) -- NOT a good run.
                # ⛔ ONE LITERAL, NOT A CONCATENATION. `@printf` requires its format to be a
                # literal string; `"a" * "b"` parses fine, passes every syntax check, and throws
                # `ArgumentError` at LOAD time -- i.e. after the run has been queued.
                if kind == "ed_torus36(exact)" && E < anchor - 1e-9
                    @printf("  ⛔ E/N=%.10f is BELOW the exact %.10f by %.2e -- the Hamiltonian is not the anchor's. Stop and check the geometry.\n",
                            E, anchor, anchor - E)
                end
            end
        end
    end
    println("\nwrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
