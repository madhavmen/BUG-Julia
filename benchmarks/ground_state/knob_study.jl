# WHICH KNOBS BUY ACCURACY PER UNIT COST IN THE GROUND-STATE SEARCH, AND WHERE THE rSVD PAYS.
#
# Three arms, and the first pair is a DIFFERENT QUESTION from the second:
#
#   2-site CBE-DMRG   `GroundState.solve!`    O(d^2 chi^3) per solve, rank from the 2-site block
#   1-site CBE-DMRG   `exact_cbe_dmrg1s!`     O(d chi^3) + an EXACT SVD of H*Theta
#   1-site CBE-rSVD   `rsvd_cbe_dmrg1s!`      O(d chi^3) + a THIN SKETCH          <- the method
#
# ⛔ `matvec` SEPARATES 1-SITE FROM 2-SITE AND IS BLIND TO exact-VS-rSVD. The two 1-site arms
# issue the SAME eigensolve calls and differ only inside `cbe_expand`, which is QR and matmul and
# issues no `expv` at all. MEASURED by `su2_gate.jl` at L=8: 75 matvec for BOTH, to every digit.
# So the exact-vs-rSVD question is WALL CLOCK ONLY, and wall clock on the dev box spreads 2.6-2.8x
# on bit-identical work -- that half belongs on the cluster, one job at a time. The columns are
# emitted here regardless so the two studies share one schema; a `seconds` column measured under
# contention must simply not be quoted.
#
# ⛔ AND EVERY ROW WHERE THE CAP DID NOT BIND MEASURES NOTHING. At a loose `maxdim` the state
# converges before the expansion matters and every arm sits on the same floor: `su2_gate.jl` at
# L=8, D=64 returned 1e-15 for all ten arms because `chi` reached 6 (SU(2)) / 16 (`:none`) and
# never approached the cap. `gse_rsvd_vs_exact.jl` recorded the same thing -- "at D=64 (full rank
# at L=12) the two arms were bit-identical". The `bound` column is therefore not decoration: a row
# with `bound = 0` is a row about roundoff, and the driver says so in the console rather than
# leaving it to be read off the numbers.
#
# ⚠ THE REFERENCE IS NOT THE SAME KIND OF OBJECT FOR ALL THREE MODELS, and the `ref` column says
# which. The chain has the Bethe ansatz -- exact, external, residual asserted. The cylinder and the
# kagome cluster do not, so their knob rows are scored against a CONVERGED RUN OF THE SAME MODEL at
# a large cap. That is legitimate for a knob study (the question is which setting gets closest for
# a given budget, which is internal) and is NOT an accuracy claim. The external cylinder anchor is
# the Loop-QMC slope of arXiv:1705.03222, which needs `Lxs = (4,6,8)` and is cluster work.
#
# ⛔ A SELF-REFERENCE COMPUTED WITH ONE ARM WOULD FAVOUR THAT ARM. `converged_reference` therefore
# runs TWO different arms (2-site exact and 1-site exact) at the large cap and REFUSES to return a
# number unless they agree -- the same discipline as asserting the Bethe residual. A disagreement
# means the reference itself is not converged, and no row scored against it could be quoted.
#
# Run:  julia -t 2 --project=. benchmarks/ground_state/knob_study.jl [A|B] [chain|cylinder|kagome]
# Out:  benchmarks/results/gs_knob_<study>_<models>.csv

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate.GroundState

include(joinpath(@__DIR__, "..", "..", "tests", "common", "analytic_reference.jl"))

const SMOKE   = "smoke" in ARGS
const WHICH   = let i = findfirst(a -> a in ("A", "B"), ARGS); i === nothing ? "A" : ARGS[i] end
const WANT    = let a = filter(x -> x in ("chain", "cylinder", "triangle", "kagome"), ARGS)
    isempty(a) ? ["chain", "cylinder", "triangle", "kagome"] : a
end
const NSWEEPS = parse(Int, get(ENV, "GS_NSWEEPS", "30"))
const ETOL    = parse(Float64, get(ENV, "GS_ETOL", "1e-12"))
const DREF    = parse(Int, get(ENV, "GS_DREF", "128"))

# ⛔ TWO chi COLUMNS, AND CONFUSING THEM IS HOW A COMPARISON GOES WRONG SILENTLY.
# `bond_dims` counts MULTIPLETS, `state_bond_dims` counts STATES; under `:none`/`:U1` they are the
# same number, under SU(2) they are not (6 multiplets = 16 states on the L=8 chain). An earlier
# version of `su2_gate.jl` read one function for the 2-site arms and the other for the 1-site arms
# and produced a `chi` column that looked like a method difference and was a units difference.
#
#   `chi_mult`  is what `maxdim` CUTS -- Telum's `Nkeep` is in multiplets -- so the `bound` test
#               (did the cap actually bind?) must use THIS one.
#   `chi_state` is the physical rank, and the only one comparable ACROSS symmetry modes.
const COLS = ["study", "model", "sym", "arm", "knob", "value", "L", "D", "bound",
              "err", "chi_mult", "chi_state", "matvec", "sweeps", "seconds", "ref"]

# ── the baseline every one-factor-at-a-time row is measured against ──────────────────────────
Base.@kwdef struct Knobs
    growth::Float64 = 2.0
    dover::Union{Nothing, Int} = nothing
    comp_ratio::Union{Float64, Nothing} = 0.5
    dex::Int = 0
    maxiter::Int = 30
    restol::Float64 = 1e-10
    grow_iters::Int = 1
    seed_expanded::Bool = false
end

# `Base.@kwdef` gives no copy-with-one-field-changed constructor, and the one-factor-at-a-time
# study is nothing but that operation, so it is written once here.
Knobs(k::Knobs; kw...) = Knobs(;
    growth        = get(kw, :growth,        k.growth),
    dover         = get(kw, :dover,         k.dover),
    comp_ratio    = get(kw, :comp_ratio,    k.comp_ratio),
    dex           = get(kw, :dex,           k.dex),
    maxiter       = get(kw, :maxiter,       k.maxiter),
    restol        = get(kw, :restol,        k.restol),
    grow_iters    = get(kw, :grow_iters,    k.grow_iters),
    seed_expanded = get(kw, :seed_expanded, k.seed_expanded))

"""
    run_arm(arm, psi0, W, D, k) -> (E, chi_mult, chi_state, matvec, sweeps, seconds)

One ground-state solve. `arm` is `:two_exact`, `:two_rsvd`, `:one_exact`, `:one_rsvd`.

⚠ `seed_expanded` is a 1-site keyword and `grow_iters` a 2-site one; passing either to the other
solver is a `MethodError`, so the split is made here rather than at every call site.
"""
function run_arm(arm::Symbol, psi0, W, D::Int, k::Knobs)
    psi = copy(psi0)
    t0 = time()
    if arm === :two_exact || arm === :two_rsvd
        opts = CBEBugOptions(; growth = k.growth, dover = k.dover, comp_ratio = k.comp_ratio,
                             dex = k.dex, maxdim = D, trunc_thresh = 1e-12,
                             maxiter = k.maxiter, tol = k.restol,
                             exact = (arm === :two_exact))
        info = solve!(psi, W; opts = opts, n_sweeps = NSWEEPS, etol = ETOL,
                      grow_iters = k.grow_iters)
        mv, swp = sum(info.krylov_dims), length(info.energies)
    else
        run = dmrg_cbe1s!(psi, W; n_sweeps = NSWEEPS, etol = ETOL, growth = k.growth,
                          dover = k.dover, comp_ratio = k.comp_ratio, dex = k.dex,
                          maxdim = D, trunc_thresh = 1e-12, maxiter = k.maxiter,
                          restol = k.restol, exact = (arm === :one_exact),
                          seed_expanded = k.seed_expanded)
        mv, swp = sum(run.krylov_dims), length(run.energies)
    end
    sec = time() - t0
    E = real(mpo_energy(copy(psi), W)) / max(norm(psi)^2, eps())
    return (E, maximum(bond_dims(psi)), maximum(state_bond_dims(psi)), mv, swp, sec)
end

"""
    converged_reference(psi0, W) -> Float64

A reference for a model with no closed form: the SAME model solved at `DREF` by TWO different
arms, which must agree. Refuses rather than returning an unconverged number.
"""
function converged_reference(psi0, W)
    k = Knobs()
    e2, = run_arm(:two_exact, psi0, W, DREF, k)
    e1, = run_arm(:one_exact, psi0, W, DREF, k)
    d = abs(e1 - e2)
    d < 1e-8 || error("reference NOT converged at D=$DREF: 2-site $e2 vs 1-site $e1 " *
                      "(differ by $d) -- raise GS_DREF; no row scored against this is quotable")
    return min(e1, e2)
end

"A model: the start state, the operator, the reference and how that reference was obtained."
struct Model
    name::String
    sym::Symbol
    L::Int
    psi0::Any
    W::Any
    eref::Float64
    ref::String
end

function build(name::String)
    if name == "chain"
        L = SMOKE ? 12 : 16
        set_symmetry!(:SU2)
        E, _, resid = bethe_xxx_open_energy(L)
        resid < 1e-12 || error("Bethe residual $resid at L=$L")
        return Model(name, :SU2, L, dimer_state(L), heisenberg_su2_mpo(L), E, "bethe")
    elseif name == "cylinder"
        # ⚠ `Ly >= 3` for `periodic_y`: at Ly = 2 the wrap bond IS the rung bond and adding both
        # doubles it. Lx x Ly = 4 x 3 keeps the smoke at 12 sites; the QUOTABLE size is Ly = 4 with
        # Lxs = (4,6,8) against the Loop-QMC slope, and that is a cluster job.
        Lx, Ly = SMOKE ? (4, 3) : (4, 4)
        L = Lx * Ly
        set_symmetry!(:SU2)
        psi0, W = dimer_state(L), square_cylinder_mpo(Lx, Ly)
        return Model(name, :SU2, L, psi0, W, converged_reference(psi0, W), "self:D=$DREF")
    elseif name == "triangle"
        # The J1-only triangular cylinder, XC wrapping -- the lattice of arXiv:2209.00739 Sec. IVB.
        # ⚠ FRUSTRATED, so `dimer_state` is a poorer guess than on the square cylinder and the rank
        # needed for a given accuracy is markedly higher at the same site count. That is physics;
        # do not read a slower convergence here as an integrator problem.
        # ⛔ THE J2 SWEEP AND THE PUBLISHED ANCHORS LIVE IN `paper_2209.jl`, NOT HERE. This entry
        # is scored against a converged run of the SAME model, which answers "which knob setting
        # gets closest for a given budget" and is not an accuracy claim.
        Lx, Ly = SMOKE ? (4, 3) : (4, 4)
        L = Lx * Ly
        set_symmetry!(:SU2)
        psi0, W = dimer_state(L), triangular_cylinder_mpo(Lx, Ly)
        return Model(name, :SU2, L, psi0, W, converged_reference(psi0, W), "self:D=$DREF")
    elseif name == "kagome"
        # ⛔ 3 SITES PER CELL, SO L = 18 IS (Lx, Ly) = (2, 3) AND NOTHING ELSE with a wrap:
        # Lx*Ly = 6 and `periodic_y` needs Ly >= 3. (3,2) would be a STRIP, not a cylinder.
        # ⛔ AND THIS MODEL HAS NO SU(2): it is the dipolar XY Hamiltonian, not S_i.S_j, so
        # `breathing_kagome_mpo` throws under :SU2 rather than building the wrong operator.
        # `h = 0` is the isotropic lattice of arXiv:2406.00098.
        set_symmetry!(:U1)
        L = 18
        psi0 = neel_state(L)
        W = breathing_kagome_mpo(2, 3; h = 0.0, shells = 7, periodic_y = true)
        return Model(name, :U1, L, psi0, W, converged_reference(psi0, W), "self:D=$DREF")
    end
    error("unknown model $name")
end

"The knob grids. `nothing` in a value list means 'the baseline', kept so the axis reads honestly."
function grids(arm::Symbol)
    g = Pair{String, Vector{Any}}[
        "growth"     => Any[1.5, 2.0, 3.0, 4.0],
        "dover"      => Any[nothing, 0, 2, 4],
        "comp_ratio" => Any[0.25, 0.5, 0.75],
        "restol"     => Any[1e-4, 1e-6, 1e-8, 1e-10, 1e-12],
        "maxiter"    => Any[5, 10, 20, 30],
    ]
    # ⚠ THE TWO SOLVER-SPECIFIC KNOBS, and they do not exist on the other solver at all.
    # `grow_iters = 0` is kept as the FROZEN CONTROL: the frames never widen, the eigensolve
    # minimises over a one-dimensional space and the sweep cannot leave the start state. A run that
    # "converges" in two sweeps at chi = 1 is what that looks like.
    if arm === :two_exact || arm === :two_rsvd
        push!(g, "grow_iters" => Any[0, 1, 2, 3])
    else
        push!(g, "seed_expanded" => Any[false, true])
    end
    return g
end

with(k::Knobs, knob::String, v) =
    knob == "growth"     ? Knobs(k; growth = v)     :
    knob == "dover"      ? Knobs(k; dover = v)      :
    knob == "comp_ratio" ? Knobs(k; comp_ratio = v) :
    knob == "restol"     ? Knobs(k; restol = v)     :
    knob == "maxiter"    ? Knobs(k; maxiter = v)    :
    knob == "grow_iters" ? Knobs(k; grow_iters = v) :
    knob == "seed_expanded" ? Knobs(k; seed_expanded = v) :
    error("unknown knob $knob")

const ARMS = (:two_exact, :two_rsvd, :one_exact, :one_rsvd)

function emit(io, study, m::Model, arm, knob, value, D, r)
    E, chim, chis, mv, swp, sec = r
    # `chi_mult` against `D`, because `maxdim` is Telum's `Nkeep` and that is in multiplets.
    bound = chim >= D ? 1 : 0
    @printf(io, "%s,%s,%s,%s,%s,%s,%d,%d,%d,%.6e,%d,%d,%d,%d,%.3f,%s\n",
            study, m.name, String(m.sym), String(arm), knob, string(value),
            m.L, D, bound, abs(E - m.eref), chim, chis, mv, swp, sec, m.ref)
    @printf("  %-9s %-10s %-14s %-8s D=%-3d %s err=%.3e chi=%d/%d mv=%-6d swp=%-3d %6.1fs\n",
            m.name, String(arm), knob, string(value), D,
            bound == 1 ? "BOUND  " : "loose ⚠", abs(E - m.eref), chim, chis, mv, swp, sec)
    flush(io); flush(stdout)
end

"""
Study A -- ONE FACTOR AT A TIME from the baseline, at a cap that BINDS.

⛔ `D` is deliberately small. The point is error-at-fixed-rank: at a loose cap every arm reaches
the same floor and the table would say the knobs do nothing, which is a statement about `D`.
"""
function study_A(io, m::Model, Ds)
    base = Knobs()
    for D in Ds, arm in ARMS
        emit(io, "A", m, arm, "baseline", "-", D, run_arm(arm, m.psi0, m.W, D, base))
        for (knob, vals) in grids(arm), v in vals
            k = with(base, knob, v)
            emit(io, "A", m, arm, knob, v, D, run_arm(arm, m.psi0, m.W, D, k))
        end
    end
end

"Study B -- the cost curve: error at fixed rank across caps, baseline knobs only."
function study_B(io, m::Model, Ds)
    base = Knobs()
    for D in Ds, arm in ARMS
        emit(io, "B", m, arm, "cap", D, D, run_arm(arm, m.psi0, m.W, D, base))
    end
end

function main()
    out = get(ENV, "GS_KNOB_OUT",
              joinpath(@__DIR__, "..", "results", "gs_knob_$(WHICH)_$(join(WANT, "-")).csv"))
    mkpath(dirname(out))
    # ⚠ A DISCARDED WARM-UP. The first arm in a session pays JIT -- measured at 263.5s against
    # 18.4s for the identical second arm in `gse_rsvd_vs_exact.jl` -- so an un-warmed `seconds`
    # column ranks compilation rather than method.
    set_symmetry!(:SU2)
    run_arm(:one_rsvd, dimer_state(8), heisenberg_su2_mpo(8), 8, Knobs())
    run_arm(:two_rsvd, dimer_state(8), heisenberg_su2_mpo(8), 8, Knobs())
    println("(warm-up done)")

    Ds = SMOKE ? (8, 16) : (16, 32, 64)
    open(out, "w") do io
        println(io, join(COLS, ","))
        for name in WANT
            m = try
                build(name)
            catch e
                @printf("\n⛔ %s SKIPPED: %s\n", name,
                        first(split(sprint(showerror, e), '\n')))
                continue
            end
            @printf("\n%s\n%s  L=%d  %s  E_ref = %.12f  (%s)\n%s\n",
                    "="^110, m.name, m.L, String(m.sym), m.eref, m.ref, "="^110)
            WHICH == "A" ? study_A(io, m, Ds) : study_B(io, m, Ds)
        end
    end
    println("\nwrote ", out)
end

if abspath(PROGRAM_FILE) == @__FILE__
    main()
end
