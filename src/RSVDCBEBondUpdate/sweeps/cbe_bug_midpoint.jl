
"""
    cbe_bug_midpoint_step!(psi, mpo, tau; kwargs...) -> CBEBugSweepInfo

One MIDPOINT BUG step (Ceruti-Kusch-Lubich, arXiv:2402.08607), in place.

TWO STAGES, and the second is an ordinary [`cbe_bug_step!`](@ref) with its bases taken from the
first:

  1. a HALF STEP, `tau/2`, whose only product is a midpoint state `Y_{1/2}`;
  2. a FULL step over `tau` whose frames are built from `Y_{1/2}` but which integrates the
     Galerkin ODE from the ORIGINAL `Y_0`, then truncates once.

⛔ IT IS EXPLICIT, DESPITE THE NAME. The midpoint *rule* is an implicit ODE method; this
integrator approximates the midpoint with a half-step of the augmented BUG integrator instead of
solving for it, so no linear system is ever formed. It is a two-stage EXPLICIT scheme, and it
does NOT remove the Krylov exponential -- stage 2's Galerkin ODE over the full step IS
`exp(tau*H_0)` on the core, i.e. the same `expv` root solve the ordinary step makes. What it buys
is Theorem 4.2: a robust SECOND-ORDER global error bound whose constants are INDEPENDENT OF THE
SINGULAR VALUES, with improved dependence on the normal component of the vector field.

⛔ STAGE 1 MUST NOT TRUNCATE. Its job is to supply DIRECTIONS, and the paper's midpoint state is
the rank-2r augmented one; cutting it back to rank r here would discard exactly what the
augmented basis of stage 2 is supposed to contain. `truncate = false` is therefore forced, not a
default, and rank control happens once at the end of stage 2.

⚠ COST IS ~2x THE ORDINARY STEP -- two sweeps and two root solves. Whether that pays has to be
MEASURED against simply halving `dt` on the ordinary sweep, which buys second-order accuracy at
the same 2x. This sweep is already measured at `O(dt^2)`, so what is on offer here is the error
CONSTANT and the proven bound, not an extra order; `benchmarks/midpoint_bug.jl` is where that
comparison lives.
"""
function cbe_bug_midpoint_step!(psi::SymMPS, mpo::MPO, tau::ComplexF64; kwargs...)
    get(kwargs, :order, 1) == 1 || throw(ArgumentError(
        "order > 1 is a single-step method (cbe_bug_step!); the midpoint scheme is its own " *
        "second-order construction and does not take an order"))
    half = copy(psi)
    # stage 1: bases only. `truncate = false` keeps the augmented rank -- see the docstring.
    info1 = cbe_bug_step!(half, mpo, tau / 2; kwargs..., truncate = false)
    # stage 2: full step, frames from the midpoint, START VECTOR still `psi`.
    info2 = cbe_bug_step!(psi, mpo, tau; kwargs..., frame_state = half)
    # Stage 1 built the directions stage 2 integrates in, so a basis test it failed is the
    # step's failure too, not a discarded intermediate.
    rfin = info2.ledger.r_final            # the PREDICTOR's rank is uncompressed, not final
    # Both stages sweep every bond; say which visit each per-bond entry is.
    for (lg, st) in ((info1.ledger, 1), (info2.ledger, 2))
        lg.bond_log = Any[merge(e, (stage = st,)) for e in lg.bond_log]
    end
    merge_ledger!(info2.ledger, info1.ledger)
    info2.ledger.r_final = rfin
    return info2
end

cbe_bug_midpoint_step!(psi::SymMPS, mpo::MPO, tau::Number; kwargs...) =
    cbe_bug_midpoint_step!(psi, mpo, ComplexF64(tau); kwargs...)

cbe_bug_midpoint_step!(psi::SymMPS, h::XXZChain, tau; kwargs...) =
    cbe_bug_midpoint_step!(psi, mpo_from_terms(h), tau; kwargs...)
