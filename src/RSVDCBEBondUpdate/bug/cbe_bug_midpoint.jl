
"""
    cbe_bug_midpoint_step!(psi, mpo, tau; kwargs...) -> CBEBugSweepInfo

One midpoint BUG step (Ceruti-Kusch-Lubich, arXiv:2402.08607), in place. A half step `tau/2` with
`truncate = false` supplies the midpoint bases; an ordinary [`cbe_bug_step!`](@ref) then builds its
frames from `Y_{1/2}`, integrates the Galerkin ODE from `Y_0`, and truncates once. It is explicit
(no linear solve), still needs one exponential root solve, and costs about twice the ordinary step.
Its gain is the second-order bound of Theorem 4.2, with constants independent of the singular
values; weigh it against halving `dt` (`benchmarks/midpoint_bug.jl`).
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
