# WHICH TFIM OBSERVABLE ACTUALLY DISCRIMINATES BETWEEN INTEGRATORS?
#
# THE PROBLEM THIS SOLVES. `<X_j>` for a kink under critical TFIM makes `cbe_bug` look 1775x
# worse than `tdvp2`, and `tfim_accumulate.jl` showed that is NOT a defect: there is no
# amplification (the error exponent is -1.36, it decays), norm holds to 1e-14 and energy to
# 6e-13, both BETTER than tdvp2. The gap is an artefact of the OBSERVABLE. Measured dt ratios:
#
#     TFIM  <X_j>    tdvp2 ~32 (dt^5)   cbe_bug ~16 (dt^4)      <- anomalous cancellation
#     XX    <Sz_j>   tdvp2 ~4  (dt^2)   cbe_bug ~4  (dt^2)      <- honest
#
# Both schemes converge FAR faster than their nominal second order on `<X_j>`, tdvp2 by one
# extra power, so the comparison at fixed dt measures which scheme happens to benefit more from
# a cancellation rather than which is more accurate. That is a degenerate probe.
#
# ⛔ THE SELECTION CRITERION IS THE dt RATIO, NOT THE ERROR SIZE. A good observable converges at
# the schemes' TRUE order -- ratio ~4 for all three -- so differences between them reflect the
# integrator. An observable showing 16 or 32 is measuring an accidental cancellation. Bigger
# errors are not the goal; HONEST SCALING is.
#
# FOUR CANDIDATES, chosen to probe different structure:
#   <X_j>            the incumbent: a one-site diagonal observable.
#   S(L/2)           entanglement entropy at the centre cut -- reads the SCHMIDT SPECTRUM, which
#                    is exactly what the schemes disagree about, and no one-site expectation can
#                    see it. Exact value from the Majorana covariance matrix restricted to the
#                    half chain, so still no 2^L object.
#   <X_i X_j>        a connected two-point function at maximum separation: sensitive to
#                    correlations that a profile averages away.
#   infidelity       the complete measure. Nothing can cancel in it, so it is the ceiling on how
#                    honest any other probe can be -- at the cost of a 2^L state.
#
# ⛔ THE INFIDELITY ARM SELF-CHECKS ITS BIT CONVENTION AND REFUSES TO REPORT IF IT CANNOT.
# `dense_state` and the dense TFIM basis are MIRRORED (site 1 is the low bit in one and the high
# bit in the other, and a set bit means the opposite spin). The kink is nearly symmetric, so a
# convention error would be almost invisible -- the check below uses <X_j> on the ASYMMETRIC
# evolved state, where a mirror shows up immediately.
#
# Run:  julia --project=. benchmarks/tfim_observables.jl

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

const L  = 8
const D  = 1024          # the L=8 Schmidt ceiling is 16: this never binds, nothing truncates
const MI = 30
const DIRS = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]

# ── the free-fermion reference: one covariance matrix serves every observable ─────────────
#
# `tfim_covariance` / `tfim_x_profile` come from `tests/common/free_fermion.jl` and are pinned
# against exact diagonalisation in `tests/common/test_analytic_reference.jl`. This file used to
# carry its own copy of the congruence; it does not, because a second copy is a second thing to
# get the Majorana factors of two wrong in, and the whole point of this benchmark is that several
# observables are read off ONE matrix.
gamma_at(t, dirs; J = 1.0, h = 1.0) = tfim_covariance(L, t, dirs; J = J, h = h)
x_exact(t, dirs) = tfim_x_profile(L, t, dirs)

"""
Entanglement entropy of sites `1:cut` from the Majorana covariance matrix.

For a Gaussian state the restricted `Gamma` block has eigenvalues `+-i*nu`, and each `nu`
contributes one binary entropy `h2((1+nu)/2)`. No `2^L` object appears.
"""
function s_exact(t, dirs; cut = L ÷ 2)
    Gr = gamma_at(t, dirs)[1:(2cut), 1:(2cut)]
    nu = filter(x -> x > 1e-12, real.(eigvals(im * (Gr - transpose(Gr)) / 2)))
    s = 0.0
    for v in nu
        p = clamp((1 + v) / 2, 1e-15, 1 - 1e-15)
        s -= p * log(p) + (1 - p) * log(1 - p)
    end
    return s
end

"`<X_1 X_L> - <X_1><X_L>`, connected, at maximum separation. Wick: the 2x2 Pfaffian block."
function xx_conn_exact(t, dirs)
    Gt = gamma_at(t, dirs)
    return -Gt[1, 2L - 1] * Gt[2, 2L] + Gt[1, 2L] * Gt[2, 2L - 1]
end

# ── the measured side ────────────────────────────────────────────────────────────────────
function s_meas(psi; cut = L ÷ 2)
    sv = bond_spectrum(copy(psi), cut)
    p  = (sv .^ 2) ./ max(sum(sv .^ 2), eps())
    return -sum(x -> x > 1e-15 ? x * log(x) : 0.0, p)
end

"Connected `<X_1 X_L>` from the MPS, via the dense state (L=8, so this is cheap)."
function xx_conn_meas(psi, Xops)
    v  = dense_state(copy(psi))
    n  = real(dot(v, v))
    e1 = real(dot(v, Xops[1] * v)) / n
    eL = real(dot(v, Xops[L] * v)) / n
    e1L = real(dot(v, Xops[1] * (Xops[L] * v))) / n
    return e1L - e1 * eL
end

"""
`X_j` as dense diagonal operators in `dense_state`'s OWN convention, or `nothing`.

Determined by TESTING, not assumed: both candidate bit orders are built and the one whose
`<X_j>` reproduces `x_profile` on an ASYMMETRIC state is returned. A symmetric state cannot
tell them apart, which is exactly how a mirrored convention survives undetected.
"""
function dense_x_ops()
    set_symmetry!(:Z2)
    probe = ising_product_state([:plus, :minus, :minus, :plus, :minus, :plus, :plus, :minus])
    want  = x_profile(copy(probe))
    # ⛔ `dense_state` DOES NOT SUPPORT :Z2 and throws rather than returning something wrong.
    # The Z2 local space is built BELOW Telum's IROP layer, so `dense_state`'s product basis is
    # U(1) and the contraction is a symmetry mismatch. Caught rather than propagated so the two
    # Majorana-based probes -- which need no dense state at all -- still run; the correlator arm
    # suppresses itself downstream. Do NOT "fix" this by running the whole file under :none:
    # that would change the model being measured, not just the bookkeeping.
    v = try
        dense_state(copy(probe))
    catch err
        @info "dense_state unavailable under :Z2, suppressing the <X_1 X_L> arm" err
        return nothing
    end
    for lowbit_is_site1 in (true, false), setbit_is_plus in (true, false)
        ops = map(1:L) do j
            b = lowbit_is_site1 ? j - 1 : L - j
            d = [(((s >> b) & 1) == 0) == setbit_is_plus ? -1.0 : 1.0 for s in 0:(2^L - 1)]
            Diagonal(ComplexF64.(d))
        end
        got = [real(dot(v, ops[j] * v)) / real(dot(v, v)) for j in 1:L]
        maximum(abs.(got - want)) < 1e-10 && return ops
    end
    return nothing
end

# ── the sweep: dt ratios per observable per scheme ───────────────────────────────────────
schemes(mpo, tau) =
    ["tdvp2"      => p -> tdvp2_step!(p, mpo, tau; maxdim = D, trunc_thresh = 1e-14, maxiter = MI),
     "tdvp_cbe1s" => p -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = D, trunc_thresh = 1e-14,
                                           maxiter = MI),
     "cbe_bug"    => p -> cbe_bug_step!(p, mpo, tau; maxdim = D, trunc_thresh = 1e-14,
                                        maxiter = MI, exact = true)]

set_symmetry!(:Z2)
const MPO  = tfim_mpo(L; J = 1.0, h = 1.0)
const XOPS = dense_x_ops()

const T   = 1.0
const DTS = (0.02, 0.01, 0.005)

# ⛔ THE INITIAL STATE IS A VARIABLE HERE, NOT A CONSTANT. The anomalous cancellation was found
# on the KINK, and a cancellation can be a property of the state as easily as of the observable
# -- so varying only the observable would leave that confounded. The three below differ in what
# they do to the ZZ coupling, which is the term `<X_j>` is blind to:
#
#   polarised  |+...+>   every ZZ bond satisfied; an EIGENSTATE of the field term alone, so the
#                        entire dynamics comes from the coupling. Also flip-symmetric.
#   Neel       |+-+-..>  every ZZ bond frustrated -- the opposite extreme, and ASYMMETRIC under
#                        reversal, which is what exposes a mirrored convention.
#   kink       |++--..>  one frustrated bond in the middle. The incumbent.
const STATES = ["polarised" => fill(:plus, L),
                "Neel"      => [isodd(i) ? :plus : :minus for i in 1:L],
                "kink"      => [i <= L ÷ 2 ? :plus : :minus for i in 1:L]]

println("="^88)
@printf("TFIM L=%d  maxdim=%d (never binds)  T=%g   WHICH OBSERVABLE AND STATE SCALE HONESTLY?\n",
        L, D, T)
println("ratio ~4 = true dt^2 order, an honest probe.  16 or 32 = accidental cancellation.")
XOPS === nothing && println("!! dense_x_ops: NO bit convention reproduced x_profile -- " *
                            "the <X_1 X_L> arm is SUPPRESSED rather than guessed.")
println("="^88)

# ⛔ NO INFIDELITY ARM. It would need a dense TFIM built in `dense_state`'s convention, and the
# only way to get that convention is to reconstruct it from the validated `X_j` -- a search over
# basis states that is both O(4^L) and fragile in exactly the place this file is trying to be
# careful about. The three probes below all rest on the SAME Majorana covariance matrix, which
# is already pinned against `x_profile` by the examples, so they share one validated reference
# and introduce no new convention risk. Infidelity would be the stronger measure; it is not
# worth a reference I cannot vouch for.
obs(dirs) = begin
    o = Any["max |dX_j|" => (p, t) -> maximum(abs.(x_profile(copy(p)) - x_exact(t, dirs))),
            "S(L/2)"     => (p, t) -> abs(s_meas(p) - s_exact(t, dirs))]
    XOPS === nothing ||
        push!(o, "<X_1 X_L> conn" =>
                 (p, t) -> abs(xx_conn_meas(p, XOPS) - xx_conn_exact(t, dirs)))
    o
end

for (statenm, dirs) in STATES
    psi0 = ising_product_state(dirs)
    println("\n" * "-"^88)
    @printf("INITIAL STATE: %s   %s\n", statenm,
            join(d === :plus ? "+" : "-" for d in dirs))
    println("-"^88)
    for (onm, of) in obs(dirs)
        @printf("\n  %-16s %-12s", onm, "scheme")
        for dt in DTS
            @printf(" %11s", "dt=$dt")
        end
        println("     ratios")
        for (snm, _) in schemes(MPO, ComplexF64(-im * DTS[1]))
            es = Float64[]
            for dt in DTS
                st = Dict(schemes(MPO, ComplexF64(-im * dt)))[snm]
                psi = copy(psi0)
                for _ in 1:round(Int, T / dt)
                    st(psi)
                end
                push!(es, of(psi, T))
            end
            @printf("  %-16s %-12s", "", snm)
            for e in es
                @printf(" %11.3e", e)
            end
            @printf("    %6.2f %6.2f\n", es[1] / max(es[2], 1e-300),
                    es[2] / max(es[3], 1e-300))
        end
        flush(stdout)
    end
end
println()
