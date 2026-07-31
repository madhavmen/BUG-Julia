# Is the expanding solver's accuracy gain real, or a Trotter prefactor?
#
#     julia --project=. benchmarks/cbe_dt_scan.jl
#
# The sweep (docs 7.6.4) put the expanding-basis Lanczos at 1.399e-06 against 1.940e-06 for
# both the one-shot default and the reorth-only control. But every scheme there sits at the
# dt-limited floor -- 2-site TDVP itself reaches 1.761e-06 at the same dt -- so the 28% gap
# could be a different constant in front of the SAME dt^2, not a better basis.
#
# This separates them. Fix t and halve dt:
#
#   * if every scheme scales as dt^2 with a different constant, the gap is a Trotter
#     PREFACTOR and says nothing about basis quality;
#   * if the expanding solver's error keeps falling while the others flatten, the others are
#     hitting a BASIS floor and the expanding basis is genuinely resolving more.
#
# Reference is the analytic free-fermion result, exact for the finite open chain.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf

set_symmetry!(:U1)

const L      = 8
const TMAX   = 0.5
const MAXDIM = 32

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(length(p) - 1)]

function analytic_sz(L::Int, t::Float64, n0::Vector{Float64}; J::Float64 = 1.0)
    h = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        h[j, j + 1] = J / 2
        h[j + 1, j] = J / 2
    end
    U = exp(-im * h * t)
    return [sum(abs2(U[j, k]) * n0[k] for k in 1:L) - 0.5 for j in 1:L]
end

const N0 = magnetisation(domain_wall_state(L)) .+ 0.5
const REF = analytic_sz(L, TMAX, N0)

function err_at(dt; kwargs...)
    n = round(Int, TMAX / dt)
    psi = domain_wall_state(L)
    gates = xx_gates(psi)
    info = cbe_bond_update_bug!(psi, gates;
                                opts = CBEBugOptions(; dt = dt, n_steps = n,
                                                     maxdim = MAXDIM, trunc_thresh = 1e-14,
                                                     kwargs...))
    return norm(magnetisation(copy(psi)) - REF), sum(info.krylov_dims)
end

const DTS = [0.04, 0.02, 0.01, 0.005]

schemes = [("one-shot default (growth=2.0)", NamedTuple()),
           ("+ reorth (CONTROL)",            (; s_reorth = true)),
           ("expanding grow=2",              (; s_iters = -2)),
           ("expanding grow=2, growth=1.5",  (; s_iters = -2, growth = 1.5))]

@printf("L = %d, t = %.2f, maxdim = %d, reference = exact free fermions\n", L, TMAX, MAXDIM)
println("A second-order scheme quarters its error when dt is halved (ratio ~ 4).\n")

for (name, kw) in schemes
    @printf("%s\n", name)
    @printf("  %-8s %-12s %-8s %s\n", "dt", "err", "ratio", "matvec")
    prev = NaN
    for dt in DTS
        e, mv = err_at(dt; kw...)
        r = isnan(prev) ? NaN : prev / e
        @printf("  %-8.4f %-12.4e %-8s %d\n", dt, e, isnan(r) ? "--" : @sprintf("%.2f", r), mv)
        prev = e
    end
    println()
end

println("DT_SCAN_DONE")
