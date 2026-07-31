# Three ways to reach the complement, scored against the exact free-fermion answer.
#
#     julia --project=. benchmarks/cbe_probe_variants.jl
#
#   V1  project-first     exact P_perp applied to H*Theta, THEN the Gaussian.
#                         comp_ratio = 0.5. FORMS THE RANK-4 OBJECT.
#   V2  two Gaussians     the sketch as shipped: Omega folded into the gate first on each
#                         side, P_perp applied to the small result. comp_ratio = 0.5, so the
#                         probe is half complement-projected, half frame-projected.
#   V3  plain Gaussians   comp_ratio = nothing: no projector in the probe at all. The
#                         complement is reached by SAMPLING, redrawn from an advancing RNG
#                         at every growth pass, and the exact P_perp still runs afterwards
#                         in the preselection.
#
# V1 and V2 must agree to rounding -- P_perp acts on (link_l, site_l), Omega on
# (site_r, link_r), disjoint legs, so they commute. Any difference between them is floating
# point, and the interesting number is the COST of having formed H*Theta for nothing.
#
# V3 is the one that can genuinely differ: it changes WHICH directions are looked at. It is
# also the only variant with no hyperparameter to tune, which was the point of iterating in
# the first place. Whether sampling over passes beats a fixed 50/50 split is what this
# measures, and it is measured WITH the iteration (s_iters = -2) since that is where the
# accumulated coverage is supposed to come from.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf

set_symmetry!(:U1)

const L      = 8
const TMAX   = 0.4
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

const N0  = magnetisation(domain_wall_state(L)) .+ 0.5
const REF = analytic_sz(L, TMAX, N0)

function evolve(dt, nsteps; kwargs...)
    p = domain_wall_state(L)
    g = xx_gates(p)
    info = cbe_bond_update_bug!(p, g;
                                opts = CBEBugOptions(; dt = dt, n_steps = nsteps,
                                                     maxdim = MAXDIM, trunc_thresh = 1e-14,
                                                     kwargs...))
    return p, info
end

const VARIANTS = [
    ("V1 project-first  (rank-4)", (; project_first = true, comp_ratio = 0.5)),
    ("V2 two gaussians  cr=0.5",   (; comp_ratio = 0.5)),
    ("V3 plain gaussian cr=none",  (; comp_ratio = nothing)),
    ("V4 two-sided (proj sketched)", (; two_sided = true)),
]

function bench(dt, label, kw)
    n = round(Int, TMAX / dt)
    evolve(dt, 2; kw...)                                  # warm-up, untimed
    best = Inf
    local p, info
    for _ in 1:3
        t = @elapsed ((p, info) = evolve(dt, n; kw...))
        best = min(best, t)
    end
    e = norm(magnetisation(copy(p)) - REF)
    @printf("  %-30s %-12.5e %-9.2f %-9d %d\n", label, e, best,
            sum(info.krylov_dims), maximum(info.max_bond_dims))
    return e
end

@printf("L = %d, t = %.2f, maxdim = %d, reference = exact free fermions\n", L, TMAX, MAXDIM)
println("wall = min of 3 after warm-up; treat gaps below ~1.3x as noise on this box.\n")

for (mode, extra) in (("ONE-SHOT (s_iters = 1)", NamedTuple()),
                      ("EXPANDING (s_iters = -2)", (; s_iters = -2)))
    println(mode)
    @printf("  %-30s %-12s %-9s %-9s %s\n", "variant", "err", "wall(s)", "matvec", "maxbd")
    println("  " * "-"^72)
    for (name, kw) in VARIANTS
        bench(0.02, name, merge(kw, extra))
    end
    println()
end

# Order of convergence: the only test that separates a better BASIS from a different
# Trotter constant (docs 7.6.5).
println("ORDER IN dt, expanding solver (ratio ~4 is clean second order)")
for (name, kw) in VARIANTS
    @printf("  %s\n", name)
    prev = NaN
    for dt in (0.02, 0.01, 0.005)
        n = round(Int, TMAX / dt)
        p, _ = evolve(dt, n; merge(kw, (; s_iters = -2))...)
        e = norm(magnetisation(copy(p)) - REF)
        r = isnan(prev) ? NaN : prev / e
        @printf("    dt=%-7.4f err=%-12.5e ratio=%s\n", dt, e,
                isnan(r) ? "--" : @sprintf("%.2f", r))
        prev = e
    end
end

println("\nVARIANTS_DONE")
