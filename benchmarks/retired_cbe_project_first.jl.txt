# P_perp before Omega, or Omega before P_perp?
#
#     julia --project=. benchmarks/cbe_project_first.jl
#
# P_perp = I - U0 U0' acts on (link_l, site_l); Omega acts on (site_r, link_r). Disjoint legs,
# so the two commute and
#
#     P_perp((H Theta) Om')  ==  (P_perp (H Theta)) Om'
#
# exactly. This benchmark does not test WHETHER that holds -- it holds by algebra -- it
# measures how closely the two implementations agree in floating point, and what the reordering
# costs. Projecting first forces the rank-4 H*Theta into existence, so the gate is applied to
# d*chi_r columns instead of Omega's npre.
#
# Three levels, because agreement at one does not imply agreement at the next:
#   1. the raw sketch Y, with the SAME Omega handed to both;
#   2. the whole expansion (ranks, err_pre, err_fnl), where an SVD cutoff could amplify a
#      1e-16 difference into a different number of admitted directions;
#   3. a full evolution, where any per-bond difference compounds over steps.

using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using LinearAlgebra, Printf, Random
using LurCGT, Telum
const R = RSVDCBEBondUpdate

set_symmetry!(:U1)

const L = 8
const DT = 0.05
const MAXDIM = 32

xx_gates(p) = Any[xx_bond_gate(p[i].inds[2], p[i + 1].inds[2]) for i in 1:(length(p) - 1)]

psi = domain_wall_state(L)
gates = xx_gates(psi)
cbe_bond_update_bug!(psi, gates;
                     opts = CBEBugOptions(dt = DT, n_steps = 3, maxdim = MAXDIM,
                                          trunc_thresh = 1e-14))

# ── 1. the raw sketch, same Omega both ways ──────────────────────────────────────────
println("1. RAW SKETCH -- same Omega handed to both orderings")
@printf("   %-6s %-14s %-14s %-12s %s\n", "bond", "||Y_omega||", "||Y_proj||",
        "abs diff", "rel diff")
println("   " * "-"^62)
worst = 0.0
for i in 1:(L - 1)
    canonical!(psi, i)
    f = bond_frame(psi, i)
    OmR = sector_graded_sketch(f.V0, :right, 8; comp_ratio = 0.5,
                               rng = MersenneTwister(0x5EED))
    OmR === nothing && continue

    # omega-first, then project (what the preselection does)
    Y1 = R.sketch_bond_left(f, gates[i], OmR)
    P1 = to_concrete(perp_component(f.U0, Y1))
    # project-first, then omega
    P2 = R.sketch_bond_left_pfirst(f, gates[i], OmR)

    # NORMALISE BY ||Y||, NOT BY ||P_perp Y||. Where the frame already spans the gate's
    # action the complement is numerically empty (1e-21), and dividing by it turns noise
    # into a "67% discrepancy". ||Y|| is the scale the difference actually matters against.
    d = norm(to_concrete(P1 - P2))
    rel = d / max(norm(Y1), 1e-300)
    global worst = max(worst, rel)
    @printf("   %-6s %-14.6e %-14.6e %-12.3e %.3e\n",
            "($i,$(i+1))", norm(Y1), norm(P2), d, rel)
end
@printf("   worst relative difference: %.3e\n\n", worst)

# ── 2. the whole expansion ───────────────────────────────────────────────────────────
println("2. FULL EXPANSION -- does the SVD cutoff amplify any difference?")
@printf("   %-6s %-16s %-16s %-12s %s\n", "bond", "new(l,r) omega", "new(l,r) proj",
        "err_pre d", "err_fnl d")
println("   " * "-"^68)
for i in 1:(L - 1)
    canonical!(psi, i)
    f = bond_frame(psi, i)
    a = cbe_expand_bond(f, gates[i])
    b = cbe_expand_bond(f, gates[i]; project_first = true)
    @printf("   %-6s (%d,%d)%-11s (%d,%d)%-11s %-12.3e %.3e\n",
            "($i,$(i+1))", a.n_new_l, a.n_new_r, "", b.n_new_l, b.n_new_r, "",
            abs(a.err_pre - b.err_pre), abs(a.err_fnl - b.err_fnl))
end

# ── 3. a full evolution, and the cost ────────────────────────────────────────────────
function analytic_sz(L::Int, t::Float64, n0::Vector{Float64}; J::Float64 = 1.0)
    h = zeros(ComplexF64, L, L)
    for j in 1:(L - 1)
        h[j, j + 1] = J / 2
        h[j + 1, j] = J / 2
    end
    U = exp(-im * h * t)
    return [sum(abs2(U[j, k]) * n0[k] for k in 1:L) - 0.5 for j in 1:L]
end

const TMAX = 0.4
const NSTEP = round(Int, TMAX / DT)
const N0 = magnetisation(domain_wall_state(L)) .+ 0.5
const REF = analytic_sz(L, TMAX, N0)

function evolve(nsteps; kwargs...)
    p = domain_wall_state(L)
    g = xx_gates(p)
    info = cbe_bond_update_bug!(p, g;
                                opts = CBEBugOptions(; dt = DT, n_steps = nsteps,
                                                     maxdim = MAXDIM, trunc_thresh = 1e-14,
                                                     kwargs...))
    return p, info
end

println("\n3. FULL EVOLUTION and cost (min of 3, after warm-up)")
@printf("   %-24s %-13s %-10s %s\n", "ordering", "err", "wall(s)", "matvec")
println("   " * "-"^58)
res = Dict{String, Any}()
for (name, kw) in (("omega first (default)", NamedTuple()),
                   ("project first", (; project_first = true)))
    evolve(2; kw...)
    best = Inf
    local p, info
    for _ in 1:3
        t = @elapsed ((p, info) = evolve(NSTEP; kw...))
        best = min(best, t)
    end
    e = norm(magnetisation(copy(p)) - REF)
    res[name] = (e, best)
    @printf("   %-24s %-13.6e %-10.2f %d\n", name, e, best, sum(info.krylov_dims))
end
a, b = res["omega first (default)"], res["project first"]
@printf("\n   error difference: %.3e      wall ratio (proj/omega): %.2fx\n",
        abs(a[1] - b[1]), b[2] / a[2])

println("\nPFIRST_DONE")
