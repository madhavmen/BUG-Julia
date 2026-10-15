# THE ADAPTIVE CBE-BUG SOLVER IN ONE PAGE -- three tolerances, one resource limit, and a status.
#
#   julia -t 2 --project=. examples/adaptive_quickstart.jl
#
# Evolves the XX chain from a Neel state with `cbe_bug_step!`, reads each step's verdict with
# `budget_status` / `budget_diagnostic`, and checks the result against the exact free-fermion
# solution. Then it asks for something infeasible (maxdim = 4) to show what a refused accuracy
# claim looks like. Uses ONLY the public API documented in README.md.

using LinearAlgebra
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))   # exact reference

set_symmetry!(:U1)
L, dt, T = 20, 0.05, 1.0
W = xxz_mpo(L; J = 1.0, delta = 0.0)                     # XX chain
sz(p) = magnetisation(copy(p)) ./ norm(copy(p))^2
exact(t) = xx_free_fermion_sz(L, t; J = 1.0, occupied = 1:2:L)

"Evolve to T with the given settings; return (max error vs exact, bond dims, statuses)."
function run(; kw...)
    psi = neel_state(L)
    statuses = Symbol[]
    for n in 1:round(Int, T / dt)
        info = cbe_bug_step!(psi, W, ComplexF64(-im * dt); parallel = true, kw...)
        push!(statuses, budget_status(info.ledger))
        n == 1 && budget_status(info.ledger) !== :met &&
            println("  step 1: ", budget_diagnostic(info.ledger))
    end
    return maximum(abs.(sz(psi) .- exact(T))), bond_dims(psi), statuses
end

println("1) defaults (tol_basis 1e-5, tol_trunc 1e-7, tol_expv 1e-10, maxdim 128)")
err, bd, st = run()
println("   err vs exact = $err, max bond dim $(maximum(bd)), all steps :met = $(all(==(:met), st))")

println("2) loose settings from ADAPTIVE_XX_L20_RESULTS.md (target ~1e-4)")
err2, bd2, st2 = run(tol_basis = 1e-5, tol_trunc = 1e-3, tol_expv = 1e-2)
println("   err vs exact = $err2, max bond dim $(maximum(bd2)), all steps :met = $(all(==(:met), st2))")

println("3) an infeasible request: tight tolerances with maxdim = 4")
err3, bd3, st3 = run(tol_basis = 1e-6, tol_trunc = 1e-8, maxdim = 4)
println("   err vs exact = $err3, max bond dim $(maximum(bd3)), steps :budget_unmet = " *
        "$(count(==(:budget_unmet), st3)) of $(length(st3))")

# The three claims this page makes, checked rather than asserted in prose:
ok = err < 1e-4 && all(==(:met), st) && err2 < 1e-3 && maximum(bd2) < maximum(bd) &&
     any(==(:budget_unmet), st3)
println(ok ? "QUICKSTART-OK" : "QUICKSTART-FAILED")
ok || exit(1)
