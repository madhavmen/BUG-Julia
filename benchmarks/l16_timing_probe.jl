# HOW EXPENSIVE IS ONE L=16 ARM? Answered before committing to a 64-arm matrix.
#
# A first attempt at the L=16 cube ran 18 minutes without completing a single arm and wrote no
# CSV, which gave no information about WHY -- reference build, one slow arm, or a hang. This
# separates the two costs and prints each as it lands, to a FILE as well as stdout: stdout
# redirected to a file is block-buffered in this environment even with `flush`, so a long run
# looks identical to a hung one.
#
# ⚠ `@printf` IS A MACRO AND NEEDS A LITERAL FORMAT STRING. `@printf(io, fmt, args...)` with
# `fmt` a variable does not compile -- it fails with "First argument to `@printf` after `io` must
# be a format string", at load time, which reads as a mysterious exit-1 if stdout went to
# /dev/null. Format with `@sprintf` at the call site and pass the finished string.
#
# Run:  julia --project=. benchmarks/l16_timing_probe.jl

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "analytic_reference.jl"))
include(joinpath(@__DIR__, "exact_sparse.jl"))

const OUT = joinpath(@__DIR__, "results", "l16_timing.txt")
mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(line::AbstractString) = (println(IO_, line); flush(IO_); println(stdout, line); flush(stdout))

set_symmetry!(:U1)
const L = 16

t0 = time()
H, states, idx = heisenberg_sparse(L)
say(@sprintf("sparse H: %d states, %.1f s", length(states), time() - t0))

t0 = time()
v = expv_sparse(H, ComplexF64(-im * 0.5), neel_vector(L, idx); m = 30, tol = 1e-13)
ref = magnetisation_dense(v, L, states)
say(@sprintf("exact reference to t=0.5: %.1f s", time() - t0))

mpo, psi0 = xxz_mpo(L; J = 1.0, delta = 1.0), neel_state(L)
for (tt, m, cap) in ((1e-4, 0, 64), (1e-4, 3, 64), (1e-8, 3, 64), (1e-10, 3, 64))
    psi, kry = copy(psi0), 0
    t0 = time()
    for _ in 1:10
        info = cbe_bug_step!(psi, mpo, ComplexF64(-im * 0.05); exact = true,
                             krylov_basis = m, krylov_tol = 0.0,
                             maxdim = cap, trunc_thresh = tt, maxiter = 16)
        kry += info.krylov_dims
    end
    el = time() - t0
    err = maximum(abs.(magnetisation(copy(psi)) ./ max(norm(psi)^2, eps()) .- ref))
    say(@sprintf("tau=%.0e m=%d cap=%d: %6.1f s / 10 steps (%.2f s/step)  chi=%3d  err=%.3e  kry=%d",
                 tt, m, cap, el, el / 10, maximum(bond_dims(psi)), err, kry))
end
close(IO_)
