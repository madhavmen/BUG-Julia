# `cbe_bug_step!` PINNED AGAINST EXACT REFERENCES, on three models at L = 12.
#
# WHAT THE SWEEP IS. CBE widens the bond, the expansion frame itself becomes the basis, the
# half-sweep deepens it by the Tests C/D Krylov stage, the environment is pushed through it, and the
# root Galerkin step is the only evolution in the step. Both frames are used across the step:
# `U_ex` becomes `W[i]` on the left half-sweep, `V_ex` becomes `Z[j]` on the right, so every bond is
# widened from both sides before the root sees it.
#
# THE PROPERTY UNDER TEST IS THAT THE RANK GROWS. There is no evolved two-site tensor to split, so
# the only thing that can carry new directions into the state is the frame the half-sweep builds.
# If the expansion were being dropped anywhere the bond dimension would sit at its initial value --
# and a product start makes that unmissable, since chi = 1 is a fixed point of every sweep that
# does not genuinely widen.
#
# ⛔ EVERY REFERENCE IS EXACT, AND THE LONG-RANGE ONE IS NOT THE FREE-FERMION FORMULA.
# Jordan-Wigner leaves a string between the two sites of an XY term and it cancels ONLY for
# nearest neighbours, so `xx_free_fermion_sz` is the right reference for the NN chain and the
# WRONG one at range. The long-range arm therefore uses an exact propagation in the `Sz = 0`
# sector -- C(12,6) = 924 states, dense-exponentiable instantly, with no solver tolerance of its
# own to inherit.
#
# TWO STEP SIZES PER MODEL, ALWAYS. A single small error says "happens to be small"; two say
# "converging", and the ratio is what the order assertions read. The bars were set on the removed
# fixed-knob path at one power of `H` (measured values in the comments); the tolerance-driven
# frame can only hold more of the Krylov space, so they are upper bounds here.

using SparseArrays

# ── the exact long-range XX reference (test-side, shares nothing with src/) ───────────────
"Sparse `H = Σ_{i<j} Jm[i,j] (S^x_i S^x_j + S^y_i S^y_j)` in the `Sz = 0` sector. Flip-flop only:
there is no `S^z S^z` term in XX, so unlike `pairs_sparse` this carries no diagonal at all."
function _xx_lr_sparse(L::Int, Jm::AbstractMatrix{Float64})
    states = [b for b in 0:(2^L - 1) if count_ones(b) == L ÷ 2]
    idx = Dict(s => k for (k, s) in enumerate(states))
    I, Jd, V = Int[], Int[], Float64[]
    bit(b, j) = (b >> (j - 1)) & 1
    for (k, b) in enumerate(states)
        for i in 1:(L - 1), j in (i + 1):L
            Jm[i, j] == 0.0 && continue
            bit(b, i) == bit(b, j) && continue
            bb = b ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
            push!(I, idx[bb]); push!(Jd, k); push!(V, Jm[i, j] / 2)
        end
    end
    return sparse(I, Jd, V, length(states), length(states)), states, idx
end

"`⟨S^z_j⟩` of a sector vector, in the bit convention of [`_xx_lr_sparse`](@ref)."
_magn_sector(v, L, states) =
    (p = abs2.(v); n = sum(p);
     [sum(p[k] * (((states[k] >> (j - 1)) & 1) - 0.5) for k in eachindex(states)) / n
      for j in 1:L])

"Evolve to `tmax` at tight tolerances and report `(err, chi, dE, norm)`.

`obs`/`exact_obs` return vectors so one helper serves both a site-resolved profile and a scalar."
function _run_tight(mpo, psi0, tmax, dt, obs, exact_obs; maxdim = 64, tol_trunc = 1e-12)
    psi = copy(psi0)
    e0 = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
    for _ in 1:round(Int, tmax / dt)
        cbe_bug_step!(psi, mpo, ComplexF64(-im * dt); exact = true, tol_basis = 1e-12,
                      tol_trunc = tol_trunc, tol_expv = 1e-14, maxdim = maxdim)
    end
    e1 = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
    return (err = maximum(abs.(obs(psi) .- exact_obs(tmax))),
            chi = maximum(bond_dims(psi)), dE = abs(e1 - e0), nrm = norm(psi))
end

_sz_profile(p) = magnetisation(copy(p)) ./ max(norm(copy(p))^2, eps())

@testset "cbe_bug_step! against exact references, L=12" begin
    L = 12

    @testset "XX nearest neighbour vs the free-fermion closed form" begin
        set_symmetry!(:U1)
        mpo, psi0 = xxz_mpo(L; J = 1.0, delta = 0.0), domain_wall_state(L)
        ex(t) = xx_free_fermion_sz(L, t; J = 1.0, occupied = 1:(L ÷ 2))
        a = _run_tight(mpo, psi0, 1.0, 0.1, _sz_profile, ex)
        b = _run_tight(mpo, psi0, 1.0, 0.05, _sz_profile, ex)

        @test b.err < 5e-4                    # measured 1.695e-04
        @test a.err / b.err > 3.0             # measured 3.85 -- second order
        @test b.chi > 1                       # the expansion reached the state at all
        @test b.chi >= 12                     # measured 15; a wall well above the chi=1 start
        @test b.dE < 1e-12                    # measured 4.6e-15, exactly conserved
        @test isapprox(b.nrm, 1.0; atol = 1e-10)
    end

    @testset "XX long range (1/r^2) vs exact sparse propagation" begin
        set_symmetry!(:U1)
        Jm = [i < j ? 1.0 / abs(i - j)^2 : 0.0 for i in 1:L, j in 1:L]
        # `delta = 0` drops the S^zS^z vertex, leaving the two CHARGED XY vertices -- the case
        # `pair_mpo` exists to handle and `long_range_mpo` refuses.
        mpo, psi0 = pair_mpo(L, Jm, xxz_chain(L; J = 1.0, delta = 0.0)), domain_wall_state(L)
        H, states, idx = _xx_lr_sparse(L, Jm)
        v0 = zeros(ComplexF64, length(states))
        v0[idx[sum(1 << (j - 1) for j in 1:(L ÷ 2))]] = 1.0
        ex(t) = _magn_sector(exp(Matrix(H) * (-im * t)) * v0, L, states)
        a = _run_tight(mpo, psi0, 1.0, 0.1, _sz_profile, ex)
        b = _run_tight(mpo, psi0, 1.0, 0.05, _sz_profile, ex)

        @test b.err < 5e-3                    # measured 1.546e-03
        # ⚠ ONLY ~FIRST ORDER HERE, and the bar says so rather than hiding it: measured ratio
        # 2.04 against 3.85 on the NN chain and 4.02 on OAT, at chi = 63/64 where truncation is
        # not the limiter. AT `m = 0` the sweep loses an order once the interaction is long
        # ranged; that is a property to investigate, not a failure of this test. Whether depth
        # recovers it is exactly the open question -- it recovers OAT completely.
        @test a.err / b.err > 1.7             # measured 2.04
        @test b.chi > 1
        @test b.dE < 1e-12                    # measured 1.4e-15
        @test isapprox(b.nrm, 1.0; atol = 1e-10)
    end

    @testset "OAT vs the closed form, and the exact Schmidt ceiling" begin
        # ⚠ AT ONE POWER OF `H` THE SWEEP WAS A LOW-ACCURACY INTEGRATOR ON OAT (2.13e-02 on the
        # removed fixed-knob path, surviving more budget, more rank and no truncation at all); the
        # Krylov depth is what closes the gap -- ~1e-14 at 30 vectors. The tolerance-driven frame
        # must reach the same.
        set_symmetry!(:none)
        mpo, psi0 = oat_mpo(L), x_polarized_state(L)
        o(p) = [total_sx(copy(p)) / max(norm(copy(p))^2, eps())]
        ex(t) = [oat_total_sx(L, t)]
        b = _run_tight(mpo, psi0, 1.0, 0.05, o, ex)
        @test b.err < 1e-10 || (@info "OAT no better than one power of H" b.err; false)
        @test b.dE < 1e-12                    # measured 1.5e-14
        @test isapprox(b.nrm, 1.0; atol = 1e-10)
        # The strongest structural check available on this model: OAT's Schmidt rank is capped at
        # `min(n, L-n) + 1` for ALL time, and the sweep must land ON that ceiling rather than
        # over-filling it. Both TDVP arms over-fill it, so this is a real property of the BUG
        # half-sweeps and not a tautology. The depth buys ACCURACY, not rank.
        @test b.chi == oat_rank(L, L ÷ 2, 1.0)
        set_symmetry!(:U1)
    end
end
