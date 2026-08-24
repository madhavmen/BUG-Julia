# THE ONE-AXIS TWISTING BENCHMARK of arXiv:2208.10972 Fig. 2, at L = 10.
#
# WHAT MAKES THIS THE STRONGEST TEST IN THE SUITE. Every other real-time benchmark here is
# measured against a magnon sector, a free-fermion profile, or a sparse propagation. OAT is
# analytic ALL THE WAY: `H = (Σ S^z)^2 / 2` is diagonal in the computational basis, so the
# evolved state, its `S^x_tot`, its Schmidt spectrum and its entanglement entropy are all closed
# forms (`tests/common/analytic_reference.jl`). Nothing below compares an integrator against
# another integrator.
#
# ORDERING. The reference is pinned against dense ED FIRST -- if the closed forms were wrong,
# every integrator test below would fail in the same direction and agree with each other, which
# is the failure mode that looks most like success.
#
# THE RANK CEILING IS THE POINT OF THE `maxdim` SPREAD. OAT's exact Schmidt rank at the cut after
# site `n` is `min(n, L-n) + 1` FOR ALL TIME -- 6 at the centre for L = 10 -- because the only
# entangling factor `exp(-i t a b)` depends on the two blocks solely through their
# magnetisations. So `maxdim >= 6` is FULL RANK and the sweeps are exact there, while
# `maxdim < 6` is a genuine truncation. Testing only the first would measure Krylov roundoff and
# call it an integrator; testing only the second would never show convergence. Both are here.

using Test
using LinearAlgebra

@testset "OAT: analytic reference vs dense ED" begin
    # `H` LITERALLY AS THE PAPER WRITES IT -- `(Σ_ℓ S^z_ℓ)^2 / 2`, squared as a dense matrix --
    # so the reference shares nothing with `oat_mpo`'s expansion of the square. If the expansion
    # (or the `L/8` c-number) were wrong, this is what would catch it.
    for L in (4, 6)
        set_symmetry!(:none)
        H = dense_total_sz(L)^2 / 2
        v0 = dense_state(x_polarized_state(L))
        @test norm(v0) ≈ 1.0 atol = 1e-12

        SXtot = sum(_embed(ComplexF64[0 0.5; 0.5 0], j, L) for j in 1:L)
        n = L ÷ 2
        for t in (0.0, 0.3, 1.0, 2.5, 2π)
            v = dense_exact_propagate(H, v0, t)
            @test real(v' * SXtot * v) ≈ oat_total_sx(L, t) atol = 1e-11

            # entropy and spectrum, from the dense state's own SVD across the same cut
            M = reshape(v, 2^n, 2^(L - n))          # site 1 is the most significant index
            s = svdvals(M)
            sref = oat_schmidt_values(L, n, t)
            ns = min(length(s), length(sref))
            @test sort(s; rev = true)[1:ns] ≈ sort(sref; rev = true)[1:ns] atol = 1e-11
            p = abs2.(s)
            Sref = -sum(q * log(q) for q in p if q > 1e-300; init = 0.0)
            @test oat_entropy(L, n, t) ≈ Sref atol = 1e-10
        end
        # THE RANK CEILING, stated in the header and checked rather than assumed.
        @test all(oat_rank(L, n, t) <= min(n, L - n) + 1 for t in (0.5, 1.7, 3.3, 9.0))
        @test oat_rank(L, n, 0.0) == 1                      # a product state
    end
end

@testset "OAT: the operator" begin
    set_symmetry!(:none)
    L = 8
    mpo = oat_mpo(L)
    # Virtual dimension 3, INDEPENDENT of L -- the whole cost claim of the `λ = 1` self-loop.
    @test mpo_virtual_dims(mpo) == [1, 3, 3, 3, 3, 3, 3, 3, 1]
    @test mpo_virtual_dims(oat_mpo(16)) == [1; fill(3, 15); 1]

    # The SAME operator through `pair_mpo`'s completely different `O(L)` automaton.
    pairs = oat_mpo_pairs(L)
    @test maximum(mpo_virtual_dims(pairs)) > 3            # it really is the expensive route
    for st in (neel_state(L), domain_wall_state(L), x_polarized_state(L))
        @test real(mpo_energy(copy(st), mpo)) ≈ real(mpo_energy(copy(st), pairs)) atol = 1e-12
    end

    # And against the dense `(Σ S^z)^2 / 2`, MINUS the c-number `oat_mpo` drops.
    Hd = dense_total_sz(L)^2 / 2
    for st in (neel_state(L), domain_wall_state(L), x_polarized_state(L))
        v = dense_state(copy(st))
        @test real(mpo_energy(copy(st), mpo)) ≈
              real(v' * Hd * v) - oat_energy_shift(L) atol = 1e-11
    end
end

@testset "OAT: the initial state and the observable" begin
    set_symmetry!(:none)
    L = 10
    psi = x_polarized_state(L)
    @test bond_dims(psi) == ones(Int, L - 1)              # the paper's "an MPS with D = 1"
    @test norm(psi) ≈ 1.0 atol = 1e-12
    @test sx_profile(copy(psi)) ≈ fill(0.5, L) atol = 1e-12
    @test total_sx(copy(psi)) ≈ oat_total_sx(L, 0.0) atol = 1e-12
    @test all(abs.([sz_expectation(copy(psi), j) for j in 1:L]) .< 1e-12)

    # `S^x`, not `-i S^y`: the sign trap named in `sx_operator`'s docstring, checked on a state
    # that is NOT symmetric under spin flip so the two would actually differ.
    @test Matrix(sx_operator().RMTs[1]) ≈ [0 0.5; 0.5 0] atol = 1e-12
    up = product_state(fill(:up, L))
    @test all(abs.(sx_profile(copy(up))) .< 1e-12)        # ⟨↑|S^x|↑⟩ = 0

    # ⛔ The :none-only guards must THROW rather than return a symmetric zero.
    set_symmetry!(:U1)
    @test_throws ArgumentError x_polarized_state(L)
    @test_throws ArgumentError sx_operator()
    set_symmetry!(:none)
end

# ── the integrators, L = 10, against the closed form ────────────────────────

"Evolve `x_polarized_state(L)` with `step`, returning `(t, S^x_tot, EE, max bond)` per sample."
function _oat_run(L::Int, step, dt::Float64, nsteps::Int; every::Int = 1)
    psi = x_polarized_state(L)
    n = L ÷ 2
    ts, sx, ee, bd = Float64[], Float64[], Float64[], Int[]
    push!(ts, 0.0); push!(sx, total_sx(copy(psi)))
    push!(ee, 0.0); push!(bd, maximum(bond_dims(psi)))
    for k in 1:nsteps
        step(psi, ComplexF64(-im * dt))
        if k % every == 0
            p = abs2.(bond_spectrum(copy(psi), n))
            p ./= max(sum(p), eps())
            push!(ts, k * dt)
            push!(sx, total_sx(copy(psi)))
            push!(ee, -sum(q * log(q) for q in p if q > 1e-300; init = 0.0))
            push!(bd, maximum(bond_dims(psi)))
        end
    end
    return ts, sx, ee, bd
end

"One `maxdim`/`dt`, run to `T`: the largest `|S^x_tot - exact|` over the trajectory."
function _oat_sx_error(mk, L::Int, maxdim::Int, dt::Float64, T::Float64)
    nst = round(Int, T / dt)
    ts, sx, _, _ = _oat_run(L, mk(maxdim), dt, nst; every = max(1, nst ÷ 5))
    return maximum(abs.(sx .- [oat_total_sx(L, t) for t in ts]))
end

@testset "OAT L=10: rank-converged at the ceiling, and O(dt^2) beyond it" begin
    set_symmetry!(:none)
    L, T = 10, 2.0                             # past the first collapse
    mpo = oat_mpo(L)

    # ⛔ "FULL RANK ⇒ EXACT" IS FALSE HERE, AND ASSUMING IT COST A ROUND OF RED TESTS. That
    # property belongs to the projector-splitting integrator applied to a NEAREST-NEIGHBOUR `H`
    # (the L=8 XX runs in `benchmarks/cbe_sweeps_l16.jl` measure ~1e-14 at every `dt`, which is
    # what the assumption was borrowed from). OAT's `H` is ALL-TO-ALL, and MEASURED here at
    # `dt = 0.05`, `T = 2`: maxdim 6 gives 4.3477e-3 and maxdim 8 gives 4.3475e-3 -- they agree
    # to 2e-7, i.e. the rank is NOT what limits them. The residual is the time step, and it is
    # `O(dt^2)`. So the two claims worth testing are exactly those, and they are separated:
    #
    #   RANK-CONVERGED  the cap stops mattering at the exact ceiling (6), not before and not after
    #   SECOND ORDER    what is left obeys `err ~ dt^2`, which is the integrator's actual claim
    mk_cbe(d) = (p, tau) -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = d, trunc_thresh = 1e-14)
    mk_t2(d)  = (p, tau) -> tdvp2_step!(p, mpo, tau; maxdim = d, trunc_thresh = 1e-14)
    # INTERLEAVED (`decoupled = false`) -- the reference's own ordering, TDVPSweepCBE1Si.m.
    mk_bug(d) = (p, tau) -> cbe_bug_step!(p, mpo, tau; maxdim = d,
                                          trunc_thresh = 1e-14)

    for (name, mk) in (("tdvp_cbe1s", mk_cbe), ("tdvp2", mk_t2), ("cbe_bug", mk_bug))
        e6  = _oat_sx_error(mk, L, 6,  0.05, T)      # AT the exact rank ceiling
        e16 = _oat_sx_error(mk, L, 16, 0.05, T)      # far above it
        # Raising the cap past the ceiling changes nothing: whatever error remains is not rank.
        @test abs(e6 - e16) < 1e-4 || (@info "$name not rank-converged at the ceiling" e6 e16;
                                       false)
        # And that remainder is second order. Halving `dt` must quarter it; the window is wide
        # because the constant is not being asserted, only the ORDER.
        #
        # ⛔ EXCEPT FOR `cbe_bug`, WHICH IS EXACT ON THIS PROBLEM -- SO IT IS ASSERTED AS EXACT
        # RATHER THAN GIVEN A WIDER ORDER WINDOW. `H_OAT` is diagonal with a finite set of `m^2`
        # eigenvalues, so the sweep's Krylov basis CLOSES the local invariant subspace and the
        # step carries no dt error at all. MEASURED at the settings below:
        #
        #     tdvp_cbe1s          4.3475e-03    ratio 4.03
        #     tdvp2               4.3475e-03    ratio 4.03
        #     cbe_bug             4.7962e-14    ratio 0.63   <- ROUNDOFF, not an order
        #
        # ⚠ A RATIO TEST HERE WOULD BE MEASURING NOISE. Both `cbe_bug` errors sit at 1e-14, so
        # `e16/eh` is the quotient of two roundoff figures and wanders either side of 1 -- the
        # same "no window" trap `docs/current_work.md` records for full-rank dt scans. Widening
        # the window to admit 0.63 would turn a real assertion into one that cannot fail.
        #
        # ⛔ THIS SUPERSEDES THE OLD 10-30 WINDOW, which encoded the RETIRED K-step sweep's
        # fourth-order superconvergence (ratio 19.59 at 2.2901e-06). That sweep was removed on
        # 2026-08-24; the current one is five orders better here and exact to roundoff.
        eh = _oat_sx_error(mk, L, 16, 0.025, T)
        if name == "cbe_bug"
            @test e16 < 1e-12 || (@info "cbe_bug no longer exact on OAT" e16 eh; false)
            @test eh  < 1e-12 || (@info "cbe_bug no longer exact on OAT at dt/2" e16 eh; false)
        else
            @test 3.0 < e16 / eh < 5.5 ||
                (@info "$name dt-order outside its window" name e16 eh e16 / eh; false)
        end
    end
end

@testset "OAT L=10: the PHYSICAL rank is 6, whatever the bond dimension says" begin
    set_symmetry!(:none)
    L, dt, nst = 10, 0.05, 40
    mpo = oat_mpo(L)
    psi = x_polarized_state(L)
    for _ in 1:nst
        tdvp_cbe1s_step!(psi, mpo, ComplexF64(-im * dt); maxdim = 16, trunc_thresh = 1e-14)
    end

    # ⛔ `maximum(bond_dims) <= 6` IS THE WRONG TEST AND IT FAILS: MEASURED 16, i.e. the bond
    # fills right up to `maxdim`. That is not a defect, and it is not disagreement with the
    # closed form either -- it is the CBE expansion proposing directions plus a `1e-14`
    # truncation keeping everything above the threshold, so numerical noise occupies the bond
    # above the 6 physically-occupied Schmidt values. It is also exactly what arXiv:2208.10972
    # Fig. 2(c) shows at L = 100, where `D(t)` climbs toward `D_max = 500` although the exact
    # rank there is only 51.
    #
    # What IS bounded is the physically occupied rank, so that is what is asserted.
    #
    # ⛔ AND NOT BY COUNTING AMPLITUDES ABOVE 1e-6, which is how an earlier version of this test
    # failed. MEASURED spectrum at this point (normalised, centre bond, T = 2):
    #
    #     k        1         2         3         4         5         6    | 7        8        9
    #     s[k]  5.96e-1   5.79e-1   5.45e-1   9.42e-2   4.44e-2   4.39e-2 | 1.15e-6  1.07e-6  7.6e-7
    #
    # so `count(>(1e-6), s)` is 8, not 6 -- the threshold sat right on the noise shoulder. The
    # noise floor is set by `trunc_thresh`, not by physics, and an amplitude of 1.15e-6 is a
    # WEIGHT of 1.3e-12, four orders below the run's own accuracy. Counting amplitudes at 1e-6
    # therefore measures the truncation threshold, not the rank.
    #
    # The gap itself is unambiguous -- `s[6]/s[7]` is 3.8e4 -- so the claim is stated as a gap, a
    # weight, and agreement with the closed form. `count(>(1e-4))` and `count(>(1e-3))` BOTH give
    # 6, i.e. 1e-4 sits on a plateau rather than on a cliff edge.
    s = bond_spectrum(copy(psi), L ÷ 2)
    s ./= max(sqrt(sum(abs2, s)), eps())
    ex = sort(oat_schmidt_values(L, L ÷ 2, dt * nst); rev = true)
    ex ./= max(sqrt(sum(abs2, ex)), eps())

    @test count(>(1e-4), s) == 6 || (@info "occupied rank is not 6" s; false)
    @test length(s) > 6                     # the noise floor really is there -- the test is live
    @test sum(abs2, s[7:end]) < 1e-10       # MEASURED 3.0e-12: it carries no weight
    @test s[6] / s[7] > 1e3                 # MEASURED 3.8e4: the gap, stated directly
    # THE STRONGEST FORM: the six occupied values are not merely six, they are the ANALYTIC six.
    # MEASURED max deviation 6.1e-4, dominated by `s[6]`; the run's own `|δS^x|` here is 4.3e-3.
    @test maximum(abs.(s[1:6] .- ex[1:6])) < 5e-3 ||
        (@info "occupied spectrum does not match the closed form" s[1:6] ex[1:6]; false)
end

@testset "OAT L=10: truncated rank converges to the ceiling" begin
    set_symmetry!(:none)
    L, dt, nst = 10, 0.05, 40
    mpo = oat_mpo(L)
    # A rank cap BELOW the exact ceiling is a genuine truncation; at and above it the error must
    # fall to the full-rank floor. Monotone in `maxdim` is the claim, and 6 is where it lands.
    errs = Float64[]
    for maxdim in (2, 3, 4, 6, 8)
        step = (p, tau) -> tdvp_cbe1s_step!(p, mpo, tau; maxdim = maxdim,
                                            trunc_thresh = 1e-14)
        ts, sx, _, _ = _oat_run(L, step, dt, nst; every = 8)
        push!(errs, maximum(abs.(sx .- [oat_total_sx(L, t) for t in ts])))
    end
    # THE CLIFF IS AT 6, and that is the whole claim: below the ceiling the cap dominates, at and
    # above it the cap is irrelevant and only the `dt` error is left. MEASURED at `dt = 0.05`,
    # `T = 2`: maxdim 6 -> 4.3477e-3, maxdim 8 -> 4.3475e-3.
    #
    # ⛔ The floor is NOT machine precision, and asserting that it was is what made an earlier
    # version of this test red -- see the header of the `O(dt^2)` testset above. The floor is the
    # time-step error, so it is compared against `dt^2` and not against zero.
    dt2 = dt^2
    @test errs[end] < 3 * dt2                    # maxdim = 8 > ceiling: at the dt floor
    @test errs[end - 1] < 3 * dt2                # maxdim = 6 == ceiling: the SAME floor
    @test abs(errs[end] - errs[end - 1]) < 1e-4  # the cap has stopped mattering
    @test errs[1] > 10 * errs[end]               # maxdim = 2 is dominated by truncation instead
    # Monotone only over the TRUNCATED caps (2, 3, 4). ⛔ Not over the full list: 6 and 8 sit on
    # the same `dt` floor, so their ORDER is roundoff and asserting it is a test that fails at
    # random.
    @test issorted(errs[1:3]; rev = true) ||
          (@info "OAT truncated-rank error not monotone" errs; false)
end

@testset "OAT L=10: the paper's Z2 symmetry, and that it is the same physics" begin
    # arXiv:2208.10972 Fig. 2 declares its symmetry in the caption -- "computed using δ=0.01 and
    # Z2 spin symmetry" -- and this is that run. The conserved quantity is the GLOBAL SPIN FLIP
    # `P = Π_ℓ 2 S^x_ℓ`, not `S^z_tot`, because `H = (Σ S^z)²/2` is EVEN in `S^z`. In the sigma^x
    # basis of `z2_ising.jl` that flip is diagonal, which inverts the whole difficulty: `S^x`
    # becomes the diagonal charge-neutral operator (measurable at all) and `⊗|→⟩` becomes a
    # single charge-0 sector at chi = 1 -- the paper's "an MPS with D = 1", literally.
    #
    # ⛔ WHAT THIS TEST IS FOR: `:Z2` and `:none` are DIFFERENT BASES (sigma^x vs sigma^z), not
    # two storage layouts for one basis. So agreement between them is a statement that the
    # physics is basis-independent, and it is the only check that would catch a wrong
    # dualisation in the Z2 pair term. That is not hypothetical -- building the vertex as `Z`/`Z`
    # instead of the adjoint pair `Z`/`Z'` gives both the opening block's outgoing virtual leg
    # and the closing block's incoming one direction `'-'`, which Telum refuses to contract, and
    # it surfaces as `_charged_transport` finding no transport block rather than as anything
    # resembling a basis error. See `_oat_sz_vertex`.
    L, dt, nst, D = 10, 0.05, 40, 6

    function trajectory(mode::Symbol)
        set_symmetry!(mode)
        mpo = oat_mpo(L)
        psi = x_polarized_state(L)
        e0 = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
        sx0 = total_sx(copy(psi))
        chi0 = maximum(bond_dims(psi))
        sxs = Float64[]
        for k in 1:nst
            tdvp_cbe1s_step!(psi, mpo, ComplexF64(-im * dt); maxdim = D, trunc_thresh = 1e-14)
            k % 8 == 0 && push!(sxs, total_sx(copy(psi)))
        end
        return (; e0, sx0, chi0, sxs, chi = maximum(bond_dims(psi)))
    end

    z2 = trajectory(:Z2)
    nn = trajectory(:none)

    # The x-polarised start, in the mode where it is a single sector: chi = 1 and S^x_tot = L/2.
    @test z2.chi0 == 1
    @test abs(z2.sx0 - L / 2) < 1e-12
    # `Σ_{ℓ<ℓ'} S^z S^z` has expectation EXACTLY zero on ⊗|→⟩ (every `⟨S^z_ℓ⟩` vanishes and the
    # sites are independent), and it commutes with itself, so this pins the whole MPO -- a wrong
    # coefficient, a wrong dualisation or a missed pair would move it off zero.
    @test abs(z2.e0) < 1e-12
    # Both modes reach the exact ceiling rather than stalling below it.
    @test z2.chi == 6
    @test nn.chi == 6

    # THE CLAIM. MEASURED max|Z2 - none| over the trajectory: 1.6e-12 (tdvp_cbe1s), 5.4e-13
    # (tdvp2), 1.5e-13 (cbe_bug). The threshold is loose against that but still three orders
    # below the run's own `dt` error (4.3e-3), so it cannot pass by accident.
    @test maximum(abs.(z2.sxs .- nn.sxs)) < 1e-9 ||
        (@info "Z2 and none disagree" z2.sxs nn.sxs; false)
    # And both must actually track the closed form, or agreeing with each other proves nothing.
    ex = [oat_total_sx(L, 8k * dt) for k in 1:length(z2.sxs)]
    @test maximum(abs.(z2.sxs .- ex)) < 3 * dt^2

    set_symmetry!(:none)          # leave the mode as the rest of the file expects
end

