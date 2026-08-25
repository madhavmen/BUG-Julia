# `cbe_bug_step!` PINNED AGAINST EXACT REFERENCES, on three models at L = 12.
#
# WHAT THE SWEEP IS. CBE widens the bond, the expansion frame itself becomes the basis, the
# half-sweep deepens it into a Krylov basis `span{Theta, H*Theta, H^2*Theta, ...}`, the environment
# is pushed through it, and the root Galerkin step is the only evolution in the step. Both frames
# are used across the step: `U_ex` becomes `W[i]` on the left half-sweep, `V_ex` becomes `Z[j]` on
# the right, so every bond is widened from both sides before the root sees it.
#
# WHY IT NEEDS ITS OWN FILE. `test_sweeps.jl` checks the sweep against `cbe_lubich_sweep` -- an
# agreement between two of OUR implementations, which would hold just as well if both were wrong.
# Nothing pinned it against an EXACT answer, on a model with growing entanglement, at an L where a
# defect is large. That is what is here.
#
# THE PROPERTY UNDER TEST IS THAT THE RANK GROWS. There is no evolved two-site tensor to split, so
# the only thing that can carry new directions into the state is the frame the half-sweep builds.
# If the expansion were being dropped anywhere the bond dimension would sit at its initial value --
# and a product start makes that unmissable, since chi = 1 is a fixed point of every sweep that
# does not genuinely widen. MEASURED here: 1 -> 15, 1 -> 64, 1 -> 7.
#
# ⛔ MOST ARMS PIN `m = 0` (one application of `H`) TO ISOLATE THE FRAME, WHICH IS NOT WHAT SHIPS.
# The defaults are `krylov_basis = 30`, `krylov_tol = 1e-6`. The third testset runs the DEFAULTS
# with nothing passed, because a suite in which every arm overrides the shipped configuration
# tests something no user runs.
#
# ⛔ EVERY REFERENCE IS EXACT, AND THE LONG-RANGE ONE IS NOT THE FREE-FERMION FORMULA.
# Jordan-Wigner leaves a string between the two sites of an XY term and it cancels ONLY for
# nearest neighbours, so `xx_free_fermion_sz` is the right reference for the NN chain and the
# WRONG one at range. The long-range arm therefore uses an exact propagation in the `Sz = 0`
# sector -- C(12,6) = 924 states, dense-exponentiable instantly, with no solver tolerance of its
# own to inherit.
#
# TWO STEP SIZES PER MODEL, ALWAYS. A single small error says "happens to be small"; two say
# "converging", and the ratio is what the order assertions read. Thresholds below are set FROM
# the measured values with roughly 3x headroom, not fitted to them.

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

"""
One step at a PINNED Krylov depth `m`, so the arms below isolate the frame rather than the
defaults. `m = 0` stops at one application of `H`, which is exactly what the CBE frame spans.
"""
_step_at_depth!(psi, mpo, tau; maxdim, cut, m::Int = 0) =
    cbe_bug_step!(psi, mpo, tau; krylov_basis = m, exact = true,
                  root_cutoff = 0.0, root_maxdim = 0, truncate = true,
                  maxdim = maxdim, trunc_thresh = cut, maxiter = 16)

"""
Evolve to `tmax` at Krylov depth `m` and report `(err, chi, dE, norm)`.

`obs`/`exact_obs` return vectors so one helper serves both a site-resolved profile and a scalar.
"""
function _run_at_depth(mpo, psi0, tmax, dt, obs, exact_obs; maxdim = 64, cut = 1e-12,
                          m::Int = 0)
    psi = copy(psi0)
    e0 = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
    for _ in 1:round(Int, tmax / dt)
        _step_at_depth!(psi, mpo, ComplexF64(-im * dt); maxdim = maxdim, cut = cut, m = m)
    end
    e1 = real(mpo_energy(copy(psi), mpo)) / max(norm(psi)^2, eps())
    return (err = maximum(abs.(obs(psi) .- exact_obs(tmax))),
            chi = maximum(bond_dims(psi)), dE = abs(e1 - e0), nrm = norm(psi))
end

_sz_profile(p) = magnetisation(copy(p)) ./ max(norm(copy(p))^2, eps())

# ── the stopping estimate, on dense matrices, before any tensor is involved ────────────────
#
# `_kry_contribution` is what decides the half-sweep basis depth, and it is pure linear algebra:
# `alpha`/`betas` in, one number out. Testing it here rather than through a sweep means a
# regression in the DEPTH CONTROL is distinguishable from a regression in the sweep, which the
# end-to-end arms below cannot do.
@testset "_kry_contribution is a CONSERVATIVE bound on the Krylov truncation error" begin
    # ⛔ THE DIRECTION OF THE INEQUALITY IS THE WHOLE POINT. Stopping when the estimate falls
    # below `tol` is only safe if the estimate is an UPPER bound on the true error. Measured
    # here it overestimates by 4x-160x and never underestimates outside roundoff -- so the rule
    # costs ~1-2 extra vectors and cannot silently stop short, which is the trade wanted.
    kry = RSVDCBEBondUpdate._kry_contribution
    Random.seed!(7)
    n = 200
    for dt in (0.05, 0.5), m in (3, 5, 8)
        A = randn(n, n); A = (A + A') / 2 / sqrt(n) * 4
        v = randn(ComplexF64, n)
        tau = ComplexF64(-im * dt)

        b0 = norm(v); V = [v / b0]; al = Float64[]; be = Float64[]
        for _ in 1:m
            w = A * V[end]
            push!(al, real(dot(V[end], w)))
            for _p in 1:2, u in V
                w -= dot(u, w) * u
            end
            b = norm(w); b <= 1e-13 && break
            push!(be, b); push!(V, w / b)
        end
        mm = length(al)
        T = zeros(mm, mm)
        for k in 1:mm
            T[k, k] = al[k]
            k < mm && (T[k, k + 1] = be[k]; T[k + 1, k] = be[k])
        end
        E = exp(tau * T)
        approx = b0 * sum(E[k, 1] * V[k] for k in 1:mm)
        truerr = norm(exp(Matrix(tau * A)) * v - approx)
        est = b0 * kry(al, be, tau)

        # Bound, with a floor so the assertion is not comparing two roundoff figures.
        @test est >= truerr * 0.5 || truerr < 1e-13
        # ...and not SO loose that the tolerance means nothing: 1e4 is far above the 160x seen.
        @test est <= max(truerr, 1e-13) * 1e4
    end
    # An empty recursion cannot certify anything and must not read as "converged".
    @test kry(Float64[], Float64[], ComplexF64(-im * 0.05)) == Inf
    # One step with no accepted residual likewise.
    @test kry([1.0], Float64[], ComplexF64(-im * 0.05)) == Inf
end

# ── ONE SVD, AT THE END -- pinned, not asserted in a comment ───────────────────────────────
#
# The half-sweep accumulates `m+1` Krylov blocks and the CALLER splits the stack ONCE. Selecting
# inside the loop is the failure mode `cbe_lubich_sweep`'s `grow_iters` demonstrates: it re-runs
# the CBE ranking every pass, re-truncates by budget, and throws away exactly the directions the
# next application of `H` needs -- MEASURED on OAT as 1.8x and then flat.
#
# ⛔ THIS INVARIANT LIVED ONLY IN A DOCSTRING, WHICH IS NOT ENOUGH. A comment cannot fail. The
# same session that wrote that comment also shipped a growth path whose "did it grow?" counter
# counted calls instead of widenings, and the no-op passed its gate. So the property is checked
# two ways here: structurally (the source has no SVD in the loop) and behaviourally (the frame
# actually carries the extra blocks into the state).
@testset "the Krylov frame is split ONCE, not per iteration" begin
    # ⚠ NORMALISE LINE ENDINGS FIRST. The checkout is CRLF on Windows, so searching for a
    # newline-delimited "end" finds NOTHING and the slice below silently fails -- which is how
    # the first version of this test errored instead of checking anything.
    src = replace(read(joinpath(@__DIR__, "..", "..", "src", "RSVDCBEBondUpdate", "sweeps",
                                "cbe_bug.jl"), String), "\r\n" => "\n")
    # The body of `_krylov_frame`, from its signature to the matching bare `end` at column 0.
    i0 = findfirst("function _krylov_frame(", src)
    @test i0 !== nothing
    body = src[first(i0):end]
    i1 = findfirst("\nend\n", body)
    @test i1 !== nothing
    body = i1 === nothing ? body : body[1:first(i1)]
    @test length(body) < length(src)          # the slice really did cut something

    # STRUCTURAL: no factorisation of any kind inside the recursion. Orthogonalisation is
    # Gram-Schmidt (`tensor_inner`), which is what makes the stack lossless.
    @test !occursin("svd(", body) ||
        (@info "an SVD appeared inside _krylov_frame -- the stack is no longer lossless"; false)
    @test !occursin("qr(", body)
    @test occursin("tensor_inner", body)      # ...and the orthogonalisation is still there

    # BEHAVIOURAL: deeper `m` must put MORE into the frame. If a cutoff ran per iteration the
    # extra Krylov directions would be discarded before the caller ever saw them, and the
    # pre-truncation bond dimension would stop responding to `m`.
    set_symmetry!(:U1)
    mpo, psi0 = xxz_mpo(L; J = 1.0, delta = 1.0), neel_state(L)
    widths = map((0, 2, 4)) do m
        psi = copy(psi0)
        info = cbe_bug_step!(psi, mpo, ComplexF64(-im * 0.05);
                             krylov_basis = m, krylov_tol = 0.0, exact = true,
                             truncate = false, split_cutoff = 0.0,
                             maxdim = 256, trunc_thresh = 1e-14, maxiter = 16)
        maximum(info.expanded)
    end
    @test widths[2] > widths[1] ||
        (@info "m=2 put nothing extra in the frame" widths; false)
    @test widths[3] > widths[2] ||
        (@info "m=4 put nothing extra in the frame" widths; false)
end

@testset "cbe_bug_step! against exact references, L=12" begin
    L = 12

    @testset "XX nearest neighbour vs the free-fermion closed form" begin
        set_symmetry!(:U1)
        mpo, psi0 = xxz_mpo(L; J = 1.0, delta = 0.0), domain_wall_state(L)
        ex(t) = xx_free_fermion_sz(L, t; J = 1.0, occupied = 1:(L ÷ 2))
        a = _run_at_depth(mpo, psi0, 1.0, 0.1, _sz_profile, ex)
        b = _run_at_depth(mpo, psi0, 1.0, 0.05, _sz_profile, ex)

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
        a = _run_at_depth(mpo, psi0, 1.0, 0.1, _sz_profile, ex)
        b = _run_at_depth(mpo, psi0, 1.0, 0.05, _sz_profile, ex)

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

    @testset "THE PACKAGE DEFAULTS -- Krylov depth is what closes the gap" begin
        # ⛔ THE ARMS ABOVE ALL PASS `m = 0` EXPLICITLY, so none of them touches the configuration
        # anyone actually runs. This one takes `cbe_bug_step!`'s own defaults
        # (`krylov_tol = 1e-6`, `krylov_basis = 30`) and pins that they are better than one power
        # of `H`, which is the whole reason the Krylov basis exists.
        #
        # ⚠ NOT CONSIDERING ENOUGH KRYLOV VECTORS IS AN EASILY-MISSED SOURCE OF ERROR, and it does
        # not look like a basis problem from outside: at `m = 0` the OAT error survives more
        # expansion budget, more rank, and the closing truncation turned off entirely. Only the
        # depth moves it. That is what these two assertions guard.
        set_symmetry!(:none)
        mpo, psi0 = oat_mpo(L), x_polarized_state(L)
        o(p) = [total_sx(copy(p)) / max(norm(copy(p))^2, eps())]
        ex(t) = [oat_total_sx(L, t)]
        one_power = _run_at_depth(mpo, psi0, 1.0, 0.05, o, ex; m = 0)
        deflt = begin
            psi = copy(psi0)
            for _ in 1:20
                cbe_bug_step!(psi, mpo, ComplexF64(-im * 0.05);   # DEFAULTS, nothing passed
                              exact = true, maxdim = 64, trunc_thresh = 1e-12, maxiter = 16)
            end
            (err = maximum(abs.(o(psi) .- ex(1.0))), chi = maximum(bond_dims(psi)))
        end
        # measured: one power 2.13e-02, defaults ~1e-14 -- twelve orders.
        @test deflt.err < 1e-10 ||
            (@info "defaults no better than one power of H" one_power.err deflt.err; false)
        @test deflt.err < one_power.err / 1e6
        # Still the exact analytic Schmidt ceiling: the depth buys ACCURACY, not rank.
        @test deflt.chi == oat_rank(L, L ÷ 2, 1.0)
        set_symmetry!(:U1)
    end

    @testset "OAT vs the closed form, and the exact Schmidt ceiling" begin
        set_symmetry!(:none)
        mpo, psi0 = oat_mpo(L), x_polarized_state(L)
        o(p) = [total_sx(copy(p)) / max(norm(copy(p))^2, eps())]
        ex(t) = [oat_total_sx(L, t)]
        a = _run_at_depth(mpo, psi0, 1.0, 0.1, o, ex)
        b = _run_at_depth(mpo, psi0, 1.0, 0.05, o, ex)

        # ⚠ THIS ARM IS PINNED AT `m = 0` and the error is ~1.5% of a signal of order 1.44. At one
        # power of `H` the sweep is a LOW-ACCURACY integrator on OAT; the testset above shows the
        # defaults take the same run to ~1e-14. This arm asserts that the FRAME alone converges
        # and lands on the right structure, not that it is accurate.
        @test b.err < 5e-2                    # measured 2.133e-02
        @test a.err / b.err > 3.0             # measured 4.02 -- second order
        @test b.dE < 1e-12                    # measured 1.5e-14
        @test isapprox(b.nrm, 1.0; atol = 1e-10)
        # The strongest structural check available on this model: OAT's Schmidt rank is capped at
        # `min(n, L-n) + 1` for ALL time, and the sweep must land ON that ceiling rather than
        # over-filling it. Both TDVP arms over-fill it (see `cbe_bug.jl`), so this is a real
        # property of the BUG half-sweeps and not a tautology.
        @test b.chi == oat_rank(L, L ÷ 2, 1.0)   # measured 7 == 7
        set_symmetry!(:U1)
    end
end
