using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG

# Direct-sum two product states of equal total Sz to get a bond that carries
# more than one charge sector (see tests/BondUpdateBUG/test_frame.jl).
function mps_sum(a::SymMPS, b::SymMPS)
    L = length(a)
    ts = Any[]
    for i in 1:L
        dims = i == 1 ? (3,) : (i == L ? (1,) : (1, 3))
        push!(ts, to_concrete(oplus(a[i], b[i], dims)))
    end
    return SymMPS(ts, L)
end

# A generator that is diagonal in the charge sectors of leg 3 but NOT a
# multiple of the identity. Hermitian iff every scale is real. Its exponential
# is known in closed form, which gives expv an exact reference to hit -- and,
# unlike `A = c*I`, it spans a Krylov space of dimension > 1.
sector_parts(v) = [(s, getsub(v, 3, q -> q == s ? Colon() : nothing; preserve_space = true))
                   for (s, _) in v.spaces[3]]

diag_apply(scale) = v -> to_concrete(reduce(+, (scale(s) * p for (s, p) in sector_parts(v))))
diag_exact(scale, tau, v) =
    to_concrete(reduce(+, (exp(tau * scale(s)) * p for (s, p) in sector_parts(v))))

@testset "expv" begin

    psi = domain_wall_state(4)
    canonical!(psi, 1)
    f = bond_frame(psi, 1)
    x = to_concrete(f.U0 * f.S0)

    @testset "identity generator reproduces exp(tau)*x" begin
        y = expv(v -> v, -1.0 + 0.0im, x; hermitian = true)
        @test isapprox(norm(y - exp(-1.0) * x), 0.0; atol = 1e-12)
    end

    @testset "zero generator is the identity map" begin
        y = expv(v -> 0.0 * v, -0.1im, x; hermitian = true)
        @test isapprox(norm(y - x), 0.0; atol = 1e-14)
    end

    @testset "hermitian generator preserves norm under imaginary rotation" begin
        y = expv(v -> 2.0 * v, -0.3im, x; hermitian = true)
        @test isapprox(norm(y), norm(x); atol = 1e-12)
    end

    @testset "arnoldi path matches lanczos on a hermitian generator" begin
        yl = expv(v -> 2.0 * v, -0.2im, x; hermitian = true)
        ya = expv(v -> 2.0 * v, -0.2im, x; hermitian = false)
        @test isapprox(norm(yl - ya), 0.0; atol = 1e-11)
    end

    # ---- generators whose Krylov space is bigger than one dimension ----
    # Everything above terminates after a single Lanczos step, so none of it
    # actually exercises the recurrence.

    ent = mps_sum(product_state([:up, :down, :up, :down]),
                  product_state([:down, :up, :up, :down]))
    canonical!(ent, 1)
    fe = bond_frame(ent, 1)
    xe = to_concrete(fe.U0 * fe.S0)
    charges = [s[1][1] for (s, _) in xe.spaces[3]]

    @testset "the multi-sector fixture really is multi-sector" begin
        @test length(xe.spaces[3]) > 1
    end

    @testset "hermitian: matches the closed-form sector-diagonal exponential" begin
        scale = s -> 0.5 * s[1][1] + 1.0            # distinct real value per sector
        for tau in (-0.4 + 0.0im, -0.7im, 0.25 + 0.3im)
            y = expv(diag_apply(scale), tau, xe; hermitian = true)
            @test isapprox(norm(y - diag_exact(scale, tau, xe)), 0.0; atol = 1e-11)
        end
    end

    @testset "arnoldi: matches the same closed form" begin
        scale = s -> 0.5 * s[1][1] + 1.0
        for tau in (-0.4 + 0.0im, -0.7im)
            y = expv(diag_apply(scale), tau, xe; hermitian = false)
            @test isapprox(norm(y - diag_exact(scale, tau, xe)), 0.0; atol = 1e-11)
        end
    end

    @testset "reorth=true spans the same space as the bare recurrence" begin
        scale = s -> 0.5 * s[1][1] + 1.0
        y0 = expv(diag_apply(scale), -0.5im, xe; hermitian = true, reorth = false)
        y1 = expv(diag_apply(scale), -0.5im, xe; hermitian = true, reorth = true)
        @test isapprox(norm(y0 - y1), 0.0; atol = 1e-11)
    end

    # ---- the reason both paths exist ----

    @testset "arnoldi terminates on a non-hermitian generator; lanczos does not" begin
        # Complex sector scales => complex eigenvalues, so the Lanczos
        # assumption alpha in R is false. This is the situation the K/L
        # generators are in once P_perp = I - U0 U0' is applied before the
        # exponential.
        #
        # NOTE: on this particular generator Lanczos still lands on the right
        # NUMBER (measured, job 93396) -- the operator is normal and the true
        # Krylov space is only 2-dimensional. So the defensible failure is not
        # "wrong answer", it is cost and conditioning: Lanczos never breaks
        # down, burns the full maxiter budget, and builds a tridiagonal whose
        # off-diagonal diverges. Nothing guarantees the number stays right.
        scale = s -> (0.5 * s[1][1] + 1.0) + 0.3im * s[1][1]
        tau = -0.35 + 0.0im
        exact = diag_exact(scale, tau, xe)
        nsectors = length(xe.spaces[3])

        enable_krylov_log()
        ya = expv(diag_apply(scale), tau, xe; hermitian = false, maxiter = 60)
        arnoldi_depth = get_krylov_log()[end]

        yl = expv(diag_apply(scale), tau, xe; hermitian = true, maxiter = 60)
        lanczos_depth = get_krylov_log()[end]
        disable_krylov_log()

        @test isapprox(norm(ya - exact), 0.0; atol = 1e-12)
        @test arnoldi_depth == nsectors        # stops at the true Krylov dimension
        @test lanczos_depth == 60              # runs the budget out instead
        @test lanczos_depth > arnoldi_depth
    end

    @testset "arnoldi terminates at the true Krylov dimension when hermitian too" begin
        scale = s -> 0.5 * s[1][1] + 1.0
        enable_krylov_log()
        expv(diag_apply(scale), -0.3im, xe; hermitian = false)
        d = get_krylov_log()[end]
        disable_krylov_log()
        @test d == length(xe.spaces[3])
    end

    @testset "zero input returns zero" begin
        z = to_concrete(0.0 * xe)
        @test isapprox(norm(expv(v -> 2.0 * v, -0.1im, z; hermitian = true)), 0.0; atol = 1e-15)
        @test isapprox(norm(expv(v -> 2.0 * v, -0.1im, z; hermitian = false)), 0.0; atol = 1e-15)
    end

    @testset "tensor_inner is zero, not an error, when no charge blocks match" begin
        # preserve_space=true keeps both leg-3 spaces identical (contract
        # asserts equal spaces) while the STORED sectors stay disjoint -- that
        # is the structurally-zero case krylov.py::tensor_inner guards.
        s1 = first(xe.spaces[3])[1]
        a = getsub(xe, 3, q -> q == s1 ? Colon() : nothing; preserve_space = true)
        b = getsub(xe, 3, q -> q == s1 ? nothing : Colon(); preserve_space = true)
        @test a.spaces[3] == b.spaces[3]
        @test tensor_inner(a, b) == 0.0 + 0.0im
        @test tensor_inner(a, a) != 0.0 + 0.0im
    end
end
