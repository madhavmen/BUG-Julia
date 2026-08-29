# CERTIFY THE ANALYTIC REFERENCE -- the ONLY place a dense Liouvillian is ever built.
#
# ⛔ WHAT THIS FILE IS FOR. `xx_boundary_covariance` claims to be the exact `<S^z_j(t)>` of the
# boundary-driven XX chain at a cost of `O(N^3)` IN THE CHAIN LENGTH. That claim rests on a chain
# of paper arguments -- the Lindbladian is quadratic, the two-point function closes, the
# Jordan-Wigner string at site N cancels in every term where it could have mattered -- and a paper
# argument is not a measurement. This file builds the `4^L x 4^L` superoperator from first
# principles at L = 2..5 and checks the two against each other.
#
# ⛔ AND THEN THE DENSE OBJECT IS RETIRED. Once this passes, every downstream use -- the MPS
# calibration, the knob studies, the log-grid suite -- measures against the ANALYTIC solution and
# never builds a `4^L` matrix again. That is the whole point: a reference that stops at L = 5 can
# certify wiring but can never certify a benchmark.
#
# ⚠ NO PACKAGE LOAD. `tests/common/free_fermion.jl` is stdlib-only, so this runs in SECONDS rather
# than paying the ~170 s `using BUGJulia` cost. Nothing here touches an MPS, an MPO or Telum, and
# it must stay that way -- the reference has to be independent of the machinery it certifies.
#
# Run:  julia benchmarks/imaginary_time/certify_xx_reference.jl

using LinearAlgebra, Printf
include(joinpath(@__DIR__, "..", "..", "tests", "common", "free_fermion.jl"))

const _SPd = ComplexF64[0 1; 0 0]      # sigma^+ : RAISES to up  == c^dag  (up = occupied)
const _SMd = ComplexF64[0 0; 1 0]
const _SZd = ComplexF64[0.5 0; 0 -0.5]

_site(op, k, L) = kron(Matrix{ComplexF64}(I, 2^(k - 1), 2^(k - 1)), ComplexF64.(op),
                       Matrix{ComplexF64}(I, 2^(L - k), 2^(L - k)))

"""
Dense `4^L x 4^L` Liouvillian of the boundary-driven XX chain, in SPIN operators.

⛔ SPIN, NOT FERMION, ON PURPOSE. The MPS side (`fused_boundary_lindblad`) is built from
`sigma^+/sigma^-` including the boundary jumps, so a fermionic dense operator would be certifying a
DIFFERENT model and the Jordan-Wigner string argument would go untested. Building it in spins is
what makes the string cancellation a measured fact rather than an assumption.

⚠ `vec` IS COLUMN-MAJOR, so `vec(A rho B) = kron(transpose(B), A) vec(rho)` -- the transpose sits
on the BRA leg. `S^z` is real symmetric, so a dephasing model cannot tell a missing transpose from
a present one; `sigma^+ rho sigma^-` can, and that is why this model is the one worth certifying.
"""
function dense_liouvillian(L::Int; J::Float64 = 1.0, eps_L::Float64 = 1.0, eps_R::Float64 = 1.0,
                           mu_L::Float64 = 1.0, mu_R::Float64 = -1.0)
    d = 2^L
    H = zeros(ComplexF64, d, d)
    for k in 1:(L - 1)
        H .+= J .* (_site(_SPd, k, L) * _site(_SMd, k + 1, L) +
                    _site(_SMd, k, L) * _site(_SPd, k + 1, L))
    end
    Id = Matrix{ComplexF64}(I, d, d)
    Ls = -im .* (kron(Id, H) - kron(transpose(H), Id))
    for (k, e, m) in ((1, eps_L, mu_L), (L, eps_R, mu_R))
        for (jump, g) in ((_site(_SPd, k, L), e * (1 + m) / 2),
                          (_site(_SMd, k, L), e * (1 - m) / 2))
            g == 0 && continue
            LdL = jump' * jump
            Ls .+= g .* (kron(transpose(jump'), jump)
                         .- 0.5 .* kron(Id, LdL) .- 0.5 .* kron(transpose(LdL), Id))
        end
    end
    return Ls
end

"`<S^z_j>` from a density matrix flattened column-major."
function dense_sz(vrho, L::Int)
    rho = reshape(vrho, 2^L, 2^L)
    t = tr(rho)
    return [real(tr(_site(_SZd, k, L) * rho) / t) for k in 1:L]
end

"Bond currents from a dense state, for the same continuity convention the analytic side uses."
function dense_current(vrho, L::Int; J::Float64 = 1.0)
    rho = reshape(vrho, 2^L, 2^L); t = tr(rho)
    return [real(tr((2im * J) .* (_site(_SPd, k, L) * _site(_SMd, k + 1, L) -
                                  _site(_SMd, k, L) * _site(_SPd, k + 1, L)) * rho) / t) / 2
            for k in 1:(L - 1)]
end

"""
    certify(; Ls = (2, 3, 4, 5), ts = (0.1, 0.5, 2.0, 8.0, 40.0)) -> Bool

Dense `exp(t*L)` versus the analytic Lyapunov solution, site by site, at several `t` and `L`.

⚠ ASYMMETRIC DRIVE (`mu_L = +1`, `mu_R = -1`). A symmetric one gives a reflection-symmetric profile
that hides left/right and ket/bra swaps -- the same trap as pinning a site convention with a Neel
state. `t = 40` is included because the two must agree in the STEADY STATE as well as in the
transient, and those are different failure modes: a wrong `G` breaks the NESS while leaving the
early transient nearly right.
"""
function certify(; Ls = (2, 3, 4, 5), ts = (0.1, 0.5, 2.0, 8.0, 40.0),
                 J::Float64 = 1.0, eps::Float64 = 1.0, tol::Float64 = 1e-11)
    @printf("\nCERTIFY  dense 4^L Liouvillian  vs  analytic N x N Lyapunov\n")
    @printf("  J=%g  eps_L=eps_R=%g  mu_L=+1  mu_R=-1   (maximally mixed start)\n", J, eps)
    @printf("\n%-4s %-8s %-12s %-12s %s\n", "L", "t", "max|dSz|", "max|dj|", "analytic <S^z_j>")
    ok = true
    for L in Ls
        Lsup = dense_liouvillian(L; J = J, eps_L = eps, eps_R = eps, mu_L = 1.0, mu_R = -1.0)
        d = 2^L
        vrho0 = vec(Matrix{ComplexF64}(I, d, d) ./ d)
        for t in ts
            ve = exp(Matrix(Lsup) .* float(t)) * vrho0
            a = dense_sz(ve, L)
            b = xx_boundary_sz(L, t; J = J, eps_L = eps, eps_R = eps, mu_L = 1.0, mu_R = -1.0)
            C = xx_boundary_covariance(L, t; J = J, eps_L = eps, eps_R = eps,
                                       mu_L = 1.0, mu_R = -1.0)
            ja = dense_current(ve, L; J = J)
            jb = xx_boundary_current(C; J = J)
            dz = maximum(abs.(a .- b))
            dj = isempty(ja) ? 0.0 : maximum(abs.(ja .- jb))
            ok &= dz < tol && dj < tol
            @printf("%-4d %-8.2f %-12.3e %-12.3e %s\n", L, t, dz, dj,
                    join([@sprintf("%+.6f", x) for x in b], " "))
        end
    end
    @printf("\n%s\n", ok ?
        "CERTIFIED -- the analytic reference reproduces the dense Liouvillian; retire the dense one" :
        "NOT CERTIFIED -- do NOT use the analytic reference")
    return ok
end

"""
    ballistic(; Ns = (8, 16, 32, 64, 128)) -> Bool

The published STRUCTURAL facts about this model, checked at sizes no dense object can reach.

⛔ THESE ARE THE ANCHORS THAT ARE NOT OUR OWN OUTPUT. Two statements about the boundary-driven XX
chain are established in the literature and are sharp enough to falsify a wrong reference:

  1. TRANSPORT IS BALLISTIC -- the NESS current is UNIFORM along the chain and INDEPENDENT of `N`.
     A diffusive reference would give `j ~ 1/N`, which at `N = 8` vs `N = 128` is a factor of 16.
  2. THE PROFILE IS FLAT IN THE BULK -- the magnetisation drop sits entirely at the two boundaries,
     again the ballistic signature, and the opposite of the linear ramp a diffusive chain shows.

⚠ Neither needs a prefactor recalled from memory, which is why they are used instead of a quoted
current formula: they are checked as SHAPES, and a shape cannot be fudged by a normalisation.
"""
# ⚠ THE DRIVE STRENGTH IS `drive`, NOT `eps`. Naming it `eps` shadows `Base.eps` INSIDE THE
# FUNCTION BODY, so the `eps()` guarding a division became a call on a Float64 -- a MethodError
# raised only after the whole table had printed, which reads like a numerical failure rather than
# a name collision.
function ballistic(; Ns = (8, 16, 32, 64, 128), J::Float64 = 1.0, drive::Float64 = 1.0)
    @printf("\nBALLISTIC TRANSPORT (published structural facts)  J=%g eps=%g\n", J, drive)
    @printf("%-6s %-14s %-14s %-14s %-14s\n", "N", "current", "max|dj| bond",
            "bulk |Sz| max", "edge Sz(1)")
    js = Float64[]
    ok = true
    for N in Ns
        C = xx_boundary_ness(N; J = J, eps_L = drive, eps_R = drive, mu_L = 1.0, mu_R = -1.0)
        j = xx_boundary_current(C; J = J)
        sz = real.(diag(C)) .- 0.5
        bulk = N > 4 ? maximum(abs.(sz[3:(N - 2)])) : 0.0
        push!(js, j[N ÷ 2])
        uniform = maximum(abs.(diff(j)))
        ok &= uniform < 1e-10
        @printf("%-6d %-14.9f %-14.3e %-14.3e %-14.9f\n", N, j[N ÷ 2], uniform, bulk, sz[1])
    end
    spread = (maximum(js) - minimum(js)) / max(abs(js[1]), 1e-300)
    @printf("\n  current spread over N = %s : %.3e  (ballistic => ~0, diffusive => O(1))\n",
            string(collect(Ns)), spread)
    ok &= spread < 1e-9
    @printf("%s\n", ok ? "BALLISTIC CONFIRMED -- uniform current, N-independent" :
                         "NOT BALLISTIC -- reference disagrees with the published transport class")
    return ok
end

"""
    zeno(; epss = ...) -> Nothing

The QUANTUM ZENO turnover: the NESS current rises linearly in the drive strength `eps` at small
`eps` and FALLS as `1/eps` at large `eps`, because a strongly measured boundary decouples from the
chain. This is the other published statement about the model and the one that makes `eps` a
meaningful knob rather than an arbitrary constant.
"""
function zeno(; N::Int = 32, J::Float64 = 1.0,
              epss = (0.01, 0.03, 0.1, 0.3, 1.0, 2.0, 4.0, 10.0, 30.0, 100.0))
    @printf("\nZENO TURNOVER  N=%d J=%g\n%-10s %-16s %-10s\n", N, J, "eps", "current", "j*eps")
    for e in epss
        C = xx_boundary_ness(N; J = J, eps_L = e, eps_R = e, mu_L = 1.0, mu_R = -1.0)
        j = xx_boundary_current(C; J = J)[N ÷ 2]
        @printf("%-10.3g %-16.9f %-10.4f\n", e, j, j * e)
    end
    println("  small eps: current ~ eps (j*eps ~ eps^2);  large eps: current ~ 1/eps (j*eps flat)")
    return nothing
end

"""
    closed_form(; ...) -> Bool

The Lyapunov solve against the CLOSED FORM `j = (mu_L - mu_R) eps J^2 / (4J^2 + eps^2)`, swept over
`N`, `J`, `eps` and the bias.

⛔ THIS IS THE CHECK THAT MAKES THE MODEL A REFERENCE RATHER THAN A SIMULATION. `certify()` above
only shows the analytic solve agrees with a dense object at `L <= 5`. This shows it agrees with a
SINGLE EXPRESSION at every `N`, so the benchmark at `N = 32` or `N = 64` has a number to be right
or wrong about -- which is precisely what `PLAN-open-systems.md` §0 lists as missing ("exact ref:
—") for every open model in the stack.

⚠ THE SWEEP CROSSES THE ZENO PEAK AT `eps = 2J` DELIBERATELY. Below it the current rises with the
drive and above it falls; a reference that had the wrong functional form but a fitted prefactor
would match on one side and fail on the other, so a sweep that stayed on one side would not
discriminate.
"""
function closed_form(; Ns = (4, 8, 32, 64), Js = (0.5, 1.0, 2.0),
                     drives = (0.1, 0.5, 2.0, 5.0, 20.0),
                     mus = ((1.0, -1.0), (1.0, 0.0), (0.5, -0.5), (0.3, -0.9)),
                     tol::Float64 = 1e-10)
    @printf("\nCLOSED FORM  j = (mu_L - mu_R) eps J^2 / (4 J^2 + eps^2)   [N-independent]\n")
    worst = 0.0; worst_at = ""
    n = 0
    for N in Ns, J in Js, e in drives, (mL, mR) in mus
        C = xx_boundary_ness(N; J = J, eps_L = e, eps_R = e, mu_L = mL, mu_R = mR)
        jl = xx_boundary_current(C; J = J)[max(1, N ÷ 2)]
        jp = xx_boundary_current_published(; J = J, eps = e, mu_L = mL, mu_R = mR)
        d = abs(jl - jp); n += 1
        if d > worst
            worst = d
            worst_at = @sprintf("N=%d J=%g eps=%g mu=(%g,%g)  lyap=%.12f  formula=%.12f",
                                N, J, e, mL, mR, jl, jp)
        end
    end
    @printf("  %d parameter points swept (N x J x eps x bias)\n", n)
    @printf("  worst |lyapunov - closed form| = %.3e\n     at %s\n", worst, worst_at)
    ok = worst < tol
    @printf("%s\n", ok ? "CLOSED FORM CONFIRMED -- exact at every N, J, eps and bias" :
                         "CLOSED FORM WRONG -- do not quote it")
    return ok
end

if abspath(PROGRAM_FILE) == @__FILE__
    ok = certify()
    ok &= ballistic()
    ok &= closed_form()
    zeno()
    println(ok ? "\nREFERENCE READY" : "\nREFERENCE NOT READY")
end
