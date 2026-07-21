using Test, LinearAlgebra, LurCGT, Telum
using BUGJulia.BondUpdateBUG

# <bra|G|ket> in the Sz product basis, via a two-site MPS so the gate meets the
# same leg layout it meets inside the KLS step.
#
# Returns `nothing` for a pair of different total Sz. Such an element is not
# "zero" in any representable sense: `product_state` trims its right boundary
# link to the total charge it actually carries, so bra and ket of different
# charge do not even share a leg SPACE and no contraction between them exists.
# That is U(1) forbidding the element structurally, which is a stronger
# statement than a numerical zero -- so the tests assert it as such rather than
# swallowing the exception.
charge(spins) = sum(s === :up ? 1 : -1 for s in spins)

function melem(G, bra::Vector{Symbol}, ket::Vector{Symbol})
    charge(bra) == charge(ket) || return nothing
    pk = product_state(ket)
    pb = product_state(bra)
    thk = to_concrete(pk[1] * pk[2])
    thb = to_concrete(pb[1] * pb[2])
    return tensor_inner(thb, apply_gate(G, thk, "S,1", "S,2"))
end

const BASIS = [[:up, :up], [:up, :down], [:down, :up], [:down, :down]]

@testset "bond gates" begin

    # The whole point of this file. Telum's Sp carries a 1/sqrt(2), so
    # Sp (x) Sp' is already (1/2) S^-S^+ and the XY term must NOT be halved
    # again. Every element of the 4x4 block is pinned so that convention -- and
    # any future change to Telum's operator normalisation -- cannot slip past.
    @testset "Heisenberg gate matches the analytic 4x4 block" begin
        G = heisenberg_bond_gate("S,1", "S,2"; J = 1.0, delta = 1.0)
        # H = 1/2 (S+S- + S-S+) + Sz Sz, rows = bra, cols = ket.
        want = [ 0.25  0.0   0.0   0.0
                 0.0  -0.25  0.5   0.0
                 0.0   0.5  -0.25  0.0
                 0.0   0.0   0.0   0.25]
        for (bi, b) in enumerate(BASIS), (ki, k) in enumerate(BASIS)
            v = melem(G, b, k)
            if v === nothing
                @test want[bi, ki] == 0.0      # forbidden, and rightly so
            else
                @test isapprox(v, want[bi, ki] + 0.0im; atol = 1e-13)
            end
        end
    end

    @testset "U(1) forbids every charge-changing element structurally" begin
        # not merely small -- there is no shared leg space to contract at all
        for b in BASIS, k in BASIS
            if charge(b) != charge(k)
                @test melem(heisenberg_bond_gate("S,1", "S,2"), b, k) === nothing
            end
        end
    end

    @testset "the flip amplitude is 1/2, not 1/4 or 1" begin
        G = heisenberg_bond_gate("S,1", "S,2")
        @test isapprox(real(melem(G, [:down, :up], [:up, :down])), 0.5; atol = 1e-13)
    end

    @testset "XX gate drops SzSz and keeps the hop" begin
        G = xx_bond_gate("S,1", "S,2")
        @test isapprox(melem(G, [:up, :up], [:up, :up]), 0.0 + 0.0im; atol = 1e-13)
        @test isapprox(melem(G, [:up, :down], [:up, :down]), 0.0 + 0.0im; atol = 1e-13)
        @test isapprox(real(melem(G, [:down, :up], [:up, :down])), 0.5; atol = 1e-13)
    end

    @testset "delta and J scale the right pieces" begin
        Gh = heisenberg_bond_gate("S,1", "S,2"; J = 1.0, delta = 1.0)
        Gd = heisenberg_bond_gate("S,1", "S,2"; J = 1.0, delta = 2.0)
        Gj = heisenberg_bond_gate("S,1", "S,2"; J = 3.0, delta = 1.0)
        # delta multiplies only the diagonal SzSz part
        @test isapprox(real(melem(Gd, [:up, :up], [:up, :up])), 0.50; atol = 1e-13)
        @test isapprox(real(melem(Gd, [:down, :up], [:up, :down])), 0.50; atol = 1e-13)
        # J multiplies everything
        for b in BASIS, k in BASIS
            v = melem(Gj, b, k)
            v === nothing && continue
            @test isapprox(v, 3.0 * melem(Gh, b, k); atol = 1e-13)
        end
    end

    @testset "the gate is Hermitian" begin
        G = heisenberg_bond_gate("S,1", "S,2"; J = 1.0, delta = 1.0)
        for b in BASIS, k in BASIS
            v = melem(G, b, k)
            v === nothing && continue
            @test isapprox(v, conj(melem(G, k, b)); atol = 1e-13)
        end
    end

    @testset "gate legs follow the (ket_l, ket_r, bra_l, bra_r) convention" begin
        G = heisenberg_bond_gate("S,1", "S,2")
        @test length(G.inds) == 4
        @test G.inds[1].dir == '+' && G.inds[2].dir == '+'   # contract theta
        @test G.inds[3].dir == '-' && G.inds[4].dir == '-'   # replace theta's legs
        @test G.inds[1].itags == G.inds[3].itags             # both are site_l
        @test G.inds[2].itags == G.inds[4].itags             # both are site_r
    end

    @testset "apply_gate preserves theta's leg structure" begin
        psi = domain_wall_state(6); canonical!(psi, 3)
        f = bond_frame(psi, 3)
        th = frame_theta(f)
        G = heisenberg_bond_gate(f.site_l, f.site_r)
        out = apply_gate(G, th, f.site_l, f.site_r)
        @test length(out.inds) == 4
        for k in 1:4
            @test out.inds[k].itags == th.inds[k].itags
            @test out.inds[k].dir == th.inds[k].dir
        end
    end

    @testset "a gate tagged for the wrong bond is rejected" begin
        psi = domain_wall_state(6); canonical!(psi, 3)
        f = bond_frame(psi, 3)
        th = frame_theta(f)
        wrong = heisenberg_bond_gate("S,1", "S,2")          # bond (1,2), not (3,4)
        @test_throws ArgumentError apply_gate(wrong, th, f.site_l, f.site_r)
        @test_throws ArgumentError heisenberg_bond_gate("S,3", "S,3")
    end

    @testset "the gate acts as the bond Hamiltonian inside a chain" begin
        # <psi|h_34|psi> on the domain wall: sites 3,4 are up/down, so the XY
        # part vanishes and SzSz gives -1/4.
        psi = domain_wall_state(6); canonical!(psi, 3)
        f = bond_frame(psi, 3)
        th = frame_theta(f)
        G = heisenberg_bond_gate(f.site_l, f.site_r)
        @test isapprox(tensor_inner(th, apply_gate(G, th, f.site_l, f.site_r)),
                       -0.25 + 0.0im; atol = 1e-13)
        # bond (2,3) is up/up: +1/4
        psi2 = domain_wall_state(6); canonical!(psi2, 2)
        f2 = bond_frame(psi2, 2)
        th2 = frame_theta(f2)
        G2 = heisenberg_bond_gate(f2.site_l, f2.site_r)
        @test isapprox(tensor_inner(th2, apply_gate(G2, th2, f2.site_l, f2.site_r)),
                       0.25 + 0.0im; atol = 1e-13)
    end
end
