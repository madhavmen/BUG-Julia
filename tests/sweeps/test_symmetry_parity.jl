# THE PHYSICS MUST BE THE SAME UNDER :none, :U1 AND :SU2.
#
# Only the MPO structure and the cost may differ. This file is that requirement, made into
# assertions, and it is the gate on every physics claim elsewhere in the suite.
#
# WHY THE OBSERVABLES HERE ARE NOT THE ENERGY. Real-time evolution CONSERVES the energy, so an
# energy trajectory agreeing across symmetry modes proves almost nothing -- three broken
# integrators that each conserve energy would agree too. The observables used are therefore ones
# that genuinely MOVE:
#
#   * the AUTOCORRELATION `|<psi(0)|psi(t)>|`, which decays from 1 and is the most sensitive
#     single number available -- it sees the whole state, not a local operator;
#   * the BOND-RESOLVED energy `<S_i . S_{i+1}>(t)`, one bond at a time, which redistributes
#     under the dynamics even though the total does not. Built with `pair_mpo` and a coupling
#     matrix carrying a single entry, so it needs no new machinery and is symmetry-native;
#   * the VARIATIONAL GROUND ENERGY, which unlike a conserved quantity is a non-trivial number
#     that a wrong operator or a wrong Clebsch-Gordan factor would move.
#
# WHY EVERY RUN HERE IS FULL RANK. `maxdim` and `Nkeep` count MULTIPLETS (see `multiplets.jl`),
# so `maxdim = 16` is a DIFFERENT truncation under `:SU2` than under `:U1` -- the SU(2) one is
# more generous. A truncated cross-symmetry comparison compares two different approximations and
# would fail for a legitimate reason, which would then mask an illegitimate one. At full rank
# there is no truncation to differ, so any disagreement is a defect.
#
# The common initial state is `dimer_state(L)`, the product of nearest-neighbour singlets. It is
# the only interesting state expressible in all three modes: under SU(2) every representable state
# has definite total spin, so Neel and domain-wall states do not exist at any bond dimension.

using Test
using LinearAlgebra
import Random

@testset "symmetry parity" begin

    nn_chain(sym, L) = sym === :SU2 ? heisenberg_su2_chain(L) : xxz_chain(L; delta = 1.0)
    renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)

    "`<S_i . S_{i+1}>` as a single-entry `pair_mpo`, so it is one operator in every mode."
    function bond_energy(psi, i)
        L = length(psi)
        J = zeros(Float64, L, L); J[i, i + 1] = 1.0
        return real(mpo_energy(psi, pair_mpo(L, J, spin_vertices(L))))
    end

    @testset "multiplet vs state accounting" begin
        # The one real difference, pinned so it is documented behaviour rather than a surprise.
        L = 6
        ms = Int[]; ss = Int[]
        for sym in (:U1, :SU2)
            set_symmetry!(sym)
            psi = dimer_state(L)
            for _ in 1:3
                tdvp2_step!(psi, nn_chain(sym, L), -0.1im; maxdim = 64,
                            trunc_thresh = 1e-14)
                renorm!(psi)
            end
            push!(ms, maximum(bond_dims(psi)))
            push!(ss, maximum(state_bond_dims(psi)))
        end
        # Under U(1) the two agree -- every irrep is one dimensional.
        @test ms[1] == ss[1]
        # Under SU(2) the multiplet count is SMALLER and the STATE count is the same physics.
        @test ms[2] < ss[2]
        @test ss[1] == ss[2]
        set_symmetry!(:U1); p = dimer_state(L)
        @test multiplet_ratio(p) ≈ 1.0 atol = 1e-12
        @info "L=6 warmed: multiplets (:U1, :SU2) = $ms, states = $ss"
    end

    @testset "real time: autocorrelation and bond-resolved energy agree" begin
        L = 6
        for (nm, step!) in (("tdvp2", (p, m, t) -> tdvp2_step!(p, m, t; maxdim = 64,
                                                               trunc_thresh = 1e-14)),
                            ("tdvp_cbe1s", (p, m, t) -> tdvp_cbe1s_step!(p, m, t; maxdim = 64,
                                                                        trunc_thresh = 1e-14)),
                            ("cbe_bug", (p, m, t) -> cbe_bug_step!(p, m, t; maxdim = 64,
                                                                   trunc_thresh = 1e-14)))
            acs = Vector{Vector{Float64}}()
            bes = Vector{Vector{Float64}}()
            for sym in (:none, :U1, :SU2)
                set_symmetry!(sym)
                # `tdvp2_step!` takes a term list, the other two an MPO; give each what it wants
                # and build both from the same constructor so they cannot disagree about `J`.
                h = nn_chain(sym, L)
                mpo = RSVDCBEBondUpdate.mpo_from_terms(h)
                arg = nm == "tdvp2" ? h : mpo
                psi = dimer_state(L)
                psi0 = copy(psi); psi0.tensors .= copy.(psi.tensors)
                ac = Float64[]; be = Float64[]
                for _ in 1:6
                    step!(psi, arg, -0.05im)
                    renorm!(psi)
                    push!(ac, abs(overlap(psi0, psi)))
                    push!(be, bond_energy(psi, 2))
                end
                push!(acs, ac); push!(bes, be)
            end
            # The observables must have MOVED, or the test proves nothing.
            @test maximum(abs.(acs[1] .- 1.0)) > 1e-3
            @test maximum(abs.(bes[1] .- bes[1][1])) > 1e-4
            for k in 2:3
                @test maximum(abs.(acs[k] .- acs[1])) < 1e-8
                @test maximum(abs.(bes[k] .- bes[1])) < 1e-8
            end
            @info "$nm parity: max |dAC| = $(maximum(abs.(acs[3] .- acs[1]))), " *
                  "max |dE_bond| = $(maximum(abs.(bes[3] .- bes[1])))"
        end
    end

    @testset "ground state: the same variational energy" begin
        # A conserved quantity agreeing proves little; a VARIATIONAL MINIMUM agreeing is a real
        # cross-check, because it is a non-trivial number that depends on the whole operator.
        L = 6
        es = Float64[]
        for sym in (:none, :U1, :SU2)
            set_symmetry!(sym)
            mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(sym, L))
            psi = dimer_state(L)
            r = dmrg_cbe1s!(psi, mpo; n_sweeps = 20, maxdim = 64, trunc_thresh = 1e-14)
            push!(es, r.energies[end])
        end
        @test es[2] ≈ es[1] atol = 1e-9
        @test es[3] ≈ es[1] atol = 1e-9
        @info "ground energies (:none, :U1, :SU2) = $es"
    end

    @testset "imaginary time: the same trajectory, not just the same limit" begin
        # Comparing only the converged energy would let two integrators with different transients
        # pass. The whole trajectory is compared, which is the stronger statement.
        L = 6
        trajs = Vector{Vector{Float64}}()
        for sym in (:none, :U1, :SU2)
            set_symmetry!(sym)
            mpo = RSVDCBEBondUpdate.mpo_from_terms(nn_chain(sym, L))
            psi = dimer_state(L)
            t = Float64[]
            for _ in 1:20
                cbe_bug_step!(psi, mpo, -0.05; maxdim = 64, trunc_thresh = 1e-14)
                renorm!(psi)
                push!(t, real(mpo_energy(psi, mpo)))
            end
            push!(trajs, t)
        end
        @test trajs[1][end] < trajs[1][1]                # it actually descended
        for k in 2:3
            @test maximum(abs.(trajs[k] .- trajs[1])) < 1e-8
        end
    end

    @testset "Haldane-Shastry: the same physics, a smaller MPO" begin
        L = 6
        es = Float64[]; ws = Int[]; acs = Vector{Vector{Float64}}()
        for sym in (:none, :U1, :SU2)
            set_symmetry!(sym)
            m = haldane_shastry_mpo(L)
            push!(ws, maximum(mpo_virtual_dims(m)))
            psi = dimer_state(L)
            psi0 = copy(psi); psi0.tensors .= copy.(psi.tensors)
            ac = Float64[]
            for _ in 1:6
                cbe_bug_step!(psi, m, -0.05im; maxdim = 64, trunc_thresh = 1e-14)
                renorm!(psi)
                push!(ac, abs(overlap(psi0, psi)))
            end
            push!(acs, ac)
            push!(es, real(mpo_energy(psi, m)))
        end
        @test es[2] ≈ es[1] atol = 1e-9
        @test es[3] ≈ es[1] atol = 1e-9
        for k in 2:3
            @test maximum(abs.(acs[k] .- acs[1])) < 1e-8
        end
        # The SU(2) MPO is genuinely smaller: ONE `S.S` vertex against U(1)'s three.
        @test ws[3] < ws[2]
        @info "HS virtual dims (:none, :U1, :SU2) = $ws"
    end
end
