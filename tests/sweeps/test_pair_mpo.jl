# `pair_mpo`: an exact MPO for an arbitrary two-body coupling, INCLUDING charged operators.
#
# THE VALIDATION CHAIN, and the order is the whole design of this file. Each step's reference is
# something already trusted, so a failure localises instead of leaving three suspects:
#
#   A  distance 1  == `mpo_from_terms`      pins the AUTOMATON LAYOUT against the already
#                                           validated builder. Distance 1 transports nothing, so
#                                           this isolates layout from transport.
#   B  distance 2  == a dense pair sum      the first case that TRANSPORTS anything, hence the
#                                           CHARGED TRANSPORT test: under `:none` the operators
#                                           are rank 2 and no charged channel exists at all,
#                                           under `:U1`/`:SU2` they are rank 3 and it does.
#   C  Haldane-Shastry == a dense HS sum     all pairs at once, and the model the whole exercise
#                                           is for.
#   D  across symmetries                     the same operator on the same physical state must
#                                           give the same energy in every mode. Only the VIRTUAL
#                                           DIMENSION may differ.
#
# `dense_state` needs `product_state`, which SU(2) has no form of, so the SU(2) arm is compared
# against the U(1) NUMBER rather than a dense vector. That is not a weaker test: U(1) is itself
# pinned to dense a few lines above, in the same run.

using Test
using LinearAlgebra
import Random

@testset "pair_mpo" begin

    "A common test state with weight on every bond, and the SAME physical state in every mode.
    The bare dimer product has zero weight on the even bonds, which would make an energy
    comparison blind to exactly the terms a wrong transport block corrupts."
    function test_state(L)
        psi = dimer_state(L)
        h = symmetry_mode() === :SU2 ? heisenberg_su2_chain(L) : xxz_chain(L; delta = 1.0)
        for _ in 1:3
            tdvp2_step!(psi, h, -0.1im; maxdim = 64, trunc_thresh = 1e-14)
            psi[psi.center] = to_concrete((1.0 / norm(psi)) * psi[psi.center])
        end
        return psi
    end

    @testset "input validation" begin
        set_symmetry!(:U1)
        L = 4
        v = spin_vertices(L)
        @test_throws ArgumentError pair_mpo(1, zeros(1, 1), v)
        @test_throws DimensionMismatch pair_mpo(L, zeros(3, 3), v)
        @test_throws ArgumentError pair_mpo(L, zeros(L, L), PairVertex[])
        # A vertex whose halves do not both carry an op-leg cannot be joined at all, and it is
        # refused rather than silently producing a rank mismatch deep inside `oplus`.
        q = local_space(:U1)
        @test_throws ArgumentError pair_mpo(L, nn_coupling_matrix(L),
                                            [PairVertex(q.Sp, q.Sz, 1.0)])
    end

    @testset "A. distance 1 == mpo_from_terms" begin
        for sym in (:none, :U1, :SU2)
            set_symmetry!(sym)
            L = 6
            psi = test_state(L)
            h = sym === :SU2 ? heisenberg_su2_chain(L) : xxz_chain(L; delta = 1.0)
            ref = real(mpo_energy(psi, RSVDCBEBondUpdate.mpo_from_terms(h)))
            got = real(mpo_energy(psi, heisenberg_chain_mpo(L)))
            @test got ≈ ref atol = 1e-10
            # The coupling matrix is read from the STRICT UPPER TRIANGLE only, so handing it the
            # symmetric matrix must not double count.
            Jsym = nn_coupling_matrix(L) .+ transpose(nn_coupling_matrix(L))
            @test real(mpo_energy(psi, pair_mpo(L, Jsym, spin_vertices(L)))) ≈ ref atol = 1e-10
        end
    end

    @testset "B. distance 2 == dense pair sum (charged transport)" begin
        L = 6
        ref_u1 = NaN
        for sym in (:none, :U1, :SU2)
            set_symmetry!(sym)
            psi = test_state(L)
            J = distance_coupling_matrix(L, 2)
            got = real(mpo_energy(psi, pair_mpo(L, J, spin_vertices(L))))
            if sym === :SU2
                @test got ≈ ref_u1 atol = 1e-10
            else
                v = dense_state(psi)
                want = real(dot(v, dense_long_range_pairs(L, J) * v))
                @test got ≈ want atol = 1e-10
                sym === :U1 && (ref_u1 = got)
            end
        end
    end

    @testset "B2. every distance, one at a time" begin
        # A single wrong sign or a channel that dies one site early shows up at ONE distance and
        # nowhere else, so sweeping `r` is what turns "it works" into "it works everywhere".
        set_symmetry!(:U1)
        L = 6
        psi = test_state(L)
        v = dense_state(psi)
        for r in 1:(L - 1)
            J = distance_coupling_matrix(L, r)
            got = real(mpo_energy(psi, pair_mpo(L, J, spin_vertices(L))))
            want = real(dot(v, dense_long_range_pairs(L, J) * v))
            @test got ≈ want atol = 1e-10
        end
    end

    @testset "B3. DISCONNECTED coupling graphs" begin
        # REGRESSION. `id -> id` used to be guarded by `!isempty(cur)` -- "is any channel open on
        # this link" -- which is not the question. A term opening at a LATER site still needs the
        # `id` channel to reach it, so on a coupling graph that is disconnected across some
        # interior bond the guard severed the automaton, and because column 1 of the matrix was
        # then entirely `nothing`, `oplus` threw
        #   "cannot infer zero TLArray at (1, 1): missing space information on leg 4"
        # (leg 1 when row 1 emptied as well -- one cause, two messages).
        #
        # Every connected coupling in this file -- nearest neighbour, each single distance,
        # Haldane-Shastry, the ring -- has `alive(i)` non-empty at every interior link, which is
        # why the whole suite passed with the bug in place. These are the cases that do not.
        #
        # Comparing against the dense pair sum rather than merely asserting it BUILDS is the
        # point: silencing the throw with a zero block would also "build".
        set_symmetry!(:U1)
        L = 6
        psi = test_state(L)
        v = dense_state(psi)

        Jsplit = zeros(L, L)                       # 1:3 and 4:6 decoupled -- link 3 is dead
        for (o, j) in ((1, 2), (2, 3), (4, 5), (5, 6))
            Jsplit[o, j] = 1.0
        end
        Jfirst = zeros(L, L); Jfirst[1, 2]     = 1.0    # links 2..5 dead
        Jlast  = zeros(L, L); Jlast[L - 1, L]  = 1.0    # links 1..4 dead
        Jfar   = zeros(L, L); Jfar[1, 2]       = 1.0    # both ends live, the middle is not
        Jfar[L - 1, L] = 1.0

        for (nm, J) in (("split 1:3 | 4:6", Jsplit), ("only J[1,2]", Jfirst),
                        ("only J[L-1,L]", Jlast), ("both ends, dead middle", Jfar),
                        ("all-zero J (the zero Hamiltonian)", zeros(L, L)))
            m = pair_mpo(L, J, spin_vertices(L))
            got = real(mpo_energy(psi, m))
            want = real(dot(v, dense_long_range_pairs(L, J) * v))
            @test got ≈ want atol = 1e-10
            @info "$nm: E = $got, virtual dims = $(mpo_virtual_dims(m))"
        end
    end

    @testset "C. Haldane-Shastry and the Heisenberg ring == dense" begin
        for sym in (:none, :U1)
            set_symmetry!(sym)
            L = 6
            psi = test_state(L)
            v = dense_state(psi)
            for (nm, m, J) in (("HS", haldane_shastry_mpo(L), haldane_shastry_couplings(L)),
                               ("ring", heisenberg_ring_mpo(L), heisenberg_ring_couplings(L)))
                got = real(mpo_energy(psi, m))
                want = real(dot(v, dense_long_range_pairs(L, J) * v))
                @test got ≈ want atol = 1e-10
            end
        end
    end

    @testset "D. same physics across symmetries, different virtual dimension" begin
        L = 6
        for (nm, build) in (("HS", haldane_shastry_mpo), ("ring", heisenberg_ring_mpo))
            es = Float64[]; ws = Int[]
            for sym in (:none, :U1, :SU2)
                set_symmetry!(sym)
                m = build(L)
                push!(es, real(mpo_energy(test_state(L), m)))
                push!(ws, maximum(mpo_virtual_dims(m)))
            end
            # THE PHYSICS IS THE SAME.
            @test es[1] ≈ es[2] atol = 1e-10
            @test es[1] ≈ es[3] atol = 1e-10
            # THE COST IS NOT. SU(2) needs ONE vertex for `S.S` where U(1) needs three, so its
            # virtual dimension is smaller -- that is the entire argument for using it, and it is
            # asserted rather than assumed.
            @test ws[3] < ws[2]
            @info "$nm virtual dims (:none, :U1, :SU2) = $ws"
        end
    end

    @testset "the virtual dimension grows with L, and that is exactness' price" begin
        # `long_range_mpo`'s geometric tail is `w = const` in L because a self-loop carries every
        # distance. An ARBITRARY coupling has no such structure -- the exact MPO's bond dimension
        # is the rank of the coupling's Hankel matrix, full for `1/sin^2` -- so `w ~ L` here is
        # not a wasteful encoding of something cheaper. This test records the scaling so a future
        # change that appears to improve it is understood as a change of MODEL, not of code.
        set_symmetry!(:SU2)
        ws = [maximum(mpo_virtual_dims(haldane_shastry_mpo(L))) for L in (4, 6, 8)]
        @test issorted(ws)
        @test ws[end] > ws[1]
    end
end
