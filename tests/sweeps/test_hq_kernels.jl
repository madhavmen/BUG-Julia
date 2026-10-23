# tests/sweeps/test_hq_kernels.jl -- updated_cbe_high_order_parallel.md Task 9, U20: the DD-QR
# kernels against the bond dimension D at FIXED sketch width, on the centre bond of the spin-1/2
# XX chain, L = 16, grown from the domain wall by TDVP2 capped at D (P06's fixture at s = 1/2).
#
# The gates are on Telum's contraction flop counter (`_flops`), which is exact and
# machine-independent. Wall time and allocated bytes are recorded per kernel, not gated: a shared
# node's time does not reproduce, and every TLArray's block metadata is a width-independent
# allocation, so bytes cannot tell a dense object from bookkeeping at these D. The timing
# regression is P06 (`benchmarks/hq_dd_scaling.jl`: one job, one node).
#
#   1. at every D, input DD, stream and output DD are EXACTLY linear in the probe width: none
#      contracts anything without the probe leg, so no X, X_mu, G_DD or dense P^D is formed;
#   2. against one environment push (cubic in D on the same sectors), each thin kernel's flops
#      fall as D grows at fixed w (thin ~ w D^2): a dense projector or X would keep pace with it;
#   3. one root application costs fewer flops than a dense root matvec on the same core.

using Test
using LinearAlgebra
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
import Random

include(joinpath(@__DIR__, "..", "common", "adaptive_fixtures.jl"))

"`(seconds, bytes)` of `g()` after one warm-up."
_tb(g) = (g(); (_, t, b) = RC._measure(g); (t, b))

@testset "U20: DD kernels against D at fixed sketch width (s = 1/2, L = 16)" begin
    L, twoS, w, dt = 16, 1, 8, 0.1
    set_symmetry!(:U1)
    mpo = spin_xx_mpo(L, twoS)
    i = L ÷ 2
    psi = spin_domain_wall_state(L, twoS)
    nstep = 0
    rows = Dict{Int, Any}()
    for D in (16, 32, 64)
        while bond_dims(psi)[i] < D && nstep < 400
            tdvp2_step!(psi, mpo, ComplexF64(-1im * dt); maxdim = D, trunc_thresh = 1e-14)
            nstep += 1
        end
        canonical!(psi, i)
        f = bond_frame(psi, i)
        lenv, renv = _envs(psi, mpo, i, L)
        @test bond_dims(psi)[i] == D
        @test leg_dim(f.U0, 1) * leg_dim(f.U0, 2) > f.old_rank   # a discarded space on both sides
        @test leg_dim(f.V0, 2) * leg_dim(f.V0, 3) > f.old_rank
        per = Dict{Symbol, Any}()
        for side in (:left, :right)
            left = side === :left
            fl = _dd_kernel_flops(f, mpo, i, lenv, renv, side, w; rng = Random.MersenneTwister(1))
            for (k, (f1, f2)) in pairs(fl)
                @test f1 > 0
                @test f2 == 2 * f1
            end
            env() = left ? push_left_channels(lenv, mpo, f.U0, i) :
                           push_right_channels(renv, mpo, f.V0, i + 1)
            # the record: seconds and bytes of every kernel of this side, including QR and the
            # sector bookkeeping of a probe draw
            frame, opp = left ? (f.U0, f.V0) : (f.V0, f.U0)
            oside = left ? :right : :left
            a, b = left ? (2, 3) : (1, 2)
            Om = RC._dd_probe(opp, oside, w; rng = Random.MersenneTwister(1))
            out = Ref{Any}(nothing)
            op = TwoSiteSketchOp(f, mpo, i, lenv, renv; adjoint = !left)
            apply_two_site_sketch!(out, Om, op)
            Z = out[]
            per[side] = (flops = merge(map(first, fl), (env = _flops(env),)),
                         input_dd = _tb(() -> RC._dd_probe(opp, oside, w;
                                                          rng = Random.MersenneTwister(1))),
                         stream = _tb(() -> apply_two_site_sketch!(Ref{Any}(nothing), Om, op)),
                         dd_select = _tb(() -> RC._dd_select(frame, Z, side, KernelCounters())),
                         sectors = _tb(() -> RC._frame_sector_dims(opp, oside,
                                           fusion_basis(opp, a, b; tag = "cbeG"), "U20")),
                         env = _tb(env))
        end
        H0 = zero_site_h(mpo, i, lenv, renv, f.U0, f.V0)
        S = apply_zero_site(H0, f.S0)                       # a generic core in the root's sectors
        N = tensor_elements(S)
        froot = _flops(() -> apply_zero_site(H0, S))
        # a dense root matvec is N^2; the matrix-free one ~ 2 w_mpo D^3 / sectors, below it once
        # D > 2 w_mpo (at D = 16 the two are comparable)
        D >= 32 && @test froot < N^2
        trunc = _tb(() -> RC.truncate_recursive!(copy(psi), i; maxdim = D, cutoff = 1e-12))
        rows[D] = (; per, froot, N, root = _tb(() -> apply_zero_site(H0, S)), trunc)
        @info "U20 D = $D (TDVP2 steps $nstep)" per[:left] per[:right] froot N2 = N^2 root = rows[D].root trunc
    end
    # 2. thin kernels against the cubic environment push, D = 16 -> 64 at fixed w: theory w / D
    # per sector, i.e. 1/4 at fixed sectors and ~1/2 as the sector count grows with D
    for side in (:left, :right), k in (:input_dd, :stream, :output_dd)
        ratio(D) = rows[D].per[side].flops[k] / rows[D].per[side].flops.env
        @info "U20 $side $k: flops / env push" r16 = ratio(16) r32 = ratio(32) r64 = ratio(64)
        @test ratio(64) <= 0.75 * ratio(16)
    end
end
