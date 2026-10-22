# benchmarks/hq_dd_scaling.jl -- HQ plan Task 8, P06: the DD-QR kernels separately, against the
# bond dimension D at FIXED sketch width w, on the centre bond of the spin-3 L = 16 XX chain.
#
# For each D the state is grown by TDVP2 (capped at D; its accuracy is irrelevant, only a generic
# bond of rank D is needed) and the PRODUCTION kernels of `dd_cbe_expand` are timed one by one,
# both sides (left = the K-sweep probe in V0's layout, right = its mirror):
#
#   input DD    `_dd_probe`: P^D on the thin input probe (kc.t_input_dd)
#   stream      `apply_two_site_sketch!`: X (P^D omega), MPO channels inside (kc.t_stream)
#   output DD   `_dd_select`'s P^D on the thin result (kc.t_output_dd)
#   QR          `_dd_select`'s two-pass thin QR (kc.t_qr)
#   sectors     `fusion_basis` + `_frame_sector_dims`: the sector bookkeeping of a probe draw
#   env         `push_left_channels` / `push_right_channels` through the bond's frame
#
# and the one-sided+QR ablation (no input projector: `sector_graded_sketch`, then the same stream,
# output DD and QR). Each is the median of HQS_REPS repetitions, with allocated bytes. The local
# exponent is gamma(D) = log2[T(2D)/T(D)]: the plan's question is whether a projector secretly
# makes the sketch kernel cubic in D (it should be ~2 at fixed w).
#
# Env: HQS_TWOS (6), HQS_L (16), HQS_DS ("32,64,128,256,512"), HQS_WS ("8,32"), HQS_REPS (5),
#      HQS_DT (0.1), HQS_OUT (hq_dd_scaling.json)

using LinearAlgebra, Printf, JSON, Random, Statistics
using LurCGT, Telum
using BUGJulia, BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "adaptive_fixtures.jl"))   # RC, _envs

const TWOS = parse(Int, get(ENV, "HQS_TWOS", "6"))
const L = parse(Int, get(ENV, "HQS_L", "16"))
const DS = parse.(Int, split(get(ENV, "HQS_DS", "32,64,128,256,512"), ","))
const WS = parse.(Int, split(get(ENV, "HQS_WS", "8,32"), ","))
const REPS = parse(Int, get(ENV, "HQS_REPS", "5"))
const DT = parse(Float64, get(ENV, "HQS_DT", "0.1"))
const OUT = get(ENV, "HQS_OUT", "hq_dd_scaling.json")

set_symmetry!(:U1)
const MPO_ = spin_xx_mpo(L, TWOS)
const I0 = L ÷ 2

"Median seconds and bytes of `f()` over `REPS` runs (after one warm-up)."
function timed(f)
    f()
    ts, bs = Float64[], Int[]
    for _ in 1:REPS
        _, t, b = RC._measure(f)
        push!(ts, t); push!(bs, b)
    end
    return median(ts), round(Int, median(bs))
end

"The `KernelCounters` field `fld` per call of `g(kc)`, median over `REPS` (after a warm-up)."
function counted(g, fld)
    g(KernelCounters())
    ts, bs = Float64[], Int[]
    bfld = Symbol(replace(string(fld), "t_" => "b_"))
    for _ in 1:REPS
        kc = KernelCounters()
        g(kc)
        push!(ts, getfield(kc, fld))
        hasfield(KernelCounters, bfld) && push!(bs, getfield(kc, bfld))
    end
    return median(ts), isempty(bs) ? 0 : round(Int, median(bs))
end

rows = Any[]
psi = spin_domain_wall_state(L, TWOS)
nstep = 0
println("HQS-CONFIG twoS=$TWOS L=$L Ds=$DS ws=$WS reps=$REPS dt=$DT threads=$(Threads.nthreads()) " *
        "blas=$(BLAS.get_num_threads())")
for D in DS
    global nstep
    while bond_dims(psi)[I0] < D && nstep < 400
        tdvp2_step!(psi, MPO_, ComplexF64(-1im * DT); maxdim = D, trunc_thresh = 1e-14)
        nstep += 1
    end
    canonical!(psi, I0)
    f = bond_frame(psi, I0)
    lenv, renv = _envs(psi, MPO_, I0, L)
    Dact = bond_dims(psi)[I0]
    @printf("HQS-STATE D=%d actual=%d after %d TDVP2 steps (t=%.1f) bonds=%s\n", D, Dact, nstep,
            nstep * DT, bond_dims(psi))
    for w in WS, side in (:left, :right)
        left = side === :left
        frame, opp = left ? (f.U0, f.V0) : (f.V0, f.U0)
        oside = left ? :right : :left
        rng = MersenneTwister(1)
        op(kc) = TwoSiteSketchOp(f, MPO_, I0, lenv, renv; adjoint = !left, counters = kc)
        Om = RC._dd_probe(opp, oside, w; rng = rng)
        Om === nothing && (println("HQS-SKIP D=$D side=$side: no discarded space"); continue)
        out = Ref{Any}(nothing)
        apply_two_site_sketch!(out, Om, op(KernelCounters()))
        Z = out[]
        a, b = oside === :left ? (1, 2) : (2, 3)
        r = Dict{String, Any}("D" => D, "D_actual" => Dact, "w" => w, "side" => side,
                              "cols" => leg_dim(Om, left ? 1 : 3))
        r["input_dd"] = counted(kc -> RC._dd_probe(opp, oside, w; rng = MersenneTwister(1), kc = kc), :t_input_dd)
        r["stream"] = counted(kc -> apply_two_site_sketch!(Ref{Any}(nothing), Om, op(kc)), :t_stream)
        r["output_dd"] = counted(kc -> RC._dd_select(frame, Z, side, kc), :t_output_dd)
        r["qr"] = counted(kc -> RC._dd_select(frame, Z, side, kc), :t_qr)
        r["sectors"] = timed(() -> RC._frame_sector_dims(opp, oside, fusion_basis(opp, a, b; tag = "cbeG"), "P06"))
        r["env"] = left ? timed(() -> push_left_channels(lenv, MPO_, f.U0, I0)) :
                          timed(() -> push_right_channels(renv, MPO_, f.V0, I0 + 1))
        # one-sided+QR: the opposite-leg projector dropped, the rest unchanged
        Om1 = sector_graded_sketch(opp, oside, w; comp_ratio = nothing, rng = MersenneTwister(1))
        r["onesided_draw"] = timed(() -> sector_graded_sketch(opp, oside, w; comp_ratio = nothing,
                                                              rng = MersenneTwister(1)))
        r["onesided_stream"] = counted(kc -> apply_two_site_sketch!(Ref{Any}(nothing), Om1, op(kc)), :t_stream)
        apply_two_site_sketch!(out, Om1, op(KernelCounters()))
        Z1 = out[]
        r["onesided_output_dd"] = counted(kc -> RC._dd_select(frame, Z1, side, kc), :t_output_dd)
        r["onesided_qr"] = counted(kc -> RC._dd_select(frame, Z1, side, kc), :t_qr)
        push!(rows, r)
        @printf("HQS D=%d w=%d %-5s cols=%d  in_dd=%.2e stream=%.2e out_dd=%.2e qr=%.2e sect=%.2e env=%.2e | 1s: draw=%.2e stream=%.2e out=%.2e qr=%.2e  (s); stream bytes=%d\n",
                Dact, w, side, r["cols"], r["input_dd"][1], r["stream"][1], r["output_dd"][1], r["qr"][1],
                r["sectors"][1], r["env"][1], r["onesided_draw"][1], r["onesided_stream"][1],
                r["onesided_output_dd"][1], r["onesided_qr"][1], r["stream"][2])
        flush(stdout)
        open(OUT, "w") do io
            JSON.json(io, Dict("config" => Dict("twoS" => TWOS, "L" => L, "bond" => I0, "reps" => REPS,
                                                "threads" => Threads.nthreads(),
                                                "blas_threads" => BLAS.get_num_threads()),
                               "rows" => rows); allownan = true)
        end
    end
end

# gamma(D) = log2[T(2D)/T(D)] per kernel, side and width
for w in WS, side in (:left, :right), k in ("input_dd", "stream", "output_dd", "qr", "env", "onesided_stream")
    rs = sort([r for r in rows if r["w"] == w && r["side"] == side]; by = r -> r["D_actual"])
    g = [log2(rs[n + 1][k][1] / rs[n][k][1]) / log2(rs[n + 1]["D_actual"] / rs[n]["D_actual"])
         for n in 1:(length(rs) - 1)]
    @printf("HQS-GAMMA w=%d %-5s %-16s %s\n", w, side, k, join((@sprintf("%.2f", x) for x in g), " "))
end
println("HQS-DONE $OUT")
