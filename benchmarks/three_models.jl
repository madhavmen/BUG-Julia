# THREE MODELS, THREE INTEGRATORS, AND THE THREE-KNOB MATRIX FOR THE BUG.
#
#   A  heisenberg_su2   Heisenberg chain, L = 16, :SU2, dimer start
#   B  square_su2       Heisenberg on a 4x4 square cylinder, 16 sites, :SU2, dimer start
#   C  kagome_xy        dipolar XY on a breathing kagome (arXiv:2603.21147), 2x3 = 18 sites,
#                       :U1, Neel start
#
# EVERY MODEL HAS AN EXACT REFERENCE at these sizes -- Sz = 0 sectors of 12870 / 12870 / 48620
# states -- so `err` below is a real error against the truth and not a spread between schemes.
# That is the whole reason for staying at 16-18 sites.
#
# ⛔ THE DIPOLAR XY MODEL IS :U1 ONLY. It is not `S_i . S_j`, so it has no SU(2) symmetry and
# `spin_vertices`' SU(2) branch would silently build the WRONG operator. `breathing_kagome_mpo`
# refuses `:SU2` for exactly that reason.
#
# ⛔ NEEL IS NOT SU(2)-REPRESENTABLE, so models A and B start from `dimer_state`. A definite-Sz
# product state spans many total-spin sectors; using it under `:SU2` is not the same physics as
# using it under `:U1`, and the comparison would be meaningless.
#
# ⚠ THE OBSERVABLE IS A BOND-ENERGY PROFILE in every case, built through `pair_mpo` so it is
# literally the same operator in every symmetry mode. For a dimer start the Sz profile is
# identically zero and stays zero under SU(2)-symmetric evolution -- it would report a flawless
# 0.000e+00 for every scheme at every tolerance, a measurement that cannot fail.
#
# ⚠ COST IS REPORTED ON TWO AXES BECAUSE THEY DISAGREE. MEASURED on an 8x8 cylinder: `cbe_bug`
# used 19x FEWER operator applications and took 1.91x LONGER in wall clock than tdvp2. `krylov`
# cannot see the CBE expansion's SVD/QR work. Quote both, always.
#
# Run:  julia --project=. benchmarks/three_models.jl [phase] [model]
#         phase = evolve | matrix        model = A | B | C | all

using LinearAlgebra, Printf, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "exact_sparse.jl"))

const PHASE = length(ARGS) >= 1 ? ARGS[1] : "evolve"
const WHICH = length(ARGS) >= 2 ? ARGS[2] : "all"
const DT    = 0.05
const TMAX  = 1.0
const MI    = 16
const CAP   = 32          # binds on all three: full rank 256 / 256 / 512

const OUTDIR = joinpath(@__DIR__, "results"); mkpath(OUTDIR)

"Everything one model needs, including its exact sparse reference."
function setup(tag)
    if tag == "A"
        set_symmetry!(:SU2)
        L = 16
        Jm = [abs(i - j) == 1 ? 1.0 : 0.0 for i in 1:L, j in 1:L]
        W  = RSVDCBEBondUpdate.mpo_from_terms(heisenberg_su2_chain(L))
        psi0 = dimer_state(L)
        Hs, states, idx = heisenberg_sparse(L)
        v0 = dimer_vector(L, states, idx)
        heis = true
    elseif tag == "B"
        set_symmetry!(:SU2)
        L = 16
        Jm = square_cylinder_couplings(4, 4)
        W  = square_cylinder_mpo(4, 4)
        psi0 = dimer_state(L)
        Hs, states, idx = pairs_sparse(L, Jm)
        v0 = dimer_vector(L, states, idx)
        heis = true
    else
        set_symmetry!(:U1)
        L = 18
        Jm = breathing_kagome_couplings(2, 3; h = 0.30)
        W  = breathing_kagome_mpo(2, 3; h = 0.30)
        psi0 = neel_state(L)
        Hs, states, idx = _xy_sparse(L, Jm)
        v0 = zeros(ComplexF64, length(states))
        v0[idx[sum(1 << (2j - 1) for j in 1:(L ÷ 2))]] = 1.0
        heis = false
    end
    bonds = [(i, j) for i in 1:L for j in (i + 1):L if Jm[i, j] != 0.0]
    # keep the observable cheap: at most 24 representative bonds, operators built ONCE
    probe = bonds[1:max(1, length(bonds) ÷ 24):end]
    ops = [begin
               A = zeros(Float64, L, L); A[i, j] = 1.0
               heis ? pair_mpo(L, A, spin_vertices(L)) :
                      pair_mpo(L, A, xxz_chain(L; J = 1.0, delta = 0.0))
           end for (i, j) in probe]
    return (tag = tag, L = L, W = W, psi0 = psi0, ops = ops, probe = probe,
            Hs = Hs, states = states, idx = idx, v0 = v0, heis = heis)
end

"Sparse XY Hamiltonian `sum J_ij (SxSx + SySy)` in the Sz=0 sector -- no diagonal at all."
function _xy_sparse(L, Jm)
    states = [b for b in 0:(2^L - 1) if count_ones(b) == L ÷ 2]
    idx = Dict(s => k for (k, s) in enumerate(states))
    I, Jd, V = Int[], Int[], Float64[]
    bit(b, j) = (b >> (j - 1)) & 1
    for (k, b) in enumerate(states), i in 1:L, j in (i + 1):L
        Jm[i, j] == 0.0 && continue
        bit(b, i) == bit(b, j) && continue
        bb = b ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
        push!(I, idx[bb]); push!(Jd, k); push!(V, Jm[i, j] / 2)
    end
    return sparse(I, Jd, V, length(states), length(states)), states, idx
end

profile(S, psi) = (n2 = max(norm(copy(psi))^2, eps());
                   [real(mpo_energy(copy(psi), op)) / n2 for op in S.ops])

function dense_profile(S, v)
    bit(b, j) = (b >> (j - 1)) & 1
    nrm = real(dot(v, v)); out = zeros(Float64, length(S.probe))
    for (n, (i, j)) in pairs(S.probe)
        acc = 0.0 + 0im
        for (k, b) in enumerate(S.states)
            si, sj = bit(b, i), bit(b, j)
            # the XY model has NO S^z S^z term; Heisenberg does. Using the wrong one here would
            # compare the MPS against a different operator than the MPO measures.
            S.heis && (acc += conj(v[k]) * v[k] * (si == sj ? 0.25 : -0.25))
            if si != sj
                bb = b ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))
                acc += conj(v[S.idx[bb]]) * v[k] * 0.5
            end
        end
        out[n] = real(acc) / nrm
    end
    return out
end

const MODELS = WHICH == "all" ? ("A", "B", "C") : (WHICH,)
const NAME = Dict("A" => "heisenberg_su2_chain_L16",
                  "B" => "square_su2_4x4_L16",
                  "C" => "kagome_dipolarXY_2x3_L18")

if PHASE == "evolve"
    io = open(joinpath(OUTDIR, "three_models_evolve.csv"), "w")
    println(io, "model,scheme,t,err,obs,obs_exact,states,krylov,secs,dE")
    for tag in MODELS
        S = setup(tag)
        nst = round(Int, TMAX / DT)
        refs = Dict(k => dense_profile(S, k == 0 ? copy(S.v0) :
                        expv_sparse(S.Hs, ComplexF64(-im * k * DT), S.v0; m = 30, tol = 1e-13))
                    for k in 0:nst)
        @printf("=== %s : %d sites, %d probe bonds, %d sector states ===\n",
                NAME[tag], S.L, length(S.probe), length(S.states)); flush(stdout)
        for (nm, mk) in (("tdvp2", () -> (p, t) -> tdvp2_step!(p, S.W, t; maxdim = CAP,
                                                trunc_thresh = 1e-10, maxiter = MI)),
                         ("tdvp_cbe1s", () -> (p, t) -> tdvp_cbe1s_step!(p, S.W, t; maxdim = CAP,
                                                trunc_thresh = 1e-10, maxiter = MI)),
                         ("cbe_bug", () -> (p, t) -> cbe_bug_step!(p, S.W, t; exact = true,
                                                krylov_basis = 30, krylov_tol = 1e-8,
                                                maxdim = CAP, trunc_thresh = 1e-10,
                                                maxiter = MI)))
            step! = mk()
            psi, kry, secs = copy(S.psi0), 0, 0.0
            e0 = real(mpo_energy(copy(psi), S.W)) / max(norm(psi)^2, eps())
            for k in 0:nst
                if k > 0
                    t0 = time_ns(); info = step!(psi, ComplexF64(-im * DT))
                    secs += (time_ns() - t0) / 1e9
                    hasproperty(info, :krylov_dims) && (kry += info.krylov_dims)
                end
                pr = profile(S, psi)
                e  = real(mpo_energy(copy(psi), S.W)) / max(norm(psi)^2, eps())
                @printf(io, "%s,%s,%g,%.8e,%.8e,%.8e,%d,%d,%.3f,%.3e\n",
                        NAME[tag], nm, k * DT, maximum(abs.(pr .- refs[k])),
                        sum(pr) / length(pr), sum(refs[k]) / length(refs[k]),
                        maximum(state_bond_dims(psi)), kry, secs, abs(e - e0))
                flush(io)
            end
            @printf("  %-12s done: err=%.3e kry=%d %.1fs\n", nm,
                    maximum(abs.(profile(S, psi) .- refs[nst])), kry, secs); flush(stdout)
        end
    end
    close(io)

elseif PHASE == "matrix"
    io = open(joinpath(OUTDIR, "three_models_matrix.csv"), "w")
    println(io, "model,m,tau_trunc,split_cutoff,err,states,krylov,secs")
    for tag in MODELS
        S = setup(tag)
        nst = round(Int, TMAX / DT)
        ref = dense_profile(S, expv_sparse(S.Hs, ComplexF64(-im * TMAX), S.v0; m = 30, tol = 1e-13))
        @printf("=== matrix %s ===\n", NAME[tag]); flush(stdout)
        for m in (0, 3, 30), tt in (1e-4, 1e-6, 1e-8, 1e-10), sc in (1e-4, 1e-6, 1e-8, 1e-10)
            psi, kry, secs = copy(S.psi0), 0, 0.0
            for _ in 1:nst
                t0 = time_ns()
                info = cbe_bug_step!(psi, S.W, ComplexF64(-im * DT); exact = true,
                                     krylov_basis = m, krylov_tol = 0.0, split_cutoff = sc,
                                     maxdim = CAP, trunc_thresh = tt, maxiter = MI)
                secs += (time_ns() - t0) / 1e9; kry += info.krylov_dims
            end
            @printf(io, "%s,%d,%g,%g,%.8e,%d,%d,%.3f\n", NAME[tag], m, tt, sc,
                    maximum(abs.(profile(S, psi) .- ref)),
                    maximum(state_bond_dims(psi)), kry, secs)
            flush(io)
        end
        @printf("  %s matrix done\n", NAME[tag]); flush(stdout)
    end
    close(io)
else
    error("unknown phase $(repr(PHASE)) -- use evolve or matrix")
end
set_symmetry!(:U1)
println("wrote to ", OUTDIR)
