# IT-1 (imaginary-time / open-system plan, Task 3): NORMALISED IMAGINARY TIME ON THE OPEN
# SPIN-1/2 HEISENBERG CHAIN, H = sum_j S_j . S_{j+1}, from the Neel MPS in S^z = 0 (strong U(1)).
#
# Arms (one process, one node, so walls compare within the job):
#   bugP      ite_step! at order P (2, 3, 4), root `IT_ROOT`, per `IT_TB` basis tolerance: Eshift =
#             the state's energy, outside-Galerkin defect vs tol_basis, re-enrich / split / FLAG
#   tdvp2     tdvp2_step! at -dbeta (the symmetric 2-site sweep), renormalised, the SAME per-cut
#             discard rule (`weight_tol = tol_trunc`) and Saad-stopped exponentials at tol_root --
#             the order-2 partner of bug2 on the same grid; it has no tol_basis
# Grids: `panel:P:k` (arXiv:2606.02930 logarithmic panels, k steps per panel) and `uniform:n`, the
# SAME list for every arm (grid ablation), so a log-grid gain is not confused with a BUG gain.
#
# References: L <= IT_EXACT_N -- exact E0, the exact trajectory E(beta) = <exp(-beta H) Neel>, and
# the exact correlation row at beta_max (sparse Krylov in S^z = 0); larger L -- a DMRG ground state
# (rsvd_cbe_dmrg1s!, D = IT_DMRG_D, cached) and its energy, labelled as DMRG not exact.
#
# Output (IT_OUT): it1_steps_L<L>.csv one row per step and arm (energies before/after the root and
# after the compression, outside defect, status, chi, walls), it1_final_L<L>.csv variance and
# correlations at beta_max, it1_ref_L<L>.csv the reference rows. Tolerances default to the
# understood fair settings: tol_trunc = tol_root = 1e-6, tol_basis the knob.
#
# Run: julia -t 4 --project=PKG benchmarks/imaginary_time/it1_heisenberg_chain.jl
# Env: IT_L (12) IT_BMAX (16) IT_GRIDS ("panel:6:4,uniform:64") IT_ARMS ("bug2,bug3,bug4,tdvp2")
#      IT_TB ("1e-3,1e-5,1e-7") IT_TT (1e-6) IT_TR (1e-6) IT_MAXDIM (256) IT_ROOT (exp)
#      IT_SEEDS ("1") IT_RETRIES (2) IT_OUTSIDE (1) IT_EXACT_N (20) IT_DMRG_D (512) IT_OUT

using LinearAlgebra, Printf, Random, Serialization, SparseArrays
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "exact_sparse.jl"))
include(joinpath(@__DIR__, "..", "sparse_krylov.jl"))

const L       = parse(Int, get(ENV, "IT_L", "12"))
const BMAX    = parse(Float64, get(ENV, "IT_BMAX", "16"))
const GRIDS   = split(get(ENV, "IT_GRIDS", "panel:6:4,uniform:64"), ",")
const ARMS    = split(get(ENV, "IT_ARMS", "bug2,bug3,bug4,tdvp2"), ",")
const TBS     = parse.(Float64, split(get(ENV, "IT_TB", "1e-3,1e-5,1e-7"), ","))
const TT      = parse(Float64, get(ENV, "IT_TT", "1e-6"))
const TR      = parse(Float64, get(ENV, "IT_TR", "1e-6"))
const MAXDIM  = parse(Int, get(ENV, "IT_MAXDIM", "256"))
const ROOT    = Symbol(get(ENV, "IT_ROOT", "exp"))
const SEEDS   = parse.(Int, split(get(ENV, "IT_SEEDS", "1"), ","))
const RETRIES = parse(Int, get(ENV, "IT_RETRIES", "2"))
const OUTSIDE = get(ENV, "IT_OUTSIDE", "1") == "1"
const EXACT_N = parse(Int, get(ENV, "IT_EXACT_N", "20"))
const DMRG_D  = parse(Int, get(ENV, "IT_DMRG_D", "512"))
const OUT     = get(ENV, "IT_OUT", joinpath(@__DIR__, "results"))
const PARALLEL = Threads.nthreads() >= 2

say(msg) = (println(msg); flush(stdout))

function grid_steps(g)
    a = split(g, ":")
    a[1] == "panel" && return log_panel_steps(BMAX, parse(Int, a[2]); per_panel = parse(Int, a[3]))
    a[1] == "uniform" && return uniform_beta_steps(BMAX, parse(Int, a[2]))
    error("grid must be panel:P:k or uniform:n, got $g")
end

energy(psi, W) = real(mpo_energy(psi, W)) / real(overlap(psi, psi))
renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)

"`<S^z_c S^z_{c+r}>` for r = 0..L-c and the staggered `S^zz(pi)` from that row."
function corr_row(psi)
    c = L ÷ 2
    phi = apply_sz!(copy(psi), c)
    row = real.(sz_overlaps(psi, phi)) ./ real(overlap(psi, psi))
    return [row[c + r] for r in 0:(L - c)]
end

# ── references ─────────────────────────────────────────────────────────────────────────────────

"""Exact (L <= IT_EXACT_N): E0, and per grid the exact E(beta) at every grid point and the exact
central correlation row at beta_max, from exp(-beta H)|Neel> by chunked sparse Krylov."""
function exact_refs()
    H, states, idx = heisenberg_sparse(L)
    E0, _, res, _, _ = sparse_ground(H, length(states))
    res < 1e-8 || error("exact ground state unconverged (resid $res)")
    v0 = neel_vector(L, idx)
    traj = Dict{String, Vector{Float64}}()
    corr = Float64[]
    for g in GRIDS
        v = copy(v0); Es = Float64[]
        for db in grid_steps(g)
            # chunks small enough for the 30-vector Krylov at ||H|| ~ L
            nch = max(1, ceil(Int, db * L / 4))
            for _ in 1:nch
                v, _ = sparse_expv(H, v, ComplexF64(-db / nch); m = 30, tol = 1e-13)
                v ./= norm(v)
            end
            push!(Es, real(dot(v, H * v)))
        end
        traj[g] = Es
        if isempty(corr)
            c = L ÷ 2
            szc = sz_on(v, states, c)
            corr = [real(dot(v, sz_on(szc, states, c + r))) for r in 0:(L - c)]
        end
    end
    return (E0 = E0, kind = "exact", traj = traj, corr = corr)
end

"DMRG reference (L > IT_EXACT_N), cached; its energy is an upper bound, labelled as such."
function dmrg_ref(W)
    file = joinpath(OUT, "it1_dmrg_L$(L)_D$(DMRG_D).jls")
    if isfile(file)
        b = deserialize(file)
        b.key == (L, DMRG_D) && return (E0 = b.E0, kind = "dmrg_D$(DMRG_D)", traj = nothing,
                                         corr = b.corr)
    end
    psi = neel_state(L)
    t = @elapsed gs = rsvd_cbe_dmrg1s!(psi, W; n_sweeps = 60, maxdim = DMRG_D, tol_basis = 1e-6,
                                       tol_trunc = 1e-9, maxiter = 30, restol = 1e-10)
    E0 = energy(psi, W)
    say(@sprintf("DMRG L=%d D=%d: E0=%.12f  E0/L=%.12f  chi=%d  var=%.2e  %.1fs", L, DMRG_D, E0,
                 E0 / L, maximum(bond_dims(psi)), mpo_variance(psi, W), t))
    corr = corr_row(psi)
    serialize(file, (key = (L, DMRG_D), E0 = E0, corr = corr))
    return (E0 = E0, kind = "dmrg_D$(DMRG_D)", traj = nothing, corr = corr)
end

# ── one arm on one grid ────────────────────────────────────────────────────────────────────────

function run_arm(io, W, ref, arm, g, tb, seed)
    steps = grid_steps(g)
    psi = neel_state(L)
    rng = MersenneTwister(seed)
    beta, wall = 0.0, 0.0
    nfail = nretry = 0
    for (k, db) in enumerate(steps)
        if startswith(arm, "bug")
            p = parse(Int, arm[4:end])
            r = ite_step!(psi, W, db; order = p, root_method = ROOT, tol_basis = tb,
                          tol_trunc = TT, tol_root = TR, maxdim = MAXDIM, retries = RETRIES,
                          outside_residual = OUTSIDE, parallel = PARALLEL, rng = rng)
            wall += r.wall
            E = r.e_after
            st, um = r.status, join(string.(r.unmet), "+")
            row = (r.e_pre_root, r.e_root, r.de_root, r.de_compression, r.resid, r.discarded,
                   r.root_err, r.nsub)
            nfail += r.status === :failed; nretry += r.status === :retried
        else                                      # tdvp2: the fair order-2 partner
            t0 = time_ns()
            info = tdvp2_step!(psi, W, ComplexF64(-db); maxdim = MAXDIM, maxiter = 30,
                               conv_tol = TR, substeps = 0, weight_tol = TT)
            renorm!(psi)
            wall += (time_ns() - t0) / 1e9
            E = energy(psi, W)
            st, um = :tdvp2, ""
            row = (NaN, NaN, NaN, NaN, NaN, info.cumulative, NaN, 1)
        end
        beta += db
        Ex = ref.traj === nothing ? NaN : ref.traj[g][k]
        @printf(io, "%d,%s,%s,%.0e,%d,%d,%.10g,%.10g,%.14f,%.6e,%.6e,%d,%.3f,%s,%s,%.14f,%.14f,%.6e,%.6e,%.6e,%.6e,%.6e,%d\n",
                L, arm, g, tb, seed, k, beta, db, E, E - ref.E0, E - Ex,
                maximum(bond_dims(psi)), wall, st, um, row[1], row[2], row[3], row[4],
                row[5], row[6], row[7], row[8])
        flush(io)
    end
    var = mpo_variance(psi, W)
    corr = corr_row(psi)
    say(@sprintf("  %-6s %-12s tb %.0e seed %d: E-E0 %.3e  E-E(beta) %.3e  var %.2e  chi %d  wall %.1fs  failed %d retried %d",
                 arm, g, tb, seed, energy(psi, W) - ref.E0,
                 ref.traj === nothing ? NaN : energy(psi, W) - ref.traj[g][end], var,
                 maximum(bond_dims(psi)), wall, nfail, nretry))
    return (var = var, corr = corr, wall = wall, nfail = nfail)
end

function main()
    mkpath(OUT)
    set_symmetry!(:U1)
    W = xxz_mpo(L; J = 1.0, delta = 1.0)
    say("== IT-1 Heisenberg chain L=$L  beta_max=$BMAX  grids $(GRIDS)  arms $(ARMS)  tol_basis $(TBS)  tol_trunc $TT  tol_root $TR  maxdim $MAXDIM  root $ROOT  threads $(Threads.nthreads())")
    ref = L <= EXACT_N ? exact_refs() : dmrg_ref(W)
    say(@sprintf("reference %s: E0 = %.12f  (E0/L = %.12f)", ref.kind, ref.E0, ref.E0 / L))
    open(joinpath(OUT, "it1_ref_L$L.csv"), "w") do io
        println(io, "kind,grid,step,E")
        println(io, "$(ref.kind),E0,0,$(ref.E0)")
        ref.traj === nothing || for (g, Es) in ref.traj, (k, E) in pairs(Es)
            println(io, "$(ref.kind),$g,$k,$E")
        end
        for (r, c) in pairs(ref.corr)
            println(io, "$(ref.kind),corr,$(r - 1),$c")
        end
    end
    io = open(joinpath(OUT, "it1_steps_L$L.csv"), "w")
    println(io, "L,arm,grid,tol_basis,seed,step,beta,dbeta,E,dE0,dEbeta,chi,wall,status,unmet," *
                "e_pre_root,e_root,de_root,de_compression,resid,discarded,root_err,nsub")
    fio = open(joinpath(OUT, "it1_final_L$L.csv"), "w")
    println(fio, "L,arm,grid,tol_basis,seed,variance,wall,nfail,corr_err_max")
    for g in GRIDS, arm in ARMS
        tbs = startswith(arm, "bug") ? TBS : [NaN]
        for tb in tbs, seed in (startswith(arm, "bug") ? SEEDS : SEEDS[1:1])
            f = run_arm(io, W, ref, arm, g, tb, seed)
            cerr = maximum(abs.(f.corr .- ref.corr); init = 0.0)
            @printf(fio, "%d,%s,%s,%.0e,%d,%.6e,%.3f,%d,%.6e\n", L, arm, g, tb, seed, f.var,
                    f.wall, f.nfail, cerr)
            flush(fio)
        end
    end
    close(io); close(fio)
    say("IT1-DONE")
end

main()
