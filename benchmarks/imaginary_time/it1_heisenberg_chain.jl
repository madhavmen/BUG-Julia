# IT-1 (imaginary-time / open-system plan, Task 3): NORMALISED IMAGINARY TIME ON THE OPEN
# SPIN-1/2 HEISENBERG CHAIN, H = sum_j S_j . S_{j+1}, from the Neel MPS in S^z = 0 (strong U(1)).
#
# Arms, grids, knobs and the per-step CSV: `it_common.jl` (bug2/3/4 via ite_step!, tdvp2 the fair
# order-2 partner; `panel:P:k` and `uniform:n`, the SAME list for every arm).
#
# References: L <= IT_EXACT_N -- exact E0, the exact trajectory E(beta) = <exp(-beta H) Neel>, and
# the exact correlation row at beta_max (sparse Krylov in S^z = 0); larger L -- a DMRG ground state
# (rsvd_cbe_dmrg1s!, D = IT_DMRG_D, cached) and its energy, labelled as DMRG not exact.
#
# Output (IT_OUT): it1_steps_L<L>.csv one row per step and arm (energies before/after the root and
# after the compression, outside defect, status, chi, walls, tdvp2's backward amplification),
# it1_final_L<L>.csv variance and correlations at beta_max, it1_ref_L<L>.csv the reference rows.
#
# Run: julia -t 4 --project=PKG benchmarks/imaginary_time/it1_heisenberg_chain.jl
# Env: IT_L (12) IT_EXACT_N (20) IT_DMRG_D (512), plus the shared knobs of it_common.jl

using Serialization, SparseArrays

include(joinpath(@__DIR__, "it_common.jl"))
include(joinpath(@__DIR__, "..", "exact_sparse.jl"))
include(joinpath(@__DIR__, "..", "sparse_krylov.jl"))

const L       = parse(Int, get(ENV, "IT_L", "12"))
const EXACT_N = parse(Int, get(ENV, "IT_EXACT_N", "20"))
const DMRG_D  = parse(Int, get(ENV, "IT_DMRG_D", "512"))

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
    for g in IT.grids
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
    file = joinpath(IT.out, "it1_dmrg_L$(L)_D$(DMRG_D).jls")
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

function main()
    mkpath(IT.out)
    set_symmetry!(:U1)
    W = xxz_mpo(L; J = 1.0, delta = 1.0)
    t = @elapsed it_warmup(xxz_mpo(6; J = 1.0, delta = 1.0), neel_state(6))
    say(@sprintf("JIT warm-up (untimed, excluded from every wall): %.1f s", t))
    say("== IT-1 Heisenberg chain L=$L  beta_max=$(IT.bmax)  grids $(IT.grids)  arms $(IT.arms)  tol_basis $(IT.tbs)  tol_trunc $(IT.tt)  tol_root $(IT.tr)  maxdim $(IT.maxdim)  root $(IT.root)  backamp $(IT.backamp)  var_every $(IT.var_every)  threads $(Threads.nthreads())")
    ref = L <= EXACT_N ? exact_refs() : dmrg_ref(W)
    say(@sprintf("reference %s: E0 = %.12f  (E0/L = %.12f)", ref.kind, ref.E0, ref.E0 / L))
    open(joinpath(IT.out, "it1_ref_L$L.csv"), "w") do io
        println(io, "kind,grid,step,E")
        println(io, "$(ref.kind),E0,0,$(ref.E0)")
        ref.traj === nothing || for (g, Es) in ref.traj, (k, E) in pairs(Es)
            println(io, "$(ref.kind),$g,$k,$E")
        end
        for (r, c) in pairs(ref.corr)
            println(io, "$(ref.kind),corr,$(r - 1),$c")
        end
    end
    io = open(joinpath(IT.out, "it1_steps_L$L.csv"), "w")
    println(io, it_step_header("L"))
    fio = open(joinpath(IT.out, "it1_final_L$L.csv"), "w")
    println(fio, "L,arm,grid,tol_basis,seed,variance,wall,nfail,corr_err_max")
    for g in IT.grids, arm in IT.arms, (tb, seed) in it_runs(arm)
        f = it_run_arm(io, string(L), W, neel_state(L), ref, arm, g, tb, seed)
        cerr = maximum(abs.(corr_row(f.psi) .- ref.corr); init = 0.0)
        @printf(fio, "%d,%s,%s,%.0e,%d,%.6e,%.3f,%d,%.6e\n", L, arm, g, tb, seed, f.var,
                f.wall, f.nfail, cerr)
        flush(fio)
    end
    close(io); close(fio)
    say("IT1-DONE")
end

main()
