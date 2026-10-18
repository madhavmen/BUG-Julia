# IT-2 (imaginary-time / open-system plan, Task 4): NORMALISED IMAGINARY TIME ON THE SPIN-1/2
# HEISENBERG SQUARE CYLINDER, H = sum_<ij> S_i . S_j, open in x, periodic in y (`Ly` even, so the
# lattice is bipartite), column-major snake (`square_cylinder_mpo`), from the CHECKERBOARD Neel
# state in S^z = 0 (strong U(1)).
#
# Arms, grids, knobs and the per-step CSV: `it_common.jl` (bug2/3/4 via ite_step!, tdvp2 the fair
# order-2 partner; `panel:P:k` and `uniform:n`, the SAME list for every arm).
#
# References: N = Lx*Ly <= IT_EXACT_N -- exact E0, the exact trajectory E(beta) of exp(-beta H)|Neel>
# and the exact observables at beta_max (sparse Krylov in S^z = 0); larger N -- a DMRG ground state
# (rsvd_cbe_dmrg1s!, D = IT_DMRG_D, cached), labelled as DMRG, not exact. A FINITE cylinder is never
# compared with a thermodynamic energy density (plan Task 4).
#
# Final observables (IT_MAPS = 1): the <S^z_i> map, the bond-energy map <S_i.S_j> on every lattice
# bond (rung, wrap, leg), and the staggered S^zz(pi,pi) = (1/N) sum_ij e_i e_j <S^z_i S^z_j> with
# e_i = (-1)^(x+y) summed over the WHOLE finite cylinder (no translation average: x is open). For
# an S = 0 state the full S(pi,pi) = 3 S^zz(pi,pi).
#
# Output (IT_OUT, tag LxxLy): it2_steps_<tag>.csv one row per step and arm, it2_final_<tag>.csv
# variance, S^zz(pi,pi) and the map errors at beta_max, it2_maps_<tag>.csv the site and bond maps
# (arm `ref` = the reference), it2_ref_<tag>.csv the reference energies.
#
# Run: julia -t 4 --project=PKG benchmarks/imaginary_time/it2_square_cylinder.jl
# Initial state (IT_INIT): `neel` (checkerboard, the plan's) or `dw`, the domain wall with the left
# half of the cylinder (snake sites 1..N/2, i.e. x <= Lx/2) up (USER 2026-10-18); tag suffix `_dw`.
#
# Env: IT_LX (4) IT_LY (4) IT_EXACT_N (24) IT_DMRG_D (1024) IT_MAPS (1) IT_INIT (neel), plus
#      it_common.jl's knobs

using Serialization, SparseArrays

include(joinpath(@__DIR__, "it_common.jl"))
include(joinpath(@__DIR__, "..", "exact_sparse.jl"))
include(joinpath(@__DIR__, "..", "sparse_krylov.jl"))

const LX      = parse(Int, get(ENV, "IT_LX", "4"))
const LY      = parse(Int, get(ENV, "IT_LY", "4"))
const N       = LX * LY
const EXACT_N = parse(Int, get(ENV, "IT_EXACT_N", "24"))
const DMRG_D  = parse(Int, get(ENV, "IT_DMRG_D", "1024"))
const MAPS    = get(ENV, "IT_MAPS", "1") == "1"
const INIT    = get(ENV, "IT_INIT", "neel")
INIT in ("neel", "dw") || error("IT_INIT must be neel or dw, got $INIT")
const TAG     = INIT == "neel" ? "$(LX)x$(LY)" : "$(LX)x$(LY)_$INIT"

iseven(LY) || error("IT-2 needs an even circumference: odd Ly frustrates the Neel pattern on the wrap")
const JM    = square_cylinder_couplings(LX, LY)
const BONDS = square_cylinder_bonds(LX, LY)
"The sublattice sign e_i = (-1)^(x+y) of snake site i = (x-1) Ly + y."
const EPS   = [(-1)^(((i - 1) ÷ LY + 1) + ((i - 1) % LY + 1)) for i in 1:N]

neel_cyl() = product_state([e > 0 ? :up : :down for e in EPS])
"The start: the checkerboard Neel or the domain wall (sites 1..N/2 up); both S^z = 0."
start_up() = INIT == "dw" ? [i <= N ÷ 2 for i in 1:N] : [e > 0 for e in EPS]
init_cyl() = product_state([u ? :up : :down for u in start_up()])
staggered(C) = sum(EPS[i] * EPS[j] * C[i, j] for i in 1:N, j in 1:N) / N

# ── MPS observables ────────────────────────────────────────────────────────────────────────────

"One `pair_mpo` per lattice bond, so `mpo_energy` reads <S_i.S_j> at any snake distance."
bond_mpos() = [pair_mpo(N, (J = zeros(N, N); J[i, j] = 1.0; J), spin_vertices(N)) for (i, j) in BONDS]

"`(sz, C, bonds)`: <S^z_i>, the matrix <S^z_i S^z_j>, and <S_i.S_j> on every bond."
function mps_maps(psi, Wb)
    n2 = real(overlap(psi, psi))
    sz = real.(sz_overlaps(psi, psi)) ./ n2
    C = zeros(N, N)
    for i in 1:N
        C[:, i] = real.(sz_overlaps(psi, apply_sz!(copy(psi), i))) ./ n2
    end
    return sz, (C + C') / 2, [real(mpo_energy(psi, W)) / n2 for W in Wb]
end

# ── references ─────────────────────────────────────────────────────────────────────────────────

"The same three maps from an exact S^z = 0 vector (bit `i-1` set = site `i` up, exact_sparse.jl)."
function exact_maps(v, states, idx)
    p = abs2.(v) ./ sum(abs2, v)
    bit(b, i) = ((b >> (i - 1)) & 1) - 0.5
    sz = [sum(p[k] * bit(states[k], i) for k in eachindex(states)) for i in 1:N]
    C = zeros(N, N)
    for (k, b) in enumerate(states), i in 1:N, j in i:N
        C[i, j] += p[k] * bit(b, i) * bit(b, j)
    end
    C = Symmetric(C, :U) |> Matrix
    nv = sum(abs2, v)
    bonds = map(BONDS) do (i, j)
        flip = 0.0im
        for (k, b) in enumerate(states)
            bit(b, i) == bit(b, j) && continue
            flip += conj(v[idx[b ⊻ (1 << (i - 1)) ⊻ (1 << (j - 1))]]) * v[k] / 2
        end
        C[i, j] + real(flip) / nv
    end
    return sz, C, bonds
end

"""Exact (N <= IT_EXACT_N): E0, per grid the exact E(beta) at every grid point, and the maps at
beta_max, from exp(-beta H)|Neel> by chunked sparse Krylov."""
function exact_refs()
    H, states, idx = pairs_sparse(N, JM)
    say("exact sector: $(length(states)) states, nnz $(nnz(H))")
    # m = 60: the Lanczos basis is n x m Float64, 1.3 GB at 6x4 (C(24,12) = 2.7e6)
    E0, _, res, _, _ = sparse_ground(H, length(states); m = 60, restarts = 80)
    res < 1e-8 || error("exact ground state unconverged (resid $res)")
    v0 = zeros(ComplexF64, length(states))
    v0[idx[sum(1 << (i - 1) for i in 1:N if start_up()[i])]] = 1.0
    traj = Dict{String, Vector{Float64}}()
    maps = nothing
    for g in IT.grids
        v = copy(v0); Es = Float64[]
        for db in grid_steps(g)
            # chunks small enough for the 30-vector Krylov at ||H|| ~ (number of bonds)
            nch = max(1, ceil(Int, db * length(BONDS) / 4))
            for _ in 1:nch
                v, _ = sparse_expv(H, v, ComplexF64(-db / nch); m = 30, tol = 1e-13)
                v ./= norm(v)
            end
            push!(Es, real(dot(v, H * v)))
        end
        traj[g] = Es
        maps === nothing && (maps = exact_maps(v, states, idx))
    end
    return (E0 = E0, kind = "exact", traj = traj, maps = maps)
end

"DMRG reference (N > IT_EXACT_N), cached; its energy is an upper bound, labelled as such."
function dmrg_ref(W, Wb)
    file = joinpath(IT.out, "it2_dmrg_$(LX)x$(LY)_D$(DMRG_D).jls")      # the ground state: any start
    if isfile(file)
        b = deserialize(file)
        b.key == (LX, LY, DMRG_D) && return (E0 = b.E0, kind = "dmrg_D$(DMRG_D)", traj = nothing,
                                              maps = b.maps)
    end
    psi = neel_cyl()
    t = @elapsed rsvd_cbe_dmrg1s!(psi, W; n_sweeps = 60, maxdim = DMRG_D, tol_basis = 1e-6,
                                  tol_trunc = 1e-9, maxiter = 30, restol = 1e-10)
    E0 = energy(psi, W)
    say(@sprintf("DMRG %s D=%d: E0=%.12f  E0/N=%.12f  chi=%d  var=%.2e  %.1fs", TAG, DMRG_D, E0,
                 E0 / N, maximum(bond_dims(psi)), mpo_variance(psi, W), t))
    maps = mps_maps(psi, Wb)
    serialize(file, (key = (LX, LY, DMRG_D), E0 = E0, maps = maps))
    return (E0 = E0, kind = "dmrg_D$(DMRG_D)", traj = nothing, maps = maps)
end

function write_maps(mio, who, g, tb, seed, maps)
    sz, C, bonds = maps
    for i in 1:N
        @printf(mio, "%s,%s,%s,%.0e,%d,site,%d,%d,%.12f\n", TAG, who, g, tb, seed, i, i, sz[i])
    end
    for (b, (i, j)) in pairs(BONDS)
        @printf(mio, "%s,%s,%s,%.0e,%d,bond,%d,%d,%.12f\n", TAG, who, g, tb, seed, i, j, bonds[b])
    end
    @printf(mio, "%s,%s,%s,%.0e,%d,szz_pipi,0,0,%.12f\n", TAG, who, g, tb, seed, staggered(C))
end

function main()
    mkpath(IT.out)
    set_symmetry!(:U1)
    W = square_cylinder_mpo(LX, LY)
    Wb = MAPS ? bond_mpos() : Any[]
    t = @elapsed it_warmup(square_cylinder_mpo(2, 4), product_state([:up, :down, :up, :down,
                                                                    :down, :up, :down, :up]))
    say(@sprintf("JIT warm-up (untimed, excluded from every wall): %.1f s", t))
    say("== IT-2 square cylinder $TAG start $INIT (N=$N, $(length(BONDS)) bonds, MPO dims $(maximum(mpo_virtual_dims(W))))  beta_max=$(IT.bmax)  grids $(IT.grids)  arms $(IT.arms)  tol_basis $(IT.tbs)  tol_trunc $(IT.tt)  tol_root $(IT.tr)  maxdim $(IT.maxdim)  root $(IT.root)  backamp $(IT.backamp)  var_every $(IT.var_every)  threads $(Threads.nthreads())")
    ref = N <= EXACT_N ? exact_refs() : dmrg_ref(W, Wb)
    rs = staggered(ref.maps[2])
    say(@sprintf("reference %s: E0 = %.12f  (E0/N = %.12f)  Szz(pi,pi) = %.10f", ref.kind,
                 ref.E0, ref.E0 / N, rs))
    open(joinpath(IT.out, "it2_ref_$TAG.csv"), "w") do io
        println(io, "kind,grid,step,E")
        println(io, "$(ref.kind),E0,0,$(ref.E0)")
        ref.traj === nothing || for (g, Es) in ref.traj, (k, E) in pairs(Es)
            println(io, "$(ref.kind),$g,$k,$E")
        end
    end
    mio = open(joinpath(IT.out, "it2_maps_$TAG.csv"), "w")
    println(mio, "geom,arm,grid,tol_basis,seed,kind,i,j,value")
    write_maps(mio, "ref", ref.kind, NaN, 0, ref.maps)
    io = open(joinpath(IT.out, "it2_steps_$TAG.csv"), "w")
    println(io, it_step_header("geom"))
    fio = open(joinpath(IT.out, "it2_final_$TAG.csv"), "w")
    println(fio, "geom,arm,grid,tol_basis,seed,E_per_site,variance,wall,nfail,szz_pipi,szz_pipi_err,bond_err_max,corr_err_max")
    for g in IT.grids, arm in IT.arms, (tb, seed) in it_runs(arm)
        f = it_run_arm(io, TAG, W, init_cyl(), ref, arm, g, tb, seed)
        Ef = energy(f.psi, W) / N
        if MAPS
            m = mps_maps(f.psi, Wb)
            write_maps(mio, arm, g, tb, seed, m); flush(mio)
            s = staggered(m[2])
            @printf(fio, "%s,%s,%s,%.0e,%d,%.14f,%.6e,%.3f,%d,%.10f,%.6e,%.6e,%.6e\n", TAG, arm,
                    g, tb, seed, Ef, f.var, f.wall, f.nfail, s, s - rs,
                    maximum(abs.(m[3] .- ref.maps[3])), maximum(abs.(m[2] .- ref.maps[2])))
        else
            @printf(fio, "%s,%s,%s,%.0e,%d,%.14f,%.6e,%.3f,%d,%.10f,%.6e,%.6e,%.6e\n", TAG, arm,
                    g, tb, seed, Ef, f.var, f.wall, f.nfail, NaN, NaN, NaN, NaN)
        end
        flush(fio)
    end
    close(io); close(fio); close(mio)
    say("IT2-DONE")
end

main()
