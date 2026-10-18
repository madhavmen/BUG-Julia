# OS-3 (imaginary-time / open-system plan, Task 7): the PARTICLE-CONSERVING quantum East chain
# (Brighi, Ljubotina & Serbyn, SciPost Phys. 15, 093 (2023)) WITH DEPHASING -- a proposed open
# extension, NOT the coherently driven open East model of Causer, Banuls & Garrahan (2025).
#
#   H_E = J sum_{j=2}^{L-1} n_{j-1} (c'_j c_{j+1} + h.c.)      hard-core bosons (= spinless fermions:
#                                                            a nearest-neighbour hop has no string)
#   L_j = sqrt(gamma) n_j
#
# H_E and every L_j commute with N, so the Lindbladian has the STRONG U(1): rho stays in one (N, N)
# block, and the plan's table assigns OS-3 exactly that grading -- the default here (OS_SYM = strong).
# The trial space keeps the block's trace vector |I_N>> (`mpdo_identity_sector`, exact, one dimension
# per link sector), so the Galerkin step is trace-exact as |I>> makes it under :weak. OS_SYM = weak
# (label n_ket - n_bra, |I>> a product state) is kept as the A/B. The facilitated hop is a THREE-site
# term, built by
# `lindblad_mpo(...; triples)` (`TripleTerm`), pinned against the dense Liouvillian in
# test_liouville.jl together with the model checks ([H, N] = 0, frozen configurations stay frozen,
# only facilitated hops).
#
# Initial states restricted to a MOBILE fragment (plan): `dw` (the first L/2 sites occupied -- the
# front is facilitated), `quarter` (the first L/4), `pair` (sites 1, 2).
#
# Observables at the output times: n_j (the heatmap), the density autocorrelation
# A(t) = (1/L) sum_j (2 n_j(t) - 1)(2 n_j(0) - 1) (exact for a product start), and the centre of mass
# x_cm = sum_j j n_j / N (the spreading). REFERENCE: exact sparse Lindblad propagation for
# 2^L <= OS_REF_MAXD; beyond that none (the gamma = 0 dynamics of Brighi et al. are TRENDS for this
# variant, not data).
#
# Run: julia -t 4 --project=PKG benchmarks/open_system/os3_east_dephasing.jl
# Env: OS_L (8) OS_J (1.0) OS_GAMMA (0.1) OS_INIT (dw | quarter | pair) OS_SYM (strong | weak)
#      OS_REF_MAXD (256), plus os_common.jl's knobs

include(joinpath(@__DIR__, "os_common.jl"))
include(joinpath(@__DIR__, "sparse_lindblad.jl"))

const L        = parse(Int, get(ENV, "OS_L", "8"))
const J        = parse(Float64, get(ENV, "OS_J", "1.0"))
const GAMMA    = parse(Float64, get(ENV, "OS_GAMMA", "0.1"))
const INIT     = get(ENV, "OS_INIT", "dw")
const SYM      = Symbol(get(ENV, "OS_SYM", "strong"))
const REF_MAXD = parse(Int, get(ENV, "OS_REF_MAXD", "256"))
const TAG      = @sprintf("L%d_g%g_%s_%s", L, GAMMA, INIT, SYM)

const SP = ComplexF64[0 1; 0 0]                 # state 1 = occupied
const SM = ComplexF64[0 0; 1 0]
const NU = ComplexF64[1 0; 0 0]

function model_terms(n)
    w = fill(J, n - 2)
    return Any[(NU, SP, SM, w), (NU, SM, SP, w)], (GAMMA > 0 ? Any[(NU, GAMMA)] : Any[])
end

function model(n)
    triples, jumps = model_terms(n)
    ls = liouville_space([1, 0]; sym = SYM)
    return ls, lindblad_mpo(ls, n; triples = triples, jumps = jumps)
end

function occupied(n)
    INIT == "dw"      && return collect(1:(n ÷ 2))
    INIT == "quarter" && return collect(1:max(2, n ÷ 4))
    INIT == "pair"    && return [1, 2]
    error("OS_INIT must be dw, quarter or pair, got $INIT")
end
local_rhos(n) = [j in occupied(n) ? NU : I - NU for j in 1:n]
const S0 = [j in occupied(L) ? 1.0 : -1.0 for j in 1:L]          # 2 n_j(0) - 1

east_obs(nj) = Dict("n" => nj, "autocorr" => [sum((2 .* nj .- 1) .* S0) / L],
                    "xcm" => [sum((1:L) .* nj) / sum(nj)])

"The trace vector kept in the trial space: |I_N>> of rho0's block (strong) or |I>> (weak)."
trace_vector(ls, n, rho0) = SYM === :strong ? mpdo_identity_sector(ls, n, rho0) : mpdo_identity(ls, n)

observe(ls) = rho -> east_obs([real(mpdo_expect(rho, ls, Dict(j => NU))) for j in 1:L])

"Exact observables at every output step of every dt (`nothing` beyond OS_REF_MAXD)."
function references()
    2^L <= REF_MAXD || return Dict(dt => nothing for dt in OS.dts)
    triples, jumps = model_terms(L)
    S = sparse_lindbladian(L, 2; pairs = Any[], jumps = jumps)
    H = spzeros(ComplexF64, 2^L, 2^L)
    for (A, B, C, w) in triples, j in 1:(L - 2)
        H += w[j] * _sp_emb(A, j, L, 2) * _sp_emb(B, j + 1, L, 2) * _sp_emb(C, j + 2, L, 2)
    end
    Id = sparse(1.0I, 2^L, 2^L)
    S += -1im * (kron(H, Id) - kron(Id, transpose(H)))
    say("exact Liouvillian: $(size(S, 1)) states, nnz $(nnz(S))")
    rho0 = kron(local_rhos(L)...)
    out = Dict{Float64, Any}()
    for dt in OS.dts
        ks = os_out_steps(os_nsteps(dt))
        r = Dict{Tuple{Int, String}, Vector{Float64}}()
        for (k, R) in zip(ks, sparse_lindblad_trajectory(S, rho0, ks .* dt))
            nj = [real(site_expect(R, NU, j, L, 2)) / real(tr(R)) for j in 1:L]
            for (name, v) in east_obs(nj)
                r[(k, name)] = v
            end
        end
        out[dt] = r
    end
    return out
end

function main()
    set_symmetry!(:U1)
    ls, W = model(L)
    rho0 = mpdo_product_state(ls, local_rhos(L))
    Iket = trace_vector(ls, L, rho0)
    ls6, W6 = model(6)
    r6 = mpdo_product_state(ls6, local_rhos(6))
    t = @elapsed os_warmup(ls6, W6, r6, trace_vector(ls6, 6, r6))
    say(@sprintf("JIT warm-up (untimed, excluded from every wall): %.1f s", t))
    say("== OS-3 U(1) East + dephasing L=$L J=$J gamma=$GAMMA init=$INIT occupied $(occupied(L)) sym=$SYM (trace vector in the trial space: $(SYM === :strong ? "|I_N>>" : "|I>>"))  MPO dims $(maximum(mpo_virtual_dims(W)))  tmax=$(OS.tmax)  dts $(OS.dts)  arms $(OS.arms)  tol_basis $(OS.tbs)  tol_trunc $(OS.tt)  tol_root $(OS.tr)  maxdim $(OS.maxdim)  root $(OS.root)  threads $(Threads.nthreads())")
    t = @elapsed refs = references()
    say(@sprintf("exact sparse reference: %.1f s", t))
    sio, oio, fio = os_open("os3", TAG)
    for dt in OS.dts, arm in OS.arms, (tb, seed) in os_runs(arm)
        f = os_run_arm(sio, oio, TAG, ls, W, rho0, Iket, observe(ls), refs[dt], arm, dt, tb, seed)
        os_final!(fio, TAG, arm, tb, seed, dt, f)
    end
    close(sio); close(oio); close(fio)
    say("OS3-DONE")
end

main()
