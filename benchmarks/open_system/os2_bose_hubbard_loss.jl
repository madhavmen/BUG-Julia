# OS-2 (imaginary-time / open-system plan, Task 6): BOSE-HUBBARD WITH LOCAL TWO-BODY LOSS as a
# weak-U(1) vectorised density matrix (`liouville.jl`, label n_ket - n_bra; the loss shifts ket and
# bra by -2 together, so the weak symmetry survives the physical loss -- the strong one does not and
# `lindblad_mpo` refuses it).
#
#   H   = -J sum_j (b'_j b_{j+1} + h.c.) + U/2 sum_j n_j (n_j - 1),   boson cutoff nmax (d = nmax+1)
#   L_j = sqrt(gamma) b_j^2                                            (a jump (Jop, g) is D[sqrt(g) Jop])
#
# EXACT IDENTITY for these conventions: d<N>/dt = -2 gamma sum_j <n_j (n_j - 1)> (each jump removes
# two bosons at rate gamma <b'^2 b^2> = gamma <n(n-1)>). Both sides are logged at every output time
# (`N` and `loss_rate`), so the identity is checked on every arm without a reference.
#
# REFERENCE: exact sparse Lindblad propagation (`sparse_lindblad.jl`) when d^L <= OS_REF_MAXD
# (plan "small L = 3-6 with nmax = 2, 3"); beyond that none -- the main sizes L = 16, 24 are graded
# by the identity, by the nmax = 2 vs 3 cutoff comparison (physical cutoff error, separate from the
# MPDO bond error) and against a tight-tolerance run, not against a quantitative literature curve
# at unmatched filling (plan: Rossini et al. give TRENDS -- the Zeno slowing of N(t) as gamma grows).
#
# Output (OS_OUT, tag L<L>_n<nmax>_U<U>_g<gamma>_<init>): os2_steps_*.csv, os2_obs_*.csv (n_j,
# double occupancy d_j = <n_j(n_j-1)>, N, loss_rate at the output times, with the exact values),
# os2_final_*.csv.
#
# Run: julia -t 4 --project=PKG benchmarks/open_system/os2_bose_hubbard_loss.jl
# Env: OS_L (4) OS_NMAX (2) OS_J (1.0) OS_U (2.0) OS_GAMMA (1.0) OS_INIT (domain | mott | double)
#      OS_REF_MAXD (1024), plus os_common.jl's knobs

include(joinpath(@__DIR__, "os_common.jl"))
include(joinpath(@__DIR__, "sparse_lindblad.jl"))

const L       = parse(Int, get(ENV, "OS_L", "4"))
const NMAX    = parse(Int, get(ENV, "OS_NMAX", "2"))
const J       = parse(Float64, get(ENV, "OS_J", "1.0"))
const U       = parse(Float64, get(ENV, "OS_U", "2.0"))
const GAMMA   = parse(Float64, get(ENV, "OS_GAMMA", "1.0"))
const INIT    = get(ENV, "OS_INIT", "domain")
const REF_MAXD = parse(Int, get(ENV, "OS_REF_MAXD", "1024"))
const D       = NMAX + 1
const TAG     = @sprintf("L%d_n%d_U%g_g%g_%s", L, NMAX, U, GAMMA, INIT)

const BOP = let b = zeros(ComplexF64, D, D)                      # state k holds k-1 bosons
    for n in 1:NMAX; b[n, n + 1] = sqrt(n); end
    b
end
const NOP = Matrix{ComplexF64}(Diagonal(0:NMAX))
const DOP = NOP * (NOP - I)                                   # n(n-1)

function model_terms(n)
    Jm = zeros(n, n); for i in 1:(n - 1); Jm[i, i + 1] = -J; end
    pairs = Any[(Matrix(BOP'), BOP, Jm), (BOP, Matrix(BOP'), Jm)]
    return pairs, fill(U / 2 * DOP, n), Any[(BOP * BOP, GAMMA)]
end

function model(n)
    pairs, hs, jumps = model_terms(n)
    ls = liouville_space(collect(0:NMAX); sym = :weak)
    return ls, lindblad_mpo(ls, n; pairs = pairs, onsite_h = hs, jumps = jumps)
end

"Initial occupation pattern: `domain` (n=1 on the left half), `mott` (n=1 everywhere), `double`
(n=2 on the central site(s), empty elsewhere)."
function occupations(n)
    INIT == "domain" && return [i <= n ÷ 2 ? 1 : 0 for i in 1:n]
    INIT == "mott"   && return fill(1, n)
    INIT == "double" && return [abs(i - (n + 1) / 2) < 1 ? min(2, NMAX) : 0 for i in 1:n]
    error("OS_INIT must be domain, mott or double, got $INIT")
end
proj(k) = (P = zeros(ComplexF64, D, D); P[k + 1, k + 1] = 1; P)

function observe(ls)
    return function (rho)
        nj = [real(mpdo_expect(rho, ls, Dict(j => NOP))) for j in 1:L]
        dj = [real(mpdo_expect(rho, ls, Dict(j => DOP))) for j in 1:L]
        Dict("n" => nj, "d" => dj, "N" => [sum(nj)], "loss_rate" => [-2GAMMA * sum(dj)])
    end
end

"Exact observables at every output step of every dt (`nothing` beyond OS_REF_MAXD)."
function references()
    D^L <= REF_MAXD || return Dict(dt => nothing for dt in OS.dts)
    pairs, hs, jumps = model_terms(L)
    S = sparse_lindbladian(L, D; pairs = pairs, onsite_h = hs, jumps = jumps)
    say("exact Liouvillian: $(size(S, 1)) states, nnz $(nnz(S))")
    rho0 = kron((proj(k) for k in occupations(L))...)
    out = Dict{Float64, Any}()
    for dt in OS.dts
        ks = os_out_steps(os_nsteps(dt))
        rs = sparse_lindblad_trajectory(S, rho0, ks .* dt)
        r = Dict{Tuple{Int, String}, Vector{Float64}}()
        for (k, R) in zip(ks, rs)
            t = real(tr(R))
            nj = [real(site_expect(R, NOP, j, L, D)) / t for j in 1:L]
            dj = [real(site_expect(R, DOP, j, L, D)) / t for j in 1:L]
            r[(k, "n")], r[(k, "d")] = nj, dj
            r[(k, "N")], r[(k, "loss_rate")] = [sum(nj)], [-2GAMMA * sum(dj)]
        end
        out[dt] = r
    end
    return out
end

function main()
    set_symmetry!(:U1)
    ls, W = model(L)
    Iket = mpdo_identity(ls, L)
    rho0 = mpdo_product_state(ls, [proj(k) for k in occupations(L)])
    ls4, W4 = model(4)
    t = @elapsed os_warmup(ls4, W4, mpdo_product_state(ls4, [proj(1), proj(1), proj(0), proj(0)]),
                           mpdo_identity(ls4, 4))
    say(@sprintf("JIT warm-up (untimed, excluded from every wall): %.1f s", t))
    say("== OS-2 Bose-Hubbard + two-body loss L=$L nmax=$NMAX J=$J U=$U gamma=$GAMMA init=$INIT ($(occupations(L)))  tmax=$(OS.tmax)  dts $(OS.dts)  arms $(OS.arms)  tol_basis $(OS.tbs)  tol_trunc $(OS.tt)  tol_root $(OS.tr)  maxdim $(OS.maxdim)  root $(OS.root)  threads $(Threads.nthreads())")
    t = @elapsed refs = references()
    say(@sprintf("exact sparse reference: %.1f s", t))
    sio, oio, fio = os_open("os2", TAG)
    for dt in OS.dts, arm in OS.arms, (tb, seed) in os_runs(arm)
        f = os_run_arm(sio, oio, TAG, ls, W, rho0, Iket, observe(ls), refs[dt], arm, dt, tb, seed)
        os_final!(fio, TAG, arm, tb, seed, dt, f)
    end
    close(sio); close(oio); close(fio)
    say("OS2-DONE")
end

main()
