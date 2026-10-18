# OS-1 (imaginary-time / open-system plan, Task 5): BOUNDARY-DRIVEN XX CHAIN + BULK DEPHASING as a
# weak-U(1) vectorised density matrix (`liouville.jl`, label q_ket - q_bra).
#
# CONVENTIONS, EXPLICIT (plan: "make spin and Lindblad-rate conventions explicit"):
#   H   = J sum_j (s+_j s-_{j+1} + h.c.)  with s+ = |up><down| (Pauli raising), i.e. by Jordan-Wigner
#         (up = occupied) the fermion hopping J sum_j (c'_j c_{j+1} + h.c.)
#   L   = sqrt(eps (1 + mu)/2) s+_1, sqrt(eps (1 - mu)/2) s-_1   (left: net gain for mu > 0)
#         sqrt(eps (1 - mu)/2) s+_L, sqrt(eps (1 + mu)/2) s-_L   (right: mu_R = -mu, net loss)
#         sqrt(gamma) n_j on every site (dephasing; gamma = 0 switches it off)
#   a jump `(Jop, g)` of `lindblad_mpo` is D[sqrt(g) Jop]; `m_j = <n_j> - 1/2 = <S^z_j>`, and the
#   bond current `j_k = -2 J Im <c'_k c_{k+1}> = -2 J Im <s+_k s-_{k+1}>` (continuity: dn_k/dt =
#   j_{k-1} - j_k at gamma = 0 and with dephasing alike, which conserves n).
#
# REFERENCE: EXACT at every size by the closed covariance ODE (`xx_open_covariance`, pinned against
# the dense Liouvillian in test_liouville.jl), up to OS_REF_MAXL (N^2 x N^2 propagator: 4096^2 at 64).
# At gamma = 0 the NESS current is the published closed form `xx_boundary_current_published`; with
# dephasing the NESS is the exact `xx_open_ness` (Znidaric's diffusive regime).
#
# Output (OS_OUT, tag L<L>_g<gamma>): os1_steps_*.csv per step (trace, purity, chi, status, tdvp2's
# backward amplification, Hermiticity / min eigenvalue at output times), os1_obs_*.csv the m_j and
# j_k profiles at the output times with the exact values (heatmaps, j(t)), os1_final_*.csv the worst
# errors, os1_ness_*.csv the exact NESS profiles and current.
#
# Run: julia -t 4 --project=PKG benchmarks/open_system/os1_driven_xx.jl
# Env: OS_L (8) OS_J (1.0) OS_EPS (1.0) OS_MU (1.0) OS_GAMMA (0.0) OS_INIT (mixed | neel)
#      OS_REF_MAXL (64), plus os_common.jl's knobs

include(joinpath(@__DIR__, "os_common.jl"))
include(joinpath(@__DIR__, "..", "..", "tests", "common", "free_fermion.jl"))

const L      = parse(Int, get(ENV, "OS_L", "8"))
const J      = parse(Float64, get(ENV, "OS_J", "1.0"))
const EPS    = parse(Float64, get(ENV, "OS_EPS", "1.0"))
const MU     = parse(Float64, get(ENV, "OS_MU", "1.0"))
const GAMMA  = parse(Float64, get(ENV, "OS_GAMMA", "0.0"))
const INIT   = get(ENV, "OS_INIT", "mixed")
const REF_MAXL = parse(Int, get(ENV, "OS_REF_MAXL", "64"))
const TAG    = @sprintf("L%d_g%g", L, GAMMA)

const SP = ComplexF64[0 1; 0 0]                 # s+ (state 1 = up = occupied)
const SM = ComplexF64[0 0; 1 0]
const NU = ComplexF64[1 0; 0 0]
const CKW = (J = J, gamma = GAMMA, eps_L = EPS, eps_R = EPS, mu_L = MU, mu_R = -MU)

function model(n)
    Jm = zeros(n, n); for i in 1:(n - 1); Jm[i, i + 1] = J; end
    at(i, g) = [k == i ? g : 0.0 for k in 1:n]
    jumps = Any[(SP, at(1, EPS * (1 + MU) / 2)), (SM, at(1, EPS * (1 - MU) / 2)),
                (SP, at(n, EPS * (1 - MU) / 2)), (SM, at(n, EPS * (1 + MU) / 2))]
    GAMMA > 0 && push!(jumps, (NU, GAMMA))
    ls = liouville_space([1, -1]; sym = :weak)
    return ls, lindblad_mpo(ls, n; pairs = Any[(SP, SM, Jm), (SM, SP, Jm)], jumps = jumps)
end

initial_rhos(n) = INIT == "neel" ? [isodd(i) ? NU : I - NU for i in 1:n] :
                  INIT == "mixed" ? [Matrix{ComplexF64}(I, 2, 2) / 2 for _ in 1:n] :
                  error("OS_INIT must be mixed or neel, got $INIT")

"`m_j` and the bond currents `j_k` of an MPDO (trace-normalised)."
function observe(ls)
    return rho -> Dict(
        "m" => [real(mpdo_expect(rho, ls, Dict(j => NU))) - 0.5 for j in 1:L],
        "j" => [-2J * imag(mpdo_expect(rho, ls, Dict(k => SP, k + 1 => SM))) for k in 1:(L - 1)])
end

cov_obs(C) = Dict("m" => real.(diag(C)) .- 0.5,
                  "j" => [-2J * imag(C[k, k + 1]) for k in 1:(L - 1)])

"Exact `m`, `j` at every output step of every dt (`nothing` beyond OS_REF_MAXL)."
function references()
    L <= REF_MAXL || return Dict(dt => nothing for dt in OS.dts)
    C0 = INIT == "neel" ? Diagonal([isodd(i) ? 1.0 : 0.0 for i in 1:L]) : nothing
    out = Dict{Float64, Any}()
    for dt in OS.dts
        ks = os_out_steps(os_nsteps(dt))
        Cs = xx_open_covariance(L, ks .* dt; C0 = C0, CKW...)
        r = Dict{Tuple{Int, String}, Vector{Float64}}()
        for (k, C) in zip(ks, Cs), (name, v) in cov_obs(C)
            r[(k, name)] = v
        end
        out[dt] = r
    end
    return out
end

function main()
    mkpath(OS.out)                     # the NESS file below is written before os_open
    set_symmetry!(:U1)
    ls, W = model(L)
    Iket = mpdo_identity(ls, L)
    rho0 = mpdo_product_state(ls, initial_rhos(L))
    ls6, W6 = model(6)
    t = @elapsed os_warmup(ls6, W6, mpdo_product_state(ls6, [isodd(i) ? NU : I - NU for i in 1:6]),
                           mpdo_identity(ls6, 6))
    say(@sprintf("JIT warm-up (untimed, excluded from every wall): %.1f s", t))
    say("== OS-1 driven XX + dephasing L=$L J=$J eps=$EPS mu=$MU gamma=$GAMMA init=$INIT  tmax=$(OS.tmax)  dts $(OS.dts)  arms $(OS.arms)  tol_basis $(OS.tbs)  tol_trunc $(OS.tt)  tol_root $(OS.tr)  maxdim $(OS.maxdim)  root $(OS.root)  threads $(Threads.nthreads())")
    t = @elapsed refs = references()
    say(@sprintf("exact covariance reference: %.1f s", t))
    if L <= REF_MAXL
        open(joinpath(OS.out, "os1_ness_$TAG.csv"), "w") do io
            println(io, "obs,i,value")
            for (name, v) in cov_obs(xx_open_ness(L; CKW...)), (i, x) in pairs(v)
                println(io, "$name,$i,$x")
            end
            GAMMA == 0 && println(io, "j_published,0,$(xx_boundary_current_published(; J = J, eps = EPS, mu_L = MU, mu_R = -MU))")
        end
    end
    sio, oio, fio = os_open("os1", TAG)
    for dt in OS.dts, arm in OS.arms, (tb, seed) in os_runs(arm)
        f = os_run_arm(sio, oio, TAG, ls, W, rho0, Iket, observe(ls), refs[dt], arm, dt, tb, seed)
        os_final!(fio, TAG, arm, tb, seed, dt, f)
    end
    close(sio); close(oio); close(fio)
    say("OS1-DONE")
end

main()
