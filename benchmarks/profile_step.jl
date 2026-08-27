# WHERE DOES A CBE-BUG STEP ACTUALLY SPEND ITS TIME? -- a STATISTICAL profile, not phase timers.
#
# The phase timers in `CBEBugSweepInfo` (`t_cbe`, `t_split`, ...) say WHICH PHASE, and they close
# to ~0.94 of the step. They cannot say WHICH FUNCTION, and that is the question a "the sketch
# should have won and did not" result actually poses: `cbe` being 45% of the step is only
# actionable once it is known whether those seconds are in the SVD the sketch replaces or in
# something incidental sitting next to it.
#
# ⛔ SELF TIME, NOT TOTAL TIME. A tree profile puts ~100% at `cbe_bug_step!` and says nothing.
# Everything below is the count of samples whose INNERMOST frame is that function, so the numbers
# are exclusive and comparable across the whole run.
#
#   julia --project=. benchmarks/profile_step.jl [model] [chi] [nsteps] [exact] [size...]

using Printf, LinearAlgebra, Profile, Serialization
include(joinpath(@__DIR__, "fused_common.jl"))

const MODEL  = length(ARGS) >= 1 ? ARGS[1] : "xx"
const CHI    = length(ARGS) >= 2 ? parse(Int, ARGS[2]) : 128
const NSTEP  = length(ARGS) >= 3 ? parse(Int, ARGS[3]) : 2
const EXACT  = length(ARGS) >= 4 ? parse(Bool, ARGS[4]) : true
const SIZEA  = length(ARGS) >= 5 ? parse.(Int, split(ARGS[5], ",")) : Int[]
const DT     = 0.05

set_symmetry!(MODEL in ("xx", "square", "kagome") ? :U1 : :none)

function build()
    if MODEL == "xx"
        L = length(SIZEA) >= 1 ? SIZEA[1] : 32
        return (L = L, W = xxz_mpo(L; J = 1.0, delta = 0.0), d = 2,
                psi0 = neel_state(L), tau = ComplexF64(-im * DT), post! = p -> p)
    elseif MODEL == "square"
        LX, LY = length(SIZEA) >= 2 ? (SIZEA[1], SIZEA[2]) : (8, 4)
        L = LX * LY
        return (L = L, W = square_cylinder_mpo(LX, LY), d = 2,
                psi0 = neel_state(L), tau = ComplexF64(-im * DT), post! = p -> p)
    elseif MODEL == "xx_open"
        N = length(SIZEA) >= 1 ? SIZEA[1] : 16
        Jm = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:N, j in 1:N]
        W, shift = fused_lindblad(N, Jm, fill(0.1, N); delta = 0.0)
        imps = identity_superket(N)
        return (L = N, W = W, d = 4,
                psi0 = rho0_product([isodd(k) ? :up : :down for k in 1:N]),
                tau = ComplexF64(DT),
                post! = p -> (apply_shift!(p, exp(DT * shift));
                              restore_trace!(imps, p, N; maxdim = CHI, cutoff = 1e-10); p))
    end
    error("unknown model $MODEL")
end

const S = build()

step!(p, cap) = cbe_bug_step!(p, S.W, S.tau; exact = EXACT, dex = 8, dover = 0,
                              comp_ratio = 1.0, krylov_basis = 3, krylov_tol = 0.0,
                              hermitian = (S.d == 2), maxdim = cap, trunc_thresh = 1e-10,
                              maxiter = 20, tol = 0.0)

function grow_to(target; maxsteps = 400)
    p = copy(S.psi0); last = 0; stalled = 0
    for _ in 1:maxsteps
        step!(p, target); S.post!(p)
        chi = maximum(state_bond_dims(p))
        chi >= target && return p, chi
        stalled = (chi == last) ? stalled + 1 : 0
        stalled > 25 && return p, chi
        last = chi
    end
    return p, maximum(state_bond_dims(p))
end

@printf("model %s  L=%d  d=%d  chi target %d  exact=%s\n", MODEL, S.L, S.d, CHI, EXACT)
let p = copy(S.psi0); step!(p, 16) end            # JIT warmup, never inside the profile

# ⚠ THE GROWTH DOMINATES THE RUN (hundreds of steps to reach the target chi) and is NOT what is
# being profiled, so it is cached. The cache key carries every parameter that changes the state.
const CACHE = joinpath(@__DIR__, "results",
                       "state_$(MODEL)_$(S.L)_$(join(SIZEA, "x"))_chi$(CHI).jls")
mkpath(dirname(CACHE))
p0, chi = if isfile(CACHE)
    st = Serialization.deserialize(CACHE)
    println("reusing cached state from $(basename(CACHE))")
    st
else
    st = grow_to(CHI)
    Serialization.serialize(CACHE, st)
    st
end
@printf("grown to chi = %d, MPO dim %d\n", chi, maximum(mpo_virtual_dims(S.W)))

# one untimed step so every method in the hot path is compiled
let p = copy(p0); step!(p, chi) end

Profile.clear()
Profile.init(n = 20_000_000, delay = 0.0005)
let p = copy(p0)
    @profile for _ in 1:NSTEP
        step!(p, chi)
    end
end

# ── SELF TIME BY FUNCTION ────────────────────────────────────────────────────────────────
#
# ⚠ ALL OF THIS LIVES IN A FUNCTION ON PURPOSE. At top level a `for` body is SOFT SCOPE, so an
# accumulator assigned inside it becomes a fresh local per iteration and the loop throws
# `UndefVarError` -- which is exactly how the first run of this file died.
function report(data, lidict)
    self = Dict{String, Int}()
    total = 0
    # `data` is a flat list of instruction pointers: one backtrace per 0-terminated block,
    # ordered innermost-first, so the first non-C frame of a block is the LEAF.
    blocks = Vector{Vector{UInt64}}()
    cur = UInt64[]
    for ip in data
        if ip == 0
            isempty(cur) || push!(blocks, cur)
            cur = UInt64[]
        else
            push!(cur, ip)
        end
    end
    for b in blocks
        lf = nothing
        for ip in b
            fs = get(lidict, ip, nothing)
            fs === nothing && continue
            for f in (fs isa Vector ? fs : [fs])
                f.from_c && continue
                lf = f
                break
            end
            lf === nothing || break
        end
        lf === nothing && continue
        key = @sprintf("%s  (%s:%d)", string(lf.func), basename(string(lf.file)), lf.line)
        self[key] = get(self, key, 0) + 1
        total += 1
    end

    @printf("\n%d samples with a Julia leaf frame\n\n", total)
    @printf("  %7s  %6s   %s\n", "samples", "self%", "function  (file:line)")
    for (k, v) in first(sort(collect(self); by = last, rev = true), 40)
        @printf("  %7d  %5.1f%%   %s\n", v, 100 * v / max(total, 1), k)
    end

    println("\n-- and the same rolled up by FILE --")
    byfile = Dict{String, Int}()
    for (k, v) in self
        m = match(r"\(([^:]+):", k)
        m === nothing && continue
        byfile[m.captures[1]] = get(byfile, m.captures[1], 0) + v
    end
    for (k, v) in sort(collect(byfile); by = last, rev = true)
        @printf("  %7d  %5.1f%%   %s\n", v, 100 * v / max(total, 1), k)
    end
    return nothing
end
report(Profile.retrieve()...)
