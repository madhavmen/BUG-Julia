# CAN WE TAKE `Tr(rho)` AND `Tr(S^z_j rho)` BY CONTRACTION ON THE DOUBLED CHAIN?
#
# Pair `j` lives on chain sites `(2j-1, 2j) = (ket, bra)`. Contract the two tensors over their
# shared link, then contract the two PHYSICAL legs with a local identity -- that is
# `sum_s T[l,s,s,r]`, the trace over that site's `d x d` block. Walking the pairs accumulates a
# bond-to-bond matrix, and no identity-superket MPS is needed anywhere.
#
# ⛔ TAGS ARE READ OFF THE TENSORS, NEVER ASSUMED. An MPS physical leg is SITE-tagged (`1,S`)
# while `local_space()`'s operators carry a generic `s`, so contracting them raises
# "Contracted legs must have matching itags". The dangerous case is the one that does NOT raise:
# `gates.jl` records that an operator tagged for the WRONG site "contracts happily against the
# wrong legs of a symmetric tensor and returns a plausible, silently wrong answer". So the
# identity is built with leg 1 carrying the KET site's tag and leg 2 the BRA site's, both taken
# from the tensors themselves.
#
# ⚠ VALIDATED AGAINST KNOWN VALUES, NOT PLAUSIBLE ONES. `rho0 = |psi><psi|` for a PRODUCT `psi`,
# so `Tr rho0 = 1` and `Tr(S^z_j rho0) = <psi|S^z_j|psi> = +-1/2` SITE BY SITE. A mis-tagged
# contraction still returns a trace -- just of the wrong object -- and only per-site known values
# catch that.

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

const OUT = joinpath(@__DIR__, "results", "trace_test.txt"); mkpath(dirname(OUT))
const IO_ = open(OUT, "w")
say(l::AbstractString) = (println(IO_, l); flush(IO_); println(stdout, l); flush(stdout))

set_symmetry!(:none)
const Q = local_space()

"Trace of the superket, optionally with `S^z` inserted on the KET leg of system site `at`."
function trace_sk(rho, L::Int; at::Int = 0)
    acc = nothing
    for j in 1:L
        A, B = rho[2j - 1], rho[2j]
        tk, tb = A.inds[2].itags, B.inds[2].itags
        if j == at
            Sz = to_concrete(setitag(setitag(Q.Sz, 1, tk), 2, tk))
            A = to_concrete(permutedims(contract(Sz, (2,), A, (2,)), (2, 1, 3)))
        end
        T   = to_concrete(contract(A, (3,), B, (1,)))           # (l, s_ket, s_bra, r)
        Ikb = to_concrete(setitag(setitag(Q.I, 1, tk), 2, tb))  # delta_{s_ket, s_bra}
        M   = to_concrete(contract(T, (2, 3), Ikb, (1, 2)))     # (l, r)
        acc = acc === nothing ? M : to_concrete(contract(acc, (2,), M, (1,)))
    end
    return acc
end

const L = 4
const SPINS = [:up, :down, :up, :up]
rho = product_state(collect(Iterators.flatten((s, s) for s in SPINS)))
say(@sprintf("doubled chain: %d sites for L = %d system sites", length(rho), L))
try
    t = real(only(trace_sk(rho, L)))
    say(@sprintf("  Tr(rho0)          = %+.12f   (expect +1.0)   %s", t,
                 isapprox(t, 1.0; atol = 1e-10) ? "OK" : "MISMATCH"))
    allok = isapprox(t, 1.0; atol = 1e-10)
    for j in 1:L
        z = real(only(trace_sk(rho, L; at = j)))
        want = SPINS[j] === :up ? 0.5 : -0.5
        ok = isapprox(z, want; atol = 1e-10)
        allok &= ok
        say(@sprintf("  Tr(S^z_%d rho0)    = %+.12f   (expect %+.1f)   %s", j, z, want,
                     ok ? "OK" : "MISMATCH"))
    end
    say("")
    say(allok ? "PASS -- the trace contraction is correct; no identity-superket MPS is needed." :
                "FAIL -- the contraction is tracing the wrong object.")
catch e
    say("THREW: " * sprint(showerror, e)[1:min(end, 400)])
end
close(IO_)
