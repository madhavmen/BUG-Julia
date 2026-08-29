# THE IDENTITY SUPERKET `|I>>`, WITHOUT HAND-ROLLING A CUP.
#
# ⛔ WHY THE OBVIOUS ROUTE FAILS. `Tr rho` means contracting the two PHYSICAL legs of a (ket, bra)
# pair against each other. Both are ket-type MPS legs, so they carry the SAME arrow, and the
# identity OPERATOR -- one leg each way -- cannot close them. Measured, exactly as `PLAN` §3
# predicted:
#     AssertionError: Contracted legs must have opposite arrow directions:
#                     q1 leg 3 has dir='-', q2 leg 2 has dir='-'
# Hand-rolling the cup is the trap `sectors.jl` warns about ("with a vacuum link the reachable set
# is its own dual, so the mislabelled version still agrees") -- it can PASS while being wrong.
#
# ⛔ SO BUILD `|I>>` OUT OF PIECES THAT ARE ALREADY VALIDATED. A singlet is `(|ud> - |du>)/sqrt2`;
# applying `S+ - S- = i*sigma^y = [0 1; -1 0]` to the BRA site of each pair sends
#     |ud> -> |uu|,   |du> -> -|dd>   =>   (|uu> + |dd>)/sqrt2
# which is one pair of `|I>>`. `_dimer_state_projected` already builds the singlet covering on
# pairs (1,2), (3,4), ... -- precisely our (ket, bra) pairs -- so `|I>>` is one existing
# constructor plus L single-site multiplications.
#
# ⚠ AND THE OPERATOR IS ORTHOGONAL (`O^T O = I`), so it preserves the canonical form: a unitary on
# one site leaves `A'A` unchanged. No re-canonicalisation, and none of the sector-resort trouble
# that writing tensors by hand invites.
#
# ⚠ NORMALISATION. `<<I|I>> = Tr(I) = 2^L`, while the flipped dimer state is NORMALISED. So the
# true identity superket is `2^(L/2)` times it, and every trace carries that factor.
#
# GATES: (0) `dense_state(|I>>)` IS the vectorised identity matrix, component by component -- the
# unambiguous check, which no sign or pairing convention can slip past; (1) `Tr rho0 = 1`;
# (2) `Tr(S^z_j rho0) = +-1/2` SITE BY SITE, on an ASYMMETRIC state, since a mis-paired or
# mis-signed `|I>>` still returns *a* number.
using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
include(joinpath(@__DIR__, "..", "tests", "common", "dense_reference.jl"))
const OUT = joinpath(@__DIR__, "results", "identity_superket.txt"); mkpath(dirname(OUT))
io = open(OUT, "w"); say(l) = (println(io, l); flush(io); println(stdout, l); flush(stdout))
set_symmetry!(:none)

"""
Apply a single-site operator to site `k`, tagging it from the tensor itself.

⛔ CONTRACT THE OPERATOR'S LEG 1, NOT LEG 2. In this codebase an operator carries
`(ket '+', bra '-')` and an MPS site leg is `'-'`, so pairing leg 2 with the site leg is two
`'-'` legs and Telum refuses it outright:
    AssertionError: Contracted legs must have opposite arrow directions:
                    q1 leg 2 has dir='-', q2 leg 2 has dir='-'
The validated path is [`apply_gate`](@ref), which contracts the gate's `(1, 2)` KET legs against
theta's site legs and lets the `'-'` bra legs come out carrying the site tags -- so the one-site
form is leg 1 against the site leg. GATE 0 below is what confirms the resulting orientation, since
this convention delivers the operator's transpose under a naive reading and `S+ - S-` is
antisymmetric.
"""
function site_apply!(p, k::Int, M)
    t = p[k].inds[2].itags
    op = to_concrete(setitag(setitag(M, 1, t), 2, t))
    p[k] = to_concrete(permutedims(contract(op, (1,), p[k], (2,)), (2, 1, 3)))
    return p
end

"`|I>>/2^(L/2)`: the NORMALISED identity superket on the 2L-site interleaved chain."
function identity_superket(L::Int)
    q = local_space()
    flip = to_concrete(q.Sp - q.Sm)              # [0 1; -1 0], orthogonal
    p = dimer_state(2L)                          # singlets on (1,2), (3,4), ...
    for j in 1:L
        site_apply!(p, 2j, flip)                 # the BRA site of pair j
    end
    return p
end

"`Tr(rho)` -- and with `at = j`, `Tr(S^z_j rho)`; `S^z` is Hermitian so it may sit on `<<I|`."
function trace_sk(Imps, rho, L::Int; at::Int = 0)
    o = copy(Imps)
    at > 0 && site_apply!(o, 2at - 1, local_space().Sz)     # the KET site of pair `at`
    return 2.0^(L / 2) * overlap(o, rho)
end

for L in (2, 3, 4)
    N = 2L
    say(@sprintf("── L = %d system sites (%d chain sites) ──", L, N))
    Imps = identity_superket(L)

    # ── GATE 0: is it the vectorised identity, component by component? ────────────────────
    dI = dense_state(Imps) .* 2.0^(L / 2)
    D  = 2^L
    want = zeros(ComplexF64, 4^L)
    for idx in 0:(4^L - 1)
        a = 0; b = 0
        for j in 1:L
            a |= ((idx >> (N - (2j - 1))) & 1) << (L - j)
            b |= ((idx >> (N - 2j))       & 1) << (L - j)
        end
        want[idx + 1] = (a == b) ? 1.0 : 0.0        # vec(I): 1 on the diagonal
    end
    d0 = maximum(abs.(dI .- want))
    say(@sprintf("   GATE 0  |I>> is vec(identity)          : %10.3e  %s",
                 d0, d0 < 1e-12 ? "PASS" : "FAIL"))

    # ── GATES 1 and 2: traces of a PRODUCT rho0, against known values ─────────────────────
    spins = [:up, :down, :up, :up][1:L]
    rho0 = product_state(collect(Iterators.flatten((s, s) for s in spins)))
    t = real(trace_sk(Imps, rho0, L))
    say(@sprintf("   GATE 1  Tr rho0 = %+.12f  (expect +1)  %s", t,
                 isapprox(t, 1.0; atol = 1e-10) ? "PASS" : "FAIL"))
    ok = true
    for j in 1:L
        z = real(trace_sk(Imps, rho0, L; at = j))
        w = spins[j] === :up ? 0.5 : -0.5
        good = isapprox(z, w; atol = 1e-10); ok &= good
        say(@sprintf("   GATE 2  Tr(S^z_%d rho0) = %+.12f  (expect %+.1f)  %s", j, z, w,
                     good ? "PASS" : "FAIL"))
    end
    say("")
end
say("GATE 0 is the one that cannot be fooled: a wrong pairing or a wrong sign changes components")
say("of vec(I), while GATES 1-2 alone would still return plausible numbers.")
close(io)
