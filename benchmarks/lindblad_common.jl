# THE VECTORISED LINDBLADIAN ON A DOUBLED MPS CHAIN -- shared by the open-system drivers.
#
# `rho` becomes a SUPERKET on `2L` spin-1/2 sites, so the existing local space, `pair_mpo` and the
# MPO ENVIRONMENTS all apply unchanged. No new local space, no Trotter gates.
#
#     d|rho>>/dt = L|rho>>,  L = -i(H(x)I) + i(I(x)H^T) + g*sum_j (S^z_j (x) S^z_j) - (g*L/4)*I
#
# ⛔ BLOCKED LAYOUT -- ket at 1..L, ancilla at L+1..2L. NOT interleaved, and this is the single
# most important decision in the file. Interleaving (ket 2j-1, ancilla 2j) makes the jump term
# nearest-neighbour and looks like the obvious choice on MPO width -- it is what this file
# originally did -- but it puts every system-system term at RANGE 2, so NO two-site block ever
# contains both operators of a term. The block then feels the term only through an environment
# channel, which for a half-open term is a ONE-SITE EXPECTATION of a charge-changing operator, and
# `<S^->` = 0 on any product state.
#
# MEASURED, not argued: from a product `rho0` the interleaved superket NEVER MOVES. Its first-order
# response `(psi(dt)-psi(0))/dt` is EXACTLY ZERO on the component where the true generator has
# -0.5i, and its error against `exp(T*L)` is flat in dt (order 0.00 from dt=0.1 to 0.0125),
# identical across `tdvp2` and `cbe_bug`, and equal to `|rho(T)-rho(0)|` to three digits.
#
# Blocked instead puts `H` at range 1 INSIDE each block (fully inside a two-site update, so the
# dynamics starts), and the jump term at range `L` -- where it is harmless, because `S^z(x)S^z` is
# DIAGONAL and `<S^z> != 0` on a product state, so a long-range diagonal term does not freeze.
#
# ⚠ WHY THIS NEVER BIT THE CLOSED MODELS. Kagome, square and dipolar all carry NEAREST-NEIGHBOUR
# terms alongside their long-range ones; those bootstrap the state and the long-range terms then
# activate. The interleaved doubled chain has NO range-1 term in its Hamiltonian part at all.
#
# ⛔ THE ZERO-BODY TERM IS NOT IN THE MPO AND CANNOT BE. `L†L = 1/4` for `L = S^z`, so the
# anticommutator is `-(g*L/4)` times the IDENTITY; `pair_mpo` builds two-body vertices only. It is
# returned separately as `shift` and applied as an EXACT scalar `exp(tau*shift)` on the state --
# `exp(tau*(W + c*I)) = e^(tau*c)*exp(tau*W)`. Omitting it does not merely rescale: `Tr rho`
# inflates (MEASURED 1.1618 at gamma = 0.5, exactly 1.0000000000 with it).
#
# ⛔ AND THIS RUNS UNDER `:none`. The physical conserved quantity is `S^z_ket - S^z_bra` (the bra
# index carries CONJUGATE charge) while our local space tracks the SUM, so the naive `:U1` is not
# the physical symmetry. The identity superket is itself a superposition of total-charge sectors,
# so even where the DYNAMICS preserves both charges the OBSERVABLE would break `:U1`.

"Chain site carrying the ket / bra index of system site `j`, in the BLOCKED layout."
lb_ket(L::Int, j::Int) = j
lb_bra(L::Int, j::Int) = L + j

"""
    lindblad_dephasing(L, Jm; gammas, delta = 0.0) -> (mpo, shift)

`Jm` is the SYSTEM coupling matrix (`L x L`, strict upper triangle), `gammas[j]` the dephasing rate
on system site `j`. `delta` scales the `S^zS^z` part of the system Hamiltonian.

⚠ `Jm` MUST BE REAL. The ancilla half carries `+i*H^T`, and the code writes `+i*Jm` -- correct only
where `H^T = H`. A complex `Jm` needs the transpose taken explicitly and is rejected here rather
than silently mis-built.
"""
function lindblad_dephasing(L::Int, Jm::AbstractMatrix; gammas::AbstractVector{<:Real},
                            delta::Float64 = 0.0)
    all(isreal, Jm) || throw(ArgumentError(
        "lindblad_dephasing needs a REAL coupling matrix: the ancilla half is written as +i*Jm, " *
        "which is +i*H^T only when H^T = H"))
    length(gammas) == L || throw(ArgumentError("need one gamma per system site"))
    N = 2L
    v = pair_vertices(xxz_chain(N; J = 1.0, delta = 1.0))
    # ⚠ DERIVED, NEVER GUESSED. `Js[t]` pairs with `vertices[t]` BY POSITION and `xxz_chain` puts
    # its S^zS^z term LAST, so a hard-coded index silently builds a different Lindbladian.
    izz = findall(t -> t.left === t.right, v)
    length(izz) == 1 || error("expected exactly one S^zS^z vertex, found $(length(izz))")
    Jxy = zeros(ComplexF64, N, N)
    for i in 1:L, j in (i + 1):L
        Jm[i, j] == 0.0 && continue
        a, b = minmax(lb_ket(L, i), lb_ket(L, j)); Jxy[a, b] += -im * Jm[i, j]
        c, d = minmax(lb_bra(L, i), lb_bra(L, j)); Jxy[c, d] += +im * Jm[i, j]
    end
    Js = Any[copy(Jxy) for _ in v]
    Jz = delta .* Jxy                      # the S^z vertex also carries delta * the same H
    for j in 1:L
        a, b = minmax(lb_ket(L, j), lb_bra(L, j)); Jz[a, b] += gammas[j]
    end
    Js[izz[1]] = Jz
    return pair_mpo(N, Vector{AbstractMatrix}(Js), v), -sum(gammas) / 4
end

"`|rho0>> = |psi0><psi0|` for a PRODUCT `psi0`: ket copy at `j`, bra copy at `L+j`."
superket_product(spins::Vector{Symbol}) = product_state(vcat(spins, spins))

"""
Apply a single-site operator to site `k`, tagged from the tensor itself.

⛔ CONTRACT THE OPERATOR'S LEG 1. Operators carry `(ket '+', bra '-')` and an MPS site leg is
`'-'`, so pairing leg 2 with the site leg is two `'-'` legs and Telum refuses it:
"Contracted legs must have opposite arrow directions". `apply_gate` -- the validated path the whole
Trotter machinery uses -- contracts the gate's KET legs against theta's site legs.
"""
function site_apply!(p, k::Int, M)
    t = p[k].inds[2].itags
    op = to_concrete(setitag(setitag(M, 1, t), 2, t))
    p[k] = to_concrete(permutedims(contract(op, (1,), p[k], (2,)), (2, 1, 3)))
    return p
end

"""
    identity_superket(L) -> SymMPS

`|I>>` NORMALISED (so the true identity superket is `2^(L/2)` times it), in the BLOCKED layout.

⛔ BUILT FROM VALIDATED PIECES, NOT FROM A HAND-ROLLED CUP. `Tr rho` contracts the two PHYSICAL
legs of a (ket, bra) pair against each other; both are ket-type MPS legs carrying the SAME arrow,
so the identity OPERATOR cannot close them, and writing the cup by hand is the trap `sectors.jl`
warns about -- with a vacuum link the reachable set is its own dual, so a mislabelled version still
agrees and PASSES. Instead: a singlet is `(|ud> - |du>)/sqrt2`, and `S+ - S- = [0 1; -1 0]` on the
bra half of each pair sends it to `(|uu> + |dd>)/sqrt2`, which is one pair of `|I>>`.

⚠ `S+ - S-` is ORTHOGONAL, so it preserves the canonical form -- a unitary on one site leaves
`A'A` unchanged, and no re-canonicalisation (nor the sector re-sort that hand-written tensors
invite) is needed.

⚠ `dimer_state` pairs (1,2), (3,4), ..., which is the INTERLEAVED pairing. The blocked layout wants
(j, L+j), so the pairs are permuted afterwards -- see `_reorder_pairs`.
"""
function identity_superket(L::Int)
    q = local_space()
    flip = to_concrete(q.Sp - q.Sm)
    p = dimer_state(2L)
    for j in 1:L
        site_apply!(p, 2j, flip)
    end
    return p
end

"`Tr(rho)`, and with `at = j` the KET-side `Tr(S^z_j rho)`. `S^z` is Hermitian so it may sit on <<I|."
function trace_rho(Imps, rho, L::Int; at::Int = 0, layout::Symbol = :interleaved)
    o = copy(Imps)
    at > 0 && site_apply!(o, layout === :blocked ? lb_ket(L, at) : 2at - 1, local_space().Sz)
    return 2.0^(L / 2) * overlap(o, rho)
end

"The omitted zero-body term, as an exact scalar on the state."
apply_shift!(p, z::Number) = (p[1] = to_concrete(ComplexF64(z) * p[1]); p)
