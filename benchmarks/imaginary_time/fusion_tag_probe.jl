# DOES `fusion_basis` HONOUR ITS `tag` KEYWORD?  A five-line question with a large consequence.
#
# ⛔ THE SUSPICION. `sectors.jl`:
#
#     function fusion_basis(t, leg_a=1, leg_b=2; tag="fused")
#         F = to_concrete(getIdentity((t, leg_a), (t, leg_b); itag = tag)')
#         return to_concrete(svd(F, (1, 2); cutoff = 0.0).U)      # <- default tags
#     end
#
# `tag` reaches `getIdentity`, and then the SVD RELABELS the bond with Telum's default (`svdL`).
# If so, the keyword is inert: `full_local_basis(tag="cbeF")` and `sector_graded_sketch(tag="cbeG")`
# both return a probe whose bond is `svdL` -- the SAME tag a bond frame carries -- and
# `cbe_core.jl:130` (`contract(_al(), (3,4), Om', (2,3))`) then builds a TLArray with two identical
# `svdL` indices, which Telum refuses:
#
#     AssertionError: Duplicate TLIndex with non-empty itag: TLIndex(svdL, '+', 0, 0, false)
#
# MEASURED: `cbe_bug` FAILS under `:U1` on BOTH the chain and the cylinder, while `tdvp2` (which
# never calls `cbe_expand`) runs fine, and `:SU2` survives on both.
#
# ⚠ VERIFY, DO NOT ASSUME. The docstring says the bond is deliberately "labelled the way
# `svd(t, (leg_a, leg_b)).U` labels its bond", so the overwrite may be INTENTIONAL and the real
# defect elsewhere. This file just prints the tag. It costs no precompile (benchmark file) and
# settles in one line what would otherwise be a ten-minute `src` edit made on a guess.
#
# Run:  julia -t 2 --project=. benchmarks/imaginary_time/fusion_tag_probe.jl

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

say(s) = (println(s); flush(stdout))

say("\n=== does fusion_basis honour `tag`? ===")
say("  If leg 3 comes back `svdL` instead of the requested tag, the keyword is INERT")
say("  and every CBE probe collides with a frame's bond leg.\n")

for sym in (:SU2, :U1, :none)
    set_symmetry!(sym)
    L = 6
    Jm = [i + 1 == j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
    W = pair_mpo(L, Jm, spin_vertices(L))
    psi = dimer_state(L)
    canonical!(psi, 3)
    t = psi[3]                                   # (link_l, site, link_r)
    try
        F = fusion_basis(t, 1, 2; tag = "cbeG")
        tags = [string(F.inds[k].itags) for k in 1:length(F.inds)]
        say(@sprintf("  %-5s fusion_basis(tag=\"cbeG\") legs = %s", string(sym), join(tags, " | ")))
        # ⚠ `tags[k]` IS AN `Itag`, NOT A `String` -- `occursin` on it throws
        # `MethodError: no method matching codeunit(::Itag)`. `string(...)` first.
        say(@sprintf("        leg 3 = %-12s  %s", tags[3],
                     occursin("cbeG", string(tags[3])) ? "tag HONOURED" :
                     "*** TAG IGNORED -- this is the collision ***"))
        # and what a frame's own bond leg is tagged, for comparison
        stags = [string(psi[3].inds[k].itags) for k in 1:3]
        say(@sprintf("        state site 3 legs = %s", join(stags, " | ")))
    catch e
        say(@sprintf("  %-5s THREW: %s", string(sym), sprint(showerror, e)))
    end
end

say("\nDONE")
