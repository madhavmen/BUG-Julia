# Pins every Telum/LurCGT API surface the bond_update_bug plan depends on.
# Every block prints the observed shape AND asserts, so the job exits nonzero
# if any signature drifts. Output is transcribed into docs/telum_api_contract.md.

using LurCGT, Telum, LinearAlgebra, Printf

failures = String[]
macro pin(desc, ex)
    return quote
        try
            $(esc(ex)) || push!(failures, $desc)
        catch err
            push!(failures, string($desc, " -- threw ", sprint(showerror, err)))
        end
    end
end

hdr(s) = println("\n", "="^70, "\n", s, "\n", "="^70)

# ── 1. U(1) spin-1/2 local space ─────────────────────────────────────────────
# NOTE: SpinOptions.spin is 2*S as an Int, so spin-1/2 is `1`, NOT `1//2`.
# NOTE: the :U1 branch returns (:I, :Sp, :Sz, :Sm). There is no `.S` field --
#       that only exists under :SU2, where Sp/Sz/Sm fuse into one IROP.
hdr("1. getLocalSpace(SpinOptions(:U1, 1), (\"site\",\"site\",\"op\"))")
q = getLocalSpace(SpinOptions(:U1, 1), ("site", "site", "op"))
println("returned fields    : ", keys(q))
println("I  legs            : ", q.I.inds)
println("I  rank            : ", length(q.I.inds))
println("I  leg-1 spaces    : ", q.I.spaces[1])
println("Sz legs            : ", q.Sz.inds)
println("Sz leg-1 spaces    : ", q.Sz.spaces[1])
println("Sp legs            : ", q.Sp.inds)
println("Sp rank            : ", length(q.Sp.inds))
for l in 1:length(q.Sp.inds)
    println("Sp leg-$l spaces    : ", q.Sp.spaces[l])
end
@pin "keys(getLocalSpace(:U1)) == (:I,:Sp,:Sz,:Sm)" Set(keys(q)) == Set((:I, :Sp, :Sz, :Sm))
@pin "I is rank-2"                    length(q.I.inds) == 2
@pin "I has 2 charge sectors"         length(q.I.spaces[1]) == 2
@pin "every U(1) sector is dim 1"     all(d == 1 for (_, d) in q.I.spaces[1])
@pin "leg dirs are (+,-)"             (q.I.inds[1].dir, q.I.inds[2].dir) == ('+', '-')
@pin "Sp carries an operator leg"     length(q.Sp.inds) == 3

# ── 2. Rank-revealing SVD (Telum has no QR; SVD stands in) ───────────────────
hdr("2. svd(q, left_legs; get_lists=true)")
res = svd(q.Sp, (1,); get_lists=true)
println("result type        : ", typeof(res).name.name)
println("fieldnames         : ", fieldnames(typeof(res)))
println("U  legs            : ", res.U.inds)
println("S  legs            : ", res.S.inds)
println("Vd legs            : ", res.Vd.inds)
println("U  leg spaces      : ", res.U.spaces)
println("kept_list          : ", res.kept_list)
println("trunc_list         : ", res.trunc_list)
@pin "SVDResult fields"  fieldnames(typeof(res)) == (:U, :S, :Vd, :kept_list, :trunc_list)
@pin "U rank = |left|+1"  length(res.U.inds) == 2
@pin "kept_list nonempty"  !isempty(res.kept_list)

# U must be an isometry. GOTCHA: `*` is GREEDY -- it contracts every leg pair
# that matches on (itag, plev, dual) with opposite dir and equal spaces. So
# `U' * U` closes the bond leg too and collapses to a rank-0 scalar. To keep
# the bond open you must prime it (or contract explicitly by leg number).
UdU_greedy = res.U' * res.U
println("(U'*U)  legs [greedy]  : ", UdU_greedy.inds, "   <- rank 0, bond closed!")
Up = prime(res.U, 2)                      # prime leg 2 (the new bond) only
UdU = contract(Up', (1,), res.U, (1,))    # contract the site leg alone
println("(U'*U)  legs [primed]  : ", UdU.inds)
println("(U'*U)  spaces         : ", UdU.spaces)
println("norm(U)                : ", norm(res.U))
bond_dim = sum(d for (_, d) in res.U.spaces[end])
@pin "greedy * closes every match"      length(UdU_greedy.inds) == 0
@pin "primed U'U keeps the bond open"   length(UdU.inds) == 2
@pin "U is isometric (norm^2 == rank)"  isapprox(norm(res.U)^2, bond_dim; atol=1e-10)
@pin "U'U == I on the bond"             isapprox(norm(UdU)^2, bond_dim; atol=1e-10)

# Nkeep truncation is how Dmax is enforced in the sweep.
res_trunc = svd(q.Sp, (1,); Nkeep=1, get_lists=true)
println("Nkeep=1 kept_list  : ", res_trunc.kept_list)
println("Nkeep=1 trunc_list : ", res_trunc.trunc_list)
@pin "Nkeep truncates"  length(res_trunc.kept_list) <= length(res.kept_list)

# ── 3. Direct sum -- the [U0 | K1] augmentation primitive ────────────────────
hdr("3. oplus(a, b, dims)")
println("Sz leg-2 spaces before : ", q.Sz.spaces[2])
opl = oplus(q.Sz, q.Sz, (2,))
println("oplus legs             : ", opl.inds)
println("oplus leg-1 spaces     : ", opl.spaces[1])
println("oplus leg-2 spaces     : ", opl.spaces[2])
dim_before = sum(d for (_, d) in q.Sz.spaces[2])
dim_after  = sum(d for (_, d) in opl.spaces[2])
@printf("leg-2 total dim  %d -> %d\n", dim_before, dim_after)
@pin "oplus doubles the summed leg"  dim_after == 2 * dim_before
@pin "oplus leaves other legs alone"  opl.spaces[1] == q.Sz.spaces[1]

# ── 4. Sector extraction -- the missing-QN table primitive ───────────────────
hdr("4. getsub / zero_qlabels")
zq = zero_qlabels(q.I)
site_sectors = [s for (s, _) in q.I.spaces[1]]
println("zero_qlabels(I)    : ", zq, "  ::", typeof(zq))
println("I leg-1 sectors    : ", site_sectors)
# GOTCHA: for U(1) spin-1/2 the site charges are 2*Sz = +/-1, so the VACUUM
# sector ((0,),) is NOT a site sector. Sector arithmetic is on 2*Sz integers.
sub_zero = getsub(q.I, 1, s -> s == zq ? Colon() : nothing)
println("getsub(q==0) spaces: ", sub_zero.spaces[1], "   <- empty; no 0 sector on a site")
up = site_sectors[end]
sub_up = getsub(q.I, 1, s -> s == up ? Colon() : nothing)
println("getsub(q==$up) spaces: ", sub_up.spaces[1])
allsub = getsub(q.I, 1, s -> Colon())
println("getsub(all) spaces : ", allsub.spaces[1])
@pin "zero_qlabels is a sector key"     zq isa Tuple
@pin "vacuum is not a site sector"      isempty(sub_zero.spaces[1])
@pin "getsub selects one sector"        length(sub_up.spaces[1]) == 1
@pin "getsub(all) is lossless"          allsub.spaces[1] == q.I.spaces[1]
@pin "U(1) site charges are 2*Sz = +-1" Set(site_sectors) == Set([((-1,),), ((1,),)])

# ── 5. Contraction / adjoint / norm -- the expv primitives ───────────────────
hdr("5. norm / adjoint / * / contract")
println("norm(I)            : ", norm(q.I))
println("norm(Sz)           : ", norm(q.Sz))
println("adjoint(Sz) legs   : ", q.Sz'.inds)
SzSz = q.Sz' * q.Sz
println("(Sz'*Sz) legs      : ", SzSz.inds)
println("(Sz'*Sz) rank      : ", length(SzSz.inds))
println("tr(Sz'Sz)          : ", norm(SzSz))
# GOTCHA: explicit `contract` does NOT auto-conjugate -- it asserts the two
# contracted legs have OPPOSITE arrow directions. `contract(Sz,(1,2),Sz,(1,2))`
# throws. You must adjoint one side first.
ex = contract(q.Sz', (1, 2), q.Sz, (1, 2))
println("contract(Sz',12,Sz,12) rank : ", length(ex.inds))
# NOTE: this must be a FUNCTION, not a bare top-level try/catch. `catch` opens a
# soft local scope, so `flag = true` inside it does not update a global in a
# script -- an earlier version of this file silently reported "false".
throws(f) = try (f(); false) catch; true end
same_dir_throws = throws(() -> contract(q.Sz, (1, 2), q.Sz, (1, 2)))
println("contract(Sz,12,Sz,12) throws: ", same_dir_throws)
@pin "norm(I)^2 == local dim"        isapprox(norm(q.I)^2, 2.0; atol=1e-10)
@pin "adjoint flips leg dirs"        q.Sz'.inds[1].dir == '-'
@pin "contract needs opposite dirs"  same_dir_throws
@pin "tr(Sz'Sz) == 1/2"              isapprox(norm(SzSz), 0.5; atol=1e-10)

# ── 6. Leg bookkeeping used by the sweep ─────────────────────────────────────
hdr("6. addSingleton / deleteSingleton / getIdentity / getvac / legflip / lock")
vac = getvac(q.I, ("Lbnd", "Rbnd"))
println("getvac legs        : ", vac.inds)
println("getvac spaces      : ", vac.spaces)
@pin "getvac is rank-2"        length(vac.inds) == 2
@pin "getvac is 1-dimensional" all(sum(d for (_, d) in sp) == 1 for sp in vac.spaces)

added = addSingleton(q.I; nlegs=1)
println("addSingleton rank  : ", length(added.inds), "  legs: ", added.inds)
@pin "addSingleton raises rank by 1"  length(added.inds) == length(q.I.inds) + 1
back = deleteSingleton(added, 3)
println("deleteSingleton rank: ", length(back.inds))
@pin "deleteSingleton inverts addSingleton"  length(back.inds) == length(q.I.inds)

idm = getIdentity(q.I, 1)
println("getIdentity legs   : ", idm.inds)
println("getIdentity spaces : ", idm.spaces)
@pin "getIdentity is rank-2"  length(idm.inds) == 2

fl = legflip(q.I, 1)
println("legflip legs       : ", fl.inds)
@pin "legflip changes leg 1"  fl.inds[1] != q.I.inds[1]

lk = lock(q.I, 1)
println("lock legs          : ", lk.inds)
println("lock levels        : ", [i.lock for i in lk.inds])
@pin "lock raises lock level"  lk.inds[1].lock > 0

# ── 7. Element type -- BUG needs complex amplitudes ──────────────────────────
hdr("7. element type / complex support")
println("eltype(q.I)        : ", eltype(q.I))
cz = q.Sz * (1.0 + 0.0im)
println("eltype(Sz*(1+0im)) : ", eltype(cz))
println("norm after cscale  : ", norm(q.Sz * (2.0 + 0.0im)))
@pin "complex scaling works"  isapprox(norm(q.Sz * (2.0 + 0.0im)), 2 * norm(q.Sz); atol=1e-10)

# ── verdict ──────────────────────────────────────────────────────────────────
hdr("VERDICT")
if isempty(failures)
    println("SMOKE_OK -- all pinned signatures verified")
else
    println("SMOKE_FAILED -- ", length(failures), " pin(s) broke:")
    for f in failures
        println("  * ", f)
    end
    exit(1)
end
