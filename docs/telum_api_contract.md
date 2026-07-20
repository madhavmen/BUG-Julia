# Telum / LurCGT API contract

Every signature and return shape below was **executed and printed** on a compute
node, not read off documentation. Later tasks cite this file instead of guessing.

Reproduce with:

```bash
cd /home/madhav.menon/BUG-Julia
sbatch --job-name=telum_smoke scripts/run_julia.sbatch scripts/telum_smoke.jl   # asserts; SMOKE_OK
sbatch --job-name=telum_probe scripts/run_julia.sbatch scripts/telum_probe.jl   # prints; PROBE_DONE
```

`telum_smoke.jl` is self-verifying: every claim here is backed by a `@pin`
assertion, and the job exits 1 if any signature drifts.

| | |
|---|---|
| Julia | 1.12.5 |
| Telum | v0.2.0 — `https://github.com/ssblee/Telum.jl#main` (UUID `206e1281-4cfe-4f93-80c6-84968ba7c8d5`) |
| LurCGT | v1.0.0 — `https://github.com/lurlurlurrrrr/LurCGT.jl#main` (UUID `4f3ab192-95b7-495a-a0ca-04b4b33dd828`) |
| Verified | job 93380 (probe) + 93381 (smoke), both `COMPLETED` |

> The two UUIDs above are the **resolved** ones from `Pkg.status()`. Do not
> hand-write package UUIDs into `Project.toml`; add by URL and let Pkg resolve.

---

## 1. Local space

```julia
q = getLocalSpace(SpinOptions(:U1, 1), ("site", "site", "op"))
```

**`SpinOptions(symmetry, spin)` takes `spin::Int` = 2·S.** Spin-½ is `1`, not `1//2`.

The `:U1` branch returns a `NamedTuple` with fields **`(:Sz, :I, :Sm, :Sp)`**.
There is **no `.S` field** — that exists only under `:SU2`, where Sp/Sz/Sm fuse
into a single IROP.

| operator | rank | legs | leg-3 (operator) charge |
|---|---|---|---|
| `q.I`  | 2 | `(site '+', site '-')` | — |
| `q.Sz` | 2 | `(site '+', site '-')` | — |
| `q.Sp` | 3 | `(site '+', site '-', op '-')` | `((2,),)` |
| `q.Sm` | 3 | `(site '+', site '-', op '-')` | `((-2,),)` |

Leg directions on a local operator are `('+', '-')` — incoming then outgoing.

Checks: `norm(q.I)^2 == 2` (local dim), `norm(q.Sz) == 1/√2`.

---

## 2. Sector labels (qlabels)

A sector key is a **tuple over symmetries of a tuple of ints**:

```
((-1,),)   # one symmetry (U1), charge -1
```

**U(1) spin-½ site charges are 2·Sz = ±1, not ±½.** All sector arithmetic is on
even/odd integers. Consequences that bite:

- `zero_qlabels(q)` returns `((0,),)`, which is **not** a site sector.
  `getsub(q.I, 1, s -> s == zero_qlabels(q) ? Colon() : nothing)` returns an
  **empty** space list. A two-site block *can* carry charge 0 (up⊗down).
- Fusion is per-symmetry on plain `Int`s:
  ```julia
  add_qn(U1, 1,  1) == 2      # NOT add_qn(U1, (1,), (1,)) -- that MethodErrors
  add_qn(U1, 1, -1) == 0
  fuse(a, b) = ((add_qn(U1, a[1][1], b[1][1]),),)   # sector-tuple wrapper
  ```
  `add_qn(::Type{Z{N}}, q1::Int, q2::Int)` reduces mod N; `add_qn(::Type{U1}, ...)`
  is plain addition.

**Task 8 (reachable-sector enumeration) must use `add_qn` on unwrapped ints and
re-wrap, and must never assume the vacuum sector appears on a physical leg.**

---

## 3. `TLArray` anatomy

```julia
struct TLArray{T, QD, N, RD, QT, PS, M, RMT}
    qlabels ::Vector{NTuple{QD, QT}}          # one entry per stored sector
    RMTs    ::Vector{RMT}                     # reduced matrix elements
    inds    ::NTuple{QD, TLIndex}             # one per leg
    spaces  ::NTuple{QD, Vector{Tuple{QT,Int}}}  # per leg: [(sector, dim), ...]
    # + wmatdata / wmatinfo / isdefined / iszero
end
```

`QD` = tensor rank, `N` = number of symmetries, `T` = element type.

```julia
struct TLIndex
    itags ::Itag     # tag string, like an ITensor tag
    dir   ::Char     # '+' incoming, '-' outgoing
    plev  ::Int      # prime level
    lock  ::Int      # >0 blocks contraction; decremented after each contraction
    dual  ::Bool
end
```

`TLIndex` equality ignores `lock`: two indices are `==` iff `(itags, dir, plev, dual)` match.

Useful derived quantities:

```julia
length(t.inds)                          # rank
sum(d for (_, d) in t.spaces[l])        # total dimension of leg l
[s for (s, _) in t.spaces[l]]           # sectors present on leg l
eltype(t)                               # Float64 by default; ComplexF64 after complex scaling
```

Rank-0 results carry the scalar in `t[]`:

```julia
s0 = contract(q.Sz', (1,2), q.Sz, (1,2))
s0[]      # 0.5   -- signed/complex value
norm(s0)  # 0.5   -- magnitude only
```

**Use `t[]`, not `norm(t)`, for inner products and expectation values** — `norm`
throws away the sign and the phase.

---

## 4. SVD (Telum has no QR — SVD stands in everywhere)

```julia
svd(q, left_legs, left_tag="svdL", right_tag="svdR";
    cutoff::Float64=1e-12, Nkeep::Union{Nothing,Int}=nothing, get_lists::Bool=false)
    -> SVDResult
```

`left_legs` must be **sorted and unique**. Returns

```julia
struct SVDResult
    U, S, Vd
    kept_list   # Vector{Tuple{Float64, Int, qlabel, Int}} = (σ, degeneracy, sector, rank)
    trunc_list  # same shape, for the discarded values
end
```

- `U` has rank `length(left_legs) + 1`; the new bond leg is **last**, tagged
  `"svdL"`, direction `'-'`.
- `Vd` has the new bond **first**, tagged `"svdR"`, direction `'-'`.
- Reconstruction is exact and chains through `*`:
  `norm(res.U * res.S * res.Vd - q) == 0.0` (measured, rank-3 bipartition).
- `Nkeep` enforces a hard bond dimension — this is how `Dmax` is applied in the sweep.
- `kept_list` / `trunc_list` are only populated when `get_lists=true`.
- **The SVD is symmetry-native**: `U.spaces[end]` lists only the sectors that
  actually carry weight. A rank-3 `svd(q.Sp, (1,2))` returns a single-sector
  bond `[(((2,),), 1)]` — the operator charge — not a padded full-dimension bond.

`U` is an isometry: `norm(U)^2 == sum(d for (_,d) in U.spaces[end])`.

---

## 5. `oplus` — the `[U0 | K1]` augmentation primitive

```julia
oplus(q1::AbstractTLArray, q2::AbstractTLArray, dimensions) -> TLArray
oplus(qs::AbstractVector, dimensions) -> TLArray
```

`dimensions` selects which legs are direct-summed; every other leg must match and
is passed through unchanged.

```julia
q.Sz.spaces[2]              # [(((-1,),),1), (((1,),),1)]   total dim 2
oplus(q.Sz, q.Sz, (2,)).spaces[2]   # [(((-1,),),2), (((1,),),2)] total dim 4
```

Summing is **per sector**: each sector's dimension adds. Sectors present in only
one operand appear at their own dimension. This is exactly the Sulz augmentation
`[U0 | K1]` at rank ≤ 2r — **it does not pad absent sectors**, which is the
scaling property Task 9 depends on.

---

## 6. `getsub` — sector selection

```julia
getsub(q, leg::Integer, pred::Function; preserve_space::Bool=false)
getsub(q, legs::LegList, pred::Function; preserve_space::Bool=false)
getsub(q, pred::Function; preserve_space=false, dir=, itag=, plev=, lock=, rev=)
```

`pred(sector)` returns:

| return | meaning |
|---|---|
| `nothing` | drop the sector |
| `Colon()` | keep the whole sector |
| `Int` / range / tuple / vector | keep those indices within the sector |

`preserve_space=false` (default) truncates `q.spaces[leg]` to the retained
sectors. `preserve_space=true` keeps every cached space list intact and only
filters stored sectors — it **rejects index-selection returns**, so `pred` must
keep whole sectors.

`getsub(q, l, s -> Colon())` is lossless (verified).

---

## 7. Contraction — two rules that will bite

### 7a. `*` is GREEDY

`q1 * q2` contracts **every** leg pair where
`inds1[i] == change_dir(inds2[j])` **and** `spaces1[i] == spaces2[j]`, over legs
with a non-empty tag and `lock == 0`. Ambiguity (one tag matching two legs) errors out.

Because `adjoint` flips directions, `U' * U` on an isometry closes the bond too
and collapses to a **rank-0 scalar** — not the bond-space identity:

```julia
(res.U' * res.U).inds        # ()  <-- bond closed!
```

To keep a leg open, prime it (or contract explicitly by leg number):

```julia
Up  = prime(res.U, 2)                     # prime the bond leg only
UdU = contract(Up', (1,), res.U, (1,))    # contract the site leg alone
UdU.inds                                   # 2 legs; norm(UdU)^2 == bond dim
```

**Every environment/overlap contraction in Tasks 6, 7, 10 and 11 must prime or
explicitly index its open legs.** Relying on `*` alone silently traces them out.

### 7b. `contract` does NOT auto-conjugate

```julia
contract(q1, legs1::NTuple{CN,Int}, q2, legs2::NTuple{CN,Int};
         reduce_lock::Bool=true, verify_legs::Bool=true)
contract(q1, l1::Int, q2, l2::Int)          # convenience
```

It **asserts opposite arrow directions** on each contracted pair:

```
contract(Sz,(1,2),Sz,(1,2))     -> AssertionError: Contracted legs must have
                                   opposite arrow directions
contract(Sz,(1,),Sz,(1,))       -> AssertionError  (both '+')
contract(Sz,(1,),Sz,(2,))       -> OK, rank 2      ('+' against '-')
contract(Sz',(1,2),Sz,(1,2))    -> OK, rank 0
```

Adjoint one side first. `adjoint(q) === conj(q)`: it conjugates values and flips
leg directions but **does not permute legs**.

---

## 8. Arithmetic (needed by `expv`, Task 7)

```julia
q1 + q2      # requires matching QD, N, RD, qlabel type, symmetry, w-matrix width
q1 - q2      # == q1 + (-1 * q2)
q * scalar   # lazy TLArrayView; promotes eltype
-q           # == q * -1
norm(q)      # Frobenius
permutedims(q, perm)   # norm-preserving; perm[new] = old
```

Verified: `norm(Sz+Sz) == 2·norm(Sz)`, `norm(Sz-Sz) == 0.0`,
`eltype(im*Sz + 2*I) == ComplexF64`.

Adding tensors with **different leg structure throws**:

```
Sz + Sp -> ArgumentError: TLArray sum requires matching QD, N, RD, qlabel type,
           product symmetry, and w-matrix tuple width
```

so a Krylov basis must be leg-aligned before any `axpy`-style combination.
Real→complex promotion happens through scaling: multiply by `(1.0 + 0.0im)`.

---

## 9. Leg bookkeeping

```julia
getvac(q, ("Lbnd","Rbnd"))    # rank-2, one trivial sector, dim 1 per leg
                              #   spaces: ([(((0,),),1)], [(((0,),),1)])
addSingleton(q; nlegs=1)      # append a dim-1 vacuum leg (untagged, '+')
addSingleton(q, legs; ...)    # at chosen positions
deleteSingleton(q, leg)       # inverse
getIdentity(q, leg)           # rank-2 identity carrying leg's space
legflip(q, leg)               # toggles `dual` on that leg
prime(q, leg; inc=1)          # plev += inc on one leg
prime(q; inc=, dir=, itag=, plev=, lock=, rev=)   # keyword leg selection
lock(q, leg)                  # lock += 1; locked legs are skipped by `*`
setitag(q, leg, "tag")        # rename a leg
empty_tlarray(q; T=ComplexF64)  # same leg structure, no stored sectors
```

`getvac` is the boundary bond for an MPS: one sector `((0,),)` of dimension 1.
`addSingleton` is the cheap route to a rank-3 boundary site tensor —
`addSingleton(q.I; nlegs=1)` gives spaces
`([±1 sectors], [±1 sectors], [(((0,),),1)])`.

---

## 10. Gotcha summary

| # | Trap | Correct form |
|---|---|---|
| 1 | `SpinOptions(:U1, 1//2)` | `SpinOptions(:U1, 1)` — arg is 2·S, an `Int` |
| 2 | `q.S` under `:U1` | `q.Sz` / `q.Sp` / `q.Sm` / `q.I`; `.S` is `:SU2`-only |
| 3 | site charges are ±½ | they are 2·Sz = ±1 |
| 4 | vacuum `((0,),)` is a site sector | it is not; `getsub` returns empty |
| 5 | `add_qn(U1, (1,), (1,))` | `add_qn(U1, 1, 1)` — plain `Int`s |
| 6 | `U' * U` gives bond identity | it gives a rank-0 scalar; prime the bond first |
| 7 | `contract` conjugates for you | it asserts opposite dirs; adjoint one side |
| 8 | `norm(t)` for an expectation value | `t[]` on the rank-0 result |
| 9 | `adjoint` permutes legs | it only conjugates + flips directions |
| 10 | `q1 + q2` on mismatched legs | align leg structure first, or it throws |

An eleventh trap is in the *harness*, not Telum: a bare top-level `try/catch`
that sets a flag inside `catch` will not update a global in a script (soft
scope), so a probe can silently report the wrong answer. Wrap it in a function.
This bit the first version of `telum_smoke.jl`.

---

## 11. Mapping to the plan

| plan needs | Telum call | notes |
|---|---|---|
| orthonormal frame (QR) | `svd(t, left_legs; cutoff=)` → `res.U` | no QR in Telum; `U` is the frame |
| bond truncation to `Dmax` | `svd(...; Nkeep=Dmax)` | `trunc_list` gives discarded weight |
| `[U0 \| K1]` augmentation | `oplus(U0, K1, (bond_leg,))` | per-sector, rank ≤ 2r, no padding |
| reachable-sector table | `t.spaces[l]` + `add_qn` | see §2 |
| missing-QN detection | compare `spaces` of `U0`/`K1` against reachable set | absent sector ⇒ needs the random seed |
| environment contraction | `contract` with primed open legs | see §7a |
| Krylov / Arnoldi vectors | `+`, `*scalar`, `norm`, `t[]` | complex via `*(1.0+0.0im)` |
| `⟨ψ\|O\|ψ⟩` | rank-0 `contract` then `t[]` | never `norm` |
