# CBE-BUG: controlled bond expansion as the BUG basis update

Design spec for the exploratory integrator. Status: **implemented and verified**
(stages 1-3 + driver). The accuracy/rank/cost comparison is deliberately NOT done here —
see "What is verified" below.

## What is verified, and what is not

Verified (886 assertions, `tests/RSVDCBEBondUpdate/`, jobs 94952-94982). Every claim is
pinned against something that shares no code with the thing under test — usually
`dense_heisenberg`/`dense_state`, since in symmetric-tensor code a leg/arrow/prime slip
does not throw, it silently traces the wrong index pair and returns a plausible number:

| claim | how it is pinned |
|---|---|
| the term list is the same operator as the gates | `env_energy` == dense matrix element, both symmetry modes |
| the two-site effective action is right | **off-diagonal** vs dense (a diagonal-only check passes with an environment on the wrong leg) |
| the sketch never needs `HΘ` | `sketch_h_left(…,Ω)` == `contract(apply_h_two_site(Θ),Ω')` for arbitrary `Ω` |
| CBE reaches sectors K/L cannot | A/B: the K/L complement provably stays inside `U0`'s sectors; CBE opens the missing one, no fill in the call |
| the expansion is a frame, and lossless | `left_isometry_defect` ~ 0; `U_ex†Θ₀V_ex†` rebuilds `Θ₀`; `‖S‖ == ‖S₀‖` (no padding needed) |
| **the step is right** | **exact at full rank: one step == `exp(-iH dt)` at L=4; 2.3e-15 over 8 steps at L=6** |
| the cheap path computes the right operator | `apply_zero_site` == `U_ex†·apply_h_two_site(U_ex S V_ex)·V_ex†` |
| no rank-4 tensor in the Krylov loop | structural: every tensor in `ZeroSiteH` is rank ≤ 3, `apply_zero_site` returns rank 2 |
| `dex` is a usable knob | more expansion never increases the error |
| the driver only does bookkeeping | equals a hand-rolled loop threading the same RNG |
| the carried environments are the rebuilt ones | link-by-link, both sides; and mid-sweep against a stack rebuilt from the updated state |

**Not** verified here, and deliberately so: accuracy vs rank, order in `dt`, and wall-clock
cost. Both CBE-BUG and 2-site TDVP are *exact* once the manifold is the whole space
(measured ~1e-13 at L=6/`maxdim`=32), so at these sizes the rank cap has to be pushed so
low to manufacture an error that the truncation floor dominates whatever is being measured.
Those comparisons belong to a large-system run where the cap binds for physical reasons.
`benchmarks/cbe_bug_vs_baselines.jl` is the harness for it, parked unrun.

Reference implementation this follows (von Delft group, QSMPSLib, `Lib_MPS_MPO/`):

| file | what it is |
|---|---|
| `RSVDpreBE0SiQS.m` | randomized-SVD bond expansion before a 0-site (bond) update, `dwqu 2025-10-31` |
| `CBE1SiQS.m` | the deterministic CBE (preselect + shrewd select) of Gleis/Li/von Delft |
| `TDVPSweepCBE1Si.m` | CBE-TDVP sweep: expand bond, forward 1-site update, backward 0-site update |
| `TDVPSweepOneSite.m` | the same sweep without CBE — the baseline |
| `LanczosUpdateCBE0SiQS.m` | expand-then-bond-update driver, `BEfun` defaults to `RSVDpreBE0SiQS` |

Local clone used while writing this:
`/tmp/.../scratchpad/QSMPSLib` (private repo; reachable with the stored
`gitlab.physik.uni-muenchen.de` credential — the `gitlab.physik.lmu.de` URL is the
same server under a different host name).

---

## 1. What is wrong with the current bond update

`kls_bond_update` (`src/BondUpdateBUG/kls_step.jl`) gets its new basis directions by
**integrating** two ODEs:

    K1 = expv(x -> P⊥_U0 · H_K(x), tau, U0·S0)     # V0 frozen
    L1 = expv(x -> P⊥_V0 · H_L(x), tau, S0·V0)     # U0 frozen

and then admits every direction of `orth(P⊥_U0 K1)` into the frame, bounded only by
Sulz's `rank ≤ 2r`. Three problems, all of which the CBE machinery addresses:

1. **The frozen opposite frame restricts what the K/L step can see.** `H_K` holds
   `V0` fixed, so the generator is `V0† H V0` — the new directions are those the
   *current* right frame can already resolve. This is the mechanism behind the
   rank-1 freeze documented in `kls_step.jl`: on a product state the off-diagonal
   generator projects to zero and the frame cannot grow at all.

2. **The discarded projector does not buy the symmetry sectors.** Under U(1) with
   the opposite frame frozen, charge conservation keeps `K1` *inside* `U0`'s charge
   sectors (`sectors.jl` header). A sector that is reachable on `link ⊗ site` but
   populated by neither `U0` nor `K1` is invisible to the augmentation forever. The
   current workaround is `random_sector_seed` — **random** orthonormal columns in
   each empty-but-reachable sector (`augment.jl:35`), pruned afterwards by
   `pairable_charges` and the truncating SVD. It works, but the directions carry no
   information about the dynamics, and `aug_k` pays for every one of them.

3. **Cost.** Two Arnoldi solves per bond (non-Hermitian, because the projector is
   applied before the exponential), plus a Galerkin S-step at rank `2r`.

## 2. The CBE replacement

CBE answers exactly question 2: *which directions outside the current frame does `H`
actually populate?* It answers it with `H` itself, so the charge sectors it opens are
the ones the dynamics needs — no random seeding. `RSVDpreBE0SiQS` computes it in two
stages:

**Preselection** (`RSVDpreBE0SiQS.m:336`). Draw a Gaussian random block `Ω` with
`Dpre ≈ 1.2·Dex` columns, apply `H` once through the environments, project off the
current frame, orthonormalize:

    U0L = orth( P⊥_{TLO} · (H · Ω) )        Dpre columns
    U0R = orth( P⊥_{TRO} · (H · Ω') )       Dpre columns

Two details that are load-bearing:

* `Ω` is drawn **sector by sector**, with `Dex` split across sectors in proportion to
  each sector's dimension (`GaussRandMat`, `RSVDpreBE0SiQS.m:557`). This is what makes
  the sketch reach every reachable charge sector — and it is why the random-fill hack
  becomes unnecessary rather than merely cheaper.
* `Ω` deliberately mixes **isometry-space and complement-space** directions
  (`CompRatio = 0.5`), *not* pure complement: *"do not project into complement space,
  1-site components are also important for bond update"* (`RSVDpreBE0SiQS.m:339`).

**Final selection** (`RSVDpreBE0SiQS.m:486`). Form the small `Dpre × Dpre` matrix

    UMU = U0L† · H · U0R

SVD it, keep `Dex`. This ranks the candidate directions **by their weight in the
two-site action of H** — the thing the current augmentation cannot do, since it admits
whatever K/L produces regardless of weight and leaves the ranking to the S-step SVD.

Then expand by direct sum (`TLex = [TLO | TLC]`, `TRex = [TRO ; TRC]`), zero-pad the
bond matrix (`ExpandOrthoCenter`), and do the cheap **0-site** update on the expanded
bond.

Guards worth copying verbatim, each of which has a counterpart already in this repo:

| reference | here |
|---|---|
| `Dex = min(Dex, ceil(2*MDB1))` "for stability, do not expand too much" (`:77`) | the Sulz `2r` cap in `augment.jl:250` |
| `RepairOrtho` — re-project and re-orthonormalize (`:743`) | the exactness pass in `_complement_basis` (`augment.jl:104`) |
| `Empty` check when left/right complement charges do not intersect (`:461`) | `pairable_charges` (`kls_step.jl:53`) |
| `ComplementQ` — fill charge sectors missing from the selection (`:654`) | `random_sector_seed` (`augment.jl:35`) — **this is what CBE should make redundant** |

## 2b. The second goal: cheaper bonds, and never a two-site tensor

Alongside correctness, this scheme has to be **more memory-efficient in how it expands a
bond** than the alternatives — cheaper than adding random vectors, cheaper than completing
the basis, and cheaper than 2-site TDVP. That is a design constraint on the
implementation, not just something to measure afterwards, and it has a sharp form:

**CBE-BUG must never allocate a rank-4 two-site tensor.** Both alternatives do:

| scheme | expanded bond | largest object per bond update |
|---|---|---|
| 2-site TDVP | `d·χ` (full local space, then truncate) | `Θ`, rank 4, `O(χ²d²)` — it *evolves* it |
| BUG with 2r K/L augmentation | `2r` | `frame_theta(f)`, rank 4, plus `apply_gate` on it |
| basis completion / random pad | up to `d·χ` | rank 4 |
| **CBE-BUG** | **`r + Dex`, `Dex` a knob** | **rank 3: `O(χ·d·Dpre)` and `O(χ²)`** |

Every step of the scheme can be written rank-4-free, and each place it would otherwise
appear has a factorisation that avoids it:

* **the sketch** — `Kpart · Vpart` (§4 stage 2), nothing bigger than `U0S0` or `V0`;
* **writing an expansion back** — `U_ex†(A_i A_{i+1}) = (U_ex†A_i)A_{i+1}`, so the
  two-site block is never materialised;
* **forming the augmented core** — `S₀ = (U_ex†A_c)(A_{c+1}V_ex†)`, likewise;
* **the Galerkin step** — this is the one that matters most, and it is also the one the
  reference makes obvious in hindsight: `RSVDpreBE0SiQS` *returns* `VLex, VRex`, the
  environments already pushed through the expanded tensors, precisely so that
  `LanczosUpdate0SiQS` can update the bond **without touching the site tensors at all**.
  Pushing the channel environments through `U_ex`/`V_ex` once per step turns each Krylov
  matvec into

      S_out = done_L·S + S·done_R^T + Σ_t c_t Σ_op open_L[t]·S·open_R[t]

  which is a handful of `bond × bond` products — `O(χ²)` per matvec instead of `O(χ²d²)`,
  with no two-site object anywhere in the Krylov loop.

The `Dex` knob is what makes the bond itself cheaper: the K/L augmentation has no knob
(it admits every direction it finds, up to `2r`), and basis completion has no knob by
definition. Here the expansion is `r + Dex` with the directions *ranked by weight in
`HΘ`*, so `Dex` can be small and still targeted.

## 3. Target scheme: CBE-BUG

Combining the above with the BUG structure, dropping the discarded projector, and
restructuring the sweep the way the TTN BUG does it:

```
one time step, chain of L sites, centre bond c:

  1.  BASIS SWEEP (no time evolution at all)
      for each bond b, outward → centre:
          expand the frame at b by CBE/RSVD:
              Ω  ← sector-graded Gaussian sketch  (isometry ⊕ complement, CompRatio)
              preselect: U0L, U0R = orth(P⊥ · H Ω)          Dpre columns
              final:     SVD(U0L† H U0R), keep Dex
              frame     ← [old | selected]     (direct sum, orthonormal)

  2.  ONE GALERKIN S-STEP at the centre bond c
          S_new = expv(P_aug H P_aug, tau, S_aug0)

  3.  truncate (symmetry-blocked SVD), redistribute
```

Three deliberate departures from `kls_bond_update`:

* **No K/L ODE solves.** The isometries come from the CBE selection, so there is no
  frozen-projector generator and nothing non-Hermitian to Arnoldi.
* **No discarded projector — "faithful" and "no-overlap" coincide here.** Faithful BUG
  needs the overlap matrices `M̂ = Û†U0`, `N̂ = V0V̂†` because its augmented basis
  `[U0 | U1]` is built from the *integrated* `U1`, which is not orthogonal to `U0`.
  The CBE expansion is an orthonormal **direct sum containing the old frame**, so
  `M̂ = [I; 0]` exactly, and `Ŝ0 = Û†Θ0V̂†` is already the faithful transport. The
  distinction that made the discarded variant a variant dissolves. (This is worth
  stating in a paper: it is not an approximation we are choosing, it is a consequence
  of expanding by direct sum.)
* **One Galerkin step per time step, at the centre bond**, in place of a per-bond S-step
  inside an odd/even Trotter sweep — the structure the TTN BUG uses (Sulz alg. 5/6/7,
  `qtci_time_integrators/exploratory_bug/src/TTN/ttn_bug.jl`). Removes the even/odd
  Trotter splitting error entirely; the step becomes one projected exponential in a
  globally-expanded subspace.

### Consequence to measure, not assume

With the time evolution confined to the single centre Galerkin step, the accuracy is set
by how well the CBE-expanded subspace captures the action of `H` — the bases are no
longer updated by evolution, they are updated by `H`'s *range*. That is arguably
better-targeted (it is the Krylov direction, not the frozen-projector direction), and
the Krylov depth inside the S-step supplies the higher powers within the subspace. But
it is a different scheme, so the order in `dt` is an **open measurement**, not an
inheritance. The `≤2r` order ceiling established earlier applies to `≤2r` *sweeps*; this
is `r + Dex ⊆ 2r`, so second order is the expectation and first order is the failure
mode to watch for.

## 4. What has to be built (BUG-Julia has none of it)

`BondUpdateBUG` is **gate-based**: `bond_gates` builds one two-site gate per bond and
`parity_sweep!` Trotterizes them. There is no MPO, no environment, and Telum itself is a
tensor library only (`TLArray`, `contract`, `svd`, `eig`, `getIdentity`, local spaces —
no MPS/MPO layer). Every ingredient above needs `H` globally.

### Stage 1 — channel environments (`henv.jl`)

Not a symmetric MPO. For a nearest-neighbour XXZ chain the sum of local terms closes
under a **four-channel recursion**, which is an MPO contraction with the virtual leg
*named* instead of fused — and naming it is what lets the charges take care of
themselves, since they arrive on the op-legs of Telum's own `Sp`/`Sm`:

    E_done'     = A†(E_done)A  +  Σ_α c_α · A†(E_open[α] ⊗ op_α^partner)A
    E_open'[α]  = A†(E_done ⊗ op_α)A                    α ∈ {Sp, Sm, Sz}

`E_done` is rank-2 on `(link', link)`; `E_open[α]` is rank-3 with the op-leg dangling
and carrying that channel's charge. O(L) build, four tensors per link, no AutoMPO and no
charge-graded virtual space to assemble by hand.

Acceptance: `⟨ψ|H|ψ⟩` from the environments equals `energy(psi, gates)` (already
validated against the analytic Heisenberg block, `test_gates.jl`) to machine precision;
the two-site effective action matches a direct gate sum on an interior bond.

### Stage 2 — CBE/RSVD expansion (`cbe.jl`)

Port `RSVDpreBE0SiQS`: sector-graded `GaussRandMat`, preselect, final selection,
`RepairOrtho`, `Empty` check, direct-sum expand, zero-pad the core.

Four things became clear while writing this and are worth recording, because they are
not obvious from the MATLAB:

**(i) The sketch must be folded in DURING the contraction, not applied to a formed
`HΘ`.** With the rank-4 `HΘ` in hand you would take its exact left SVD and beat any
sketch of it, in both accuracy and cost — the randomization would be a strawman. This is
why `contractHPsi_R2L` (`RSVDpreBE0SiQS.m:826`) orders its network to put `Ω` next to the
right environment first, and why `sketch_h_left`/`sketch_h_right` here never build a
two-site object.

**(ii) One decomposition makes all five contributions uniform.** Every term of `H` at a
bond factorises into something acting on the left legs and something on the right,
sharing at most one op-leg:

    Y[link_l, site_l, g] = Σ_{bond, op} Kpart[link_l, site_l, bond, op] · Vpart[bond, g, op]

with `Kpart` the left operator applied to `K0 = U0 S0` and `Vpart` the right operator
applied to `V0` and then contracted with `Ω`. Nothing bigger than `K0` or `V0` is ever
formed, and the sketch dimension replaces `site_r ⊗ link_r` at the first opportunity.
`K0` and `V0` both carry their site leg at position 2 — an MPS tensor's layout — so the
site-operator helper applies to them unchanged.

**(iii) The final-selection matrix is one more sketch call.** Unwinding the `Scheme`
branches of `RSVDpreBE0SiQS.m:435-457` (which do include the core `OC`), the small
matrix is exactly

    UMU = U0L† (HΘ) V0R†

so it is `sketch_h_left(…, V0R)` contracted with `U0L†` — no new machinery, and it makes
plain what the selection ranks candidates by: their weight in the two-site action of `H`.

**(iv) Zero-padding the core is automatic, not a step.** `ExpandOrthoCenter`
(`RSVDpreBE0SiQS.m:864`) pads `OC` with explicit zero blocks. In BUG's no-overlap form
the core is obtained as `Ŝ₀ = Û†Θ₀V̂†`, and since the new columns are orthogonal to the
old frame by construction, that projection *is* `[[S₀,0],[0,0]]`. Nothing to pad.

And the mechanism that makes the whole exercise worthwhile, in one line: `GaussRandMat`
allocates complement columns per sector in proportion to `DC(q) = Dfull(q) − DT(q)`
(`RSVDpreBE0SiQS.m:590-620`). A charge sector the frame does not touch at all has
`DT(q) = 0`, hence the **largest** complement share — so an empty-but-reachable sector is
sampled by construction. That is the replacement for `random_sector_seed`: not a fill
bolted on beside the augmentation, but a property of the sketch itself.

Acceptance:
1. expanded frames are isometries (`left_isometry_defect` at machine zero);
2. the expansion contains the old frame, so `Û†Θ0V̂†` is lossless and `tau = 0` is the
   identity on the state;
3. **the headline claim** — with `Dex` large enough, the selected directions populate
   exactly the empty-but-reachable charge sectors that `sector_report` flags as
   `missing`, with **no random fill anywhere in the call**;
4. at `Dex = Dfull` the expansion recovers the full two-site space, so the bond update
   reproduces the exact two-site update.

### Stage 3 — the step (`cbe_bug.jl`)

Entry point `cbe_bug_bond_update`; `bond_update_bug!` and `kls_bond_update` are left
untouched (validated reference, and the unitarity-critical path).

```
c = centre bond

1.  right-to-centre basis sweep   j = L-1 … c+1 : expand V, write back, centre moves left
2.  left-to-centre basis sweep    i = 1 … c-1   : expand U, write back, centre moves right
3.  centre bond: expand both frames, then ONE Galerkin step
        S₀    = U_ex† Θ V_ex†                    (lossless; see stage 2c)
        S_new = expv(x -> U_ex† H (U_ex x V_ex) V_ex†, tau, S₀)
    then the truncating SVD sets the new centre rank
4.  truncation sweep (Sulz algorithm 1) over the remaining bonds
```

Writing an expansion back is exact and needs no correction term: with `U_ex ⊇ U0`,

    psi[i] <- U_ex ,   psi[i+1] <- U_ex† (psi[i] psi[i+1])

leaves the state unchanged, because `U_ex†U0 = [I;0]`.

**A subtlety worth recording, because the scheme looks wrong at first glance.** With the
basis update now purely algebraic (CBE expands; it does not evolve), and only one core
evolved, it can seem that the expansions at bonds far from the centre can never acquire
amplitude — the centre core lives on bond `c` alone. They do, and the reason is that
expanding bond `c-1` enlarges `dim(link_c)`, hence the *ambient space* `link_c ⊗ site_c`
that the centre frame is chosen from. The outer expansions raise the ceiling on the
centre's variational subspace rather than carrying weight themselves. What the step then
computes is one projected exponential in a globally CBE-enlarged subspace, whose error is
`(1-P)H|ψ⟩` — precisely the quantity `dex` controls.

Step 4 is not optional housekeeping: after step 3 the non-centre bonds are
over-dimensioned relative to the weight they carry, and the unpopulated directions have
exactly zero singular values, so the truncating sweep removes them and the rank does not
ratchet up step after step.

Environment work is **`O(L)` per step**. Each sweep CARRIES its own side's channels — one
`push_*_channels` per bond — and reads the other side from a single prebuilt stack, so
nothing is rebuilt per bond. The centre step needs no rebuild either: the carried left
channels already sit on link `c` and the prebuilt right stack already holds link `c+2`.
This is the structure of the chain BUG in qtci's `exploratory_bug/src/BUG/discarded_bug.jl`,
whose K-sweep carries `aps`/`aph` site to site and computes its environments exactly once.

Note the gauge transport needs no `M̂`/`N̂` matrices at all, for the same reason that
formulation drops them: `U_ex ⊇ U0` makes the overlap exactly `[I;0]`, so the write-back
*is* the transport.

ORDERING IS LOAD-BEARING, and a per-bond rebuild hides it. `canonical!` rewrites tensors,
so each sweep canonicalises to its far end BEFORE building the stack it will read, and the
sweep direction is chosen so the write-back only ever touches sites that stack does not
depend on (`_absorb_right!` at bond `j` touches sites `j, j+1`, while the left stack is
read at links `<= j`; mirror for the other sweep). Getting this backwards reads a stale
environment, which does not throw — it silently evolves with the wrong Hamiltonian, so
`test_carry.jl` checks mid-sweep that the carry equals a stack rebuilt from the UPDATED
state while the opposite stack still equals the untouched side.

### Stage 4 — measurement

* order in `dt` against the XX free-fermion / dense references in `tests/common/`;
* cost against `bond_update_bug!` at matched accuracy (matvec counts + wall time);
* rank trajectories on the domain-wall benchmark;
* `aug_k` with CBE vs with `missing_fill` — the direct test of claim 3 at the
  integrator level.
