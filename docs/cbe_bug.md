# CBE-BUG: controlled bond expansion as the BUG basis update

Design spec for the exploratory integrator. Status: **spec + stage 1**.

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

Basis sweep + single centre Galerkin S-step + truncation. New entry point
`cbe_bug!`; `bond_update_bug!` and `kls_bond_update` are left untouched (they are the
validated reference and the unitarity-critical path).

### Stage 4 — measurement

* order in `dt` against the XX free-fermion / dense references in `tests/common/`;
* cost against `bond_update_bug!` at matched accuracy (matvec counts + wall time);
* rank trajectories on the domain-wall benchmark;
* `aug_k` with CBE vs with `missing_fill` — the direct test of claim 3 at the
  integrator level.
