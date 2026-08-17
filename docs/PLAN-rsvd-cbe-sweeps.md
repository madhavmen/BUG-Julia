# Plan — RSVD-CBE for DMRG and 1-site TDVP, and the CBE + fixed-rank-BUG sweep

Working branch `feature/cbe-rsvd-sweeps`, base `3452a51` (which is the snapshot commit of
everything `docs/HANDOFF-sweep-variants.md` describes — nothing was uncommitted when this
started, so every removal below is reversible).

This document is the structure. It is written phase by phase because the phases have a real
dependency order: a physics claim about Haldane–Shastry is worthless if the non-abelian
primitives underneath are wrong, and a sweep comparison is worthless if the two sweeps are
not running the same Hamiltonian.

---

## 0. Reference corrections, before anything is built on them

Two of the references named in the request point somewhere other than assumed. Both were
checked against the papers themselves, not from memory.

**arXiv:2606.28169 — the fixed-rank BUG sweep is §1, not §2.**
§1 is "The BUG algorithm": Eq. (1) the Schrödinger equation, Eqs. (4)–(5) the canonical form
about the root, Eq. (6) the tensor ODE `d/dt T^[c] = -i H_eff^[c] T^[c]`, Eq. (7) `H_eff` as
one contraction of `L · W · R`, Eq. (8) the rank-3 left-environment recursion, Eq. (9) the
augmentation projector, and **Remark 1** the frozen-environment assumption that makes the
node updates independent (hence "parallel"). §2 is "The Hybrid BUG Algorithm" — the
quantum/classical split, §§2.1–2.5, `|Ψ_sub⟩ = Π U^[k]|0⟩`, the bookkeeping tensors `A`/`B`
of Definition 3, the isometrisation matrix `M` of Eq. (17). There is no MPS sweep spec in §2.
**So the scheme being implemented is §1**, which is also the one described in the request in
words (left-to-centre and right-to-centre, single-site basis updates, one final root update).

The request also asks for **no frozen environments**, i.e. deliberately *not* Remark 1: the
sweep is sequential, and the environment carried past bond `i` is pushed through the basis
just written. That is a departure from the paper and it is the right one for a serial code —
Remark 1 exists to buy parallelism, and parallelism is not what is being bought here. It is
recorded as a departure so the comparison is honest.

**arXiv:2208.10972 is the CBE-TDVP paper** (Li, Gleis, von Delft), i.e. the *same* paper the
MATLAB reference implements — and it is indeed where the SU(2) Haldane–Shastry ring benchmark
lives, alongside XX, one-axis twisting and Peierls–Hubbard. Its Hamiltonian is

```
H_HS = J Σ_{ℓ<ℓ'≤L}  π² / ( L² sin²( π(ℓ-ℓ')/L ) )  S_ℓ · S_ℓ'
```

so the "Haldane–Shastry reference" and the "CBE reference" are one document. Its CBE
tolerances are `ε' = 1e-4` (preselection), `ε̃ = 1e-6` (final selection), `ε = 1e-12`
(trim).

## 0b. How often is the shrewd selection done? — every bond, every sweep

This was the question asked, and both the reference code and the paper answer it the same
way: **once per bond per half-sweep, at every time step.** Not once at the start.

- `cbe_matlab/TDVPSweepCBE1Si.m:30` calls `CBE(TL, TR, VL, VR, HL, HR, '>>')` *inside* the
  `si = 1:(L-1)` loop, and line 70 calls it again inside the `si = L:-1:2` loop. Both
  half-sweeps, every bond.
- `cbe_matlab/DMRGSweepCBE1Si.m:29` and `:64` — identical placement.
- `cbe_matlab/mainDMRG_CBE.m:90-94` binds `Funcs.CBE` once per *sweep schedule entry* only to
  fix `Dmax`/`Dex`; the call itself is per bond.
- The paper's step 0 at each bond is `expand A_ℓ, C_{ℓ+1}, H^{1s}_{ℓ+1} → ex`, immediately
  before that bond is time-evolved.

The reason is structural, not a tuning choice: a 1-site update **cannot change a bond
dimension at all**. The expansion is the only thing that can, so if it is not done at a bond
in a given step, that bond's rank is frozen for that step, and for a time integrator a
direction not admitted is not deferred — it is gone. (Our `tdvp1_cbe_step!` already does it
per bond in both half-sweeps, so this needed no change; it is now asserted in the tests
rather than left implicit.)

---

## Phase 0 — is the non-abelian path actually correct? (blocking)

The stated requirement is that **the physics is identical under `:none`, `:U1` and `:SU2`**;
only the MPO structure and the cost may differ. Everything downstream is gated on that, so it
is measured first and independently of the new code.

Specific hazards, each with a concrete way it would show up:

| hazard | why it bites under SU(2) | how it is measured |
|---|---|---|
| `leg_dim` counts **multiplets**, not states | every budget (`growth*dmax - r`), every `Nkeep`, every reported rank is then in multiplets, so "maxdim = 16" is a *different physical rank* per mode | print both counts; run parity at full rank so the difference cannot hide |
| singular-value weights | Telum's `kept_list` rows are `(σ, deg, sector, rank)`; a discarded weight without `deg` mis-ranks sectors, one *with* `deg` double-counts if σ is already state-normalised | check `Σ σ²·deg == ‖T‖²` against `Σ σ²` |
| `_alloc_columns` splits the sketch budget by multiplet count | a sector holding one high-spin multiplet carries many states but gets one probe column | compare sketched vs exact CBE per mode |
| `fused_dim` / `room` | `leg_dim(U0,1)*leg_dim(U0,2)` is wrong non-abelian; already fixed via `reachable_sectors` (`cbe_core.jl:519`) | parity at full rank |
| operator normalisation | Telum scales SU(2) generators by Clebsch–Gordan factors; `heisenberg_bond_gate` already carries a measured `-1` on its `:none` branch for exactly this | dimer-state energy `= -0.75` per singlet, in all modes |

Order of work: `0.1` primitive semantics → `0.2` parity of the **validated baseline**
`tdvp2_step!` (no CBE in it, so a disagreement there is a primitive defect, not a selection
defect) → `0.3` parity of `cbe_lubich_sweep` / `jan_bug_step!` / `GroundState.solve!` →
`0.4` fix and re-measure.

The common initial state is `dimer_state(L)`, the product of nearest-neighbour singlets. It
is the only interesting state expressible in all three modes: under SU(2) every representable
state has definite total spin, so Néel and domain-wall states do not exist at any bond
dimension.

## Phase 1 — the models, in both symmetries

`1.1` **A charged-channel all-pairs MPO.** Haldane–Shastry needs `Σ_{i<j} J_ij S_i·S_j` with
arbitrary `J_ij`, and the existing `long_range_mpo` **throws** on it: its geometric tail is one
self-loop channel, and a rank-3 operator (`Sp` under U(1), `S` under SU(2)) puts its op-leg on
the *virtual* leg, so that channel's self-loop would need the identity on a **charged** space
rather than `λ·Iid` (`mpo.jl:239-247`). That restriction is the blocker, and it is lifted the
exact way rather than the fitted way:

- one channel per **opening site**, `w = 2 + n·(L-1)`, so `J_ij` is applied at closing and any
  coupling matrix is represented **exactly** — no exponential fit, hence no fit error to
  report, which matters because a fit error is a property of the Hamiltonian and is not
  separable from the result afterwards;
- the transport block is `δ_{w_l w_r} ⊗ I_site` on the charged space — the one genuinely new
  tensor;
- `n = 1` under SU(2) (one irreducible `S·S`) against `n = 3` under U(1) (`SpSm`, `SmSp`,
  `SzSz`), so `w = L+1` vs `w = 3L-1`. **This is the place the two symmetries are allowed to
  differ**, and it is the cost argument for SU(2), not a physics difference.
- On a ring, `J_ij = π²/(L² sin²(π(i-j)/L))` already encodes the periodicity: the pair `(1,L)`
  has `sin²(π(L-1)/L) = sin²(π/L)`, i.e. it is a *near* pair. No wrap-around term is added.

`1.2` **The models.** Haldane–Shastry ring and the Heisenberg chain/ring, each built in every
symmetry mode from the same coupling matrix, with the cross-symmetry equality asserted:
`:none` validated against a dense reference, then `:U1` and `:SU2` against the `:none` number.
That chain is what validates the charged transport, since `:none` has rank-2 operators and no
charged channel at all.

## Phase 2 — the sweeps, in one structured place

New directory `src/RSVDCBEBondUpdate/sweeps/`, all three on the **MPO path** (so long-range
and site-dependent `H` work, which the `XXZChain` channel path cannot express):

```
sweeps/expansion.jl    the per-bond expansion, at the MATLAB call site's granularity:
                       CBE(TL,TR,VL,VR,HL,HR,dir) -> (TLex, TRex, Vex, Info)
                       i.e. cbe_expand + lossless write-back + one env push. ONE place.
sweeps/dmrg_cbe1s.jl   port of DMRGSweepCBE1Si.m -- expand, 1-site Lanczos eigensolve,
                       truncated split, absorb. No 0-site update (DMRG has none).
sweeps/tdvp_cbe1s.jl   port of TDVPSweepCBE1Si.m -- expand, 1-site +τ, 0-site -τ, absorb.
sweeps/cbe_bug.jl      THE FINAL SWEEP (below).
```

**The final sweep.** Same spirit as CBE-TDVP — expand the bond first, then update — but the
update is the fixed-rank BUG of §1 instead of a 1-site TDVP:

1. root `r = ⌊L/2⌋`;
2. **left → root**, bond by bond: CBE-expand bond `i` (lossless, so the state is untouched),
   then the BUG **K-step**: evolve the one-site object `K = C·A_i` (widened) under `H^{1s}_i`
   for the *full* step, take `W_i = orth(K(τ))` and **discard `R`** — the basis moves where the
   dynamics points, no evolution accumulates. Carry `C ← W_i†K(0)`, i.e. the *original*
   amplitude re-expressed in the new basis. That is "not dynamically evolving the basis".
3. **right → root**, the mirror, giving `Z_{c+2} … Z_L`;
4. **one** Galerkin update at the root bond: seed `S₀ = ⟨W,Z|ψ(t₀)⟩`, one 0-site
   `exp(τ H_eff)`. This is the only evolution that is kept, so there is no over-counting and
   no backward substep;
5. one truncation round trip.
6. **Fixed rank** means `augment = false`: `W_i = orth(K(τ))`, not `orth([K(τ)|A_i])`. The
   rank therefore comes **only** from the CBE expansion in step 2 — which is the standing
   constraint that `cbe_rsvd` *is* the method.
7. Environments are **not** frozen: the left environment is pushed through `W_i` before bond
   `i+1` is touched. Departure from Remark 1, recorded above.

Where it sits against what already exists: `cbe_lubich_sweep` is this sweep with the K-step
*omitted* (CBE alone supplies the basis), and `jan_bug_step!` is a 0-site S-step at every bond
with the root at a boundary. So the three differ in exactly one axis each, which is what makes
a comparison between them attributable.

## Phase 3 — physics validation against analytic results

`3.1` an analytic-reference module, separate from the integrators and sharing no code with
them:

- **Haldane–Shastry ring**, exact: `E₀/J = -π²(N²+5)/(24N)` for the normalisation above, plus
  the paper's quantities. The formula's *constant* is checked against exact diagonalisation at
  `N = 4, 6, 8` before it is used as a reference, and the measured constant is what gets
  reported if they disagree — a reference asserted from memory is not a reference.
- **Heisenberg**, exact: Bethe-ansatz finite-`N` ground energies for the ring, the Hulthén
  thermodynamic value `E/N = 1/4 - ln 2` as the large-`N` check, and — for **real time**, where
  XXX has no closed-form solution — the **one- and two-magnon sectors**, which are exact
  eigenspaces, so a wavepacket in them evolves by a plain Fourier/Bethe sum.

`3.2` the matrix: `{Haldane–Shastry, Heisenberg} × {ground state, imaginary time, real time} ×
{the new sweep, jan_bug_step!}`, in both symmetries.

**An honest limitation, stated rather than papered over:** SU(2) XXX real-time evolution from a
generic state has no analytic solution. Ground state and imaginary time do (they converge to
the Bethe/HS ground energy). For real time the analytic references available are the
magnon-sector wavepackets above and the exact conservation laws (norm, energy, total spin);
outside those, the only reference is a numerically exact one. Where that is what is used, it is
labelled as such.

## Phase 4 — remove the experimental sweeps

Keep `jan_bug_step!`, `cbe_lubich_sweep` and the new sweep. Everything reachable only from the
removed arms goes with it. Done **last**, after the replacements pass, and reversible because
of the snapshot commit.

## Phase 5 — the explanatory notebooks (Python, Alice + Nicole)

Nine notebooks, `{1-site CBE TDVP, the new BUG sweep, 1-site CBE DMRG} × {GSE, ITE, RTE}`, on
**L = 6 XX with U(1)**, one cell per line or block of the workflow, printing every intermediate
so that both the physics and the computation are visible. These are written against Alice
(`C:\AliceRepo`, `src/alice/algorithm/cbe_bug/*`) and Nicole 0.3.7 tensors — the notebook *is*
the expanded implementation, not a wrapper around a library call.
