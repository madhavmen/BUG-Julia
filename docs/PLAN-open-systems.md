# PLAN — open quantum systems on MPS, and the L=18 model suite

**Architectural constraint, and it is not negotiable: everything goes through MPOs and
ENVIRONMENTS.** No Trotter gates, no bare two-site gate layers. That is how `cbe_bug`,
`tdvp_cbe1s` and `tdvp2` are already built (MPO channel environments, QR gauge transport), it is
what makes the three comparable at all, and it is what makes a long-range or doubled-chain
generator expressible in the first place. Every construction below is an MPO acting through the
existing environment stacks.

---

## 0. Where we are

| model | symmetry | sites | exact ref | status |
|---|---|---|---|---|
| Heisenberg chain | SU(2) | 16 | yes | ✅ measured |
| XX chain | U(1) | 12–16 | yes | ✅ measured |
| Heisenberg square cylinder 4×4 | SU(2) | 16 | yes | ✅ measured |
| dipolar XY, breathing kagome 2×3 | U(1) | 18 | yes | ✅ measured |
| **XX boundary-driven** (MPDO) | none (fused d=4) | 8–16 | ✅ **exact, analytic** | ✅ **VALIDATED 2026-08-28** |
| XX **open** (dephasing) | U(1) | 18 | — | ⛔ blocked, see §2 |
| Heisenberg square **open** | U(1) | 18 | — | ⛔ blocked |
| lattice gauge theory **open** | U(1) | 18 | — | ⛔ blocked |

Machinery already in place: `pair_mpo` with **one coupling matrix per vertex** (needed for any
model whose operator types have different ranges) and **complex couplings** (verified:
`E = −1.75 + 0.075im`). Both were added for this and are guard-tested.

✅ **THE `exact ref` COLUMN IS NO LONGER EMPTY FOR EVERY OPEN MODEL.** The boundary-driven XX chain
is solvable by third quantization — an `N x N` non-Hermitian Lyapunov solve, `O(N^3)`, no object of
size `4^N` — so it turns that column into a real number rather than a published *trend*. Measured at
`L = 8` with truncation live, against `xx_boundary_sz`:

| arm | observed order | err at `dt = 0.005` |
|---|---|---|
| `tdvp2` | 2.99 ⚠ one ratio, two points — quote as ">= 2" | **1.437e-10** |
| `bug_rsvd` | **1.98** (the expected 2) | 2.665e-06 |

⚠ The ~4-order gap is an **ORDER** gap and therefore WIDENS as `dt` shrinks ⇒ any BUG-vs-TDVP cost
claim on this model must equalise ACCURACY, not `dt`. Consistent with §5's standing warning that
the two cost axes disagree, and with the finding that the advantage holds in imaginary time and not
in real time.

⛔ Getting here required fixing two defects that produced PLAUSIBLE WRONG NUMBERS rather than
errors: `pair_mpo` shares `site_apply!`'s TRANSPOSE, so `JP`/`JM` entered the Lindbladian swapped
and the jump term exactly cancelled the anticommutator — making `|I>>` a genuine fixed point that
read as a dead integrator; and `evolve_arm` scored the NOMINAL snapshot time while stepping
`round(Int, ...)` times, so `round(12.5) = 12` made `t = 0.25` really `0.24`. **Both announced
themselves as `order 0.00`** — a residual that does not move when `dt` moves is never a
discretisation error, whatever its magnitude.

---

## 1. ⛔ THE BLOCKER: every sweep assumes a HERMITIAN generator

```
cbe_bug.jl:579        expv(..., hermitian = true, ...)
tdvp_cbe1s.jl:129,145 expv(..., hermitian = true, ...)
cbe_core.jl:1204      expv(..., hermitian = true, ...)
```

and inside `_krylov_frame`:

```julia
push!(alpha, real(tensor_inner(vs[end], w)))     # Rayleigh quotient, real ONLY if H = H†
```
with `_kry_contribution` assembling a **symmetric tridiagonal** `T_m`.

On a non-Hermitian generator all three of those are wrong, and **none of them errors** — Lanczos
on a non-symmetric operator returns a plausible vector. This affects `tdvp2`, `tdvp_cbe1s` and
`cbe_bug` equally, so it is not a fairness issue between schemes; it is a correctness issue for
all of them.

**Work item 1 (prerequisite for everything open).** Thread `hermitian::Bool` through the three
step functions to `expv` (which already implements the Arnoldi branch), and give `_krylov_frame`
a non-Hermitian path: complex `alpha`, an upper-Hessenberg projection instead of the symmetric
tridiagonal, and the contribution estimate read off that. Gate: on a Hermitian problem the new
path must reproduce the old numbers **bit for bit**, or the switch has changed the closed-system
results too.

---

## 2. ⛔ AND THE OBVIOUS DISSIPATORS ARE TRIVIAL — the physics constrains the construction

Two separate traps, both of which produce a well-formed run that measures nothing:

**Uniform one-body loss does nothing under U(1).** `Σⱼ nⱼ` is CONSERVED, so `−iΓ Σⱼ nⱼ` is a
constant times the identity: a global decay factor and no dynamics. A one-body dissipator must be
**staggered** to matter (which is exactly what the Schwinger paper uses,
`O(n) = (−1)ⁿ(Zₙ+1)/2`).

**Dephasing cannot be done with a non-Hermitian `H_eff` at all.** For `L_j = S^z_j`,
`L†L = ¼` — a constant — so the no-jump Hamiltonian is `H − iγN/8`, again pure global decay.
**Dephasing acts entirely through the jump term.** This matters because dephasing is precisely
what the XX transport literature uses (§4).

So there are two distinct open-system capabilities, and they need different machinery:

| dissipator | route | machinery |
|---|---|---|
| loss / gain / correlated loss | non-Hermitian `H_eff` | ✅ ready (complex `pair_mpo`) + work item 1 |
| **dephasing** | full Lindbladian | work item 2 |

---

## 3. Work item 2 — the vectorised Lindbladian as an MPO on a DOUBLED CHAIN

Vectorise `ρ → |ρ⟩⟩`, but lay the doubled space out as **2L spin-½ sites** rather than an
`L`-site chain with local dimension 4. That keeps the existing local space, the existing symmetry
handling, and — the point — the existing MPO/environment machinery:

```
L(ρ) = −i(H⊗I) + i(I⊗Hᵀ) + γ Σⱼ (Lⱼ ⊗ L̄ⱼ) − (γ/2)(L†L⊗I + I⊗(L†L)ᵀ)
```

Every term is **two-body between a specific pair of sites** — system–system, ancilla–ancilla, or
system `j` to its ancilla `j'` — so all of it goes through `pair_mpo` with one coupling matrix per
vertex. For dephasing (`Lⱼ = S^z_j`) the jump term is `γ Σⱼ S^z_j ⊗ S^z_{j'}` and the
anticommutator is a constant.

**Layout decision to make and then never change:** system at odd sites, ancilla at even, so each
system site is ADJACENT to its own ancilla and the `S^z⊗S^z` coupling is nearest-neighbour. The
alternative (system 1..L, ancilla L+1..2L) makes every jump term range-`L` and inflates the MPO.

⚠️ **Observables change.** `⟨O⟩ = Tr(Oρ) = ⟨⟨I|O|ρ⟩⟩` — measurement is an overlap with the
vectorised identity, not `⟨ψ|O|ψ⟩`. And `|ρ⟩⟩` is **not normalised**; `Tr ρ = ⟨⟨I|ρ⟩⟩ = 1` is the
conservation law to monitor, in place of the norm.

⚠️ **The Lindbladian is not Hermitian**, so work item 1 is a hard prerequisite here too.

**Gates, in order:** (a) `γ = 0` must reproduce the closed-system result exactly, since the
doubled chain then factorises; (b) `Tr ρ` conserved to machine precision; (c) `ρ` stays positive
(smallest eigenvalue of a small reshaped block ≥ 0 within truncation error).

### 3.1 Status — the construction is NOT yet validated

Measured at `L = 4` (`benchmarks/smoke_lindblad.jl`), against a dense superoperator built by
applying `L` to every basis matrix:

```
GATE 0 (layout, asymmetric psi = up,down,up,up): max |MPS - dense| = 0.000e+00   PASS

   gamma max|MPS-dense|         Tr rho          |dTr|    verdict
    0.00      1.455e-01      1.0000000000      0.000e+00       FAIL
    0.50      2.040e-01      1.1618342427      1.618e-01       FAIL
```

Gate 0 passing pins down the interleaved layout, the `dense_state` bit convention and the
column-stacked `vec` mapping as exact. Gate 1 failing **with the trace exactly conserved** says the
error is in the dynamics, not the bookkeeping.

**The diagnostic ladder, and what each rung eliminates.** Each rung was chosen to make one suspect
answerable on its own; a time-evolution comparison alone cannot separate them.

| rung | question | result |
|---|---|---|
| `half_check.jl` (a) | is `pair_mpo` + XY vertices the same operator as `xxz_mpo(delta=0)`? | ✅ `\|<q1\|q2>\| = 1.000000000000` — identical |
| `half_check.jl` (b) | does the KET half alone reproduce `ψ(t) ⊗ ψ(0)`? | ❌ 1.489e-01 — **but see below: the reference was unfair** |
| `mpo_vs_dense.jl` | is the doubled-chain MPO the dense superoperator, **as an operator**? | see table |
| `arnoldi_check.jl` | does the same failure appear under schemes with no CBE and no growth frame? | ❌ **identical** in all five |

⚠️ **Rung (a) does not cover the doubled chain.** It ran on the plain `L`-site chain with REAL
couplings at range 1. The doubled chain asks for COMPLEX couplings at range 2 with the partner site
sitting in between — different transport-channel bookkeeping — so it is not implied by (a).

**`mpo_vs_dense.jl` — the MPO as an OPERATOR, `<v|W|v>` against the dense matrix, no solver:**

| piece | max rel dev | |
|---|---|---|
| ket half `−i(H⊗I)` | 1.325e-15 | PASS |
| bra half `+i(I⊗Hᵀ)` | 1.805e-15 | PASS |
| both halves, `γ=0` | 3.521e-15 | PASS |
| full, `γ=0.5` | **8.572e-01** | **FAIL** |

**`arnoldi_check.jl` — the ket half through every scheme:** `tdvp2`, `tdvp_cbe1s`, `cbe_bug` at
`m=3` and `m=0`, and `tdvp2` forced through the **Lanczos** branch, all returned **1.489e-01 —
the same number to four digits**.

### 3.2 What those two tables actually say

**FAULT 1 (real, and now understood): the dissipator's zero-body term is missing.** `L†L = ¼` for
`L = Sᶻ`, so the anticommutator is `−(γL/4)·𝟙` — a multiple of the IDENTITY. `pair_mpo` builds
**two-body vertices only** and has nowhere to put a zero-body term, so it dropped it silently.
That is the whole of the `γ=0.5` failure, and it is exactly what §3's own note predicted
("the trace term is not decoration") — the note was written and then not implemented.

The fix needs **no MPO change**: `exp(τ(W + c𝟙)) = e^{τc}·exp(τW)`, so the constant is an EXACT
scalar prefactor on the state, applied per step so `Tr ρ` is right at every step and not only at
the end.

**FAULT 2 (retracted — it was never a fault): "the ket half is wrong".** ⚠️ The `half_check` (b)
and `smoke_lindblad` gate-1 references were **themselves produced by `tdvp2`, on a DIFFERENT
chain** (`L` sites versus the `2L`-site doubled chain). That subtracts one Trotter splitting error
from another and calls the remainder a fault. The `arnoldi_check` table is what exposes it: five
code paths — one of them (Lanczos on a non-Hermitian generator) **deliberately wrong** — cannot
agree to four digits on a number any of them produced. A constant across all five is a property of
the *reference*, not of the solvers.

⚠️ **The interleaved layout is intrinsically harder for a sweep, and that has to be measured
rather than assumed.** Interleaving puts every system–system coupling at RANGE 2 with the
partner's ancilla in between, so no two-site block ever contains a whole interaction term. A
larger splitting-error prefactor than the plain chain is expected; a non-vanishing floor is not.
`lindblad_diag.jl` distinguishes them by **dt-refinement at fixed total time against the exact
dense propagator** — `dev ~ dt²` is splitting error, `dev ~ const` is a bug.

### 3.3 ⛔ FAULT 3, and it is the real one: THE INTERLEAVED SUPERKET NEVER EVOLVES

`lindblad_diag.jl`, against the exact `exp(T·𝓛)`, `T = 0.3`:

```
── L = 2, gamma = 0.00           ── L = 3, gamma = 0.00
   steps   max dev      order       steps   max dev      order
       3  1.478e-01                     3  1.455e-01
       6  1.478e-01     0.00           6  1.455e-01     0.00
      12  1.478e-01     0.00          12  1.455e-01     0.00
      24  1.478e-01     0.00          24  1.455e-01     0.00
```

**`Tr ρ = 1.0000000000` in every row, `γ = 0.5` included — fault 1 is fixed.** But the deviation is
FLAT in dt, identical for `tdvp2` and `cbe_bug`, and barely moves with `L` or `γ`.

**It is not an error at all — it is the whole of the dynamics.** At `L=2, γ=0, T=0.3` the XX
propagator gives `ψ(T) = cos(0.15)|↑↓⟩ − i·sin(0.15)|↓↑⟩`, so the largest change in `ρ` is the
coherence `|sin(0.15)·cos(0.15)| = 0.14776`. Measured: **1.478e-01**. And `0.1372/0.1478 = 0.928
= exp(−0.25·0.3)`, exactly the dephasing damping of that same coherence at `γ = 0.5`. **The
superket sits at `ρ₀` and never moves.**

⛔ **THE CAUSE IS THE LAYOUT, NOT A BUG.** Interleaving puts every system–system term at range 2,
so **no two-site block ever contains both of its operators**. The block feels such a term only
through an MPO environment channel — and a half-open channel is a **one-site expectation of a
charge-changing operator**, which is `⟨S⁻⟩ = 0` on any product state. There is nothing for the
first step to act on, so the dynamics can never start. Every scheme inherits it identically, which
is exactly what the table shows.

⛔ **THE LAYOUT DECISION IN §3 IS THEREFORE WRONG AND IS REVERSED.** It was chosen to keep the MPO
narrow ("interleaved wins in both"); that reasoning counted MPO width and not bootstrapping. With
**ket at `1..L` and ancilla at `L+1..2L`**:

| term | interleaved | blocked |
|---|---|---|
| `H` on ket / on ancilla | range 2 — **cannot bootstrap** | range 1, wholly inside a two-site block ✅ |
| jump `Sᶻ⊗Sᶻ` | range 1 | range `L` — but **diagonal**, and `⟨Sᶻ⟩ ≠ 0` on a product state, so no freeze ✅ |
| MPO width | narrow | `L` jump channels |

⚠️ **The extra width is not a regret.** §6 wants a wide MPO: the rSVD sketch can only pay off when
`npre/(d·χ)` is small, and the open-system generators are the first in this study wide enough for
that. The blocked layout produces one.

⛔ **AND IT IS NOT A BROKEN TRANSPORT BLOCK.** `transport_check.jl` shows range 1 passing and
ranges 2–3 failing at exactly `5.0e-01`, identically under `:none` and `:U1` — which reads like
`pair_mpo`'s transport being wrong, and would impugn kagome, square and dipolar, all of which use
it. Two results rule that out: `mpo_vs_dense.jl` reproduces the **range-2** doubled-chain operator
to `1.3e-15` with ENTANGLED probes, and the response here is `got = 0` **exactly** rather than a
wrong nonzero value. The closed models never hit the freeze because they all carry
nearest-neighbour terms that bootstrap the state first; the interleaved doubled chain has no
range-1 term in its Hamiltonian part at all.

### 3.4 Layouts measured, and the one that is kept

`layout_bootstrap.jl`, `L = 3`, `T = 0.3`, against `exp(T·𝓛)`:

| layout | γ | `moved by` | max dev | order |
|---|---|---|---|---|
| interleaved | 0.0 | **0.0000e+00** | 1.4554e-01 (= the whole dynamics) | 0.00 |
| interleaved | 0.5 | ~1e-14 | 1.3519e-01 | 0.00 |
| **blocked** | 0.0 | 1.4554e-01 ✅ | **7.5e-14** `tdvp2` / **2.6e-15** `cbe_bug` | machine |
| **blocked** | 0.5 | 1.3519e-01 ✅ | 4.8e-08 → | **2.99** |

`moved by = 0.0000e+00` is the freeze, exactly. Blocked moves by precisely `|ρ(T)−ρ(0)|` and
converges cleanly.

**Kept layout: FUSED `d = 4` sites** — one chain site per system site, carrying `(ket, bra)` with
basis `2k+b+1`. Blocked fixes the dynamics but not the observables: there `|I⟩⟩` is maximally
entangled across the ket/ancilla cut (`χ = 2^L`), and a **spatial** cut needs TWO MPS cuts — so
operator entanglement, which is the observable this study was asked for, is not readable from one
bond. Fusing gets all three at once:

| | interleaved | blocked | **fused d=4** |
|---|---|---|---|
| `H` range | 2 — **frozen** | 1 ✅ | 1 ✅ |
| jump term | range 1 | range `L` | **on-site** ✅ |
| `\|I⟩⟩` | χ=2 | **χ=2^L** | **product** ✅ |
| spatial cut | 1 bond ✅ | 2 bonds ✗ | 1 bond ✅ |

Two small pieces of machinery this needed, both done:

* **`pair_mpo` gained an `Iloc` keyword.** It picked the transported identity from
  `symmetry_mode()`, but the mode does not determine the SITE — the fused chain runs under `:none`
  and its identity is 4×4. `Iloc` is a tensor, not a mode, so the override keeps `pair_mpo`
  agnostic.
* **The on-site jump term goes through a two-body builder** as `O_j ⊗ I_{j+1}` on bonds `1..L−1`
  plus `I_{L−1} ⊗ O_L` on the last — the same distribute-over-bonds trick `ising_bond_gates` uses
  for its field. Per-vertex coupling matrices are what make the two different supports
  expressible at once.

⚠️ **Operator normalisation is written out explicitly in the fused space.** Telum's `q.Sp` carries
a `1/√2` (`gates.jl` says so), which silently rescaled a first attempt at `|I⟩⟩` — `Tr ρ₀` came
back `−0.5` instead of `+1`. The `d=4` matrices are given as literal `kron`s, so there is nothing
to inherit.

### 3.5 Fused layout — gates measured

`fused_lindblad.jl`, `L = 2`, against the exact dense propagator:

```
GATE 0  |I>> = vec(identity), bond dims [1] / [1,1] :  2.220e-16  PASS
GATE 1  Tr rho0 = +1.000000000000                     PASS
GATE 2  Tr(S^z_j rho0) = +-0.500000000000, site by site   PASS
```

**`bond dims [1]`** is the load-bearing part of gate 0: `|I⟩⟩` really is a product state, which is
the whole reason for fusing, and it makes every trace observable an O(χ²) overlap instead of a
fold. `|ρ(T)−ρ(0)| = 1.4776e-01` is itself an independent check: for `L=2`, XX, `T=0.3`, the
largest change in `ρ` is the coherence `|sin(0.15)·cos(0.15)| = 0.14776` by hand.

GATE 3, against `exp(T·𝓛)`, `T = 0.3` — **the construction is validated by `tdvp2`**:

| L | γ | scheme | 3 steps | 12 | 48 | order | `Tr ρ` at 48 |
|---|---|---|---|---|---|---|---|
| 2 | 0.0 | `tdvp2` | 1.055e-14 | 6.095e-14 | 3.515e-13 | roundoff | 1.0000000000 |
| 2 | 0.5 | `tdvp2` | 9.243e-15 | 2.054e-14 | 1.116e-13 | roundoff | 1.0000000000 |
| 3 | 0.0 | `tdvp2` | 8.282e-14 | 1.125e-13 | 2.429e-13 | roundoff | 1.0000000000 |
| 3 | 0.5 | `tdvp2` | 1.932e-14 | 3.542e-14 | 5.481e-13 | roundoff | 1.0000000000 |
| 3 | 0.0 | `cbe_bug m=0` | 4.769e-02 | 1.183e-02 | **2.953e-03** | **1.00** | 0.9995279374 |
| 3 | 0.5 | `cbe_bug m=0` | 4.204e-02 | 1.023e-02 | **2.541e-03** | **1.00** | 0.9995290779 |

`tdvp2` is at machine precision in every row (the negative "orders" are roundoff over roundoff), so
**the fused Lindbladian is the operator we think it is** — layout, `|I⟩⟩`, the zero-body constant
and the on-site jump term all confirmed against an exact reference.

### 3.7 RESOLVED: the order is set by KRYLOV DEPTH, not by the expansion budget

⛔ **RETRACTED: "`cbe_bug` is only first order on a `d = 4` chain."** That was measured on the
`m = 0` arm — the CHEAPEST one, with **no Krylov growth frame at all** — and read as if it were the
method. It is not. `benchmarks/fused_growth_scan.jl`, `L = 3`, against `exp(T·𝓛)`:

| arm | 3 steps | 12 | 48 | order | χ | `Tr ρ` |
|---|---|---|---|---|---|---|
| `tdvp2` (reference) | 8.28e-14 | 1.12e-13 | 2.43e-13 | roundoff | 4 | 1.0000000000 |
| `m=0`, growth 2 | 4.77e-02 | 1.18e-02 | 2.95e-03 | **1.00** | **4** | 0.9995279374 |
| `m=0`, growth 4 | 7.38e-03 | 1.85e-03 | 4.62e-04 | **1.00** | **4** | 0.9995180589 |
| `m=0`, growth 8 | 7.38e-03 | 1.85e-03 | 4.62e-04 | **1.00** | **4** | 0.9995180589 |
| `m=3`, growth 4 | 2.46e-03 | 1.53e-04 | 9.56e-06 | **2.00** | 4 | 0.9999804688 |
| **`bug_interleaved`, growth 2** ← shipped default | 4.79e-02 | 1.19e-02 | **2.95e-03** | **1.00** | 4 | 0.9999902344 |
| **`bug_interleaved`, growth 4** | 2.46e-03 | 1.53e-04 | **9.56e-06** | **2.00** | 4 | 0.9999804688 |

**`χ = 4` — full rank — in every row, so it was never starvation.**

⛔ **BUDGET AND DEPTH ARE BOTH REQUIRED, AND NEITHER ALONE SUFFICES.**

* At `growth = 2` the canonical arm collapses to the `m = 0` result (2.95e-03, order 1.00) **even
  though it computed a depth-30 adaptive basis** — the budget cannot ADMIT the Krylov directions,
  so the depth is paid for and thrown away.
* At `growth = 4` with `m = 0` it is still order 1.00 — budget alone does not fix it either.
* At `growth = 4` with adaptive depth it is order 2.00 and **bit-identical to `m = 3`**.

⛔ **THE SHIPPED `growth = 2.0` THEREFORE CAPS THE ACHIEVABLE ORDER AT 1 ON A `d = 4` CHAIN**, and
would have made every open-system run silently FIRST order while looking perfectly healthy —
`Tr ρ` is 0.99999 there, better than several second-order rows, so the conservation law does not
flag it. `growth = 4.0` is wired into all three fused drivers.

The reason is dimensional: `budget = ceil(growth·dmax) − r`, and "a bond at the neighbourhood max
may DOUBLE" is the right unit for a `d = 2` site. **A fused `d = 4` site can QUADRUPLE a bond.**

⚠️ **Corrected twice on the way here, both recorded so the reasoning is auditable:** first
"`cbe_bug` is only first order on `d = 4`" (measured on `m = 0` and read as the method), then "the
order is set by depth, not budget" (measured on the `m = 0` rows alone, before the canonical arm
was in the scan). The depth arms must never be quoted as "BUG" against `tdvp2` / `tdvp_cbe1s`.

### 3.8 ⛔ ONE BUG SCHEME ACROSS EVERY RUN: `bug_interleaved`

`cbe_bug`'s sweep **is** the interleaved ordering — CBE expands bond `i`, then bond `i`'s frame is
built and the environment pushed through it immediately, so bond `i+1`'s selection sees the
widening at bond `i` (`sweeps/cbe_bug.jl:447`, the reference's own ordering,
`TDVPSweepCBE1Si.m:30/:34`). The DECOUPLED variant is the one that no longer exists: its flag lived
on the K-step path, removed with `kaug` and `rexpand` on 2026-08-24.

⚠️ **Consequence that had gone unnoticed:** `bug_decoupled` and `bug_interleaved` in
`cbe_sweeps_l16.jl` had since been calling `cbe_bug_step!` with **identical arguments** — two
entries producing one curve, which doubles the array and reads as agreement between methods in a
plot. `bug_decoupled` is removed; `bug_interleaved` is the canonical arm in the closed suite and in
all three open drivers, with the same settings in each.

The suffixed arms `bug_interleaved_{m0,m3,tol4,tol6}` are **that same sweep at different Krylov
depths** — a separate axis, belonging to the depth study. Any comparison against `tdvp2` or
`tdvp_cbe1s` uses the unsuffixed name, or it reports a tuning difference as a method difference.

---

## 3.6 What is written and where it runs

| file | item | status |
|---|---|---|
| `benchmarks/fused_common.jl` | shared d=4 machinery (space, states, `fused_lindblad`, `|I⟩⟩`, traces, `operator_entanglement`) | ✅ |
| `benchmarks/fused_lindblad.jl` | the local correctness gates, L=2..3 vs exact | ✅ measured |
| `benchmarks/xx_open_transport.jl` | **item 3** — ballistic→diffusive crossover | written |
| `benchmarks/heis_square_open.jl` | **item 4** — Heisenberg 5×5 open | written |
| `benchmarks/lgt_open.jl` | **item 5** — Schwinger chain + bath | written |
| `benchmarks/submit_xx_open_transport.slurm`, `submit_open_suite.slurm` | array jobs, one (model, γ, scheme) per task | written |
| `benchmarks/plot_open_suite.py` | figures, log-log for the exponent claim | written |

⛔ **The split is deliberate: correctness LOCAL and small, size and wall clock on the CLUSTER.**
Local runs settle the question "is this the right operator" at `L = 2..4` against an exact
reference. `L = 18` and every timing number goes through sbatch, one scheme per array task, threads
pinned to 1 — otherwise `elapsed` measures a machine running twelve other things.

### 3.9 Driver smoke, and two flags for the production runs

All three drivers run end to end at tiny size with `bug_interleaved` (`t = 0.2`, `dt = 0.05`,
cap 64):

| driver | observable | χ | `Tr ρ` |
|---|---|---|---|
| XX, L=4, γ=0.5 | `dM` 0 → 0.009636, `S_op` 0 → 0.103 | 1 → 15 | 0.9993792877 |
| Heisenberg 2×3, γ=0.5 | `M_stag` 0.500 → 0.4813, `S_op` 0 → 0.192 | 1 → **64 = cap** | 0.9962968641 |
| Schwinger N=4, γ=0.5 | condensate −0.500 → −0.4444, **`n_part` = 0.0000** | 1 → 16 | 0.9925856726 |

✅ **`n_part = 0.0000` to machine precision is an independent check on the Schwinger
construction.** The staggered charge `Σ Q_n` is conserved, which the all-to-all `ZZ` term, the
one-body field and the last-site compensation all had to be right to produce — none of which the
`L = 2..4` dephasing gates exercise.

### 3.10 `Tr ρ` under BUG — measured, and what "fix it" actually means

`benchmarks/trace_preservation.jl`, fused L=3, γ=0.5, T=0.3. The same arm at a cap that **cannot**
bind and one that **certainly** does, with `tdvp2` as the control:

**FULL RANK (cap 256, cut 1e-14 — nothing discarded):**

| arm | 3 steps | 12 | 48 | reading |
|---|---|---|---|---|
| `tdvp2` `1−Tr` | 1.7e-14 | 1.5e-14 | 3.0e-13 | **exact** |
| `bug_interleaved` `1−Tr` | 4.91e-03 | 3.11e-04 | 1.95e-05 | **÷16 per ÷4 dt ⇒ O(dt²)** |

**CAP 2 (truncation binds):** `tdvp2` `1−Tr` = 2.49e-02 → 2.22e-02 → **2.15e-02**;
`bug_interleaved` = 2.85e-02 → 2.31e-02 → **2.17e-02**. Roughly flat in dt, and **the two schemes
are within 1 % of each other**.

**THREE CONCLUSIONS, AND THEY POINT DIFFERENT WAYS:**

1. **At full rank the loss is the GALERKIN PROJECTION, exactly as the algebra predicts.** `tdvp2`
   is a product of local exponentials and each local generator annihilates `⟨⟨I|`, so it conserves
   the trace to roundoff. BUG evolves `exp(τ·P𝓛P)`, and `Tr(Pρ) = ⟨⟨PI|ρ⟩⟩`, which needs
   `P|I⟩⟩ = |I⟩⟩`.
2. ⚠️ **But it is the SAME ORDER as the method's own error, not a separate pathology.** At 48 steps
   the state error is 9.56e-06 and `1−Tr` is 1.95e-05 — both `O(dt²)`, both vanishing together.
   The trace is conserved *to the accuracy of the integrator*.
3. ⛔ **Under truncation `tdvp2` loses just as much trace as BUG** (2.15e-02 vs 2.17e-02). So the
   truncation-driven loss is a **shared** mechanism, not a BUG defect — and in production, where
   the cap binds, it will dominate both schemes. **That makes the rank cap the operative fix, not
   the integrator** (see FLAG 2: the square already hits χ = cap).

### 3.10b Is the trace loss a DEFECT? No — `cbe_bug` is exact when the basis is complete

The obvious objection to §3.10 is that it is strange for BUG to need a correction where `tdvp2`
does not, and that this smells like an implementation bug. It was worth testing rather than
arguing, and the test is sharp: **at full rank the Galerkin basis is the whole space, so
`ψ_new = W·exp(τH₀)·W†ψ = exp(τ𝓛)ψ` exactly** — BUG must be exact to roundoff or something is wrong.

`benchmarks/bug_closed_exactness.jl`, closed spin-½, against the dense propagator:

| system | arm | 3 steps | 48 steps | `b_L` | `b_R` |
|---|---|---|---|---|---|
| L=2 (space 4 = 2×2) | `tdvp2` | 7.22e-16 | 4.56e-15 | — | — |
| L=2 | **`bug_interleaved`** | **4.58e-16** | **3.48e-15** | **2** | **2** |
| L=3 (χ=2 manifold already spans 8) | **`bug_interleaved`** | **1.44e-15** | **3.52e-15** | 2 | 3 |

✅ **`cbe_bug` IS EXACT AT A COMPLETE BASIS**, and growth-independent there (`g` = 2 / 8 / 32
bit-identical) — exactly the invariance completeness predicts. **No implementation defect, and the
closed campaign results are unaffected.**

⛔ **THE FUSED `d = 4` BASIS IS NEVER COMPLETE, AND THAT IS THE WHOLE EXPLANATION.**
`benchmarks/bug_exactness_probe.jl` at L=2 fused, reporting the MINIMUM frame width over steps:
`b_R` is **2** at `growth = 2` and **3** at `growth ≥ 4` — never the available 4. The Galerkin space
is therefore ≤ 4×3 = 12 of 16, and the O(dt²) error is the manifold projection. It also explains
the growth-dependence: `growth` moves `b_R` (2 → 3) and the error tracks it
(2.63e-03 → 9.55e-06). CBE grows the basis toward `H|ψ⟩`, which is the right thing per unit cost,
but it does **not** saturate the local space even where saturation is affordable.

⚠️ **THREE PROBE DEFECTS BIT THIS ONE QUESTION — record them, because each produced a confident
wrong answer:**
1. `expanded[c]` records only the LEFT frame (`:620`; the right half-sweep skips `j == c`), so
   completeness was being judged on half the criterion. Fixed by adding `root_right` to
   `CBEBugSweepInfo` — pure observability, one field, one construction site.
2. `max` over steps where the claim ("complete at EVERY step") needs `min`. A max reports the best
   step and hides a starved first one.
3. `minimum(x; init = 0)` seeds a MINIMUM at zero and therefore always returns 0 — it reported
   `b_L = 0` universally, which reads as a catastrophically starved basis and is pure artefact.
   `init = 0` is right for `maximum` and wrong for `minimum`; the two are not symmetric.

### 3.11 ✅ FIXED — restore the trace ALONG `|I⟩⟩`, do not rescale

Two remedies, both measured at full rank (`benchmarks/trace_preservation.jl`):

| arm | 3 | 12 | 48 | `Tr ρ` | vs uncorrected |
|---|---|---|---|---|---|
| `bug_interleaved` | 2.4236e-03 | 1.5265e-04 | 9.5550e-06 | drifts | — |
| `+rescale Tr` (`ρ/Tr ρ`) | 4.6628e-03 | 2.9225e-04 | 1.8287e-05 | 1 exactly | **1.91× WORSE** |
| **`+restore along \|I⟩⟩`** | **1.8096e-03** | **1.1375e-04** | **7.1162e-06** | **1 to 3e-15** | **0.745× BETTER** |

⛔ **RESCALING OVERCORRECTS.** `ρ/Tr(ρ)` multiplies the TRACELESS part too, and the BUG error is
not a pure trace deficit — so it buys the conservation law at a consistent 1.9× accuracy cost and
merely stops the loss advertising itself.

✅ **THE FIX IS THE MINIMAL CORRECTION:** `ρ ← ρ + (ε/2^{L/2})·|I⟩⟩` with `ε = 1 − Tr ρ`. It moves
ONLY along the identity direction, leaves the traceless part exactly alone, and is therefore not a
trade at all — **exact trace AND a 25 % better state**, at every dt (0.747 / 0.745 / 0.745).
It works because `|I⟩⟩` is a PRODUCT state in the fused layout (χ = 1, measured bond dims `[1]`),
so the addition costs one bond dimension and truncates back. `TRACE_FIX = :along_I` is the default
in all three drivers.

⚠️ **IT IS NOT EXACT UNDER A BINDING CAP.** At cap 2 the closing truncation sheds part of the added
direction: `1−Tr` goes 2.17e-02 → **2.49e-03** (≈9× better, improving with more steps as each
correction is re-applied), but not to machine precision. Exactness there needs the truncation
itself to preserve the `|I⟩⟩` direction — i.e. the structural change inside `cbe_bug.jl`
(`_union_left`/`_union_right` at `:526`/`:574` already do this kind of union). ⚠️ That is a change
to the **production sweep every campaign uses**, so it belongs behind an opt-in keyword with its
own gate, and is not needed while the caps are chosen not to bind.

**Machinery added** (`fused_common.jl`, benchmark layer only — no library change):
`add_mps` (bond direct sums: row at the left boundary, block-diagonal interior, column at the
right) and `restore_trace!`. ⚠️ Two traps handled: `b`'s **bond** tags are retagged to `a`'s before
each `oplus` (Telum refuses mismatched itags, and the two states come from different constructors)
while PHYSICAL legs are left alone — retagging those would silently pair the wrong sites; and the
direct sum destroys the canonical form, so the centre is reset and the state re-gauged rather than
left stale.

⚠️ **FLAG 2 — the square hits the rank cap almost immediately.** χ reached **64 = the cap** after
four steps on a 2×3 patch. At 5×5 the shipped `cap = 256` will bind early, so those runs will be
rank-limited rather than dt-limited, and a rank-limited comparison measures the CAP and not the
integrator. Either raise the cap for the square or state plainly that it is an error-at-fixed-rank
comparison.

⚠️ **5×5 with `periodic_y` is NON-BIPARTITE.** An odd circumference makes the wrap bond join two
sites of the SAME sublattice, so a Néel pattern is frustrated on every wrap and the staggered
magnetisation measures a pattern the lattice cannot support — it would still return a number, and
that number would still decay under dephasing. The driver therefore defaults to an OPEN 5×5 patch
(still 2D-on-MPS, still no PEPS), with the cylinder behind a flag and a warning for odd `Ly`.

⚠️ **`mpo_vs_dense.jl` involves no solver at all**, which is what makes it decisive. It compares
`<v|W|v>` from `mpo_energy` (which returns `ComplexF64` and takes no `real` part, so it is valid on
a non-Hermitian MPO) against `dv' * W_chain * dv`. Probe states come from `tdvp2` under a Hermitian
MPO — any generic state probes an operator, so the probe generator's own accuracy is irrelevant.
What is NOT allowed is a product state: the XY part is purely off-diagonal, so `<a|W|a> = 0` for
both the right and the wrong operator.

⚠️ **Truncation is not a confound in either rung.** With the bra half untouched the interleaved
chain needs only `χ = χ_ket ≤ 2^(L/2)`, far under the cap; and the step functions do **not**
renormalise (verified — the docstrings state the caller renormalises), so `Tr ρ = 1.1618` at
`γ = 0.5` is a real failure and not a normalisation artefact.

**Found along the way (fixed):** `_grow_frame` — Jan's §4 growth path — still built a REAL
`alpha`/`betas` pair and called `_kry_contribution` with its old three-argument signature, which
the non-Hermitian rewrite had removed. That is a `MethodError` on the `krylov_grow = true` **and**
`ktol > 0` path, which nothing in the suite exercised. It now accumulates the same upper-Hessenberg
projection as `_krylov_frame`.

---

## 4. Work item 3 — literature validation, per model

A benchmark only validates the method if the target is a published, qualitative trend that a small
cluster can show. One per model:

| model | published signature | reference |
|---|---|---|
| **XX open** | dephasing turns BALLISTIC transport DIFFUSIVE — front spreads like `√t`, current `j ~ 1/n` | [Žnidarič, NJP 12 043001](https://iopscience.iop.org/article/10.1088/1367-2630/12/4/043001/pdf); [arXiv:2311.07375](https://arxiv.org/abs/2311.07375) |
| **Heisenberg square open** | **ZENO SLOWING**: relaxation of the magnetisation slows as the dephasing rate rises — reported for XXX and XX chains under onsite dephasing, against Bethe ansatz and ED | [arXiv:2408.10304](https://arxiv.org/pdf/2408.10304); 2D open-lattice TN methods: [arXiv:2512.01781](https://arxiv.org/pdf/2512.01781), [arXiv:2012.12233](https://arxiv.org/pdf/2012.12233) |
| **kagome (dipolar XY)** | Dirac→chiral spin liquid; correlations algebraic → exponential; entropy feature near `h ≈ 0.22` | [arXiv:2603.21147](https://arxiv.org/html/2603.21147v1) |
| **LGT open** | thermalisation time RISES with dissipation, temperature, background field and mass | [arXiv:2501.13675](https://arxiv.org/html/2501.13675) |

⛔ **AND THE SQUARE'S OBSERVABLE HAD TO BE CHOSEN AROUND A CONSERVATION LAW.** Pure dephasing
(`L_j = Sᶻ_j`) leaves `d⟨Sᶻ_k⟩/dt` from the dissipator **exactly zero** — `Sᶻ_k` commutes with every
`Sᶻ_j`, so the jump and anticommutator terms cancel identically. The bath therefore does **not**
damp the magnetisation directly; it acts only by destroying the coherences the Hamiltonian uses to
move it. So the signature is not "dephasing makes `M_stag` decay" (it would be measuring the
Hamiltonian) but **"larger γ makes it decay MORE SLOWLY"** — the Zeno trend above. Reporting the
decay itself as a dissipation effect would be a plausible-looking claim about the wrong term.

⚠️ **All three are TREND validations, not number reproductions**, and the write-up must say so.
The kagome paper uses iDMRG at `D = 10⁴` on infinite cylinders and states its critical point moves
with width; ours is an 18-site open cluster. Already measured: `S_vN` peaks at `h = 0.22`, which is
suggestive — but `states = 96` is the cap at every `h` and `|dE| ~ 2.5e-3`, so the ground states are
neither rank-converged nor fully relaxed.

⚠️ **⟨χ⟩ is exactly zero on any finite real ground state** (verified by ED at every `h`): every
term of the chiral order parameter carries one `σʸ`, so it is purely imaginary, and a real
Hamiltonian has a real ground state. That is a theorem about real matrices, not a property of our
solvers, and it is why the kagome validation uses the T-EVEN observables.

---

## 5. Work item 4 — the L=18 suite

Every model at **L = 18**, integrators `tdvp2`, `tdvp_cbe1s`, and `cbe_bug` at **m ∈ {0, 3}** and
**adaptive `krylov_tol` ∈ {1e-4, 1e-6}**. Tracked on all three axes: **error** (against an exact
sector reference where one exists), **operator applications**, and **wall clock**.

⚠️ **THE TWO COST AXES DISAGREE AND BOTH MUST BE QUOTED.** Measured on an 8×8 cylinder: `cbe_bug`
used **19× fewer** operator applications and took **1.91× longer**. `krylov` cannot see the CBE
expansion's SVD/QR work. At 18 sites they have agreed so far; that is not a reason to quote one.

⚠️ **Starts differ by symmetry and it must be stated, not silently mixed.** Néel is NOT
SU(2)-representable, so SU(2) models start from `dimer_state`; and a dimer's `Sᶻ` profile is
identically zero under SU(2)-symmetric evolution, so those models are measured on **bond
energies**, never on `Sᶻ`. `dimer_state` also needs even `L`, so an 18-site SU(2) chain needs a
different start or a different length — decide once and record it.

---

## 6. Order of work — status

1. ✅ **Work item 1** — `hermitian` threading + non-Hermitian `_krylov_frame`. Committed
   (`8160827`), both gates measured, closed-system numbers bit-for-bit identical.
   ⚠️ Two follow-ups found later: `_grow_frame` (Jan's §4 path) still called `_kry_contribution`
   with its removed 3-argument signature — a `MethodError` on `krylov_grow = true` **and**
   `ktol > 0`, which nothing in the suite exercised; and the existing Arnoldi test covers only a
   **normal** (sector-diagonal) non-Hermitian generator, not a non-normal one. Both fixed/noted.
2. ✅ **Work item 2** — the vectorised Lindbladian. Validated against the exact dense propagator;
   see §3.1–3.7. The layout decision was reversed on measurement (§3.4).
3. **XX open** — driver + sbatch written, awaiting the cluster.
4. **Heisenberg square open** — 5×5, written. ⚠️ open patch, not a cylinder (§3.6).
5. **LGT open** — Schwinger chain written, with the electric term expanded to all-to-all `ZZ` plus
   a one-body field.
6. **L=18 closed suite** — ✅ `benchmarks/submit_cbe_sweeps_l18.slurm` written. It reuses
   `benchmarks/cbe_sweeps_l16.jl`, which already does this with ANALYTIC references (Bethe, free
   fermions, Haldane-Shastry closed form) and takes `L` as an argument. ⛔ Every reference is a
   FORMULA, so nothing builds a `2^L` object and L=18 costs no more in reference terms than L=16.
   ⚠️ L=18 is EVEN, which `dimer_state` requires — the SU(2) arms start from a dimer covering
   because a definite-`Sᶻ` product state is not SU(2)-representable at any bond dimension. The
   L=16 script is kept as the record of the completed campaign.
   ⛔ Its scheme list contained `bug_decoupled` and `bug_interleaved`, which became **the same
   run** when the K-step / `kaug` / `rexpand` were removed on 2026-08-24 — both branches called
   `cbe_bug_step!` with identical arguments. Two entries producing one curve doubles the array and
   reads as agreement between methods in a plot. Replaced by `cbe_bug_m0 / m3 / tol4 / tol6`,
   which is also what item 2 asked for (`m` varied, adaptive to `1e-4` and `1e-6`).
7. **rSVD**, deliberately last: the wide LGT MPO is the first generator in this study where the
   sketch has a chance, since `npre/(d·χ) → 0` needs a FIXED `dex` at large `χ`, and the shipped
   `growth = 2.0` makes it a constant 60% of full. ⚠️ §3.7 is a prerequisite — `growth` must be
   re-derived for `d = 4` first, or every fused-chain cost number carries a tuning artefact.

**Trees are explicitly out of scope.** A single bath is a chain and needs no tree; trees earn
their place only for *multiple/structured* environments. `exploratory_bug/src/TTN/` already has a
Ceruti–Lubich tree BUG (2123 lines, `ttn_bug_step!`, algorithms 5/6/7), but it is ITensor-based
while `cbe_bug` is Telum/LurCGT, and it has no CBE — so that would be a port across tensor
libraries plus a CBE implementation, not a small generalisation.

---

## 7. THE THREE-MODEL CAMPAIGN — definition and status (2026-08-28)

The claim under test is **BUG is more robust than 2-site TDVP**, shown on log grids where a log
grid is legitimate and on a uniform grid where it is not, with symmetry as our addition.

| # | model | time | grid | symmetry | status |
|---|---|---|---|---|---|
| 1 | Heisenberg chain, L=16 | imaginary | log | SU(2) ✅, U(1) ✅ | **BUG 8.8× fewer matvec, 4 orders more accurate** vs Bethe |
| 2 | Heisenberg cylinder, L=16 | imaginary | log | SU(2) ✅, U(1) ✅ | **BUG 12.2–12.4× fewer matvec**; QMC anchor ⛔ NOT yet quotable |
| 3 | **Boundary-driven XX chain (open)** | real | uniform | ⛔ `:none` only | **TWO REGIMES** — tdvp2 more accurate at `eps=1` (§7.1c); tdvp2 **blows up** at `eps≳24`, BUG stable (§7.1d) |

`square_open` (the 8×2 dephasing patch) is **dropped**.

⚠ **TWO FURTHER OPEN MODELS ARE BUILT AND CERTIFIED BUT ARE NOT MODEL 3.** Keep them; do not
promote them into the headline without being asked.

* **Dissipative Heisenberg** (Hryniuk & Szymańska, Comms Phys 2026 9:45) — `dissipative_heisenberg.jl`.
  Generator certified against a dense Liouvillian to 2.1e-15; the observable path (`trace_op`,
  `pauli_1site`, `pauli_2site`) certified in three gates, including the one that actually pins the
  `σʸ` SIGN (see §7.5). `protocol` + `submit_dissipative_heisenberg.slurm` are written.
* **Cui–Cirac–Bañuls dissipative Ising** (PRL 114, 220601) — `dissipative_ising.jl`, certified to
  1.4e-14 with the purity normalisation pinned separately. Exists to support a ROBUSTNESS argument
  about dark states, recorded in §7.6.

### 7.1 What the grid argument actually rests on

⚠ **THE FIGURE IS DRAWN ON THE DISSIPATIVE *HEISENBERG* CHAIN, NOT ON MODEL 3.** `spectrum()` here
is `dissipative_heisenberg.jl`'s, and the two numbers below are that model's. Model 3 makes the
same argument with its own, sharper constants (a QUADRATIC observable ⇒ Nyquist step `π/4`, not
`π/2`; measured at N=30, `n = 2000` log steps overshoot by 79×) — see `submit_dissipative_xx.slurm`.
Attributing 10.55/14.29 to the boundary-driven XX chain would be wrong.

`spectrum()` + `plot_spectrum.py`. `−H` is Hermitian ⇒ `max|Im λ| = 0.000e+00` ⇒ every mode is a
monotone decay, nothing can alias, and a log grid's exponentially growing steps cost nothing.
`L` is non-Hermitian ⇒ `max|Im λ| = 10.55 (L=3), 14.29 (L=4)` ⇒ decaying **oscillations**.

⛔ **THE SHARP ARGUMENT IS NOT "THE LOG GRID ALIASES", IT IS THAT THE LOG GRID DEGENERATES.**
Aliasing alone (2×–15× over) invites "then use more log points". But Nyquist caps the **largest**
step at 0.22–0.30 regardless of how many points there are, and a log grid spends its resolution at
small `t` — so it needs ~100 steps where a uniform grid needs ~21–27 to cover the same horizon.
The log grid is not merely wrong here, it is *worse than the thing it was meant to beat*.

⚠ The Liouvillian gap is ≈ 0.50 and essentially **N-independent** (the dissipation is uniform, not
boundary-driven), so the steady state sits at `t ≈ 6` at every size. That is what makes a uniform
grid affordable — and it is also why this model, unlike the XX chain, does not need `t ~ 1300`.

### 7.1b Model 3 as re-scoped: what the XX open run must show

Three things on one trajectory per arm, which is what `dissipative_xx.jl chigrowth` emits:

1. **Accuracy against an ANALYTIC solution** — `xx_boundary_sz`, the closed-form Lyapunov result,
   `O(N³)` and valid at *every* `t`, certified to 7.2e-16 against a dense Liouvillian at N=2..5.
   Each checkpoint is scored at the time the run actually reached. No dense object is ever built.
2. **χ(t)** — the superket's operator entanglement, which is what carries transport.
3. **Wall-clock**, beside `matvec`. If they disagree about which arm is cheaper, believe `matvec`.

⛔ **THE NESS IS NOT REACHED AND MUST NOT BE CLAIMED.** This model's Liouvillian gap closes as
`N⁻³` — already `2.5412e-02` at N=8, so a true steady state needs `t ≈ 236` (~12,000 steps at
dt=0.02), and `t ≈ 1300` at N=16. Everything local is the **transient**, which is exactly where χ
grows and where the integrators can differ. The closed-form `j = 0.4` is drawn as an asymptote for
orientation only. Charging an integrator for the residual physical transient would report
relaxation as integrator error.

⚠ **MEASURED COST, and it decides the run parameters.** At L=8, dt=0.05, cap=64, maxiter=30:

| t | χ | matvec | seconds | s/step |
|---|---|---|---|---|
| 0.25 | 16 | 3,601 | 24.4 | 4.9 |
| 0.50 | 35 | 7,501 | 93.4 | 13.8 |
| 0.75 | **64 = cap** | 11,401 | 226.1 | 26.6 |

χ saturates the cap at `t ≈ 0.75` and cost climbs as χ³. Two consequences: `maxiter` is dropped to
**8** (`lanczos_expv` exits on BREAKDOWN only, so every solve burns its full depth — a measured
3.64× at 2e-13, four orders below this run's truncation floor, applied identically to every arm);
and past `t ≈ 0.75` this is an **error-at-fixed-rank** comparison, which is the right question once
a rank budget binds.

### 7.1c ✅ MODEL 3 HAS TWO REGIMES — read 7.1c AND 7.1d together, never one alone

⚠ **THIS SECTION WAS ORIGINALLY TITLED "BUG LOSES ON MODEL 3" AND THAT WAS AN UNDER-POWERED
CONCLUSION.** Everything in it is correct *at the campaign's default dissipation* (`eps = 1`). At
strong dissipation the ranking REVERSES and tdvp2 becomes unstable — §7.1d. Quoting this section on
its own would misrepresent the result.

#### Weak dissipation (`eps = 1`), small `dt`: tdvp2 wins on accuracy

L=8, dt=0.02, cap=64, maxiter=8, t=0.8, against the analytic Lyapunov solution:

| arm | matvec | seconds | χ | err |
|---|---|---|---|---|
| 2-site TDVP | 8,271 | **138.9** | 64 | **4.98e-08** |
| BUG (rSVD-CBE) | **315** | 204.7 | 64 | 4.90e-05 |
| BUG (parallel) | **315** | 169.6 | 64 | 4.90e-05 |

⛔ **26× FEWER MATVEC AND STILL SLOWER.** On this model `matvec` is NOT a cost proxy — BUG's CBE
expansion (rSVD sketch, QR, SVDs) dominates and the matvec count cannot see it. The campaign's
"believe matvec over seconds" rule is about CONTENTION and does **not** license quoting 26× as a
speedup here. Models 1–2 are matvec-bound; this one is not.

**Three candidate excuses, all tested, all eliminated:**

| suspect | test | verdict |
|---|---|---|
| Krylov depth | `maxiter` 8 → 30 | err 5.3e-05 → 5.644e-05, matvec 139 → 565 ⇒ **innocent** |
| rank budget | χ at t=0.1 | err already 1.7e-05 at χ=12, cap is 64 ⇒ **innocent** |
| CBE `growth` | 2 / 4 / 8 | 1.395e-04 / 5.644e-05 / **5.644e-05 (identical)** ⇒ **saturated at 4** |

✅ **It is an ORDER gap**, and it was already on record: on the fused MPDO tdvp2 converges at
~2.98 and bug_rsvd at ~1.98. An order gap *widens* as dt shrinks.

⇒ **ANY BUG-vs-TDVP CLAIM ON AN OPEN MODEL MUST EQUALISE ACCURACY, NOT dt.** Extrapolating (BUG
2nd order ⇒ needs dt/31.6 for 1000×): ~10,000 matvec against tdvp2's 8,271 — comparable in matvec,
clearly worse in wall time. ⚠ EXTRAPOLATED, NOT MEASURED; measure before quoting.

⚠ `bug_par` is numerically IDENTICAL to `bug_rsvd` (same matvec, same err to every digit) and 1.21×
faster — the parallel half-sweep works, it just cannot change the order.

⚠ **A SIGN BUG WAS FOUND HERE AND FIXED**: `mps_current` returned the NEGATIVE of the true current
(magnitudes matching to 3 figures at every checkpoint). Its docstring had counted `overlap`'s
conjugation but not `site_apply!`'s transpose — the two CANCEL. It survived because a wrong current
sign leaves `⟨Sᶻ⟩` untouched, and `ness()` (the only gate comparing currents) is cluster work that
had never run.

### 7.1d ✅ STRONG DISSIPATION: tdvp2 IS CONDITIONALLY STABLE, BUG IS NOT — the robustness result

TDVP2 removes the double-counting of its two-site sweep with a **backward** substep
(`tdvp2_baseline.jl:124,136` — `expv(..., -tau, M)`, verified by line). On a Lindbladian `tau` is
**real** and the spectrum sits in the left half-plane, so that step carries `exp(+|Re λ|·dt/2)` and
is **anti-dissipative by construction**. `cbe_bug.jl:844` has one forward `expv` and none backward.

⛔ **THE AXIS IS DISSIPATION STRENGTH, NOT `dt` AND NOT TIME.** `|Re λ| ~ eps`, so at `eps = 1` no
reachable `dt` makes the exponent O(1) — three separate scans (purity at γ=0.1, `dt` to 0.8,
long-time at fixed rank to t=2) all came back NULL for exactly this reason. Escalating `eps` finds
it at once. ⚠ A null from an under-powered scan is a statement about the parameter, not the method.

`dt_robustness`, L=8, cap=32, t_max=0.8:

| eps | dt | tdvp2 err | tdvp2 raw `Tr ρ` | BUG err | BUG `Tr ρ` |
|---|---|---|---|---|---|
| 16 | 0.4 | 7.85e-04 | 1.000000 | 2.64e-02 | 1.000000 |
| 16 | 0.8 | **2.30e-01** | 1.000000 | **6.04e-02** | 1.000000 |
| 24 | 0.8 | **TRACE LOST** | **6.27e+07** | 4.47e-02 | 1.000000 |
| 32 | 0.4 | **TRACE LOST** | **2.82e+09** | 1.45e-02 | 1.000000 |
| 64 | 0.4 | **TRACE LOST** | **2.04e+15** | 7.77e-03 | 1.000000 |
| 64 | 0.8 | **TRACE LOST** | **1.98e+31** | 1.90e-02 | 1.000000 |

⛔ **`Tr ρ = 1.98e+31` IS THE RESULT, NOT THE `err` COLUMN.** `exp(Lt)` is trace-preserving
*exactly*, so any deviation is pure integrator error — 1e31 cannot be truncation. It grows
monotonically in **both** `eps` and `dt`, the signature `exp(+|Re λ|·dt/2)` predicts. BUG holds
`Tr = 1.000000` at every point.

✅ **There is also an accuracy crossover INSIDE the stable region**: at `eps=16, dt=0.8` BUG is
**3.8× more accurate** (6.04e-02 vs 2.30e-01) *before* tdvp2 fails at all. So the claim is not only
"survives vs crashes".

⚠ **BUG'S OWN FAILURE MODE, which must be reported beside it**: at `dt = 0.8` its `χ` collapses to
**1** at every `eps` — CBE cannot build rank from one huge step — so its survival there is a
product state with 2–6e-02 error. At `dt = 0.4` it keeps `χ = 13..32` and is genuinely usable
(7.77e-03 at `eps=64`). ⛔ Its `Tr` reads a perfect 1.000000 the whole time, so a trace-only gate
would score the collapsed state a pass.

⚠ `eps ≳ 16 J` is the strong-measurement / Zeno regime (the NESS current peaks at `eps = 2J` and
falls as `1/eps`). Physically legitimate — but name the regime rather than stating the claim flat.

**Suggested figure:** the `(eps, dt)` stability boundary with raw `Tr ρ` on a log axis. The 1e31 is
the entire argument in one number.

### 7.2 ⛔ NO open model here has a symmetry arm — the fused space itself is `:none`

`fused_space()` builds the d=4 site through **`_getLocalSpace_no_symmetry`** — the route
`z2_ising.jl` also had to take, because `getLocalSpace` derives its space from a *known* symmetry
type and cannot be handed an arbitrary one. **Every fused superket in this repo is `:none`**,
whatever the model. A symmetry arm needs a new d=4 local space graded by the difference charge
`q = n_ket − n_bra`, sectors `{−1, 0, 0, +1}` — real work, not a parameter change.

⚠ **THIS IS SHARPEST FOR THE XX CHAIN, WHICH *DOES* HAVE THE SYMMETRY.** XX conserves `Sᶻ` and the
boundary jumps are the only thing that breaks it, so U(1) is physically available here and we
still cannot exploit it. (For the Hryniuk Heisenberg model there is a second, independent
obstruction: `Jx ≠ Jy` opens the `S⁺S⁺`/`S⁻S⁻` channel, so the *model* has no U(1) either —
⚠ note the dissipator is innocent, since `kron(SM,SM)` lowers ket and bra together and preserves
`n_ket − n_bra`. Moving to XXZ would fix the model and still leave the space `:none`.)

⇒ **the symmetry axis is carried by models 1 and 2**, and must be *stated* rather than left to
read as an oversight. Model 3 is a clean BUG-vs-TDVP comparison at `:none`.

### 7.3 U(1) — ✅ FIXED (was blocking every symmetry result), and what is still open

**U(1) was unusable for every BUG arm**, on the chain *and* the cylinder, returning
`E = NaN, mv = 0, chi = 0` in under two seconds while `tdvp2` ran normally. Two independent itag
collisions, both now fixed:

1. `fusion_basis` discarded its `tag` through the final SVD, so every CBE probe came back `svdL`
   and collided inside `cbe_expand` (`src/BondUpdateBUG/sectors.jl`).
2. `_dimer_state_projected` wrote `res.U` / `res.S*res.Vd` back **without restoring the bond tag**
   (`src/BondUpdateBUG/sweep.jl:121`). Its loop is `for i in 1:2:(L-1)` — exactly the odd bonds,
   which is the observed pattern. `move_right!` then built `res.S * res.Vd` with `S = (svdL,svdR)`
   and `Vd = (svdR,svdL)`: the same itag twice in one TLArray.

⛔ **It had silently confined every BUG result in this repo to `:SU2` or `:none`** — `dimer_state`
delegates to the projected constructor only when the mode is *not* `:SU2`, and the `:SU2` branch
builds by fusion and never SVDs. ✅ Audited: this was the only such site in `src`.

✅ **Verified — and not merely that it stops throwing.** `benchmarks/imaginary_time/u1_cylinder_repro.jl`:

    chain / cylinder, SU2 and U1: all OK      svd-tagged sites = 0 in ALL modes, before and after
    E(20 imaginary-time steps, L=8):  SU2 = U1 = none = -3.3499056613
    |U1 - SU2| = 7.994e-15    |none - SU2| = 9.326e-15

`:none` is the control — it never hit the crash but *does* go through the patched constructor, so a
numerical regression from the retag would surface there. ⇒ symmetry changes the COST, not the
physics, which is the claim the symmetry arm rests on.

⚠ **`χ` IS NOT COMPARABLE ACROSS MODES.** Same state, same step: `χ = 6` under SU(2), `16` under
U(1)/`:none`, because SU(2) counts MULTIPLETS. ⛔ MEASURED on the L=16 cylinder: `tdvp2` at
`:U1, maxdim=128` gives `err = 2.9e-02` (and `1.05e-01` at n=12) against `2.76e-03` at
`:SU2, maxdim=70` — the U(1) run is **under-resolved, not worse-behaved**. A fair SU(2)-vs-U(1)
comparison must match the EFFECTIVE dimension, never `maxdim`.

**Still open:** the cylinder QMC anchor is not quotable at `Lxs = (2,3,4)`. All four arms agree on
`e_bulk = −0.6805250308` to ten digits, but `resid = 8.0e-03` and the driver's own "FIT NOT LINEAR,
do not quote" guard fires: three short lengths cannot reach the bulk. Needs `Lxs = (4,6,8)`
(16/24/32 sites) — `benchmarks/imaginary_time/submit_logtime_suite.slurm` task 1.

### 7.4 The `σʸ` gate — why the observable path needed a third test

Recorded because the *first two gates passed vacuously* and it would have been easy to stop there.
`certify` pins the generator with `⟨σᶻ⟩`, which is **diagonal**: it returns the same number whether
`site_apply!` applies `M` or `Mᵀ`. `σˣ` is real symmetric, so it cannot tell them apart either.
Gate (a) uses `|+x⟩` and gate (b) a Néel state — and **both have `⟨σʸ⟩ = 0` identically**, so each
compared 0 against 0. The `⟨yy⟩` correlator does exercise `σʸ`, but a sign enters it **squared**, so
it is blind to a global flip.

⇒ gate (c) evolves `|+x⟩`, which the field rotates *into* y, and compares a **nonzero** `⟨σʸ⟩`:

| gate | state | pins | result |
|---|---|---|---|
| (a) | `\|+x⟩`, t=0 | normalisation; connected correlators vanish | **0.000e+00** |
| (b) | Néel, t=0.25 | two-site operator order, site placement | 2.4e-07 / 1.9e-07 |
| (c) | `\|+x⟩`, t=0.25 | **the `σʸ` sign** (`⟨σʸ⟩ = 0.4118`) | **2.8e-09** |

⚠ The gate is **two-part**: a small deviation proves nothing if `⟨σʸ⟩` never left zero, so it
requires the signal to be PRESENT (`max\|⟨σʸ⟩\| > 1e-3`) before it accepts the agreement.

### 7.5 The dark-state robustness argument (Cui–Cirac–Bañuls)

Their method finds the NESS **directly**, minimising `‖L|Φ(ρ)⟩‖²` subject to `‖Φ‖₂ = 1` — the
smallest eigenvalue of `L†L`. That constraint is **Hilbert–Schmidt**: it enforces neither
positivity nor unit trace. With `dim ker L = 1` this costs nothing, because the unique null vector
is proportional to the physical state. Their conclusion states the failure otherwise, verbatim:

> "When the steady state is degenerate, the simplest method described in this paper might have
> problems to find a valid guess for the steady state. Specially in the situation of several dark
> states, the null subspace of the Lindbladian contains infinitely many vectors which do not
> correspond to positive operators and hence do not constitute valid physical states."

✅ **An evolution-based method cannot fail that way, and not by luck.** `exp(Lt)` is CPTP at every
`t`, so `ρ(t)` is positive and trace-one at every step and any `t→∞` limit of physical states is
physical. We do not *enforce* positivity; we inherit it. Degeneracy becomes a **selection rule**:
`ρ(∞) = Σ_m Tr(P_m ρ₀) · I_m/d_m` — the initial state picks the sector.

⚠ **AND THE CLAIM MUST STAY NARROW.** Theirs is *faster* than evolution when the steady state is
unique and low-rank; that is the result of their paper and it stands. The claim here is about
ROBUSTNESS only: where their functional has no way to return a physical answer, ours has no way to
return an unphysical one.

⛔ **The symmetry half of this argument is blocked by §7.2.** Their suggested remedy is to use
symmetries to reduce the degeneracy — which we cannot do at the tensor level for any open model.
What *is* demonstrable without a symmetric space is the physics: different `ρ₀` in different
sectors flow to different, valid steady states. Do not present that as a symmetric-tensor result.

### 7.6 ⚠ The Hryniuk Heisenberg model carries no `err` column, by construction

`4^16` is 4.3 billion amplitudes, so there is no exact reference and **our own finest grid is not
an anchor**. Two honest statements replace it: a three-point `dt` ladder (self-convergence, one
array task each) and *qualitative* agreement with the paper's published Fig. 3a-c — theirs is
N=200, ours N=16, so a numeric deviation between them measures the size difference, not an
integrator. `plot_dissip_heis.py` panel (d) is labelled SELF-convergence for this reason.

⚠ **`χ` at the cap invalidates the correlators specifically.** Operator entanglement is what
carries the connected part, so a too-hard truncation keeps a smooth `⟨σᵃ(t)⟩` and a perfect
`Tr ρ = 1` while `Cᵃᵃ_d` decays for a purely numerical reason. Check peak `χ` against `DH_CAP`
before reading panels (b) and (c).

## 8. THE LONG-RANGE RTE CAMPAIGN — `dissipative_longrange.jl` (2026-08-29)

Same paper (Hryniuk & Szymańska, Comms Phys 2026 **9**:45), same operator algebra, **different
coupling matrix and nothing else**. Three integrators — 2-site TDVP, 1-site CBE-TDVP
(arXiv:2208.10972, the published baseline), CBE-BUG — on the paper's own protocols.

| mode | model | geometry | their figure |
|---|---|---|---|
| `fig3` | nearest-neighbour XYZ + spin decay | chain | Fig. 3a–c |
| `fig4` | Eq. (6), diagonal `σᶻσᶻ`, α = 3 and 6 | chain | Fig. 4a,b |
| `fig4sq` | Eq. (6) | 4×4 open patch | Fig. 4c,d |
| `fig6` | Eq. (7), **non-diagonal** XYZ, `rmax = 4` | chain | Fig. 6c,d |
| `fig5` | Eq. (6), **steady state**, Kac-normalised, swept over `J₂` | chain | Fig. 5 |

`pair_mpo` weights each `PairVertex` by an arbitrary `J[i,j]` over every pair `i<j`, so a power law
is an **exact** MPO with **no exponential fitting and no extra channels** — `nn(c)` becomes
`c .* D` with `D[i,j] = d_ij^(-α)`. The cost shows up as bond dimension: **242** at L=16 for the
untruncated double power law, 66 for Fig. 6 where the paper itself truncates at r=4.

### 8.1 ✅ Certified against a dense Liouvillian, and why the `⟨σʸ⟩` start matters

`certify_lr` builds an independent dense Liouvillian at L=3–4 from the *same* site positions and
compares 1- and 2-site observables:

    Eq. (6):  dt 0.05 -> 2.4875e-05,  0.025 -> 6.1025e-06,  0.0125 -> 2.1877e-06   (order 4.08, 2.79)
    Eq. (7):  dt 0.05 -> 6.6311e-06,  0.025 -> 1.8643e-06,  0.0125 -> 5.3103e-07   (order 3.56, 3.51)

⛔ The start is `⟨σʸ⟩ = −1`, which makes this **strictly stronger** than the Néel-start gate:
`σᶻ` is diagonal and `σˣ` is real symmetric, so both read back the same number whether
`site_apply!` applies `M` or `M'`. `σʸ` is the only one of the three that can catch the transpose,
and here it is large from `t = 0`. `rho0_minusy` additionally *asserts* its own sign, because
`P_y` is Hermitian but not symmetric and the naive form silently prepares the **+y** pole.

### 8.2 ⛔ Fig. 5 is a CLAIM about a steady state, so it is gated

`steady_lr` reports `drift` — the largest change in `S_zz(q)` over the last quarter of the window,
relative to its final value. Above 2% the point is written `converged=0`, drawn **hollow**, and
excluded from the plotted line. An unconverged transient traces a smooth, entirely plausible curve
against `J₂`; nothing else in the output would distinguish it from a NESS.

`kac = true` **here and nowhere else**: the Kac factor rescales energy and therefore *time* with
`N`, so switching it on for the dynamics figures would shift every curve along `t` against theirs.

`szz_at` is shared by `protocol_lr` and `steady_lr` so the two cannot drift into different
normalisations. Its `t=0` value is `1/N` for **every** `q` (product state, `(σᶻ)² = I`) — the
cheapest available check that the prefactor and the q-grid are right; measured `0.1666666667` at
L=6.

### 8.3 Cost, measured — and two ways it was misread

L=16 chain, tdvp2, MPO dim 242, against a recorded cap-32/maxiter-16 baseline:

| config | s/step | speedup | `krylov` | `d⟨σᵃ⟩` |
|---|---|---|---|---|
| cap 32, maxiter 16 | 214.4 | 1.00× | 1856 | — |
| cap 20, maxiter 16 | 132.8 | 1.61× | 1856 | 4.4e-06 |
| **cap 20, maxiter 8** | **59.8** | **3.59×** | **928** | **4.4e-06 — identical** |

Halving the Krylov depth changed the observables by **nothing** (the 4.4e-06 is entirely the cap):
`arnoldi_expv` exits on breakdown only, so the other 8 vectors were burned for free. ⚠ Measured,
not assumed — §7 records the opposite case, where too few vectors survived every other check.

`rmax` is the one lever that changes the model, so it was priced and **declined**: 242 → 98 in MPO
dim bought 18% of the wall clock and cost **15.6×** in `d⟨σˣ⟩`. The χ-dependent linear algebra
dominates, not the MPO bond.

⛔ **Two misreadings of this table, both mine, both worth keeping:**

1. **The probe measured 2 steps from a product state**, where χ is still climbing 1→20, and so
   priced the cheapest steps of the run and called them typical. Production is ~198 s/step at
   χ = 20. **A cost probe must run past the rank-growth transient.**
2. That 3.3× was then blamed on BLAS threading and "fixed" — costing a restart. Total CPU demand
   was **4.27 cores of 12**; the box was never the constraint, and raising `OPENBLAS_NUM_THREADS`
   to 4 made it *worse* (31.5 s/step vs 20.4) by thrashing threads on χ=20, d=4 tensors. The tell
   that the machine was innocent: **`krylov` was 464 per step in both readings** — identical work,
   different wall clock.

⇒ **`krylov` is the cost axis; `seconds` is a sanity check.** The four figures run concurrently
(`benchmarks/run_dissipative_l16.ps1`), which makes wall clock non-comparable by construction. For
publication wall-clock, run one job at a time.

### 8.4 ⛔ No SU(2) or U(1) arm here either — and it is the fused space, not the couplings

Exactly as §7.2. `Jx ≠ Jy` in Eq. (7) opens the `S⁺S⁺`/`S⁻S⁻` channel and kills U(1) at the
Hamiltonian level, but that is not the binding constraint: `fused_space()` builds the d=4 site
through `_getLocalSpace_no_symmetry`, so **every fused superket in this repo is `:none` whatever
the couplings are**. The campaign's symmetry axis is carried by the *closed* models — Heisenberg
chain, cylinder, kagome. Say so explicitly rather than letting a missing `sym` column read as an
oversight.
