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
| XX **open** | U(1) | 18 | — | ⛔ blocked, see §2 |
| Heisenberg square **open** | U(1) | 18 | — | ⛔ blocked |
| lattice gauge theory **open** | U(1) | 18 | — | ⛔ blocked |

Machinery already in place: `pair_mpo` with **one coupling matrix per vertex** (needed for any
model whose operator types have different ranges) and **complex couplings** (verified:
`E = −1.75 + 0.075im`). Both were added for this and are guard-tested.

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

---

## 4. Work item 3 — literature validation, per model

A benchmark only validates the method if the target is a published, qualitative trend that a small
cluster can show. One per model:

| model | published signature | reference |
|---|---|---|
| **XX open** | dephasing turns BALLISTIC transport DIFFUSIVE — front spreads like `√t`, current `j ~ 1/n` | [Žnidarič, NJP 12 043001](https://iopscience.iop.org/article/10.1088/1367-2630/12/4/043001/pdf); [arXiv:2311.07375](https://arxiv.org/abs/2311.07375) |
| **kagome (dipolar XY)** | Dirac→chiral spin liquid; correlations algebraic → exponential; entropy feature near `h ≈ 0.22` | [arXiv:2603.21147](https://arxiv.org/html/2603.21147v1) |
| **LGT open** | thermalisation time RISES with dissipation, temperature, background field and mass | [arXiv:2501.13675](https://arxiv.org/html/2501.13675) |

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

## 6. Order of work

1. **Work item 1** — `hermitian` threading + non-Hermitian `_krylov_frame`. Prerequisite for
   everything open, and gated on reproducing closed-system numbers bit for bit.
2. **L=18 closed suite** (work item 4 for the four closed models). Independent of 1, runnable now.
3. **XX open** via the vectorised Lindbladian (work item 2), validated against the ballistic →
   diffusive crossover. This is the one that proves the open-system machinery.
4. **Heisenberg square open** — same machinery, 2D, no new physics.
5. **LGT open** — the Schwinger spin Hamiltonian (NN hopping + all-to-all Coulomb, needs the
   per-vertex `pair_mpo` already added), then the same vectorised bath.
6. **rSVD**, deliberately last: the wide LGT MPO is the first generator in this study where the
   sketch has a chance, since `npre/(d·χ) → 0` needs a FIXED `dex` at large `χ`, and the shipped
   `growth = 2.0` makes it a constant 60% of full.

**Trees are explicitly out of scope.** A single bath is a chain and needs no tree; trees earn
their place only for *multiple/structured* environments. `exploratory_bug/src/TTN/` already has a
Ceruti–Lubich tree BUG (2123 lines, `ttn_bug_step!`, algorithms 5/6/7), but it is ITensor-based
while `cbe_bug` is Telum/LurCGT, and it has no CBE — so that would be a port across tensor
libraries plus a CBE implementation, not a small generalisation.
