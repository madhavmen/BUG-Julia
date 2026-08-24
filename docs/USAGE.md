# Using RSVD-CBE for the bond-update BUG

How to run the rank-adaptive BUG with RSVD controlled bond expansion. There is **one path**:
an `MPS`, an `MPO`, and a step function.

Runnable companions: [`examples/oat_z2.jl`](../examples/oat_z2.jl),
[`examples/xx_domain_wall.jl`](../examples/xx_domain_wall.jl),
[`examples/tfim_z2.jl`](../examples/tfim_z2.jl) — see [`examples/README.md`](../examples/README.md).
Design and measurements: [`docs/cbe_bug.md`](cbe_bug.md), [`docs/current_work.md`](current_work.md).

---

## 1. Quick start

```julia
using BUGJulia.BondUpdateBUG, BUGJulia.RSVDCBEBondUpdate

set_symmetry!(:U1)                                  # :none | :U1 | :SU2 | :Z2 — FIRST
psi = domain_wall_state(8)                          # any state; chi = 1 is fine
W   = xxz_mpo(8; J = 1.0, delta = 1.0)              # the Hamiltonian, as an MPO

for _ in 1:40
    cbe_bug_step!(psi, W, ComplexF64(-im * 0.05); maxdim = 32, trunc_thresh = 1e-10)
end

magnetisation(copy(psi))     # <Sz_j> per site
bond_dims(psi)               # how far the bond dimension grew: 1 -> [2,4,8,16,8,4,2]
mpo_energy(psi, W)
```

That is the whole surface: **pick a symmetry, build a state, build an MPO, call a step
function.** `tau = -im*dt` is real time and `tau = -dt` imaginary time — the same call either
way, and in imaginary time you rescale the state each step.

```bash
julia --project=. examples/xx_domain_wall.jl
julia --project=. examples/oat_z2.jl
```

## 2. Switching symmetry

One global call, before you build any tensors:

```julia
set_symmetry!(:none)    # no symmetry — one dense block per bond
set_symmetry!(:U1)      # Sz conservation — separate charge sectors  (default)
set_symmetry!(:SU2)     # total spin — multiplets
set_symmetry!(:Z2)      # spin-flip parity — the TRANSVERSE-FIELD ISING model
symmetry_mode()         # what is currently set
```

**`:Z2` is a different MODEL, not just a different symmetry.** The other three are spin-½
Heisenberg/XXZ spaces sharing `local_space()`; `:Z2` is the TFIM
`H = -J Σ Z_i Z_{i+1} - h Σ X_i`, whose symmetry is the global flip `P = Π_i X_i`. It has its
own local space (`ising_local_space()` → `I, X, Z`, no `Sp/Sz/Sm`), its own gates
(`ising_bond_gates`) and its own states (`ising_polarised_state`, `ising_kink_state`).
`local_space(:Z2)` deliberately **throws** rather than returning something spin-like, and
`set_symmetry!(:U1)` would be plainly wrong for the TFIM — `Z_i Z_{i+1}` does not conserve `Sz`.
See §3b.

**It must come before state construction**, because it decides how every tensor is laid out.
Switching mid-run leaves you with tensors from two different worlds.

Everything downstream is symmetry-agnostic — the same `evolve!` call, the same options, the
same observables. The example runs `:none` and `:U1` through an identical code path and gets
`max |<Sz>_none − <Sz>_U1| = 4.7e-15`: **symmetry changes the storage, not the physics.** A
visible difference between two symmetry modes is a bug, not a finding.

What each mode can represent:

| | `product_state` / `neel_state` / `domain_wall_state` | `dimer_state` | `<Sz_j>` | `<S_j·S_j+1>` |
|---|---|---|---|---|
| `:none` | ✅ | ✅ | ✅ | ✅ |
| `:U1` | ✅ | ✅ | ✅ | ✅ |
| `:SU2` | ❌ spans many spin sectors | ✅ | ❌ identically 0, and `Sz` is not exposed | ✅ |

So an SU(2)-vs-abelian comparison must use `dimer_state` + `bond_energies`, never a domain
wall + magnetisation. Asking for `magnetisation` under `:SU2` throws rather than silently
returning zeros.

## 3. One path: MPO environments

⛔ **THE GATE-BASED PATH IS GONE.** `RealTime.evolve!`, `ImaginaryTime.evolve!`,
`RealTime.evolve_mpo!` and the engine under them (`cbe_gate_evolve!`,
`cbe_bond_update_bug!`, `cbe_lubich_bug!`) were removed with the other BUG implementations.
Everything now goes through **MPO environments**, which is what the surviving sweeps take.

Nothing was lost in accuracy by that: the gate path carried a **Trotter splitting error of order
dt²** that the MPO path does not have at all.

```julia
W = xxz_mpo(8; J = 1.0, delta = 1.0)       # or heisenberg_su2_mpo(8) under :SU2
for _ in 1:n_steps
    cbe_bug_step!(psi, W, ComplexF64(-im * dt); maxdim = 32, trunc_thresh = 1e-10)
end
```

**Time direction is the `tau` argument, not a mode object.** `tau = -im*dt` is real time and
`tau = -dt` is imaginary time; imaginary time is not norm-preserving, so rescale each step:

```julia
renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)
```

The three sweeps all share this signature — `cbe_bug_step!`, `tdvp_cbe1s_step!`, `tdvp2_step!` —
so swapping one for another is a one-word change and any difference between them is the
integrator. `GroundState.rsvd_cbe_dmrg!` remains the variational ground-state mode.

For a single expansion sweep rather than a step loop: `cbe_lubich_sweep(psi, h, tau; maxdim, ...)`.

## 3b. The transverse-field Ising model under Z2

```julia
set_symmetry!(:Z2)                                   # or :none, same basis, same results
psi = ising_kink_state(8)                            # |+...+ -...->, chi = 1
W   = tfim_mpo(8; J = 1.0, h = 1.0)                  # H = -J ZZ - h X, virtual dimension 3
for _ in 1:40
    cbe_bug_step!(psi, W, ComplexF64(-im * 0.05); maxdim = 32)
end
x_profile(psi)                                       # <X_j>, the light-cone observable
```

Runnable, with the exact free-fermion reference and figures: [`examples/tfim_z2.jl`](../examples/tfim_z2.jl).

**Everything is in the σ^x eigenbasis, and that is what makes Z2 useful.** In the conventional
σ^z basis the flip `P = Π X_i` is entirely off-diagonal, so nothing is block-sparse and the
symmetry buys nothing. In the σ^x basis `|+⟩` is Z2 charge 0 and `|−⟩` is charge 1, so `X` is
diagonal and `Z` flips the charge — and `Z_i Z_{i+1}` changes it by `1+1 = 0 (mod 2)`. H
conserves the charge exactly, and the tensors go block-sparse.

The operator *names* refer to the operators, which are basis-independent: `x_profile` is the
genuine transverse magnetisation. Only the matrix representations are rotated.

**`⟨Z_j⟩` is identically zero in any Z2-symmetric state** — `Z` changes the charge, so it has no
diagonal matrix element at all. That is the symmetry, not a missing feature: the order parameter
must be probed with the correlator `⟨Z_i Z_j⟩`, never a single site. This is the Z2 analogue of
`⟨Sz_j⟩` vanishing under SU(2).

Bond dimensions under `:Z2` count **states** per charge sector, so unlike SU(2) they *are*
directly comparable with `:none`.

**The MPO path is now the supported one** — `tfim_mpo` builds the automaton explicitly, so the
TFIM drives the same environments and Krylov solves as every other model. The old note that only
the gate path supported ℤ₂ is obsolete: it was true when `XXZChain` was the only route to an MPO
and its channels had no on-site slot, which `tfim_mpo` does not go through.

⛔ **The two halves of the coupling are the adjoint pair `Z`/`Z'`, not `Z`/`Z`.** Mod 2 the flip
is its own charge partner (`1+1 = 0`), so `Z Z` is *charge*-correct and looks obviously right —
the failure is an **arrow**. An op-leg is created with direction `'-'`, so `Z`/`Z` gives the
opening block's outgoing virtual leg and the closing block's incoming one the same direction, and
Telum will not contract two `'-'` legs. It does not surface as an arrow error either: `oplus`
fails first, with *"entry 4 has no leg matching reference leg 1"*, which reads like a missing
mod-2 capability. `_oat_sz_vertex` documents the identical trap for `(Σ S^z)²`.

⛔ **A product-state energy pins the FIELD and not the coupling.** `Z` is the flip here, so
`⟨s|Z_i Z_{i+1}|s⟩ = 0` for *every* σˣ product state — an MPO with the coupling deleted or at the
wrong `J` reproduces `⟨H⟩` on all of them. Only a state that is not diagonal in this basis sees
it, which is why the coupling is pinned by a short quench against `exp(-iHt)` (measured 1.6e-13
at full rank, L=6).

The gate builders remain, and their field distribution over bonds is not uniform: an interior
site touches two bonds and takes `h/2` from each, while sites `1` and `L` touch one bond and must
take the **full** `h` there. Halving every site's field instead would quietly evolve a chain with
weakened boundary fields — a different Hamiltonian whose error *shrinks* with `L`, so small-size
tests would miss it. `tests/BondUpdateBUG/test_z2_ising.jl` pins both the gates and the MPO
against a dense `H` built independently of the tensor code.

⚠️ **The Z2 local space rests on Telum-private API.** `getLocalSpace` cannot build it: LurCGT
derives an operator's charge from a Lie-algebra commutator `[Q,O] = q·O`, and the Z2 flip admits
no such `q` (it needs `+1` and `−1` at once — the same charge mod 2, opposite as integers). The
blocks are therefore built directly via `Telum._localspace_cgt_fields`. Every such call lives in
`src/BondUpdateBUG/z2_ising.jl` and nowhere else, so a Telum upgrade breaks one file, and the
tests fail loudly if the private contract changes. If LurCGT ever grows a `Z{N}`-aware
`getweight_irop`, delete `_z2_local_space` and call `getLocalSpace` — nothing else changes.

## 3c. One-axis twisting — the arXiv:2208.10972 Fig. 2 benchmark

```julia
set_symmetry!(:Z2)                         # the paper's own symmetry -- see below
psi = x_polarized_state(10)                # |Psi(0)> = (x)|->, ONE sector, bond dims all 1
mpo = oat_mpo(10)                          # H = sum_{l<l'} Sz_l Sz_l'
for k in 1:251
    tdvp_cbe1s_step!(psi, mpo, -im * 0.05; maxdim = 16, trunc_thresh = 1e-14)
end
total_sx(copy(psi))                        # the observable: S^x_tot(t)
```

`H_OAT = (Σ_ℓ S^z_ℓ)²/2`. Expanding the square and using `(S^z)² = I/4` exactly on spin-½ gives
`Σ_{ℓ<ℓ'} S^z_ℓ S^z_ℓ' + (L/8)·I` — a **uniform all-to-all Ising coupling** plus a c-number.
`oat_mpo` builds the first part; `oat_energy_shift(L)` is the second, which is a global phase and
cancels from every expectation value and from `δE(t) = E(t) − E(0)`.

**Under `:none` the MPO has virtual dimension 3 for every `L`, and it is exact.** A constant
coupling is the `λ = 1` degenerate case of the geometric tail that `long_range_mpo` carries on one
self-loop channel — there is no fit and nothing truncated. Contrast Haldane-Shastry, whose `1/sin²`
has no finite-dimensional MPO and costs `O(L)` through `pair_mpo` to stay exact. `oat_mpo_pairs`
builds the same operator the `O(L)` way; `test_oat.jl` pins the two against each other, which is
the only test in the suite that exercises `λ = 1` at all.

**`:Z2` is the paper's own choice**, stated in the Fig. 2 caption: *"computed using δ=0.01 and Z2
spin symmetry"*. The conserved quantity is **not** `S^z_tot` — it is the global spin flip
`P = Π_ℓ 2 S^x_ℓ`, because `H_OAT` is *even* in `S^z`. In the σˣ basis of `z2_ising.jl` that flip
is diagonal, and the whole difficulty inverts: `S^x` becomes the diagonal, charge-neutral operator
(so it can be measured at all) and `⊗|→⟩` becomes a single charge-0 sector at `χ = 1` — the paper's
"an MPS with `D = 1`", literally.

⚠️ **`:Z2` costs `O(L)` virtual dimension, not 3, and that is structural.** In the σˣ basis `S^z`
is the charge-1 *flip*, hence rank 3, so its channel needs a charged transport block.
`long_range_mpo` refuses rank-3 terms outright (a self-loop cannot carry a charged channel), so
`:Z2` routes through `pair_mpo`, which builds charged transport. Exact either way; at `L = 10` the
cost is nothing, and at the paper's `L = 100` the symmetry is why they could run it at all. One
consequence worth knowing: under `:Z2`, `oat_mpo` *is* `oat_mpo_pairs`, so comparing them there is
a tautology rather than a cross-check.

⚠️ **The Z2 pair term is an adjoint pair `Z`/`Z'`, never `Z`/`Z`.** This is the same convention
`XXZTerm` documents for `Sp`/`Sp'`, and getting it wrong fails in a thoroughly misleading way: an
op leg is created with direction `'-'`, so `Z`/`Z` gives the opening block's outgoing virtual leg
and the closing block's incoming one the *same* direction, Telum refuses to contract two `'-'`
legs, and the symptom is `_charged_transport` reporting that no transport block satisfies its three
tests. That reads like a missing mod-2 capability and is actually a malformed vertex — the mod-2
charge itself is fine (`1 + 1 = 0`, so the flip is its own partner, where U(1) needs `+2` and `-2`).

**`:U1` is refused, and the reason is physics, not convenience.** `H_OAT` conserves `S^z_tot`, so
the *operator* is fine there — but the *initial state* `⊗|→⟩` has weight in every total-`S^z`
sector, and the *observable* `S^x_tot` is charge-raising. On a `SymMPS` of definite total charge
`⟨S^x_tot⟩` is not small, it is **exactly zero by symmetry**: a U(1) run returns a clean flat line
instead of the collapse-and-revival signal. `x_polarized_state` and `sx_operator` therefore throw
outside `:none`/`:Z2` rather than return it.

**`:none` is the dense control**, in the σᶻ basis. It is a *different basis* for the same physics,
so every observable must agree with `:Z2` digit for digit — measured over a 40-step run at
`maxdim = 6`, `max|Z2 − none|` on `S^x_tot(t)` is `1.6e-12` (`tdvp_cbe1s`), `5.4e-13` (`tdvp2`),
`1.5e-13` (`cbe_bug`), with `E(0)`, `S^x_tot(0)` and the norm bit-identical. `test_oat.jl` asserts
it, and it is the only check that would catch a wrong dualisation in the Z2 term.

**Everything about this benchmark is closed form**, which is why it is the sharpest test here.
`H` is diagonal in the computational basis, so `tests/common/analytic_reference.jl` supplies

| reference | what it gives |
|---|---|
| `oat_total_sx(L, t)` | `(L/2)cos^{L-1}(t/2)` — the paper's Fig. 2(a) curve |
| `oat_schmidt_values(L, n, t)` | the exact Schmidt spectrum across the cut after site `n` |
| `oat_entropy(L, n, t)` | exact von Neumann entropy (natural log, the paper's base) |
| `oat_rank(L, n, t)` | the exact bond dimension |

No diagonalisation and no propagation anywhere — these are formulas, and they share no code with
`src/`.

⚠️ **The rank ceiling is hard, and it sets every sensible `maxdim`.** The only entangling factor
is `exp(-i t a b)` in the two block magnetisations `a`, `b`, so the exact Schmidt rank across the
cut after site `n` is `min(n, L−n) + 1` **for all time** — 6 at the centre for `L = 10`, and it is
*saturated*. So `maxdim ≥ 6` is full rank (where a projector-splitting sweep is exact and the only
error left is the Krylov tolerance) and `maxdim < 6` is a genuine truncation. A run reporting
`D > 6` at `L = 10` is reporting noise, not physics. Choose caps on both sides of it or the
comparison measures only one regime.

The revival period is `4π`; `t = 2π` is the *negative* revival (`S^x_tot = −L/2`, and the state is
a product state again there, rank 1). The paper's `t = 12π` is three cycles.

```
julia --project=. benchmarks/oat_l10.jl 10 0.05 12.566 all 2,3,4,6,16
python3 benchmarks/plot_oat.py
```

## 3a. The TDVP integrators, and where they live

Beside the two BUG paths there are two TDVP integrators, each in its own directory under
`src/RSVDCBEBondUpdate/`:

```
src/RSVDCBEBondUpdate/
  tdvp2/tdvp2_baseline.jl      2-site TDVP — the accuracy/memory baseline
  tdvp1_cbe/tdvp1_cbe.jl       1-site TDVP + CBE — the step
  tdvp1_cbe/variants.jl        its two selection rules, named
```

```julia
tdvp2_step!(psi, h, tau; maxdim = 32, trunc_thresh = 0.0, maxiter = 8)

tdvp1_rsvd_cbe_step!(psi, h, tau; maxdim = 32, maxiter = 8)   # the RSVD extension — the method
tdvp1_cbe_exact_step!(psi, h, tau; maxdim = 32, maxiter = 8)  # plain CBE — the A/B reference
tdvp1_cbe_step!(psi, h, tau; exact = false, ...)              # the general entry point
```

**2-site TDVP has no CBE, and that is the point.** A two-site block supplies its own rank —
the SVD that splits it is free to return more singular values than went in — so it adapts
without an expansion. That is what makes it the baseline the CBE arms are measured against.

**1-site TDVP + CBE needs the expansion; it is not optional.** A one-site sweep moves the
centre through tensors of fixed shape, so it *cannot* change the bond dimension by itself.
Without CBE the rank is frozen at whatever the initial state had — start from a product state
and it stays at χ=1 forever. CBE is what makes a 1-site sweep rank-adaptive at all.

**The two variants are one implementation, not two.** `tdvp1_rsvd_cbe_step!` and
`tdvp1_cbe_exact_step!` both call the same step with `exact = false` / `true`; the flag reaches
`cbe_expand_bond` and chooses a Gaussian sketch of `H·Θ` or the complete fused basis. They are
deliberately not separate implementations — the A/B only means something if the two paths are
otherwise bit-for-bit identical code, so any difference in output comes from the selection rule
and nothing else.

## 4. Real time, imaginary time, ground state

Same engine. `τ` is the only thing that separates real from imaginary time — there is no
real/imaginary branch anywhere in the expansion or the S-step, which is why there are no
separate entry points for them.

⛔ **`RealTime` and `ImaginaryTime` NO LONGER EXIST.** They were thin forwarders to the
gate-based engine removed in §3, so their bodies named functions that were already gone. Pass
`τ` to a step function instead:

```julia
using BUGJulia.RSVDCBEBondUpdate: GroundState

cbe_bug_step!(psi, W, ComplexF64(-im * dt); maxdim = 32)   # τ = -i·dt   exp(-iHt)
cbe_bug_step!(psi, W, ComplexF64(-dt);      maxdim = 32)   # τ = -dt     exp(-βH)

GroundState.rsvd_cbe_dmrg!(psi, h; opts, n_sweeps = 40, etol = 1e-12)   # the method
GroundState.cbe_dmrg!(psi, h; opts, n_sweeps = 40, etol = 1e-12)        # exact-CBE control
```

`tdvp_cbe1s_step!` and `tdvp2_step!` take the identical signature, so swapping the integrator
is a one-word change and nothing else moves.

Things worth knowing, now that the caller drives the loop rather than an `evolve!` driver:

- **Real time.** `exp(-i dt H)` is unitary, so rescaling is hygiene — but `norm(psi)` drifting
  from 1 is a *diagnostic* that truncation is biting, and worth watching for that reason. The
  energy is conserved, so it carries no convergence information and is not a calibration
  signal.
- **Imaginary time.** `exp(-dt H)` is a contraction: the norm decays, so **rescaling every step
  is load-bearing, not hygiene** — there is no `normalize` flag on the step functions, the
  caller does it. Writing the scale into the CENTER tensor is what keeps the canonical form
  intact:

  ```julia
  renorm!(p) = (p[p.center] = to_concrete((1.0 / norm(p)) * p[p.center]); p)
  ```

  ⛔ A dt² splitting error is *invisible* in the energy at any β (O(dt²) state error → O(dt⁴)
  energy error), so **do not use the energy to calibrate `dt`.**
- **Ground state.** `GroundState` still takes `CBEBugOptions`, and its `maxiter` drives a
  Lanczos **eigensolver**, not an exponential — the Krylov-depth guidance in §5 does not
  transfer to it.

## 5. Options that matter

`CBEBugOptions` has many fields; these are the ones to actually set.

```julia
opts = CBEBugOptions(;
    dt           = 0.05,
    n_steps      = 40,
    maxdim       = 32,        # rank cap
    trunc_thresh = 1e-12,     # discard threshold; inert once maxdim binds
    s_iters      = -1,        # expanding-basis Lanczos — keeps the step 2nd order in dt
    maxiter      = 8,         # Krylov depth (see below)
    exact        = false,     # false = RSVD-CBE (the method); true = exact CBE (A/B only)
    normalize    = true,
    growth       = 2.0,       # expansion budget — PROBLEM-DEPENDENT, see cbe_bug.md §7.1
    seed         = 0x5EED,    # one RNG threads the run, so runs are reproducible
)
```

**`exact = false` is the method and the default.** `exact = true` replaces the thin Gaussian
sketch with the complete fused basis, which forms the rank-4 `H·Θ` that the sketch exists to
avoid. It is the **reference arm of an A/B**, never a better default and never a fallback.
Measured: the two agree to every printed digit on all time-evolution arms, and part only where
the rank cap binds.

**`maxiter = 8`, not the default 30.** `lanczos_expv` has a *breakdown-only* exit — `tol` gates
the off-diagonal Lanczos norm, not convergence of the exponential — so for a generic `Θ` it
never fires and every solve burns its full budget. Measured at L=18: mean Krylov dimension
**27.4 against a cap of 30**. Setting 8 is **3.64× faster and changes the trajectory by
2.9e-13** with the rank cap binding. It is not a universal constant: depth is set by `‖τH‖`
(≈0.3 at dt=0.05, L≈18), so a larger `dt` or a wider spectrum needs more. The cliff is at 4
(3.2e-07). To check your own case:

```julia
enable_krylov_log(); step!(psi); d = get_krylov_log(); disable_krylov_log()
# mean(d) near the cap  =>  the budget is being spent, not converged into
```

**`s_iters = -1`** selects the expanding-basis Lanczos for the S-step. The one-shot solver
loses second-order convergence to a basis floor (measured error ratios 3.46, 3.01 instead of
4.00). Note `cbe_lubich_sweep` does **not** accept `s_iters` — it has no S-step solver choice.

**Rank grows only through RSVD-CBE.** There is no `missing_fill`, no `pad`, no KLS augmenter
anywhere in this path, and a source-level test pins that. If a bond will not grow, investigate
the expansion — do not add seeding. The proof the growth is not seed-driven: it grows under
`:none`, where no empty charge sector exists to seed into.

## 6. Reading the result

`CBEBugRunInfo` — every field has length `n_steps`:

| field | what it tells you |
|---|---|
| `max_bond_dims` | did the rank cap bind? **check this before reading any error** |
| `norms` | real time: drift from 1 = truncation biting. Imaginary: the decay factors |
| `max_expanded` | largest bond proposed *before* truncation, i.e. what CBE asked for |
| `krylov_dims` | operator applications per step — the cost half of the pair |
| `mean_aug` | mean expanded area those solves acted on — the size half |
| `err_pre`, `err_fnl` | preselection and final truncation errors |
| `energies` | only if you passed `energy_gates` |

**A knob is inert unless the run is rank-starved.** If `max_bond_dims` never reaches `maxdim`,
then `maxdim`, `growth` and `trunc_thresh` changed nothing and any comparison across them is
measuring noise. This has produced several wrong conclusions in this repo — check it first.

## 6b. Krylov depth is what governs `cbe_bug`'s accuracy

**One knob moves the error on a hard model, and it is not the one anyone reaches for.** The
half-sweeps build an orthonormal Krylov basis `span{Θ, HΘ, H²Θ, …}` per bond, and its DEPTH is
what separates a good answer from one that is eight orders out.

> **`krylov_basis`** (default `30`) caps the depth; **`krylov_tol`** (default `1e-6`) ends the
> recursion when the next vector's CONTRIBUTION to `exp(τH)Θ` falls below it — Saad's estimate
> `β_m·|[exp(τT_m)]_{m,1}|`. `0.0` disables the test and runs to the cap.
>
> `krylov_basis = 0` stops after one application of `H`, which is exactly what the CBE frame
> already spans. So `m = 1` reproduces `m = 0` bit-for-bit, and **the first genuine addition is
> `m = 2`** — a scan that starts at 1 will look inert and prove nothing.

MEASURED at L = 12, `dt = 0.05`:

| model | `krylov_basis = 0` | `= 2` | `= 3` |
|---|---|---|---|
| XX `:U1` | 1.6948e-04 | 1.6948e-04 | 1.6948e-04 |
| Heisenberg | 8.8424e-05 | — | **2.6697e-06** |
| OAT | 2.1329e-02 | 1.1358e-04 | **7.0610e-14** |

⚠️ **Both schemes are the SAME ORDER**, so a `dt` scan cannot separate them. One power of `H` and
the full basis agree to O(τ); the entire difference is a prefactor, and it only becomes visible
where `τ‖H_loc‖` is not small. That is why XX is bit-identical across the whole range while OAT
spans eight orders.

### ⛔ Why this is easy to miss

Depth is the only axis that changes which **directions** the step has. Every other knob changes
how much **room** it has, so all of them report that nothing is wrong:

| what was tried | result |
|---|---|
| more expansion budget (`growth` 2/4/8, `dex` 16/32/64, `comp_ratio`) | **bit-identical** across 8 variants, `err_fnl ≈ 1e-14` — the selection discarded nothing |
| more rank (closing truncation off entirely) | χ 7 → 24, error moved by **zero digits** |
| more CBE passes (`grow_iters`) | 1.8×, then flat — each pass re-runs the *selection* and so re-truncates by budget, discarding what the next application of `H` needs |
| energy drift | 1.5e-14, i.e. perfect, on the arm that is eight orders wrong |
| bond profile | lands exactly on the analytic Schmidt ceiling `min(n, L−n)+1`, on the same wrong arm |

A run at `krylov_basis = 0` exits 0, writes its CSV, conserves energy and reports the correct
rank. **Nothing but a comparison against an exact reference will tell you.**

⚠️ **β alone is a breakdown detector, not a convergence criterion — and thresholding it was
tried first.** Heisenberg and XX returned bit-identical results for anything from 1e-6 to 1e-13;
the test never fired and every bond ran to the cap (5580 operator applications against 1038 for a
pinned depth 3). It is now the `_KRY_BREAKDOWN` guard only, and `krylov_tol` compares the Saad
contribution instead. That estimate is a **conservative** bound — measured on dense Hermitian
problems it overshoots the true Krylov error by 4×–160× and never undershoots outside roundoff, so
the rule costs ~1–2 extra vectors and cannot stop short (`test_cbe_bug_exact_refs.jl` pins the
direction of that inequality).

⚠️ **Build the basis with full re-orthogonalisation** — two Gram-Schmidt passes, which is what
`_krylov_frame` does. A raw power iteration collapses onto the dominant eigenvector by `k ≈ 4` and
the closing SVD then deletes the vectors as linearly dependent: saturation for a *conditioning*
reason that is indistinguishable from a physical one.

### What this replaced

Until 2026-08-24 this section documented `kstep`, `kaug` and `rexpand` — three knobs around a
half-sweep that evolved a one-site tensor per bond and then had to repair the width its split had
lost. **All three are gone.** The half-sweeps now take the CBE frame as the basis directly and
deepen it with `H`, so there is no split to lose width and nothing to repair.

The retirement was measured, not assumed (L = 12, `dt = 0.05`, `krylov` = operator applications):

| model | this sweep | old K-step sweep | |
|---|---|---|---|
| XX | 1.6948e-04 @ 289 | 1.6948e-04 @ 2888 | identical, **10× cheaper** (at `m = 0`) |
| Heisenberg | 2.6697e-06 @ 1038 | 2.7869e-06 @ 3423 | better, **3.3× cheaper** |
| OAT | 7.0610e-14 @ 1036 | 5.2449e-06 @ 3740 | **8 orders better**, 3.6× cheaper |

The old defect is worth keeping on record because its *signature* recurs: the basis update wrote
`psi[i] = W[i]`, so the bases WERE the state, but each half-sweep kept the CBE frame on the
opposite side from the basis it was building and dropped the one that would become the state.
`n_new` was identical either way while `expanded` differed — one admitted direction per interior
bond discarded — and the damage was ordered by the Hamiltonian's RANGE (dE 4.9e-07 nearest
neighbour, 1.5e-03 on Haldane-Shastry and OAT). **Nearest-neighbour benchmarks could not see it**,
which is the same reason XX cannot see Krylov depth today.

### The 10π campaign (2.5 revival periods), against `tdvp_cbe1s`

`tdvp_cbe1s` is the method to beat — it is the paper's own, and it matches `tdvp2`'s accuracy to
four digits at every `D` while costing less, so it is the honest baseline for **both** axes.

| L | D (= exact ceiling) | `cbe_bug` | `tdvp_cbe1s` | accuracy | cost |
|---|---|---|---|---|---|
| 10 | 6 | 5.467e-06 | 4.491e-03 | **821×** | **3.11× cheaper** |
| 12 | 7 | 2.588e-05 | 8.376e-03 | **324×** | **3.26× cheaper** |

Per revival the error is **bounded**: `cbe_bug` 5.460e-06 → 5.467e-06 → 5.386e-06 at L = 10. Under
truncation (`D = 4`, below the ceiling) `cbe_bug` stays flat while `tdvp_cbe1s` *grows*
(3.680 → 4.443 → 4.488) — a stability difference that a single-period run cannot show, since one
revival cannot distinguish a bounded error from an accumulating one.

⚠️ **The accuracy ratio SHRINKS with L** (821× → 324×) while the cost ratio holds at ~3.2×. Two
points are not a trend; L = 16/20 (`benchmarks/submit_oat_large.slurm`) is what would settle it.
Do not quote 821× as "the" advantage.

⚠️ **`cbe_bug` does NOT self-truncate to the exact rank over long windows.** At `D = 16` the bond
reaches χ = 8 against an exact rank of 6 (L = 10) and **χ = 14 against 7** (L = 12) — nearly the
whole cap. Accuracy is unchanged (identical to the `D = ceiling` arm), so this is noise above the
physically-occupied rank, the same effect §6 documents for `tdvp_cbe1s`. An earlier claim that it
converges to exactly `min(n, L-n)+1` held only at 4π and L = 10.

⚠️ **Do not read a `dt` order off OAT at all.** `H_OAT` is diagonal with commuting terms and a
finite `m²` spectrum, so the Krylov basis closes the local invariant subspace exactly: the current
sweep is at **machine precision** on this model (4.80e-14 at `dt`, 7.59e-14 at `dt/2`) and there is
no `dt` error left to have an order. The ratio there is roundoff over roundoff and wanders either
side of 1 — `tests/sweeps/test_oat.jl` therefore asserts **exactness** on this arm rather than a
ratio window. (The retired K-step sweep showed 19.59 here, which was read as fourth-order
superconvergence; it was the same effect seen through a larger error.)

⚠️ **On Heisenberg `cbe_bug` measures THIRD order** (ratios 7.92 / 7.96), not second as this
document previously stated. Take the order from a rank-limited scan on the model you care about:
at full rank both sweeps are exact and the "order" you measure is the roundoff floor.

## 7. Gotchas

- **`set_symmetry!` before building tensors**, never mid-run.
- **A `χ = 1` start no longer costs a bootstrap step** — dE is 1.4e-15 on the first step from a
  product state, against 1.561e-3 historically. The old `warmup` grading is kept in
  `benchmarks/oat_l10.jl` for its own sake, not because the scheme needs it (§6b).
- **Scan `krylov_basis` before blaming truncation, rank or scheme order** (§6b). It is the only
  knob that changes which *directions* the step has, and every other diagnostic — energy drift,
  bond profile, `err_fnl` — reports a healthy run while it is eight orders wrong.
- **`maxbd` must reach the cap** before an error column means anything (§6).
- **First call costs minutes of JIT.** A silent log is not a hang; `1site_tdvp_cbe` 87.5 s vs
  `1site_tdvp_cbe_rsvd` 4.8 s at L=8 is compilation reuse, not an 18× speedup.
- **One timing job at a time.** Contention has produced non-monotonic nonsense here (less work
  taking longer). Accuracy numbers survive contention; timing numbers do not.
- **Profile with `threads = 1:1`.** MKL parks a thread pool whose idle threads the sampler
  counts as ~26% of "time", which reads convincingly as a BLAS bottleneck and is not one.
- **`parity_bonds` names are 0-based**: `:even` is bonds `1,3,5,…` and `:odd` is `2,4,6,…`.
  Asking for the wrong group makes every gate `nothing`, so the sweep silently does nothing and
  returns the input unchanged — indistinguishable from a genuine rank freeze.
- **Bond dimension is not comparable across symmetries.** `bond_dims` counts *multiplets* under
  `:SU2` and *states* under `:none`; χ=8 SU(2) ≈ χ=16 `:none`. Compare observables, not χ.
