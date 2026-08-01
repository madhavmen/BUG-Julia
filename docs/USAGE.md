# Using RSVD-CBE for the bond-update BUG

How to run the rank-adaptive BUG with RSVD controlled bond expansion, on either of its two
paths — the **gate-based** sweep and the **Lubich** (MPO-environment) sweep.

Runnable companion: [`examples/heisenberg_domain_wall.jl`](../examples/heisenberg_domain_wall.jl).
Design and measurements: [`docs/cbe_bug.md`](cbe_bug.md), [`docs/current_work.md`](current_work.md).

---

## 1. Quick start

```julia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate
using BUGJulia.RSVDCBEBondUpdate: RealTime

set_symmetry!(:U1)                                  # :none | :U1 | :SU2
psi   = domain_wall_state(8)                        # any state; chi = 1 is fine
gates = bond_gates(psi; J = 1.0, delta = 1.0)       # one gate per bond

info = RealTime.evolve!(psi, gates;
                        opts = CBEBugOptions(dt = 0.05, n_steps = 40, maxdim = 32))

magnetisation(copy(psi))     # <Sz_j> per site
bond_dims(psi)               # how far the bond dimension grew
```

That is the whole surface: **pick a symmetry, build a state, build gates, call `evolve!`.**

```bash
julia --project=. examples/heisenberg_domain_wall.jl
L=10 TMAX=3 SYMS=none,U1 julia --project=. examples/heisenberg_domain_wall.jl
```

## 2. Switching symmetry

One global call, before you build any tensors:

```julia
set_symmetry!(:none)    # no symmetry — one dense block per bond
set_symmetry!(:U1)      # Sz conservation — separate charge sectors  (default)
set_symmetry!(:SU2)     # total spin — multiplets
symmetry_mode()         # what is currently set
```

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

## 3. The two paths

Both use the **same** RSVD-CBE expansion and the **same** S-step. They differ only in how `H`
is applied.

### Gate-based — `RealTime.evolve!`

```julia
gates = bond_gates(psi; J = 1.0, delta = 1.0)
info  = RealTime.evolve!(psi, gates; opts = opts)
```

Strang composition over bond gates: `even τ/2, odd τ, even τ/2`. No MPO environments at all.
Cheapest per step, and the path the benchmarks exercise most. Carries a **Trotter splitting
error of order dt²**.

### Lubich sweep — `RealTime.evolve_mpo!`

```julia
h    = xxz_chain(8; delta = 1.0)          # or heisenberg_su2_chain(8) under :SU2
info = RealTime.evolve_mpo!(psi, h; opts = opts)
```

One projected exponential per step over a globally expanded subspace, through MPO
environments. **No Trotter splitting error.** More work per step; measured ~1 `expv` call per
sweep against ~66 for a 2-site TDVP step at L=18.

For a single sweep rather than a driver loop: `cbe_lubich_sweep(psi, h, tau; maxdim, ...)`.

### Which to use

| | gate | Lubich |
|---|---|---|
| Trotter error | O(dt²) | none |
| needs an MPO | no | yes |
| cost per step | lower | higher |
| non-nearest-neighbour `H` | ❌ | ✅ |

**The two paths need not agree to round-off** — the example measures a 3.4e-05 gap at dt=0.05,
which is the gate path's splitting error and expected physics. Both are checked against the
exact reference, which is what settles correctness. Do not use one as the other's oracle.

## 4. The three modes

Same engine; the mode fixes `τ` and what gets recorded. Nothing else differs — there is no
real/imaginary branch anywhere in the expansion or the S-step.

```julia
using BUGJulia.RSVDCBEBondUpdate: RealTime, ImaginaryTime, GroundState

RealTime.evolve!(psi, gates; opts)                    # τ = -i·dt   exp(-iHt)
RealTime.evolve_mpo!(psi, h; opts)

ImaginaryTime.evolve!(psi, gates; opts, energy_gates = gates)   # τ = -dt  exp(-βH)
ImaginaryTime.cool!(psi, gates; opts)

GroundState.rsvd_cbe_dmrg!(psi, h; opts, n_sweeps = 40, etol = 1e-12)   # the method
GroundState.cbe_dmrg!(psi, h; opts, n_sweeps = 40, etol = 1e-12)        # exact-CBE control
```

Mode-specific things worth knowing:

- **Real time.** `exp(-i dt H)` is unitary, so `opts.normalize` is hygiene, and `info.norms`
  drifting from 1 is a *diagnostic* that truncation is biting. No energy recorded by default —
  it is conserved, so it carries no convergence information.
- **Imaginary time.** `exp(-dt H)` is a contraction: the norm decays and `opts.normalize` is
  **load-bearing**, not hygiene. Pass `energy_gates` — the energy is the convergence signal.
  Note a dt² splitting error is *invisible* in the energy at any β (O(dt²) state error →
  O(dt⁴) energy error), so do not use the energy to calibrate `dt`.
- **Ground state.** `maxiter` here drives a Lanczos **eigensolver**, not an exponential — the
  time-evolution guidance in §5 about Krylov depth does not transfer.

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

## 7. Gotchas

- **`set_symmetry!` before building tensors**, never mid-run.
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
