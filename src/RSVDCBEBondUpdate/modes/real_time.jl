"""
    RSVDCBEBondUpdate.RealTime

Real-time evolution, `exp(-i t H)`.

THIS MODULE ADDS NO ALGORITHM. It fixes `tau = -im*dt` and forwards to the shared engine in
`cbe_core.jl` (gate path) or `cbe_lubich.jl` (MPO-environment path). It exists so the three
modes are named and discoverable in one place, and so the differences between them are
visible as three short files rather than buried in a `tau` argument.

`evolve!` IS `cbe_bond_update_bug!` -- the historical entry point that every existing test
and benchmark calls -- not a wrapper that could drift from it.

WHAT IS SPECIFIC TO THIS MODE:

  - `exp(-i dt H)` is UNITARY, so the norm is conserved up to Trotter and truncation error.
    `opts.normalize` is hygiene here, not a load-bearing step, and `info.norms` is a
    DIAGNOSTIC: a norm drifting from 1 means truncation is biting, and is worth watching.
  - No energy is recorded by default. Energy is conserved in real time, so it carries no
    convergence information and a full contraction per step would be paid for nothing. Ask
    for it explicitly with `energy_gates` if you want it as a conservation check.
  - Second order in `dt` from the Strang composition. With the expanding-basis solver
    (`s_iters = -g`) this is a clean 4.00 error ratio per halving; the one-shot solver loses
    it to a basis floor (3.46 then 3.01). See `docs/current_work.md` 7.6.5.
"""
module RealTime

using ..RSVDCBEBondUpdate: CBEBugOptions, CBEBugRunInfo, cbe_gate_evolve!,
                          cbe_bond_update_bug!, cbe_lubich_bug!, XXZChain
using ...BondUpdateBUG: SymMPS

"""
    evolve!(psi, gates; opts=CBEBugOptions(), energy_gates=nothing) -> CBEBugRunInfo

Real-time gate-based CBE-BUG. `tau = -im*opts.dt`, Strang composition, `opts.n_steps` steps.

Pass `energy_gates` to record the energy each step as a conservation check; it is off by
default because energy is conserved here and the contraction is not free.
"""
evolve!(psi::SymMPS, gates; opts::CBEBugOptions = CBEBugOptions(), energy_gates = nothing) =
    cbe_gate_evolve!(psi, gates, ComplexF64(-im * opts.dt);
                     opts = opts, energy_gates = energy_gates)

"""
    evolve_mpo!(psi, h; opts=CBEBugOptions()) -> CBEBugRunInfo

Real time on the MPO-environment path: one projected exponential per step over a globally
expanded subspace, no Trotter splitting. `cbe_lubich_bug!` under its mode name.
"""
evolve_mpo!(psi::SymMPS, h::XXZChain; opts::CBEBugOptions = CBEBugOptions()) =
    cbe_lubich_bug!(psi, h; opts = opts)

export evolve!, evolve_mpo!

end # module RealTime
