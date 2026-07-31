"""
    RSVDCBEBondUpdate.ImaginaryTime

Imaginary-time evolution, `exp(-beta H)`, and the ground state you get by cooling.

Same engine, same growth passes, same Krylov basis as [`RealTime`](@ref); the ONLY
difference is `tau = -dt` instead of `tau = -im*dt`. What that one sign change does to the
physics, though, is worth being explicit about, because three things stop being free:

  1. `exp(-dt H)` IS NOT UNITARY. It is a contraction: every eigencomponent is damped by
     `exp(-dt E)`, so the norm decays and `opts.normalize` becomes LOAD-BEARING rather than
     hygiene. Left off, the state underflows. `info.norms` records the decay factor BEFORE
     the rescale, which is the useful quantity -- `-log(norm)/dt` is an energy estimate.

  2. THE NORM DECAY IS THE CONVERGENCE SIGNAL, not an error. In real time a norm drifting
     from 1 means truncation is biting; here it is the algorithm working. Do not read
     `info.norms` the same way in the two modes.

  3. THERE ARE TWO INDEPENDENT ERRORS AND THEY MUST NOT BE CONFLATED. Reading a residual
     against the true ground energy without separating them will misattribute it -- which is
     easy to do, because both shrink under "run it harder":

       incomplete cooling  ~ `exp(-2*beta*(E1-E0))`  falls with BETA, FLAT in `dt`
       Trotter splitting   ~ `dt^2`                  falls with `dt`, FLAT in beta

     MEASURED at L=8, delta=1, half filling: at beta = 10 the residual is 2.48e-04 at
     `dt = 0.1` and 2.48e-04 at `dt = 0.05` -- unmoved by halving `dt`, so at that beta it is
     ENTIRELY the cooling term and says nothing about the splitting.

     AND THE `dt^2` SCALING DOES NOT SIMPLY APPEAR ONCE BETA IS LARGE, which is what this note
     used to predict. Cooling further does drop the residual -- 1.220e-02 at beta=5, 2.478e-04
     at beta=10, 9.625e-08 at beta=20 -- but at beta=20 it is STILL flat in `dt` (9.626e-08 at
     `dt=0.1` against 9.625e-08 at `dt=0.05`, ratio 1.00).

     The reason is that ENERGY IS THE WRONG INSTRUMENT for the splitting error. The ground state
     is stationary, so an `O(dt^2)` error in the STATE is second order in the energy -- it shows
     up as `O(dt^4)` -- and it stays buried under the cooling term at every beta reachable here.
     To see the splitting, measure the STATE (a profile or an overlap), not the energy.

     The converged-in-beta state is the ground state of the TROTTERISED Hamiltonian, `O(dt^2)`
     from the true one, and no amount of extra cooling removes that. It is why the fixed point
     here is not identical to [`GroundState`](@ref)'s, which has no splitting error at all.

WHY IMAGINARY TIME WORKS ON THE GATE PATH WHEN GROUND-STATE SEARCH DOES NOT. Trotterisation
is a statement about `exp(tau*(A+B))`, and it does not care whether `tau` is real or
imaginary: `exp(-dt*H)` factorises into gates exactly as `exp(-im*dt*H)` does, and across a
full sweep every term of `H` is applied. A local EIGENSOLVE has no such composition -- it
sees one bond's operator and nothing else -- which is why that one needs environments. See
`CBEBugOptions` for the full argument.

COOLING TO THE GROUND STATE. `exp(-beta H)|psi>` converges to the lowest eigenstate that
`|psi>` overlaps, at a rate set by the gap: the first excited state is suppressed relative
to the ground state by `exp(-beta*(E1 - E0))`. So a small gap means slow cooling, and a
start state ORTHOGONAL to the ground state never gets there at all. On a U(1)-symmetric
chain the start state also pins the particle-number sector, so this finds the ground state
OF THAT SECTOR -- `neel_state` gives half filling.
"""
module ImaginaryTime

using ..RSVDCBEBondUpdate: CBEBugOptions, CBEBugRunInfo, cbe_gate_evolve!
using ...BondUpdateBUG: SymMPS

"""
    evolve!(psi, gates; opts=CBEBugOptions(), energy_gates=gates) -> CBEBugRunInfo

Imaginary-time gate-based CBE-BUG. `tau = -opts.dt`, Strang composition, `opts.n_steps`
steps of inverse temperature `opts.dt` each.

`opts.normalize` MUST be true (the default) -- see the module docstring; it is checked.

Energy is recorded every step by default, unlike real time, because here it is the
convergence signal rather than a conserved quantity. Pass `energy_gates = nothing` to skip
the contraction if you only want the final state.
"""
function evolve!(psi::SymMPS, gates; opts::CBEBugOptions = CBEBugOptions(),
                 energy_gates = gates)
    opts.normalize || throw(ArgumentError(
        "imaginary time needs opts.normalize = true: exp(-dt*H) is a contraction, so " *
        "without the rescale the state decays to underflow"))
    return cbe_gate_evolve!(psi, gates, ComplexF64(-opts.dt);
                            opts = opts, energy_gates = energy_gates)
end

"""
    cool!(psi, gates; opts=CBEBugOptions(), etol=1e-10) -> (info, converged)

[`evolve!`](@ref) with an early stop once the energy stops moving.

Returns the run record and whether it converged. `etol` is on the ABSOLUTE change in energy
between consecutive steps; it is not a bound on the distance to the true ground state, which
also carries the `O(dt^2)` Trotter error the module docstring describes.

Runs in blocks of `opts.n_steps` so the check costs one energy contraction per block rather
than restarting the driver every step.
"""
function cool!(psi::SymMPS, gates; opts::CBEBugOptions = CBEBugOptions(),
               etol::Float64 = 1e-10, max_blocks::Int = 100)
    infos = CBEBugRunInfo[]
    prev = Inf
    for _ in 1:max_blocks
        info = evolve!(psi, gates; opts = opts)
        push!(infos, info)
        isempty(info.energies) && break
        e = info.energies[end]
        abs(e - prev) < etol && return (infos, true)
        prev = e
    end
    return (infos, false)
end

export evolve!, cool!

end # module ImaginaryTime
