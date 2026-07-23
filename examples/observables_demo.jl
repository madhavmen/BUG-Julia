# =============================================================================
#  How to retrieve observables from a bond_update_bug! run.
#
#  Run from the package root with:
#
#      julia --project=. examples/observables_demo.jl
#
#  It evolves a Heisenberg quench once WITH U(1) symmetry and once WITH NONE,
#  and pulls out exactly the quantities used for the standard plots:
#
#    * magnetisation of every site at every dt   -> light-cone heat-map, errors
#    * discarded weight at every dt              -> truncation monitor
#    * center-bond dimension at every dt         -> entanglement growth
#    * entanglement spectrum of ALL bonds        -> the "spectral spectrum" plot,
#      recorded at a few chosen timestamps
#
#  Two ways to get them, both shown below:
#    (A) turn on the built-in recorders and read them off `info`  -- easiest;
#    (B) call the observable functions yourself on any state       -- most direct.
# =============================================================================

using BUGJulia.BondUpdateBUG

"Run one quench in symmetry mode `sym` and report what came back."
function demo(sym::Symbol)
    set_symmetry!(sym)                       # :U1 or :none -- state and gates share it

    N       = 8
    n_steps = 20
    dt      = 0.05

    # A domain wall (up...up | down...down) is a product state, so it exercises
    # the rank-growth path of BOTH modes: the missing-sector fill under :U1, the
    # criterion-2 auto-default under :none.
    psi   = domain_wall_state(N)
    gates = bond_gates(psi; J = 1.0, delta = 1.0)

    # Record the t = 0 magnetisation ourselves, so the light-cone starts at t=0.
    mz0 = magnetisation(copy(psi))

    # ---- (A) built-in recorders -------------------------------------------------
    # spectrum_steps picks 4 evenly spaced timestamps (like t=5,10,15,20 of 20).
    spectrum_steps = round.(Int, range(n_steps ÷ 4, n_steps; length = 4))

    info = bond_update_bug!(psi, gates; opts = BondUpdateOptions(
        dt = dt, n_steps = n_steps, order = :strang, maxdim = 64,
        record_magnetisation = true,      # -> info.magnetisations (one row per step)
        spectrum_steps = spectrum_steps,  # -> info.spectra[step] (all bonds)
    ))

    # everything is now a plain Julia array, ready to plot or save:
    times      = info.times                         # length n_steps
    mz_series  = vcat([mz0], info.magnetisations)    # (n_steps+1) profiles incl. t=0
    discarded  = info.discarded                      # length n_steps
    center_bd  = info.center_bond_dims               # length n_steps

    println("\n── symmetry = :$sym ──────────────────────────────────────────")
    println("sites N=$N, steps=$n_steps, dt=$dt")
    println("magnetisation profiles recorded : $(length(mz_series))  (incl. t=0)")
    println("  ⟨Sz⟩ at t=0        : ", round.(mz0; digits = 3))
    println("  ⟨Sz⟩ at t=$(times[end]) : ", round.(mz_series[end]; digits = 3))
    println("center-bond dim over time       : ", center_bd)
    println("max discarded weight over time  : ", round(maximum(discarded); sigdigits = 3))
    println("entanglement spectrum recorded at steps ", sort(collect(keys(info.spectra))),
            "  (t = ", round.([info.times[s] for s in sort(collect(keys(info.spectra)))]; digits=2), ")")
    for s in sort(collect(keys(info.spectra)))
        spec = info.spectra[s]                        # spec[b] = Schmidt values of bond b
        cb   = center_bond(psi)
        println("  step $s: center-bond spectrum (top 4) = ",
                round.(first(spec[cb], min(4, length(spec[cb]))); sigdigits = 3))
    end

    # ---- (B) call the functions directly on the final state ---------------------
    # Same numbers, no run required -- use this to inspect any state on the fly.
    println("direct read-out of the FINAL state:")
    println("  magnetisation()          -> length ", length(magnetisation(copy(psi))))
    println("  center_bond_dimension()  -> ", center_bond_dimension(psi))
    println("  bond_spectrum() (center) -> top 4 ",
            round.(first(bond_spectrum(copy(psi)), 4); sigdigits = 3))
    println("  entanglement_spectrum()  -> ", length(entanglement_spectrum(copy(psi))),
            " bonds")

    return info
end

demo(:U1)
demo(:none)
println("\nDone. `info.magnetisations`, `info.discarded`, `info.center_bond_dims`,")
println("and `info.spectra` hold the series; the same functions work on any state.")
