
#
# Run:  julia --project=. benchmarks/midpoint_bug.jl
# Out:  benchmarks/results/midpoint_bug.csv

using LinearAlgebra, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.BondUpdateBUG
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "tests", "common", "free_fermion.jl"))

const OUT  = joinpath(@__DIR__, "results", "midpoint_bug.csv")
const COLS = ["L", "T", "scheme", "dt", "nsteps", "err", "matvec", "chi", "seconds"]

_kry(i) = hasproperty(i, :krylov_dims) ? i.krylov_dims : 0

"Max |<X_j>(T) - exact| over the chain -- the profile, not a single site, so a scheme cannot
look good by being right in one place."
function xerr(psi, L, T, dirs)
    return maximum(abs.(x_profile(copy(psi)) .- tfim_x_profile(L, T, dirs)))
end

function main(; L = 16, T = 1.0, D = 64, J = 1.0, h = 1.0, maxiter = 30)
    set_symmetry!(:Z2)
    W = tfim_mpo(L; J = J, h = h)
    # ⛔ THE Z2 TFIM LIVES IN THE sigma^x BASIS: `:plus`/`:minus`, NOT `:up`/`:down`, and the state
    # and the reference MUST BE THE SAME ONE. `ising_kink_state` is the repo's kink -- `:plus` on
    # the left half, `:minus` on the right -- and `dirs` below is exactly its `tfim_covariance`
    # spelling, which is the pairing every other TFIM benchmark here uses.
    #
    # ⚠ WHY NOT BUILD MY OWN: `ising_product_state` REJECTS spin-z labels outright, but
    # `tfim_covariance` maps anything that is not `:plus` to `-1` -- so a plausible-but-wrong label
    # would silently describe a DIFFERENT reference state and produce a wrong error curve instead
    # of an error. Reusing the validated pair removes that whole failure mode.
    #
    # The kink is asymmetric under reflection (reversed it is `:minus` then `:plus`), so it can
    # still catch a reversed site convention -- see the `dense_state`/`exact_sparse` trap.
    dirs = [i <= L ÷ 2 ? :plus : :minus for i in 1:L]
    psi0 = ising_kink_state(L)
    @printf("TFIM Z2  L=%d  J=%g h=%g  T=%g  D=%d   (real time, analytic Majorana reference)\n\n",
            L, J, h, T, D)

    kw = (maxdim = D, trunc_thresh = 1e-12, maxiter = maxiter, exact = true)
    schemes = [
        "bug"      => (p, dt) -> cbe_bug_step!(p, W, ComplexF64(-im * dt); kw...),
        "midpoint" => (p, dt) -> cbe_bug_midpoint_step!(p, W, ComplexF64(-im * dt); kw...),
    ]

    mkpath(dirname(OUT))
    open(OUT, "w") do io
        println(io, join(COLS, ","))
        @printf("  %-10s %-8s %6s %13s %9s %5s %8s\n",
                "scheme", "dt", "steps", "max|dX|", "matvec", "chi", "secs")
        prev = Dict{String, Float64}()
        for dt in (0.25, 0.125, 0.0625, 0.03125)
            n = round(Int, T / dt)
            for (nm, st) in schemes
                psi = copy(psi0); mv = 0; t0 = time()
                ok = true
                for _ in 1:n
                    try
                        mv += _kry(st(psi, dt))
                    catch err
                        @info "arm broke" nm dt err; ok = false; break
                    end
                end
                el = time() - t0
                e = ok ? xerr(psi, L, T, dirs) : NaN
                @printf(io, "%d,%g,%s,%g,%d,%.6e,%d,%d,%.3f\n",
                        L, T, nm, dt, n, e, mv, maximum(bond_dims(psi); init = 0), el)
                flush(io)
                # the observed order between successive dt -- the number Theorem 4.2 is about
                ord = haskey(prev, nm) && isfinite(e) && e > 0 ? log2(prev[nm] / e) : NaN
                @printf("  %-10s %-8g %6d %13.4e %9d %5d %7.2fs%s\n",
                        nm, dt, n, e, mv, maximum(bond_dims(psi); init = 0), el,
                        isfinite(ord) ? @sprintf("   order %.2f", ord) : "")
                isfinite(e) && (prev[nm] = e)
                flush(stdout)
            end
        end
    end

    # ── THE FAIR COST COMPARISON ────────────────────────────────────────────────────────────
    # midpoint at `dt` costs about what the ordinary sweep costs at `dt/2`. Put them side by side
    # at MATCHED COST and let the errors decide; the table above cannot settle that, because its
    # rows are matched on `dt` instead.
    println("\n-- matched cost: midpoint at dt vs ordinary bug at dt/2 --")
    @printf("  %-8s %-22s %13s %9s %8s\n", "dt", "arm", "max|dX|", "matvec", "secs")
    for dt in (0.25, 0.125, 0.0625)
        for (nm, st, step) in (("midpoint @ dt", schemes[2][2], dt),
                               ("bug      @ dt/2", schemes[1][2], dt / 2))
            n = round(Int, T / step)
            psi = copy(psi0); mv = 0; t0 = time()
            for _ in 1:n; mv += _kry(st(psi, step)); end
            @printf("  %-8g %-22s %13.4e %9d %7.2fs\n",
                    dt, nm, xerr(psi, L, T, dirs), mv, time() - t0)
            flush(stdout)
        end
    end
    println("\nwrote ", OUT)
end

main()
