# tests/common/ramped_tdvp.jl -- the Task 7 reference solver: order-p TDVP2
# (`tdvp2_composed_step!`) whose FIRST step is split geometrically.
#
# From a product state (the domain wall) the first projected TDVP2 step leaves an error that no
# later step removes: Var(H) - s^2 = 2.18e-3, 5.44e-4, 1.36e-4 at s = 3, dt = 0.025, 0.0125,
# 0.00625 (hq22a/b/c), exactly dt^2, flat afterwards. Replacing step 1 by
# dt/2^k, dt/2^k, dt/2^(k-1), ..., dt/2 (summing to dt) shrinks it by 4^-k (hq26b, s = 3 L = 6
# against ED: 3.48e-4, 1.36e-6, 5.31e-9 at k = 0, 4, 8). Steps 2..n are the plain dt steps.

"The first step's sub-steps: `dt/2^k, dt/2^k, dt/2^(k-1), ..., dt/2` (`[dt]` at `k = 0`)."
first_step_ramp(dt, k::Int) = k == 0 ? [dt] : [dt / 2^k; [dt / 2^r for r in k:-1:1]]

"""
    ramped_tdvp2_step!(psi, mpo, dt, n; ramp, kw...) -> info

Step `n` (1-based) of real-time order-p TDVP2 by `dt` (`exp(-i dt H)`): step 1 runs
`first_step_ramp(dt, ramp)`, later steps one `dt` step. `kw` goes to `tdvp2_composed_step!`;
`info.cumulative` is summed over the sub-steps.
"""
function ramped_tdvp2_step!(psi, mpo, dt, n::Int; ramp::Int, kw...)
    hs = n == 1 ? first_step_ramp(dt, ramp) : [dt]
    disc = 0.0
    for h in hs
        disc += tdvp2_composed_step!(psi, mpo, ComplexF64(-1im * h); kw...).cumulative
    end
    return (cumulative = disc,)
end
