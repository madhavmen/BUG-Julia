# Dynamical spin structure factor of the Haldane-Shastry ring, the Li et al. (arXiv:2208.10972)
# Fig. 3 protocol.
#
# WHY THIS EXISTS. Every other arm of the L=16 campaign is graded against a closed form -- Bethe
# for the Heisenberg chain, free fermions for XX, `-pi^2 (L^2+5) / (24 L)` for the HS ground
# energy. Real-time HS had none, so it was graded on ENERGY CONSERVATION, and at L=16 that
# measures nothing: `S^z_{j0}|0>` starts with a centre bond of 247 against a 256 cap, i.e. the
# state is essentially full rank before the first step, nothing is truncated, and conservation is
# satisfied by construction (measured: 2.7e-15). A degenerate test that always passes.
#
# The paper's observable is not conserved and does discriminate:
#
#     C(x, t) = e^{i E_0 t} <0| S^z_{j0+x} e^{-i H t} S^z_{j0} |0>
#     S(k, w) = sum_x int dt  e^{i (w t - k x)} C(x, t) W(t)
#
# with `W` a window that kills the ringing from the finite time record. `S(k,w)` for HS is the
# two-spinon continuum: weight strictly inside `w_lower(k) <= w <= w_upper(k)`, a square-root edge
# at the lower boundary, and ZERO outside. Those are the features to check against the paper --
# the support, the edges, the sum rule -- not a pointwise number, because the paper's figure is a
# thermodynamic-limit object and this is L=16.
#
# ⛔ NO EXACT DIAGONALISATION AND NO SPARSE PROPAGATION ANYWHERE HERE, by directive: for exactly
# solvable models the reference is analytic. The finite-L checks below are SUM RULES and SYMMETRY
# identities, which are exact at any L and cost nothing:
#
#   * C(x, 0) = <0| S^z_{j0+x} S^z_{j0} |0>  -- the static correlator, real and symmetric in x
#   * sum_k S(k, w) summed over w  ==  L * <(S^z)^2> = L/4  (total spectral weight, since
#     S^z_j has eigenvalues +-1/2 and the HS ground state is a spin singlet)
#   * S(k, w) = 0 for w < 0 (the operator acts on the GROUND state, so only positive
#     excitation energies contribute)
#   * C(-x, t) = C(x, t) on a RING (translation + reflection symmetry)
#
# A violation of any of those is a bug in the pipeline, independent of the paper.

using Printf

"""
    sz_correlation(psi0, E0, step, j0, nt, dt) -> (ts, C)

`C[d + 1, n] = e^{i E_0 t_n} <0| S^z_{j0 + d} |phi(t_n)>` with `|phi> = S^z_{j0}|0>`, where `d` is
the RING DISPLACEMENT from `j0`, running `0:L-1` with the site index taken mod `L`.

⛔ THE FIRST INDEX IS A DISPLACEMENT, NOT A SITE. `structure_factor` transforms with `e^{-i k d}`
using the row index directly, so this re-centring is what makes that transform correct. Returning
raw site order instead leaves a constant factor `e^{-i k (j0 - 1)}` in `C_k(t)`: it is
time-INDEPENDENT, so it cannot be spotted in any single-time diagnostic, but it rotates the whole
complex time series and the `2 * real(...)` in `structure_factor` then returns a MIXTURE of the
real and imaginary parts of the true transform. `S(k, w)` comes out neither positive nor
band-limited, and the failure looks like a broken integrator rather than a mislabelled axis.
Re-centring here rather than in the transform keeps `j0` out of `structure_factor`'s signature and
makes the row index mean one thing only.

`step` is the integrator under test, called as `step(psi, tau)` exactly as in the campaign, so
this measures THE SCHEME and not some private propagator.
"""
function sz_correlation(psi0::SymMPS, E0::Float64, step, j0::Int, nt::Int, dt::Float64)
    L = length(psi0)
    phi = deepcopy(psi0)
    apply_sz!(phi, j0)

    ts = collect(0:nt) .* dt
    C = zeros(ComplexF64, L, nt + 1)

    # The L bras `S^z_x|0>` are built ONCE, outside the time loop: `psi0` never changes, and
    # `apply_sz!` runs two full `canonical!` sweeps, so rebuilding them every step cost `L * nt`
    # canonicalisations for `L` distinct states. `overlap` copies and retags the bra internally
    # rather than mutating it, so reuse is safe.
    bras = map(0:(L - 1)) do d
        x = mod1(j0 + d, L)              # the ring wrap; `mod1` because sites are 1-based
        b = deepcopy(psi0)
        apply_sz!(b, x)
        b
    end

    # `<0| S^z_{j0+d} |phi>` for every displacement at one time. S^z is applied to the GROUND
    # state, never to `phi` -- `phi` must stay the evolving state.
    row!(col, phi) = for d in 0:(L - 1)
        C[d + 1, col] = overlap(bras[d + 1], phi)
    end

    row!(1, phi)
    for n in 1:nt
        step(phi, ComplexF64(-im * dt))
        # The `e^{i E_0 t}` factor removes the ground state's own phase, so a STATIC correlator
        # stays static instead of winding at `E_0` -- without it every C(x,t) rotates and the
        # spectrum is rigidly shifted by `E_0`.
        row!(n + 1, phi)
        C[:, n + 1] .*= cis(E0 * ts[n + 1])
    end
    return ts, C
end

"""
    structure_factor(C, ts, L; nw=400, wmax=6.0, window=:hann) -> (ks, ws, S)

`S(k, w) = sum_d int dt e^{i(w t - k d)} C(d, t)`, on the ring momenta `k = 2*pi*m/L`, with `d` the
displacement `C`'s first index already carries (see `sz_correlation`).

NO `1 / 2pi` IS APPLIED. `S` is the plain `int dt`, so the spectral-weight identity carries the
`2pi` on the other side: `sum_k int dw S(k,w) / (2pi) = L * <(S^z)^2> = L/4`. `check_sum_rules`
divides by `2pi` for exactly this reason -- dropping it there reports `2pi * 0.25 = 1.571` against a
target of `0.25` and reads as a broken normalisation in the integrator.

The time integral is over a FINITE record `[0, T]`, which convolves the true spectrum with the
window's transform. A rectangular window's sinc sidelobes put spurious weight OUTSIDE the
two-spinon continuum -- exactly the feature being checked -- so a Hann window is the default and
the choice is reported rather than hidden. The window does NOT spoil the weight identity above,
because `w(0) = 1` and the `dw` integral of the smeared spectrum returns `w(0) * C_k(0)`.

Only `t >= 0` is computed: `C(d,-t) = conj(C(d,t))` and `C(-d,t) = C(d,t)` on a RING, so
`C_k(-t) = conj(C_k(t))` and the two-sided integral is `2 Re` of the one-sided one. ⛔ That rests on
the ring's reflection symmetry through `j0`; on an OPEN chain `C(-d,t) != C(d,t)` and this
shortcut is wrong. Haldane-Shastry is the only ring in the campaign.
"""
function structure_factor(C::Matrix{ComplexF64}, ts::Vector{Float64}, L::Int;
                          nw::Int = 400, wmax::Float64 = 6.0, window::Symbol = :hann)
    nt = length(ts)
    T = ts[end]
    w = window === :hann   ? [0.5 * (1 + cos(pi * t / T)) for t in ts] :
        window === :none   ? ones(nt) :
        throw(ArgumentError("window must be :hann or :none, got $window"))

    ks = [2pi * m / L for m in 0:(L - 1)]
    ws = collect(range(-wmax / 8, wmax; length = nw))
    S = zeros(Float64, L, nw)
    dt = length(ts) > 1 ? ts[2] - ts[1] : 1.0

    for (ik, k) in enumerate(ks)
        # Spatial transform first: one complex time series per k. `row - 1` IS the displacement
        # `d` from `j0`, not a site offset -- `sz_correlation` re-centred so that this line needs
        # no knowledge of where `j0` sits.
        Ck = [sum(C[d + 1, n] * cis(-k * d) for d in 0:(L - 1)) for n in 1:nt]
        for (iw, om) in enumerate(ws)
            # Re over t>=0 and doubling IS the full two-sided integral, given the conjugate
            # symmetry noted above; writing it this way avoids storing the mirrored record.
            acc = sum(Ck[n] * cis(om * ts[n]) * w[n] for n in 1:nt) * dt
            S[ik, iw] = 2 * real(acc) - real(Ck[1]) * dt * w[1]
        end
    end
    return ks, ws, S
end

"""
    two_spinon_band(k) -> (w_lower, w_upper)

The exact HS two-spinon continuum in the thermodynamic limit, with `J = 1` and the convention
`H = sum_{i<j} J/d(i,j)^2 S_i.S_j` that `haldane_shastry_couplings` builds.

One spinon at momentum `p in [0, pi]` costs `eps(p) = p (pi - p) / 2`. A two-spinon state of total
momentum `k = p1 + p2` therefore spans

    w_lower(k) = eps(k)            = k (pi - k) / 2     (one spinon carries everything)
    w_upper(k) = 2 * eps(k / 2)    = k (2 pi - k) / 4   (the two share it equally)

These bound the support; the spectral weight has a square-root divergence at `w_lower`. Used as a
QUALITATIVE overlay on the figure -- the finite-L spectrum is a set of delta peaks broadened by
the window, so it is the SUPPORT and the edge shape that should match, never a pointwise value.
"""
function two_spinon_band(k::Float64)
    kk = mod(k, 2pi)
    kk > pi && (kk = 2pi - kk)          # the band is symmetric about k = pi
    return (kk * (pi - kk) / 2, kk * (2pi - kk) / 4)
end

"""
    check_sum_rules(C, L, S, ws) -> NamedTuple

The finite-L identities that must hold EXACTLY, so a pipeline bug cannot hide behind "the paper
is a thermodynamic-limit figure".
"""
function check_sum_rules(C::Matrix{ComplexF64}, L::Int,
                         S::Matrix{Float64}, ws::Vector{Float64})
    dw = length(ws) > 1 ? ws[2] - ws[1] : 1.0
    # 1. equal-time correlator is real
    c0_imag = maximum(abs.(imag.(C[:, 1])))
    # 2. on-site value is <(S^z)^2> = 1/4 exactly. Row 1 IS d = 0 by construction, so index it
    #    directly rather than searching for the largest entry -- `argmax` happens to find d = 0
    #    here only because the off-site correlators are smaller, which is a fact about HS and not
    #    something this check should depend on.
    onsite = real(C[1, 1])
    # 3. total spectral weight. The `2pi` belongs here because `structure_factor` returns the
    #    plain `int dt` with no `1/2pi` folded in; see its docstring.
    total = sum(S) * dw / (2pi * L)
    # 4. no weight at negative frequency
    negidx = findall(<(0.0), ws)
    negw = isempty(negidx) ? 0.0 : maximum(abs.(S[:, negidx])) / max(maximum(abs.(S)), eps())
    # 5. ring reflection: C(-d, t) = C(d, t), i.e. row `d+1` equals row `L-d+1`. This was
    #    DOCUMENTED as an identity and relied on by the `2 Re` one-sided time integral, but never
    #    measured. It is also the sharpest available check that the re-centring is right: a
    #    mislabelled first index breaks the pairing immediately.
    refl = 0.0
    for d in 1:(L - 1)
        refl = max(refl, maximum(abs.(C[d + 1, :] .- C[L - d + 1, :])))
    end
    # 6. THE k = 0 CHANNEL IS EXACTLY EMPTY, AT EVERY TIME. `sum_d C(d,t) = <0| S^z_tot e^{-iHt}
    #    S^z_{j0} |0>` and the HS ground state is a spin singlet, so `<0| S^z_tot = 0` and the sum
    #    vanishes identically -- not approximately, and not only in the thermodynamic limit.
    #    This is the sharpest check available because it exercises the ENTIRE chain at once: the
    #    ring wrap in `mod1`, every one of the L bras, the time evolution's conservation of the
    #    U(1) charge, and the row indexing. A single mis-wrapped or duplicated displacement leaves
    #    a residue here while every other identity above still passes.
    ktot = maximum(abs.(sum(C; dims = 1)))
    return (; c0_imag, onsite, total, negw, refl, ktot)
end
