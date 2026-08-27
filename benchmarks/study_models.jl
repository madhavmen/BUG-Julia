# THE SIX MODELS THE rSVD / PARALLEL STUDY RUNS ON, in one place.
#
# `rsvd_models.jl` (does the sketch pay?) and `rsvd_parallel.jl` (does splitting the half-sweeps
# pay, and do the two compose?) must measure the SAME systems or their numbers cannot be put in one
# table. They used to carry a copy each.
#
#   xx           NN chain, `:U1`, d = 2, MPO dim 4    -- the NARROWEST generator
#   square       2D cylinder, `:U1`, d = 2            -- MPO width grows with Ly
#   kagome       breathing kagome + dipolar tail      -- the widest CLOSED MPO
#   xx_open      fused Lindblad, `:none`, d = 4, NN   -- narrow MPO at d = 4
#   square_open  fused Lindblad on a 2D patch, d = 4  -- wide AND d = 4
#   lgt          Schwinger + bath, all-to-all ZZ      -- the widest generator here
#
# ⚠ MPO WIDTH IS THE AXIS THAT SEPARATES THEM, and it cuts BOTH ways, which is the whole reason
# the list spans it. A wide MPO makes the CBE contraction the sketch shrinks more expensive (good
# for rSVD) and makes the environment work that the sketch cannot touch more expensive too (bad).
# Only measuring across the range says which wins.
#
# ⚠ `set_symmetry!` IS GLOBAL, so `setup` sets it and nothing may change it mid-run. Running two
# models in one process therefore means re-running `setup` between them, never interleaving.

"The staggered-fermion Schwinger chain: NN hopping, all-to-all electric ZZ, staggered mass."
function schwinger_terms(N::Int, x::Float64, mu::Float64, eps0::Float64)
    Jxy = zeros(Float64, N, N)
    for n in 1:(N - 1)
        Jxy[n, n + 1] = 2 * x
    end
    Jzz = zeros(Float64, N, N)
    for k in 1:(N - 1), l in (k + 1):N
        l <= N - 1 || continue
        Jzz[k, l] = 2.0 * (N - l)
    end
    a(n) = eps0 + (isodd(n) ? -0.5 : 0.0)
    hz = [2.0 * sum(a(n) for n in k:(N - 1); init = 0.0) + mu * (isodd(k) ? -1.0 : 1.0)
          for k in 1:N]
    return Jxy, Jzz, hz
end

"""
Closed `:U1` spin-1/2 lattice: real time `tau = -im*dt`, profile = SITE MAGNETISATIONS plus the
TOTAL ENERGY from the model's own MPO.

⛔ NOT A SPREAD OF PER-BOND `pair_mpo`s, WHICH IS WHAT THIS USED TO BUILD. MEASURED at L=10:
`setup` took **279 seconds** -- one `pair_mpo` per probe bond, nine of them, before a single step
ran. That is harness cost charged to every process invocation of every driver, and it dwarfed the
thing being timed.
⚠ It also made the accuracy guard WEAKER than it looks: nine one-bond energies all read the same
few sites, whereas `sz_expectation` at every site plus the total energy covers the whole chain and
carries the conserved quantity. Cheaper AND more sensitive.
"""
function _closed(nm, L, W, Jm, label, dt)
    prof = function (psi)
        p = copy(psi)
        n2 = max(norm(copy(p))^2, eps())
        vals = Float64[real(sz_expectation(p, j)) for j in 1:L]
        push!(vals, real(mpo_energy(copy(psi), W)) / n2)
        return vals
    end
    # ⚠ THE NO-OP `shift!` / `restore!` ARE HERE SO A DRIVER CAN RUN A CLOSED MODEL AS THE CONTROL
    # ARM OF AN OPEN-SYSTEM EXPERIMENT without branching on `S.d`. A closed unitary step needs
    # neither: there is no zero-body dissipator term and the 2-norm is already the invariant.
    return (nm = nm, label = label, L = L, W = W, d = 2, tau = ComplexF64(-im * dt),
            psi0 = neel_state(L), profile = prof, post! = (p) -> p,
            shift! = (p) -> p, restore! = (p) -> p, imps = nothing, shift = 0.0,
            params = nothing)
end

"""
Fused open system: `tau = dt` REAL (the generator already carries the `-i`), profile =
`Tr(S^z_j rho)`, and the zero-body constant plus the trace correction applied after each step.

⚠ THE POST-STEP WORK IS OUTSIDE EVERY TIMED REGION -- it is common to all arms and is not what is
being compared.
"""
function _open(nm, N, W, shift, label, dt, cap; params = nothing)
    imps = identity_superket(N)
    prof = rho -> [real(trace_rho(imps, rho, N; at = j)) for j in 1:N]
    # ⛔ THE TWO HALVES OF `post!` ARE ALSO EXPOSED SEPARATELY, and the reason is a measurement.
    # `shift!` supplies the dissipator's ZERO-BODY term (`exp(tau*c)` as an exact scalar, since
    # `pair_mpo` has nowhere to put a constant), so `Tr rho` is only meaningful AFTER it.
    # `restore!` then projects the drift away. A driver that wants to USE `Tr rho = 1` as an exact
    # error measure -- it is a conservation law of any Lindbladian, so its violation is an error
    # with no reference state required -- must read the trace BETWEEN the two. Calling `post!`
    # first destroys exactly the number it wanted to measure.
    shift! = function (p)
        shift != 0.0 && apply_shift!(p, exp(dt * shift))
        return p
    end
    restore! = (p) -> restore_trace!(imps, p, N; maxdim = cap, cutoff = 1e-10)
    post! = function (p)
        shift!(p)
        restore!(p)
        return p
    end
    # ⛔ `params` CARRIES THE COUPLINGS THAT BUILT `W`, AND THAT IS THE WHOLE POINT. An exact
    # dense-Liouvillian reference (`tests/common/dense_lindblad.jl`) must be built from the SAME
    # numbers the MPO was built from. A driver that re-derives the lattice itself is not validating
    # the integrator -- it is comparing two independent lattice constructions, and a disagreement
    # there is indistinguishable from an integrator error. Passing them through makes the reference
    # a function of the model rather than a second guess at it.
    return (nm = nm, label = label, L = N, W = W, d = 4, tau = ComplexF64(dt),
            psi0 = rho0_product([isodd(k) ? :up : :down for k in 1:N]),
            profile = prof, post! = post!,
            shift! = shift!, restore! = restore!, imps = imps, shift = shift,
            params = params)
end

"""
    shape_for(name, sites) -> Tuple

The 2D shape that gives `sites` sites, so every model in a table is the SAME LENGTH OF CHAIN.

⛔ ONE SITE COUNT ACROSS THE STUDY, NOT ONE SHAPE. A timing table that puts a 32-site cylinder
beside an 18-site chain is measuring the size difference as much as the method, and the sweep cost
is linear in `L` before anything else happens. The lattices are therefore folded to the requested
length rather than kept at their natural aspect: `L = 18` is a 6x3 cylinder and a 3x2 kagome
(3 sites per unit cell), both exactly 18.

⚠ A REQUEST THAT DOES NOT FACTOR IS ROUNDED DOWN, and the actual count is printed. Silently
running 20 sites for a table headed `L = 18` is the failure this returns rather than hides.
"""
function shape_for(nm, sites::Int)
    if nm in ("square", "square_open")
        ly = gcd(sites, 3) == 3 ? 3 : (iseven(sites) ? 2 : 1)
        return (sites ÷ ly, ly)
    elseif nm == "kagome"
        cells = sites ÷ 3                       # 3 sites per kagome unit cell
        ly = gcd(cells, 2) == 2 ? 2 : 1
        return (cells ÷ ly, ly)
    end
    return (sites,)
end

"""
    setup(name; sites = 18, size = Int[], dt = 0.05, gamma = 0.1, cap = 256) -> NamedTuple

Build one model at `sites` sites (or at an explicit `size`, which overrides). `gamma` is SMALL ON
PURPOSE for the open ones: dephasing suppresses operator entanglement, so a strongly damped run
never reaches the chi where any of this is decidable.
"""
function setup(nm; sites::Int = 18, size::Vector{Int} = Int[], dt::Float64 = 0.05,
               gamma::Float64 = 0.1, cap::Int = 256)
    sz = isempty(size) ? shape_for(nm, sites) : Tuple(size)
    if nm == "xx"
        set_symmetry!(:U1)
        L = sz[1]
        W = xxz_mpo(L; J = 1.0, delta = 0.0)
        Jm = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:L, j in 1:L]
        return _closed(nm, L, W, Jm, "XX chain, L = $L (NN MPO -- the narrow-generator control)",
                       dt)
    elseif nm == "square"
        set_symmetry!(:U1)
        LX, LY = sz[1], sz[2]
        return _closed(nm, LX * LY, square_cylinder_mpo(LX, LY),
                       square_cylinder_couplings(LX, LY),
                       "$(LX)x$(LY) cylinder, $(LX * LY) sites", dt)
    elseif nm == "kagome"
        set_symmetry!(:U1)
        LX, LY = sz[1], sz[2]
        return _closed(nm, 3 * LX * LY, breathing_kagome_mpo(LX, LY; h = 0.30),
                       breathing_kagome_couplings(LX, LY; h = 0.30),
                       "$(LX)x$(LY) breathing kagome, $(3 * LX * LY) sites, h = 0.30", dt)
    end
    set_symmetry!(:none)
    if nm == "xx_open"
        N = sz[1]
        Jm = [abs(i - j) == 1 && i < j ? 1.0 : 0.0 for i in 1:N, j in 1:N]
        W, shift = fused_lindblad(N, Jm, fill(gamma, N); delta = 0.0)
        return _open(nm, N, W, shift,
                     "XX + dephasing, N = $N, NN only (narrow MPO at d = 4)", dt, cap;
                     params = (Jm = Jm, gammas = fill(gamma, N), delta = 0.0,
                               hz = nothing, Jzz = nothing))
    elseif nm == "square_open"
        LX, LY = sz[1], sz[2]
        N = LX * LY
        Jm = square_cylinder_couplings(LX, LY; periodic_y = false)
        W, shift = fused_lindblad(N, Jm, fill(gamma, N); delta = 1.0)
        return _open(nm, N, W, shift,
                     "Heisenberg $(LX)x$(LY) open patch + dephasing, $N sites", dt, cap;
                     params = (Jm = Jm, gammas = fill(gamma, N), delta = 1.0,
                               hz = nothing, Jzz = nothing))
    elseif nm == "lgt"
        # ⚠ THE STAGGERED-FERMION CHAIN NEEDS AN EVEN N -- an odd request is rounded DOWN, not
        # thrown, so a mixed table still runs and the printed site count shows what happened.
        N = iseven(sz[1]) ? sz[1] : sz[1] - 1
        Jxy, Jzz, hz = schwinger_terms(N, 1.0, 0.5, 0.0)
        W, shift = fused_lindblad(N, Jxy, fill(gamma, N); hz = hz, Jzz = Jzz)
        return _open(nm, N, W, shift,
                     "Schwinger + bath, N = $N (the widest generator here)", dt, cap;
                     params = (Jm = Jxy, gammas = fill(gamma, N), delta = 0.0,
                               hz = hz, Jzz = Jzz))
    end
    error("unknown model $nm -- one of xx | square | kagome | xx_open | square_open | lgt")
end
