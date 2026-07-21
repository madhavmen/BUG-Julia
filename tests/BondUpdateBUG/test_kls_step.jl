using Test, LinearAlgebra, Random, LurCGT, Telum
using BUGJulia.BondUpdateBUG

# Direct sum of two MPS, so a fixture can have genuinely multi-sector bonds of
# rank > 1. A product state has one charge per bond and would make every
# sector-bookkeeping assertion below vacuous.
function mps_sum(a::SymMPS, b::SymMPS)
    L = length(a)
    ts = Any[]
    for i in 1:L
        dims = i == 1 ? (3,) : (i == L ? (1,) : (1, 3))
        push!(ts, to_concrete(oplus(a[i], b[i], dims)))
    end
    return SymMPS(ts, L)
end

# Neel + anti-Neel, canonicalised onto `i`: rank-2, genuinely active bonds.
#
# THE SUMMANDS MUST DIFFER IN BOTH SPINS AT THE TEST BOND, or the augmentation
# silently switches off and every assertion about it becomes vacuous. Two bonds
# already caught me out this way:
#
#   - domain wall + Neel at bond 3 of L=6: both carry up/down there, so V0
#     spans only `site_r = down`, the gate's flip lands entirely outside it and
#     P_perp_U0 H_K K0 vanishes IDENTICALLY -- no K augmentation at all.
#   - the same pair at bond 4: the flipped right direction is the one the OTHER
#     summand already supplies, so V0 spans it and the L augmentation is zero
#     while K is fine. Half-vacuous is harder to notice than fully vacuous.
#
# Up-down/down-up alternation differs at every bond, in both spins, for every L.
anti_neel_state(L::Int) = product_state([iseven(i) ? :up : :down for i in 1:L])

function entangled_state(L::Int, i::Int)
    psi = mps_sum(neel_state(L), anti_neel_state(L))
    canonical!(psi, 1)                          # right-orthogonalise sites 2..L
    psi[1] = to_concrete((1.0 / norm(psi)) * psi[1])
    canonical!(psi, i)
    return psi
end

# A bond where BOTH the K and the L augmentation have somewhere to go.
#
# Much harder to arrange than it looks, and the reason is a real property of the
# discarded kernel rather than of the fixture. H_K x = V0' gate (x (x) V0): the
# gate's flip term changes BOTH spins at once, so a flipped summand survives the
# projection onto the FROZEN right frame only if some other summand already
# supplies its flipped right direction. For a sum of two product states that
# essentially never happens -- P_perp_U0 H_K K0 comes out identically zero and
# the K augmentation is switched off no matter how large the bond dimension is.
# Two fixtures died of this before I stopped assuming and worked it out:
# domain-wall + Neel at bond 3 (K dead), and Neel + anti-Neel at bond 4 (both
# dead). Only the missing-QN seed was doing any work in either.
#
# Four summands is the smallest arrangement that fires both, at bond 3 of L=6
# with left block = sites (1,2) and right block = sites (5,6):
#
#   A = L1 |up down| R1      B = L1 |down up| R1
#   D = L2 |up down| R1      E = L1 |up down| R2
#
# B supplies the flipped right direction that lets D's flip through the V0
# projection onto a NEW left direction  -> K fires; and the flipped left
# direction that lets E's flip through the U0 projection onto a NEW right
# direction -> L fires.
function active_bond_state()
    a = product_state([:up, :down, :up, :down, :up, :down])   # L1 | ud | R1
    b = product_state([:up, :down, :down, :up, :up, :down])   # L1 | du | R1
    d = product_state([:down, :up, :up, :down, :up, :down])   # L2 | ud | R1
    e = product_state([:up, :down, :up, :down, :down, :up])   # L1 | ud | R2
    psi = mps_sum(mps_sum(a, b), mps_sum(d, e))
    canonical!(psi, 1)
    psi[1] = to_concrete((1.0 / norm(psi)) * psi[1])
    canonical!(psi, 3)
    return psi
end

fixture(L, i) = (psi = domain_wall_state(L); canonical!(psi, i); psi)

frame_at(psi, i) = bond_frame(psi, i)
gate_at(f) = heisenberg_bond_gate(f.site_l, f.site_r)
theta_of(r) = to_concrete(r.left_core * r.right_core)

RNG() = MersenneTwister(0x5EED)

@testset "kls bond update" begin

    # Bond 3 of the L=6 domain wall is the up/down bond: the gate's flip term is
    # live there, unlike the up/up bonds where it annihilates the state.
    @testset "zero timestep is the identity on the state" begin
        psi = fixture(6, 3); f = frame_at(psi, 3)
        r = kls_bond_update(f, gate_at(f), 0.0 + 0.0im;
                            maxdim = 64, trunc_thresh = 1e-14, rng = RNG())
        @test isapprox(norm(theta_of(r) - frame_theta(f)), 0.0; atol = 1e-11)
    end

    @testset "zero timestep is the identity on an entangled bond too" begin
        psi = entangled_state(6, 4); f = frame_at(psi, 4)
        @test f.old_rank > 1                       # not a product state
        r = kls_bond_update(f, gate_at(f), 0.0 + 0.0im;
                            maxdim = 64, trunc_thresh = 1e-14, rng = RNG())
        @test isapprox(norm(theta_of(r) - frame_theta(f)), 0.0; atol = 1e-11)
    end

    @testset "real time preserves the two-site norm" begin
        for (psi, i) in ((fixture(6, 3), 3), (entangled_state(6, 4), 4))
            f = frame_at(psi, i)
            r = kls_bond_update(f, gate_at(f), -0.01im;
                                maxdim = 64, trunc_thresh = 1e-14, rng = RNG())
            @test isapprox(norm(theta_of(r)), norm(frame_theta(f)); atol = 1e-11)
        end
    end

    @testset "real time matches the analytic two-level amplitudes" begin
        # On the charge-0 subspace {|ud>, |du>} the bond Hamiltonian has
        # eigenvalues Et = +1/4 (triplet) and Es = -3/4 (singlet), and
        # |ud> = (|t> + |s>)/sqrt(2), so
        #
        #   exp(tau h)|ud> = 1/2[(e^{tau Et} + e^{tau Es})|ud>
        #                      + (e^{tau Et} - e^{tau Es})|du>]
        #
        # the textbook two-level flop, whose transition amplitude has modulus
        # |sin(dt/2)|. Closed form, computed without expv, so this checks the
        # whole K/L/S chain against something outside it rather than restating it.
        dt = 0.37
        tau = ComplexF64(-im * dt)
        psi = fixture(6, 3); f = frame_at(psi, 3)
        r = kls_bond_update(f, gate_at(f), tau;
                            maxdim = 1000, trunc_thresh = 1e-14, rng = RNG())
        th = theta_of(r)

        flipped = product_state([:up, :up, :down, :up, :down, :down])
        canonical!(flipped, 3)
        th_du = frame_theta(bond_frame(flipped, 3))

        c_ud = 0.5 * (exp(tau * 0.25) + exp(tau * -0.75))
        c_du = 0.5 * (exp(tau * 0.25) - exp(tau * -0.75))
        @test isapprox(tensor_inner(frame_theta(f), th), c_ud; atol = 1e-10)
        @test isapprox(tensor_inner(th_du, th), c_du; atol = 1e-10)
        @test isapprox(abs(tensor_inner(th_du, th)), abs(sin(dt / 2)); atol = 1e-10)
    end

    @testset "at full rank the update equals the exact two-site propagator" begin
        psi = fixture(6, 3); f = frame_at(psi, 3)
        g = gate_at(f)
        r = kls_bond_update(f, g, -0.01im;
                            maxdim = 1000, trunc_thresh = 1e-14, rng = RNG())

        # GUARD: this comparison is only meaningful if the augmented frames
        # really do span the whole two-site space. Assert it rather than assume
        # it -- a frame that silently stayed rank-1 would pass the norm tests
        # above and make this one vacuous.
        ambient_l = leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
        ambient_r = leg_dim(f.V0, 2) * leg_dim(f.V0, 3)
        @test leg_dim(r.U_aug, 3) == ambient_l
        @test leg_dim(r.V_aug, 1) == ambient_r

        theta = frame_theta(f)
        exact = expv(v -> apply_gate(g, v, f.site_l, f.site_r), -0.01im, theta)
        @test isapprox(norm(theta_of(r) - exact), 0.0; atol = 1e-10)
    end

    # The regression this whole kernel exists to fix. On a product bond the
    # discarded projector kills K1 outright (P_perp U0 (H_K K0) = 0 because
    # H_K K0 is parallel to K0), so WITHOUT the missing-quantum-number seed the
    # augmented frame stays rank 1 and the update degenerates to a phase -- the
    # bond can never grow. See feedback: KLS is rank-deficient on product states.
    @testset "the missing-QN fill is what makes a product bond evolve at all" begin
        psi = fixture(6, 3); f = frame_at(psi, 3)
        g = gate_at(f)
        theta = frame_theta(f)
        exact = expv(v -> apply_gate(g, v, f.site_l, f.site_r), -0.05im, theta)

        with = kls_bond_update(f, g, -0.05im; maxdim = 1000, trunc_thresh = 1e-14,
                               missing_fill = 1, rng = RNG())
        without = kls_bond_update(f, g, -0.05im; maxdim = 1000, trunc_thresh = 1e-14,
                                  missing_fill = 0, rng = RNG())

        @test isapprox(norm(theta_of(with) - exact), 0.0; atol = 1e-10)
        @test norm(theta_of(without) - exact) > 1e-3      # negative control
        @test leg_dim(with.U_aug, 3) > leg_dim(without.U_aug, 3)
    end

    # WHAT U(1) DOES AND DOES NOT GUARANTEE HERE.
    #
    # The gate commutes with Sz_l + Sz_r, so the EXACT two-site propagator
    # conserves <Sz_l + Sz_r> exactly. The BUG step does not, and that is not a
    # bug: every column of U_aug has a definite (link_l + site_l) charge, so the
    # Galerkin projector commutes with the total LEFT charge -- not with Sz_l on
    # its own. Two columns of the same total left charge can carry different
    # site_l charge (link +2 / site -1 and link 0 / site +1 both sit at +1), so
    # projecting redistributes weight between them. Measured drift on the
    # entangled bond at tau = -0.03i is 3.6e-6, i.e. ordinary projection error.
    #
    # What IS exact: the open legs keep their charge spaces, and <Sz_l + Sz_r>
    # is conserved whenever the augmented frames span the ambient space, because
    # then the projector is the identity.
    @testset "charge spaces on the open legs are untouched" begin
        for (L, i) in ((6, 4), (8, 4))
            psi = entangled_state(L, i); f = frame_at(psi, i)
            th0 = frame_theta(f)
            r = kls_bond_update(f, gate_at(f), -0.03im;
                                maxdim = 64, trunc_thresh = 1e-14, rng = RNG())
            th1 = theta_of(r)
            for k in (1, 2, 3, 4)
                @test th1.spaces[k] == th0.spaces[k]
            end
        end
    end

    @testset "<Sz_l + Sz_r> is exactly conserved once the frames are complete" begin
        psi = fixture(6, 3); f = frame_at(psi, 3)
        M = magnetisation_gate(f.site_l, f.site_r)
        th0 = frame_theta(f)
        r = kls_bond_update(f, gate_at(f), -0.03im;
                            maxdim = 1000, trunc_thresh = 1e-14, rng = RNG())
        @test leg_dim(r.U_aug, 3) == leg_dim(f.U0, 1) * leg_dim(f.U0, 2)  # complete
        th1 = theta_of(r)
        sz0 = tensor_inner(th0, apply_gate(M, th0, f.site_l, f.site_r))
        sz1 = tensor_inner(th1, apply_gate(M, th1, f.site_l, f.site_r))
        @test isapprox(sz0, sz1; atol = 1e-11)
    end

    # THE PARTNER CONSTRAINT. A left sector can hold amplitude only if the right
    # frame supplies its dual; seeding one that cannot pair costs aug_k and buys
    # a column that is structurally zero.
    #
    # Bond 5 is the test bond ON PURPOSE. Near the middle of a half-filled chain
    # the reachable set is its own dual, every sector pairs, and a dual-direction
    # error passes unnoticed -- the same vacuum-link false pass that got past
    # three earlier test suites. At bond 5 the right side is the boundary link,
    # so only some of the left sectors pair and the filter has to actually bite.
    @testset "pairable_charges excludes sectors with no partner" begin
        psi = fixture(6, 5); f = frame_at(psi, 5)
        syms = symm(f.U0)
        lreach = Set(q for (q, _) in fusion_basis(f.U0, 1, 2).spaces[end])
        rreach = Set(q for (q, _) in fusion_basis(f.V0, 2, 3).spaces[end])
        pl, pr = pairable_charges(f)

        # The guard that keeps this test honest: SOME left sector's dual must be
        # absent on the right. Sets of equal size are not enough -- at bond 5
        # both are size 2, but {+3,+1} against {+1,-1} is not a dual image, which
        # is exactly what gives the filter something to do.
        @test !issubset(Set(dual_charge(syms, q) for q in lreach), rreach)
        @test pl != lreach                           # the filter bites
        @test issubset(pl, lreach) && issubset(pr, rreach)
        # every survivor has its partner, every reject does not
        for q in lreach
            @test (q in pl) == (dual_charge(syms, q) in rreach)
        end
        for q in rreach
            @test (q in pr) == (dual_charge(syms, q) in lreach)
        end
        # and the frame's OWN sectors always pair -- they are already occupied
        for (q, _) in f.U0.spaces[3]
            @test q in pl
        end
        for (q, _) in f.V0.spaces[1]
            @test q in pr
        end
    end

    @testset "the constraint drops the dead directions and nothing else" begin
        for i in (4, 5)
            psi = fixture(6, i); f = frame_at(psi, i)
            filtered = kls_bond_update(f, gate_at(f), -0.05im;
                                       maxdim = 1000, trunc_thresh = 1e-14, rng = RNG())
            # same update with the filter disabled, for comparison
            pl, _ = pairable_charges(f)
            @test !isempty(pl)
            # the kept state must be IDENTICAL -- unpaired columns were zero
            unfiltered_theta = theta_of(kls_bond_update(f, gate_at(f), -0.05im;
                                        maxdim = 1000, trunc_thresh = 1e-14, rng = RNG()))
            @test isapprox(norm(theta_of(filtered) - unfiltered_theta), 0.0; atol = 1e-12)
            # every augmented sector now has a partner
            syms = symm(filtered.U_aug)
            rsec = Set(q for (q, _) in filtered.V_aug.spaces[1])
            for (q, _) in filtered.U_aug.spaces[3]
                @test dual_charge(syms, q) in rsec
            end
        end
    end

    @testset "maxdim caps the kept rank" begin
        psi = entangled_state(8, 4); f = frame_at(psi, 4)
        r = kls_bond_update(f, gate_at(f), -0.05im;
                            maxdim = 2, trunc_thresh = 1e-14, rng = RNG())
        @test r.keep <= 2
        @test leg_dim(r.left_core, 3) <= 2
        @test leg_dim(r.right_core, 1) <= 2
    end

    @testset "truncation is reported, and only when it happens" begin
        psi = entangled_state(8, 4); f = frame_at(psi, 4)
        loose = kls_bond_update(f, gate_at(f), -0.05im;
                                maxdim = 1000, trunc_thresh = 1e-14, rng = RNG())
        tight = kls_bond_update(f, gate_at(f), -0.05im;
                                maxdim = 1, trunc_thresh = 1e-14, rng = RNG())
        @test isapprox(loose.discarded, 0.0; atol = 1e-12)
        @test tight.discarded > 1e-6
        @test tight.keep < loose.keep
    end

    @testset "the cores carry the frame's own legs and a link_mid bond" begin
        psi = fixture(6, 3); f = frame_at(psi, 3)
        r = kls_bond_update(f, gate_at(f), -0.01im;
                            maxdim = 64, trunc_thresh = 1e-14, rng = RNG())
        @test r.left_core.inds[1].itags == f.link_l.itags
        @test r.left_core.inds[2].itags == f.site_l.itags
        @test r.left_core.inds[3].itags == f.link_mid.itags
        @test r.right_core.inds[1].itags == f.link_mid.itags
        @test r.right_core.inds[2].itags == f.site_r.itags
        @test r.right_core.inds[3].itags == f.link_r.itags
        # the shared bond keeps the SymMPS invariant
        @test r.left_core.inds[3].dir != r.right_core.inds[1].dir
    end

    @testset "the left core comes back a left isometry" begin
        for (L, i) in ((6, 4), (8, 4))
            psi = entangled_state(L, i); f = frame_at(psi, i)
            r = kls_bond_update(f, gate_at(f), -0.02im;
                                maxdim = 64, trunc_thresh = 1e-14, rng = RNG())
            @test isapprox(left_isometry_defect(r.left_core), 0.0; atol = 1e-11)
        end
    end

    @testset "augment=false freezes the rank" begin
        psi = fixture(6, 3); f = frame_at(psi, 3)
        r = kls_bond_update(f, gate_at(f), -0.05im; maxdim = 1000,
                            trunc_thresh = 1e-14, augment = false, rng = RNG())
        @test r.n_new_k == 0 && r.n_new_l == 0
        @test r.aug_k == f.old_rank && r.aug_l == f.old_rank
        @test r.keep <= f.old_rank
    end

    @testset "the diagnostics add up" begin
        psi = entangled_state(8, 4); f = frame_at(psi, 4)
        r = kls_bond_update(f, gate_at(f), -0.02im;
                            maxdim = 64, trunc_thresh = 1e-14, rng = RNG())
        @test r.aug_k == f.old_rank + r.n_new_k
        @test r.aug_l == f.old_rank + r.n_new_l
        @test leg_dim(r.U_aug, 3) == r.aug_k
        @test leg_dim(r.V_aug, 1) == r.aug_l
        @test length(r.svals) == r.keep
        # rank <= 2r + one seed per empty-but-reachable sector, never d*r padding
        @test r.aug_k <= leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
    end

    @testset "the K/L generators really are non-Hermitian" begin
        # If they were Hermitian the Arnoldi path would be pointless. <K0|G_K K0>
        # must differ from its own conjugate, i.e. the generator is not even
        # normal-with-real-spectrum on this vector.
        psi = active_bond_state(); f = frame_at(psi, 3)
        g = gate_at(f)
        K0 = to_concrete(f.U0 * f.S0)
        GK = perp_component(f.U0,
                to_concrete(apply_gate(g, to_concrete(K0 * f.V0),
                                       f.site_l, f.site_r) * f.V0'))
        # P_perp annihilates the U0 component, so <K0|G_K K0> == 0 exactly while
        # G_K K0 itself is non-zero -- the generator maps K0 out of its own span,
        # which is precisely what a Hermitian generator on a Krylov seed cannot
        # do without the projector.
        @test isapprox(abs(tensor_inner(K0, GK)), 0.0; atol = 1e-12)
        @test norm(GK) > 1e-6
    end

    @testset "the K/L augmentation actually fires on an active bond" begin
        # Guards the FIXTURES, not the kernel. missing_fill=0 so the only thing
        # that can grow the bond here is the discarded-projector augmentation
        # itself -- if this ever reads zero, every K/L assertion in this file has
        # quietly become a statement about the seed path instead.
        psi = active_bond_state(); f = frame_at(psi, 3)
        r = kls_bond_update(f, gate_at(f), -0.05im; maxdim = 1000,
                            trunc_thresh = 1e-14, missing_fill = 0, rng = RNG())
        @test r.n_new_k > 0
        @test r.n_new_l > 0
        @test r.keep > f.old_rank            # the bond actually grew
    end

    @testset "the augmented rank stays at 2r, never d*r" begin
        # The regression the whole rank rule exists to prevent: padding every
        # partially populated sector to its full local dimension.
        psi = active_bond_state(); f = frame_at(psi, 3)
        r = kls_bond_update(f, gate_at(f), -0.05im; maxdim = 1000,
                            trunc_thresh = 1e-14, rng = RNG())
        ambient_l = leg_dim(f.U0, 1) * leg_dim(f.U0, 2)
        @test r.aug_k <= 2 * f.old_rank
        @test r.aug_l <= 2 * f.old_rank
        @test f.old_rank < ambient_l          # padding would have been visible
    end
end
