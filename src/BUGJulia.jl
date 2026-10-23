module BUGJulia

# `BondUpdateBUG` is the single consolidated symmetry-native BUG integrator.
include("BondUpdateBUG/BondUpdateBUG.jl")

# `RSVDCBEBondUpdate` is the exploratory CBE-BUG variant (docs/cbe_bug.md): the
# randomized-SVD controlled bond expansion in place of the K/L discarded-projector
# augmentation. Separate submodule -- it imports from `BondUpdateBUG` and changes
# nothing there.
include("RSVDCBEBondUpdate/RSVDCBEBondUpdate.jl")

# ── PRECOMPILE WORKLOAD ──
# Runs the real entry points so Telum's block-sparse specialisations land in the .ji cache;
# otherwise every process pays minutes of JIT at run time. L = 6 at maxdim = 8 specialises the same
# types as larger runs, keeping each `Pkg.precompile` cheap. It is guarded: a throw would fail
# precompilation itself, so failure must degrade to slow.
using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference

# ── WORKLOAD PREFERENCE ──
# A preference rather than an ENV switch, since it is part of the cache key. `adaptive` (default)
# covers the U(1) campaign paths; `full` adds :none, non-Hermitian, SU(2) and fused d = 4 Hubbard;
# `none` skips the workload. It pays off for jobs (break-even at about two processes per edit); the
# imaginary-time and open-system paths get no benefit from it, so stage those with `none`.
const _WORKLOAD = @load_preference("workload", "adaptive")

@setup_workload begin
    @compile_workload begin
        if _WORKLOAD in ("adaptive", "full")
        try
            BondUpdateBUG.set_symmetry!(:U1)
            psi0 = BondUpdateBUG.neel_state(6)
            mpo = RSVDCBEBondUpdate.xxz_mpo(6; J = 1.0, delta = 1.0)
            tau = ComplexF64(-0.05im)
            # maxdim = 4 so the cap BINDS and the truncation branch compiles (see the d = 4 note
            # below: one step from a product state never truncates).
            akw = (tol_basis = 1e-4, tol_trunc = 1e-6, tol_expv = 1e-6, maxdim = 4)
            for (exact, parallel) in ((false, false), (true, false), (false, true))
                p = copy(psi0)
                for _ in 1:3
                    RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, tau; exact = exact,
                                                    parallel = parallel, akw...)
                end
            end
            let p = copy(psi0)
                for _ in 1:2
                    RSVDCBEBondUpdate.cbe_bug_midpoint_step!(p, mpo, tau; parallel = true, akw...)
                end
                for ord in (2, 4)
                    RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, tau; order = ord, parallel = true, akw...)
                end
                for _ in 1:2
                    RSVDCBEBondUpdate.tdvp2_step!(p, mpo, tau; maxdim = 4, trunc_thresh = 1e-10,
                                                  maxiter = 8)
                    RSVDCBEBondUpdate.tdvp_cbe1s_step!(p, mpo, tau; maxdim = 4,
                                                       trunc_thresh = 1e-10, maxiter = 8)
                end
                BondUpdateBUG.magnetisation(p)
                BondUpdateBUG.overlap(p, psi0)
                q = copy(psi0); BondUpdateBUG.apply_sz!(q, 3)
            end
            for ex in (false, true)
                RSVDCBEBondUpdate.dmrg_cbe1s!(copy(psi0), mpo; n_sweeps = 2, tol_basis = 1e-3,
                                              tol_trunc = 1e-4, maxdim = 4, exact = ex)
            end
            let p = copy(psi0)
                RSVDCBEBondUpdate.dmrg_cbe1s!(p, mpo; n_sweeps = 2, maxdim = 4)
                RSVDCBEBondUpdate.mpo_energy(p, mpo)
                RSVDCBEBondUpdate.mpo_variance(p, mpo)
            end
        catch err
            @warn "BUGJulia precompile workload: the adaptive U(1) block did not complete; " *
                  "those runs will pay full JIT" exception = (err, catch_backtrace())
        finally
            BondUpdateBUG.set_symmetry!(:U1)
        end
        end   # adaptive block

        if _WORKLOAD == "full"   # the original blocks, kept for SU(2) / d = 4 campaigns
        try
            for sym in (:U1, :none)
                BondUpdateBUG.set_symmetry!(sym)
                psi = BondUpdateBUG.neel_state(6)
                mpo = RSVDCBEBondUpdate.xxz_mpo(6; J = 1.0, delta = 1.0)
                # both probe draws (exact DD reference / randomised) and both sweep
                # executions (sequential / concurrent half-sweeps)
                for (exact, parallel) in ((true, false), (false, false), (true, true))
                    p = copy(psi)
                    RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, ComplexF64(-0.05im);
                                                   exact = exact, parallel = parallel,
                                                   tol_basis = 1e-4, tol_trunc = 1e-6,
                                                   tol_expv = 1e-6, maxdim = 8)
                    BondUpdateBUG.sz_expectation(p, 3)
                    RSVDCBEBondUpdate.mpo_energy(copy(p), mpo)
                end
                # ── BASELINE SWEEPS ──
                # `tdvp2` and `tdvp_cbe1s` are in the workload too: otherwise the first arm to run
                # absorbs its own JIT and skews the timing ratios. `hermitian = false` (Arnoldi) is
                # a different code path, the one open-system runs take, so both `hermitian` branches
                # are compiled.
                for herm in (true, false)
                    p = copy(psi)
                    RSVDCBEBondUpdate.tdvp2_step!(p, mpo, ComplexF64(-0.05im);
                                                  maxdim = 8, trunc_thresh = 1e-10,
                                                  maxiter = 5, hermitian = herm)
                    q = copy(psi)
                    RSVDCBEBondUpdate.tdvp_cbe1s_step!(q, mpo, ComplexF64(-0.05im);
                                                       maxdim = 8, trunc_thresh = 1e-10,
                                                       maxiter = 5, hermitian = herm)
                    p = copy(psi)
                    RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, ComplexF64(-0.05im);
                                                    parallel = true, tol_basis = 1e-4,
                                                    tol_trunc = 1e-6, tol_expv = 1e-6,
                                                    maxdim = 8, hermitian = herm)
                end
            end
        catch err
            # ⛔ CARRY THE BACKTRACE. `exception = err` alone prints the message and nothing
            # about where it came from, and this failure has now survived three wrong
            # root-cause guesses (SU(2), the Julia version, the thread count) precisely
            # because the report said WHAT and never WHERE. `(err, catch_backtrace())` is the
            # form `@warn` unpacks into a stack trace; it costs nothing on the success path.
            @warn "BUGJulia precompile workload: the U(1)/:none block did not complete; those " *
                  "runs will pay full JIT (~313 s measured for a first step)" *
                  " -- symmetry at failure: $(BondUpdateBUG.symmetry_mode())" exception =
                  (err, catch_backtrace())
        end

        # ── SU(2) BLOCK ──
        # Its own try/catch/finally, not the U(1) block's try: the restore to `:U1` must run on
        # every path, because the `_SYMMETRY` Ref is serialised into the .ji and a skipped restore
        # would make the package load in :SU2. The separate messages say which coverage was lost,
        # since the U(1) coverage runs first and may already have succeeded.
        try
            # SU(2) is a separate compilation: the reduced-multiplet code is a different
            # specialisation of Telum's kernels, so compiling `:U1` teaches the cache nothing about
            # it. Its start state is `dimer_state`, since `neel_state` is not an SU(2) singlet.
            BondUpdateBUG.set_symmetry!(:SU2)
            let psi = BondUpdateBUG.dimer_state(6),
                mpo = RSVDCBEBondUpdate.heisenberg_su2_mpo(6)
                for herm in (true, false)
                    p = copy(psi)
                    RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, ComplexF64(-0.05im);
                                                   tol_basis = 1e-4, tol_trunc = 1e-6,
                                                   tol_expv = 1e-6, maxdim = 8,
                                                   hermitian = herm)
                end
                RSVDCBEBondUpdate.tdvp2_step!(copy(psi), mpo, ComplexF64(-0.05im);
                                              maxdim = 8, trunc_thresh = 1e-10, maxiter = 5)
                RSVDCBEBondUpdate.tdvp_cbe1s_step!(copy(psi), mpo, ComplexF64(-0.05im);
                                                   maxdim = 8, trunc_thresh = 1e-10, maxiter = 5)
                # The ground-state entry point: a different reduction (`LowestEigen`) through the
                # same expanding-Krylov core, and the one this campaign starts every run with.
                for ex in (true, false)
                    RSVDCBEBondUpdate.GroundState.solve!(copy(psi), mpo;
                        opts = RSVDCBEBondUpdate.CBEBugOptions(; maxdim = 8, maxiter = 5,
                                                               exact = ex),
                        n_sweeps = 2)
                end
            end
        catch err
            # A broken workload must not break the package; the only cost is a JIT in each process.
            # It must not be silent either, so that a renamed function or changed keyword that drops
            # coverage shows up in the next `Pkg.precompile`.
            @warn "BUGJulia precompile workload: the SU(2) block did not complete; SU(2) runs " *
                  "will pay full JIT (~4300 s measured for one L=12 step). U(1)/:none coverage " *
                  "is unaffected — that block runs first and completes." exception = err
        finally
            # ⛔ `finally`, NOT a trailing statement inside the `try`. As the last line of the try
            # this never ran on the throwing path, and precompilation baked :SU2 into the image.
            BondUpdateBUG.set_symmetry!(:U1)     # leave the global where it started
        end

        # ── FUSED d = 4 BLOCK ──
        # A fused Hubbard site is another type. One step from a product state never truncates, so
        # maxdim = 4 over three steps: the cap must bind, and a step must follow the first
        # truncation.
        try
            BondUpdateBUG.set_symmetry!(:none)
            let occ = RSVDCBEBondUpdate.hub4_cdw_occupations(6; doublon_frac = 0.5, seed = 1),
                psi = RSVDCBEBondUpdate.hub4_product_state(occ),
                mpo = RSVDCBEBondUpdate.hubbard_aa_mpo_d4(6; J = 1.0, Delta = 8.0, U = 4.0,
                                                          beta = 532 / 738, phi = 0.0),
                tau = ComplexF64(-0.05im)
                # The four schemes a d = 4 campaign steps with -- a different keyword set is a
                # different specialisation.
                for mk in (
                    (p, t) -> RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, t; maxdim = 4,
                                  tol_basis = 1e-4, tol_trunc = 1e-6, tol_expv = 1e-6),
                    (p, t) -> RSVDCBEBondUpdate.cbe_bug_midpoint_step!(p, mpo, t; maxdim = 4,
                                  tol_basis = 1e-4, tol_trunc = 1e-6, tol_expv = 1e-6),
                    (p, t) -> RSVDCBEBondUpdate.tdvp2_step!(p, mpo, t; maxdim = 4,
                                  trunc_thresh = 1e-10, maxiter = 5, conv_tol = 0.0),
                    (p, t) -> RSVDCBEBondUpdate.tdvp_cbe1s_step!(p, mpo, t; maxdim = 4,
                                  trunc_thresh = 1e-10, maxiter = 5, conv_tol = 0.0),
                )
                    p = copy(psi)
                    for _ in 1:3          # 3, NOT 1 -- see the truncation note above
                        mk(p, tau)
                    end
                    # The sampling path is timed inside the same loop, so it belongs in the cache
                    # too: an uncached observable sweep would land on whichever arm ran first.
                    RSVDCBEBondUpdate.hub4_density(p, 3)
                    RSVDCBEBondUpdate.mpo_energy(copy(p), mpo)
                end
            end
        catch err
            @warn "BUGJulia precompile workload: the fused d=4 Hubbard block did not complete; " *
                  "the spinful campaign will pay full JIT (~334 s MEASURED for the first " *
                  "truncating step at chi=32)" exception = (err, catch_backtrace())
        finally
            BondUpdateBUG.set_symmetry!(:U1)
        end
        end   # full block
    end
end

end # module
