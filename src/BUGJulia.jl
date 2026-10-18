module BUGJulia

# `BondUpdateBUG` is the single consolidated symmetry-native BUG integrator.
include("BondUpdateBUG/BondUpdateBUG.jl")

# `RSVDCBEBondUpdate` is the exploratory CBE-BUG variant (docs/cbe_bug.md): the
# randomized-SVD controlled bond expansion in place of the K/L discarded-projector
# augmentation. Separate submodule -- it imports from `BondUpdateBUG` and changes
# nothing there.
include("RSVDCBEBondUpdate/RSVDCBEBondUpdate.jl")

# ── PRECOMPILE WORKLOAD ───────────────────────────────────────────────────────────────────
#
# ⛔ WITHOUT THIS, EVERY `julia` INVOCATION PAID ~580 SECONDS OF JIT BEFORE IT COULD TIME
# ANYTHING. MEASURED (L = 10, chi = 32, XX): `setup` 265 s and the FIRST `cbe_bug_step!` 313 s,
# against 0.25 s for every step after it. Loading the package was never the problem -- that is
# 7 s from the cache -- the cost was specialising Telum's generic block-sparse code at RUN time,
# which a plain `using` does not cache because no method was ever called during precompilation.
#
# Running the real entry points here moves that work into the `.ji` cache, where it is paid ONCE
# per source change instead of once per process. A benchmark sweep that launches several Julia
# processes was spending the overwhelming majority of its wall clock compiling the same code over
# and over.
#
# ⚠ KEEP THE WORKLOAD SMALL. Everything in here is paid on every `Pkg.precompile`, i.e. after
# every edit to `src/`. `L = 6` at `maxdim = 8` is enough to specialise the whole chain -- the
# types do not depend on the sizes -- and a bigger system would only make editing slower.
#
# ⚠ AND IT IS GUARDED. A workload that throws fails the PRECOMPILATION, which would turn a
# performance optimisation into a package that does not load at all. A failure here must degrade
# to "slow again", never to "broken".
using PrecompileTools: @setup_workload, @compile_workload
using Preferences: @load_preference

# ⛔ WHICH WORKLOAD IS A PREFERENCE, NOT AN ENV VARIABLE: a preference is part of the cache key,
# so changing it recompiles; an ENV switch would silently keep whichever image was built first.
#
#   "adaptive" (default)  the U(1) spin paths the current campaign runs -- adaptive CBE-BUG at
#                         orders 1/2/4 and midpoint, adaptive + legacy CBE-DMRG, TDVP2, CBE-TDVP,
#                         energy / variance / S^z. MEASURED 2026-10-16: the old "full" workload
#                         cost 1802 s of precompilation on EVERY src edit (job 17208254) while
#                         covering none of the adaptive entry points, so each process still paid
#                         their JIT on top.
#   "full"                adds the original blocks: `:none`, non-Hermitian branches, SU(2), fused
#                         d = 4 Hubbard -- set it for those campaigns
#                         (`Preferences.set_preferences!(BUGJulia, "workload" => "full")`).
#   "none"                no workload (fastest edit cycle; every process pays its own JIT).
#
# THE TRADE, MEASURED on one node (jobs 17208934, 17230469; x86-64-v3, 2 threads):
#                     precompile   `using`   first adaptive BUG step   first process total
#   "adaptive"          565 s       96 s            2 s                      107 s
#   "none"               32 s        3 s          218 s                      385 s
# The `using` cost IS the cached specialisations (dependencies load in ~2 s), so it cannot be
# had without them. Break-even is ~2 processes per edit: keep "adaptive" for jobs, and run many
# test files in ONE process; "none" only for a single quick process.
#
# ⛔ A WORKLOAD FOR THE IMAGINARY-TIME / OPEN-SYSTEM PATHS DOES NOT HELP. MEASURED 2026-10-18 on the
# same stages in ONE process (`run_stages.jl`, Intel E5-2680 v4): an "imagopen" workload
# (`ite_step!`, MPDO Lindblad step, tdvp2 at -dbeta) cost 627 s of precompilation plus 126 s of
# `using` (s43comp 17343333), while "none" cost 23 s plus 2.6 s and ran all four test files and
# the IT-1/OS-1/2/3 smokes in 25.6 min total (s43none 17343409). Any workload over
# `cbe_bug_step!`'s call graph is ~10 min, so for one-process jobs stage with "none".
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
                RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, tau; dex = 4, dover = 0,
                                                comp_ratio = 1.0, krylov_basis = 3,
                                                krylov_tol = 0.0, maxdim = 4,
                                                trunc_thresh = 1e-10, maxiter = 5, tol = 0.0)
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
                RSVDCBEBondUpdate.dmrg_cbe1s!(p, mpo; n_sweeps = 2, maxdim = 4,
                                              trunc_thresh = 1e-10)
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
                # both selection paths (exact CBE / randomised sketch) and both sweep
                # executions (sequential / concurrent half-sweeps)
                for (exact, parallel) in ((true, false), (false, false), (true, true))
                    p = copy(psi)
                    RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, ComplexF64(-0.05im);
                                                   exact = exact, parallel = parallel,
                                                   dex = 4, dover = 0, comp_ratio = 1.0,
                                                   krylov_basis = 3, krylov_tol = 0.0,
                                                   maxdim = 8, trunc_thresh = 1e-10,
                                                   maxiter = 5, tol = 0.0)
                    BondUpdateBUG.sz_expectation(p, 3)
                    RSVDCBEBondUpdate.mpo_energy(copy(p), mpo)
                end
                # ⛔ THE TWO BASELINE SWEEPS BELONG IN THE WORKLOAD TOO, AND LEAVING THEM OUT
                # CORRUPTED A MEASUREMENT RATHER THAN MERELY COSTING TIME. Any driver that ran
                # `tdvp2` or `tdvp_cbe1s` first had that arm absorb its own specialisation:
                # measured on an L=8 campaign smoke run, `tdvp2` 61.6 s against `cbe_bug` 2.7 s,
                # where the true ratio is far smaller. A benchmark driver's own warmup loop fixes
                # its numbers, but only if whoever wrote it remembered -- precompiling the arms
                # removes the trap instead of documenting it.
                #
                # ⚠ BOTH `hermitian` BRANCHES ARE COMPILED. `hermitian = false` selects Arnoldi and
                # a full Hessenberg projection -- a genuinely different code path, and the one every
                # open-system run takes. Compiling only the Hermitian branch would leave the fused
                # `d = 4` drivers paying full JIT while looking covered.
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
                                                    exact = false, parallel = true,
                                                    dex = 4, dover = 4, comp_ratio = 1.0,
                                                    krylov_basis = 3, krylov_tol = 0.0,
                                                    maxdim = 8, trunc_thresh = 1e-10,
                                                    maxiter = 5, tol = 0.0, hermitian = herm)
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

        # ⛔ THE SU(2) BLOCK GETS ITS OWN try/catch/finally, AND THAT SEPARATION IS LOAD-BEARING.
        # It used to share one `try` with the block above, with the `set_symmetry!(:U1)` restore as
        # the LAST STATEMENT INSIDE it. MEASURED 2026-09-05 on the cluster: the SU(2) block throws
        # (`ArgumentError: reducing over an empty collection`, raised inside LurCGT's SU(2) irrep
        # generation, not in our code), and that had two consequences the single `try` hid:
        #
        #   1. THE RESTORE WAS SKIPPED, so precompilation ended with `_SYMMETRY[] == :SU2`. That Ref
        #      is module state and is serialised into the `.ji`, so the package would LOAD in :SU2 —
        #      a silent global-mode flip for any caller that does not set the symmetry itself.
        #      `finally` now restores it on every path, thrown or not.
        #   2. ONE WARNING FOR TWO VERY DIFFERENT FAILURES. "workload did not complete" read as
        #      "nothing is cached", when in fact the U(1)/:none coverage above had already
        #      succeeded — it runs first. The two messages now say which coverage was actually lost.
        try
            # ⛔ SU(2) IS A SEPARATE BLOCK BECAUSE IT IS A SEPARATE COMPILATION, and leaving it out
            # meant the whole non-abelian campaign paid full JIT while the workload looked complete.
            # MEASURED before this block existed: one `cbe_bug_step!` at L=12 SU(2) took 4314 s in a
            # freshly precompiled process, against sub-second for the U(1) path the workload did
            # cover. The fused/reduced-multiplet code is a different specialisation of Telum's
            # block-sparse kernels, so compiling `:U1` teaches the cache nothing about it.
            #
            # ⚠ THE START STATE DIFFERS TOO: `neel_state` is not an SU(2) singlet, so the S = 0
            # sector this campaign works in is reached with `dimer_state`.
            BondUpdateBUG.set_symmetry!(:SU2)
            let psi = BondUpdateBUG.dimer_state(6),
                mpo = RSVDCBEBondUpdate.heisenberg_su2_mpo(6)
                for herm in (true, false)
                    p = copy(psi)
                    RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, ComplexF64(-0.05im);
                                                   exact = false, dex = 4, dover = 4,
                                                   krylov_basis = 3, krylov_tol = 0.0,
                                                   maxdim = 8, trunc_thresh = 1e-10,
                                                   maxiter = 5, tol = 0.0, hermitian = herm)
                end
                # The growth path is its own specialisation -- `_grow_frame` rebuilds `H1` and
                # re-expands between Lanczos vectors, which the fixed-width builder never calls.
                let p = copy(psi)
                    RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, ComplexF64(-0.05im);
                                                   exact = false, dex = 4, dover = 4,
                                                   grow_iters = 2, krylov_tol = 0.0,
                                                   maxdim = 8, trunc_thresh = 1e-10,
                                                   maxiter = 5, tol = 0.0)
                end
                RSVDCBEBondUpdate.tdvp2_step!(copy(psi), mpo, ComplexF64(-0.05im);
                                              maxdim = 8, trunc_thresh = 1e-10, maxiter = 5)
                RSVDCBEBondUpdate.tdvp_cbe1s_step!(copy(psi), mpo, ComplexF64(-0.05im);
                                                   maxdim = 8, trunc_thresh = 1e-10, maxiter = 5)
                # The ground-state entry point: a different reduction (`LowestEigen`) through the
                # same expanding-Krylov core, and the one this campaign starts every run with.
                for ex in (true, false)
                    RSVDCBEBondUpdate.GroundState.solve!(copy(psi), mpo;
                        opts = RSVDCBEBondUpdate.CBEBugOptions(; maxdim = 8, dex = 4,
                                                               maxiter = 5, exact = ex),
                        n_sweeps = 2, grow_iters = 1)
                end
            end
        catch err
            # A broken workload must not break the package; the cost is only that the first
            # call in each process pays its own JIT again.
            #
            # ⛔ BUT IT MUST NOT BE SILENT EITHER. Swallowing this bare meant a workload that had
            # stopped covering anything -- a renamed function, a changed keyword -- looked exactly
            # like a working one, and the only symptom was runs mysteriously taking minutes again.
            # Precompilation prints this, so the next `Pkg.precompile` says which entry point fell
            # out of the cache.
            @warn "BUGJulia precompile workload: the SU(2) block did not complete; SU(2) runs " *
                  "will pay full JIT (~4300 s measured for one L=12 step). U(1)/:none coverage " *
                  "is unaffected — that block runs first and completes." exception = err
        finally
            # ⛔ `finally`, NOT a trailing statement inside the `try`. As the last line of the try
            # this never ran on the throwing path, and precompilation baked :SU2 into the image.
            BondUpdateBUG.set_symmetry!(:U1)     # leave the global where it started
        end

        # ⛔⛔ THE FUSED d=4 BLOCK, AND IT IS HERE BECAUSE THE MISSING COVERAGE WAS MEASURED AT
        # 334 SECONDS IN A SINGLE STEP. Job 16341790, `phase_hub4prod`, L=20, **chi = 32**:
        #
        #     step 1   0.8 s      (product state -> bd 32)
        #     step 2   334.3 s    <- at bond dimension 32
        #
        # A bd=32 step is microseconds of arithmetic, so 334 s is compilation, and it matches the
        # ~313 s this file already documents for an uncached first step. Two separate holes put it
        # there and BOTH are closed below:
        #
        #   1. d=4 WAS NEVER IN THE WORKLOAD. Every entry point above runs the SPIN chain, whose
        #      local space is a different type, and this file's own SU(2) note says it plainly --
        #      "compiling `:U1` teaches the cache nothing about it". The fused Hubbard site is
        #      another such type, so the whole spinful campaign ran uncached.
        #
        #   2. ⛔ ONE STEP FROM A PRODUCT STATE DOES NOT COMPILE THE TRUNCATION PATH, which is the
        #      subtler half and it also invalidates the `warm!` docstring in `benchmarks/mbl_scars.jl`
        #      ("one step is enough -- specialisation is per method, not per step"). Starting at
        #      bd = 1, the augmented rank never reaches `maxdim`, so the truncate-to-cap branch is
        #      never entered and its kernels are never specialised. Step 2 is the FIRST step that
        #      has to truncate, and it pays for all of them at once. That is exactly why the cost
        #      appeared at step 2 rather than step 1, and why it was identical at bd=32 and bd=43 --
        #      a compile does not care how big the tensors are.
        #
        # ⇒ `maxdim = 4` WITH THREE STEPS IS THE LOAD-BEARING CHOICE, not an arbitrary size. The cap
        # must BIND (a d=4 site reaches bd 4 after one step, so step 2 truncates) and there must be
        # a step AFTER the first truncating one, or the branch stays cold. `L = 6` keeps the tensors
        # tiny; the runtime here is negligible and the compile time is the entire point.
        #
        # ⚠ DOUBLONS ARE DELIBERATE (`doublon_frac = 0.5`). They are the physically relevant start
        # state for arXiv:1501.05661 Figs. 4-5 and the cells where the integrators differ most, so
        # the workload should exercise the same branches production does.
        try
            BondUpdateBUG.set_symmetry!(:none)
            let occ = RSVDCBEBondUpdate.hub4_cdw_occupations(6; doublon_frac = 0.5, seed = 1),
                psi = RSVDCBEBondUpdate.hub4_product_state(occ),
                mpo = RSVDCBEBondUpdate.hubbard_aa_mpo_d4(6; J = 1.0, Delta = 8.0, U = 4.0,
                                                          beta = 532 / 738, phi = 0.0),
                tau = ComplexF64(-0.05im)
                # The four schemes `benchmarks/mbl_scars.jl::stepper` can return, at the keywords it
                # actually passes -- a different keyword set is a different specialisation.
                for mk in (
                    (p, t) -> RSVDCBEBondUpdate.cbe_bug_step!(p, mpo, t; maxdim = 4,
                                  trunc_thresh = 1e-10, split_cutoff = 1e-10, maxiter = 5,
                                  krylov_basis = 5, root_conv_tol = 0.0, root_substeps = 1),
                    (p, t) -> RSVDCBEBondUpdate.cbe_bug_midpoint_step!(p, mpo, t; maxdim = 4,
                                  trunc_thresh = 1e-10, split_cutoff = 1e-10, maxiter = 5,
                                  krylov_basis = 5, root_conv_tol = 0.0, root_substeps = 1),
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
