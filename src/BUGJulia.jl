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

@setup_workload begin
    @compile_workload begin
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
            BondUpdateBUG.set_symmetry!(:U1)     # leave the global where it started
        catch err
            # A broken workload must not break the package; the cost is only that the first
            # call in each process pays its own JIT again.
            #
            # ⛔ BUT IT MUST NOT BE SILENT EITHER. Swallowing this bare meant a workload that had
            # stopped covering anything -- a renamed function, a changed keyword -- looked exactly
            # like a working one, and the only symptom was runs mysteriously taking minutes again.
            # Precompilation prints this, so the next `Pkg.precompile` says which entry point fell
            # out of the cache.
            @warn "BUGJulia precompile workload did not complete; first call in each process " *
                  "will pay full JIT (SU(2) measured at ~4300 s)" exception = err
        end
    end
end

end # module
