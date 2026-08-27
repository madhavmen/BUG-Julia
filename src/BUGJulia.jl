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
            end
            BondUpdateBUG.set_symmetry!(:U1)     # leave the global where it started
        catch
            # A broken workload must not break the package; the cost is only that the first
            # call in each process pays its own JIT again.
        end
    end
end

end # module
