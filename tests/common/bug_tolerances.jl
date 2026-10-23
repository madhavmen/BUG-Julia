# tests/common/bug_tolerances.jl -- the tight tolerances a physics test runs the adaptive solvers
# at. `BUG_TIGHT` is `cbe_bug_step!`'s basis and root (the fixtures' own, `adaptive_fixtures.jl`);
# each call names its compression `tol_trunc`. `DMRG_TIGHT` is CBE-DMRG's basis and split, the
# settings at which `test_adaptive_dmrg.jl` drives the variance below 1e-8.

const BUG_TIGHT = (tol_basis = 1e-12, tol_expv = 1e-14)
const DMRG_TIGHT = (tol_basis = 1e-7, tol_trunc = 1e-12)
