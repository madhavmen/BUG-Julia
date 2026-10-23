# tests/common/bug_tolerances.jl -- the tight basis and root tolerances a physics test runs
# `cbe_bug_step!` at (the fixtures' own, `adaptive_fixtures.jl`); each call names its compression
# `tol_trunc`.

const BUG_TIGHT = (tol_basis = 1e-12, tol_expv = 1e-14)
