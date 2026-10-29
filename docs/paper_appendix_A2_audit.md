# Connor, Duncan & Daley, arXiv:2507.01149v1 -- Appendix A.2 / B source audit

Read from https://arxiv.org/html/2507.01149v1 on 2026-10-30 (QTT plan, "Source-vs-implementation
acceptance test").

## (i) What Appendix A.2 says

- TDVP "is a local time-evolution method which evolves an MPS under the action of some Hamiltonian";
  implemented with **ITensorTDVP** on ITensors.
- The Hamiltonian is built as an MPO; the diagonal part `H_d = V(r) - mu + g|Psi|^2` is "a diagonal
  MPO in real space".
- The Laplacian can be built by finite differences or through the QFT; "TDVP performed using the QFT
  Laplacian produces smaller errors", so the QFT Laplacian is used for the TDVP runs.
- Figure 26: vortex shedding, `256 x 256`, `gamma = 0.1`, `v = 1.5`, `dt = 0.001`, errors against
  XMDS2 for finite-difference RK4, second-order split-step and TDVP at cutoffs `1e-12` and `1e-15`.
  At `1e-12` TDVP and split-step give "very similar errors"; at `1e-15` split-step is smaller; TDVP
  beats finite-difference RK4.
- Main text: sequential (all `x` bits, then all `y` bits) ordering gives a much smaller bond
  dimension than alternating, and is used throughout.

## (ii) What it does not specify

- one-site vs two-site TDVP;
- how often, or whether within a sweep, the state-dependent `g|Psi|^2` MPO is refreshed;
- any nonlinear stage consistency or the temporal order of the nonlinear treatment;
- the local exponential solver and its tolerances.

Appendix B: the paddle is the product of a top-hat MPS and an extended rounded function ("such as
sin(x)^2"); **no amplitudes, widths or motion law are given**, and the boundary condition at the
strong potential is not enforced.

## (iii) Our chosen variants (all labelled, none attributed to the paper)

| ours | file / function | relation to A.2 |
|---|---|---|
| `TDVP2_lagged` | `qtt_tdvp2_step!(...; density = :lagged)`, or `tdvp2_step!` on `gp_hamiltonian(m, psi_n, t_n)` | two-site, `H[psi_n]` held through the step |
| `TDVP2_consistent` | `qtt_tdvp2_step!(...; density = :consistent)` | every substep re-evaluates `H` from its midpoint trial state |
| split-dissipation control | `qtt_split_tdvp_step!` | Strang: `exp(-gamma h/2 H)` (global TT Krylov) around conservative TDVP2 |
| `BUG_nonlinear` | `qtt_bug_step!(psi, GPFlow(m), ...)` | Radau/trapezoid stages, density rebuilt at every iterate |
| frozen-density ablation | `gp_bug_step!` (`:frozen_density`) | Chen et al.'s per-step refresh, lagged |
| Laplacian | `qtt_derivative` (second-order FD) | **differs**: the paper's TDVP uses the QFT Laplacian -- comparisons must state this |
| paddle | `GPPaddle` (tanh-rounded separable box, linear sweep, linear ramp-down) | engineering fixture; Appendix B gives no parameters |

**Never** claim the paper demonstrated a projector-splitting instability: Figure 26 reports stable,
competitive TDVP errors on a small-step benchmark.
