# Audit of the CBE-BUG Integrator

This document explains the CBE-BUG time integrator which has been implemented., including the randomized controlled
bond expansion, the individual K and L sweep augmentations, and the root
zero-site S step. The goal is to hope to pass this to an AI to create tensor network diagrams because I am lazy.

The midpoint variant is also audited and compared with the ordinary CBE-BUG
update.

The method has two distinct roles:

1. The K and L sweeps construct enlarged left and right bases.
2. The root S step performs the actual time evolution inside the enlarged
   Galerkin space.

The K and L sweeps do not themselves integrate an ODE. They determine which
additional bond directions are available to the root solve.

## 1. Starting point

Let the current MPS be

\[
Y_0 = \psi,
\]

with a chosen root bond `c`. The algorithm constructs left and right frames
around this root:

\[
W_1,\ldots,W_c,
\qquad
Z_c,\ldots,Z_{L-1}.
\]

The root tensor is represented by a small matrix `S` connecting the left and
right frames. The final update has the form

\[
Y_1 = W_1\cdots W_c\,S_1\,Z_c\cdots Z_{L-1}.
\]

The difference between ordinary BUG and midpoint BUG is the state from which
the frames `W` and `Z` are constructed.

## 2. Randomized controlled bond expansion

The controlled bond expansion is implemented through `cbe_expand`. Its purpose
is to enlarge a local bond basis from rank `r` to approximately

\[
r_{\mathrm{new}} = r + D_{\mathrm{ex}},
\]

where `D_ex` is controlled by parameters such as `dex`, `growth`, `dover`,
and the truncation thresholds.

### 2.1 Local two-site object

At a bond, the method forms a local two-site object, schematically

\[
\Theta = A_i A_{i+1}.
\]

The current left and right bond bases are extracted from `Theta`. The
expansion identifies directions in the complement of the current frame that
are important under the Hamiltonian.

The relevant quantity is the Hamiltonian action

\[
H\Theta.
\]

The code applies the Hamiltonian using left and right environments and tensor
contractions rather than forming a large dense global vector. The local
`H*Theta` action is cached during a CBE call so the left and right sketches can
reuse the same expensive contraction.

### 2.2 Randomized sketch

A sector-aware random test matrix `Omega` is generated. In the
symmetry-preserving case, random vectors are constructed separately in the
allowed quantum-number sectors.

The sketch contains:

- components in the existing frame;
- components in its orthogonal complement;
- sector-dependent complement dimensions;
- optional oversampling controlled by `comp_ratio`.

The random sketch probes the range of the Hamiltonian action without explicitly
constructing the entire range of `H*Theta`.

A simplified left-side sketch is

\[
Y_L = P_L^\perp H\Theta\,\Omega_R,
\]

and similarly on the right,

\[
Y_R = P_R^\perp H\Theta\,\Omega_L.
\]

Here `P_L^perp` and `P_R^perp` project away the current bond frames. This
prevents the expansion from rediscovering directions already contained in the
MPS.

### 2.3 Preselection

The sketched vectors are orthogonalized:

\[
Q_L = \operatorname{orth}(Y_L),
\qquad
Q_R = \operatorname{orth}(Y_R).
\]

These are candidate directions, not yet the final expansion basis.

The preselection stage is controlled by the preselection tolerance, associated
with `stol_pre`. Candidates with numerically negligible norm are removed.

#### Preselection audit checks

The preselection stage is correct only if:

- candidates are orthogonal to the existing frame;
- candidates respect the symmetry sectors;
- zero or numerically negligible candidates are removed;
- reachable empty sectors are sampled;
- the same local `H*Theta` is used consistently for left and right selection.

A failure here can appear as an expansion that adds no useful rank, produces
symmetry violations, or is excessively sensitive to the random seed.

### 2.4 Final selection

The preselected candidates are ranked using a smaller projected Hamiltonian,

\[
M = Q_L^\dagger H Q_R.
\]

The method computes

\[
M = U_M \Sigma_M V_M^\dagger,
\]

and retains the strongest singular directions. The selected vectors are formed
from the corresponding columns of `Q_L` and rows of `Q_R`.

The final selection is controlled by `stol_fnl`, `dex`, growth limits, and any
requested rank caps. The final directions are selected according to their
importance in the Hamiltonian action, not merely according to their norm in
the random sketch.

### 2.5 Direct-sum augmentation

The existing and selected bases are combined by direct sum:

\[
U_{\mathrm{ex}} = [\,U_{\mathrm{old}}\mid U_{\mathrm{sel}}\,].
\]

The old basis is retained exactly and the new directions are added
orthogonally. In exact arithmetic,

\[
U_{\mathrm{ex}}^\dagger U_{\mathrm{old}}
=
\begin{bmatrix} I \\ 0 \end{bmatrix}.
\]

This is why the implementation does not require a discarded-projector
correction at this stage: the augmented frame explicitly contains the old
frame.

### CBE audit conclusion

The randomized expansion is responsible for the approximation error associated
with the enlarged space. The root solve can only be accurate if the CBE frames
capture the important directions of `H*psi`.

The main failure modes are:

- insufficient `dex` or growth;
- overly aggressive `stol_pre` or `stol_fnl`;
- insufficient random oversampling;
- poor sector allocation;
- an inadequate complement ratio;
- loss of numerical orthogonality;
- CBE directions that are inconsistent with the later root environments.

## 3. K sweep: left-to-right augmentation

The left sweep is implemented in `half_left!`. It constructs the left frame
`W_i` bond by bond.

### 3.1 Canonicalization and environments

The working left state is canonicalized at the left boundary:

\[
\psi_L \leftarrow \operatorname{canonical}(\psi_L,1).
\]

Right environments are built before the sweep. They provide the Hamiltonian
action needed by the local CBE expansion.

The left sweep carries an intermediate tensor `C`. At site `i`, the local
object is schematically

\[
K_i =
\begin{cases}
\psi_L[i], & i=1,\\
C\,\psi_L[i], & i>1.
\end{cases}
\]

The carry transports the already-constructed left frame toward the next bond.

### 3.2 Local frame construction

A local frame is formed from `K_i` and the next site tensor:

\[
\text{frame}_i = \operatorname{frame}(K_i,\psi_L[i+1]).
\]

This identifies the current left and right subspaces at the bond before
augmentation.

### 3.3 CBE augmentation at bond `i`

The local frame is passed to `cbe_expand`, which returns enlarged CBE frames,
denoted schematically by

\[
U_{\mathrm{ex}}^{(i)},
\qquad
V_{\mathrm{ex}}^{(i)}.
\]

The expansion uses the local Hamiltonian action and the current environments.
It does not update the physical MPS.

### 3.4 Optional one-site Krylov enlargement

After CBE, the code can enlarge the basis further using a local one-site
Hamiltonian. It constructs a projected object such as

\[
K_{\mathrm{ex}} = K_i M,
\]

where `M` connects the local tensor to the CBE-expanded opposite frame.

It then generates and orthogonalizes a Krylov sequence,

\[
K_{\mathrm{ex}},\quad H_1K_{\mathrm{ex}},\quad
H_1^2K_{\mathrm{ex}},\quad \ldots.
\]

These blocks are combined with the CBE directions and split into the final
left-sweep frame `W_i`.

If `krylov_basis=0`, the left frame is based only on the CBE result.

The Krylov vectors are basis vectors only. They are not a partial time
evolution of the physical MPS.

### 3.5 Carry update

Once `W_i` has been constructed, the carry is updated using the original local
amplitude:

\[
C \leftarrow W_i^\dagger K_i.
\]

The next site receives the original state represented in the new frame. No
exponential or Euler update is applied in the K sweep.

At the root, the left environment is saved before the root frame is pushed
through. This environment is later used to construct the zero-site root
Hamiltonian.

### K-sweep audit

The K sweep passes if:

- every `W_i` contains the previous left frame;
- added CBE directions are orthogonal and symmetry-compatible;
- the carry is contracted using the current frame;
- the Hamiltonian action uses the correct environments;
- no physical state evolution occurs inside the sweep;
- the root environment is saved at the correct position.

## 4. L sweep: right-to-left augmentation

The right sweep is implemented in `half_right!`. It constructs the right frame
`Z_j` from right to left.

### 4.1 Canonicalization and environments

The working right state is canonicalized at the right boundary:

\[
\psi_R \leftarrow \operatorname{canonical}(\psi_R,L).
\]

Left environments are built before the sweep.

The right sweep carries a tensor `D`. At site `j`, the local object is
schematically

\[
B_j =
\begin{cases}
\psi_R[j+1], & j=L-1,\\
\psi_R[j+1]D, & j<L-1.
\end{cases}
\]

This is the right-to-left counterpart of the K-sweep carry.

### 4.2 Local CBE expansion

A local frame is formed from `psi_R[j]` and `B_j`. `cbe_expand` then generates
the right-side CBE directions. The randomized preselection and final singular-
value selection are the same as in the K sweep, with the tensor contractions
reversed.

### 4.3 Optional right-side Krylov enlargement

The code constructs a projected object

\[
B_{\mathrm{ex}} = M B_j
\]

and generates

\[
B_{\mathrm{ex}},\quad H_1B_{\mathrm{ex}},\quad
H_1^2B_{\mathrm{ex}},\quad \ldots.
\]

The vectors are orthogonalized, combined with the CBE directions, and split
into the final right frame `Z_j`.

These are basis vectors only. No right-to-left time evolution occurs.

### 4.4 Carry update

The right carry is updated as

\[
D \leftarrow B_j Z_j^\dagger.
\]

The root right environment is saved before the root frame is pushed through.
This ensures that the later zero-site Hamiltonian is constructed at the
correct root bond.

### L-sweep audit

The L sweep passes if:

- every `Z_j` retains the old right subspace;
- right CBE directions use the correct left and right environments;
- the right carry is propagated with `Z_j^dagger`;
- the root right environment is saved before root-frame contraction;
- no physical state evolution occurs inside the sweep.

The K and L sweeps can be run in parallel because they construct independent
frame information. Their results must still be combined consistently in the
root step.

## 5. Root projection before the S step

After the sweeps, the original state is projected into the enlarged left-right
basis. The left and right projections are schematically

\[
A_L = W_c^\dagger\cdots W_1^\dagger Y_0,
\]

\[
A_R = Y_0 Z_{L-1}^\dagger\cdots Z_c^\dagger.
\]

The root tensor is then

\[
S_0 = A_L A_R
      = \langle W,Z\mid Y_0\rangle.
\]

This is the root coefficient matrix in the enlarged Galerkin basis.

The distinction between the state used to create the frames and the state
being projected is essential for midpoint BUG.

## 6. Zero-site root Hamiltonian

The K and L environments are pushed through the enlarged frames. This produces
a Hamiltonian acting only on the root tensor:

\[
H_0 = P_{\mathrm{aug}} H P_{\mathrm{aug}}.
\]

The implementation represents this operator without constructing a large
rank-four two-site tensor. Its action has the structure

\[
H_0(S)
= H_L S + S H_R^T
  + \sum_t C_t^{(L)} S C_t^{(R)T}.
\]

The first two terms represent left and right boundary contributions. The
channel terms represent MPO interactions crossing the root.

The root problem is therefore a matrix-valued projected evolution problem:

\[
\frac{dS}{dt} = H_0S.
\]

For a complex or imaginary-time step, the code applies the exponential over
the full requested step:

\[
S_1 = \exp(\tau H_0)S_0.
\]

The `expv` Krylov iteration is the actual time integration in the BUG method.

## 7. Ordinary CBE-BUG update without midpoint

The ordinary method is

\[
Y_0
\longrightarrow
\text{K/L frames from }Y_0
\longrightarrow
S_0
\longrightarrow
\exp(\tau H_0(Y_0))S_0
\longrightarrow
Y_1.
\]

The steps are:

1. Build left and right working states from the current MPS.
2. Construct the K-sweep frames `W`.
3. Construct the L-sweep frames `Z`.
4. Project the original state into the root basis.
5. Build the root Hamiltonian `H_0`.
6. Evolve the root tensor for the full step `tau`.
7. Split the updated root tensor by SVD.
8. Reassemble the MPS.
9. Apply the final recursive truncation if requested.

The root SVD is

\[
S_1 = U\Sigma V^\dagger.
\]

The updated MPS is assembled using the left factor `W_c U` and the right
factor `Sigma V^dagger Z_c`. By default, the root SVD is lossless because
`root_cutoff=0` and `root_maxdim=0`. Truncation is normally deferred to the
final recursive truncation step.

### Ordinary-method audit

The main approximation sources are:

- CBE range approximation;
- optional Krylov basis truncation;
- root `expv` tolerance and iteration limits;
- final MPS truncation;
- rank limits from `maxdim`.

The K/L sweep itself is not the time integration. Its role is to approximate
the Galerkin subspace in which the root integration is performed.

## 8. Audit of truncation after the S step

There are two related operations after the root exponential:

1. The root-core SVD splits the updated root tensor `S1` into two centre
   tensors.
2. The optional closing `truncate_recursive!` pass reduces ranks throughout
   the assembled MPS.

They should be distinguished. The root-core SVD is always needed to reconnect
the left and right parts of the MPS. It is not necessarily a truncation. The
closing recursive pass is the operation that normally enforces the final
`maxdim` and `trunc_thresh` limits.

### 8.1 Root-core split

After the root evolution,

\[
S_1 = \exp(\tau H_0)S_0,
\]

the code computes

```julia
res = svd(S1, (1,); cutoff=root_cutoff,
          Nkeep=root_maxdim > 0 ? root_maxdim : nothing,
          get_lists=true)
```

The resulting factors are distributed as

\[
S_1 = U\Sigma V^\dagger.
\]

The left centre tensor becomes

\[
\psi[c] = W_c U,
\]

and the right centre tensor becomes

\[
\psi[c+1] = \Sigma V^\dagger Z_c.
\]

The new MPS centre is placed at `c+1`.

This factorization is required because the root solve produces one matrix
connecting the enlarged left and right frames, while the MPS stores separate
site tensors. It is therefore a structural split even when no singular values
are discarded.

### 8.2 Default root truncation behavior

The default parameters are

```julia
root_cutoff = 0.0
root_maxdim = 0
```

With these defaults, the root SVD keeps all nonzero singular values, subject
only to exact zeros and numerical behavior of the SVD implementation. The
root-core split is therefore lossless by default.

This design avoids a third unswitchable truncation between the half-sweep
splits and the closing recursive truncation. The normal rank controls are:

- `split_cutoff` and optional `split_maxdim` during the K and L frame splits;
- `trunc_thresh` and `maxdim` during `truncate_recursive!`.

`root_cutoff` and `root_maxdim` remain available for controlled experiments or
for deliberately applying a rank restriction at the root.

### 8.3 Root-core truncation when explicitly enabled

If `root_cutoff > 0`, singular values below the root cutoff criterion can be
discarded. If `root_maxdim > 0`, at most that many root singular directions are
kept.

The discarded root weight is measured using `_trunc_weight(res)`, which is

\[
w_{\mathrm{root}}
=
\left(
\frac{\sum_{k\in\mathrm{discarded}}\sigma_k^2}
     {\sum_{k\in\mathrm{kept}}\sigma_k^2
      +\sum_{k\in\mathrm{discarded}}\sigma_k^2}
\right)^{1/2}.
\]

This is a relative discarded 2-norm weight, including symmetry-sector
degeneracies. It measures the error introduced by the root split itself, not
the error of the preceding CBE expansion or the root `expv` tolerance.

Root truncation can be useful when the root bond becomes the dominant rank
bottleneck. However, it can also remove directions that the K/L expansion
worked to construct. Applying it too aggressively can therefore make the
expanded Galerkin space ineffective immediately after it has been used.

### 8.4 Closing recursive truncation

After the two centre tensors have been assembled, the code applies the closing
pass when `truncate=true`:

```julia
truncate_recursive!(psi, c + 1;
                    maxdim=maxdim,
                    cutoff=cut)
```

The effective closing cutoff is

```julia
cut = max(trunc_thresh, 1e-14)
```

The recursive pass starts at the root and moves toward the leaves.

On the left subtree, it visits bonds

\[
c,\ c-1,\ \ldots,\ 1,
\]

using an SVD of each tensor. The factorization is arranged so that the
retained right factor stays on the child tensor while the retained `U*S`
factor is pushed outward into its parent.

The state is then canonicalized at the root before processing the right
subtree. The right pass visits bonds

\[
c+1,\ c+2,\ \ldots,\ L-1.
\]

On the right, the retained `U` factor stays on the current tensor and the
retained `S*Vd` factor is pushed into the next tensor.

This ordering is important. It prevents the same bond from being cut twice and
avoids compounding truncation errors from repeatedly refactorizing an already
truncated bond.

The closing pass returns the largest relative discarded weight over all bonds,

\[
w_{\mathrm{close}}
=
\max_i w_i.
\]

The step-level `discarded` diagnostic is updated as

\[
\texttt{discarded}
=
\max(w_{\mathrm{root}}, w_{\mathrm{close}}).
\]

Thus the reported value is a worst single-SVD relative discarded weight. It is
not the sum of errors over all bonds and should not be interpreted as a full
global norm bound without additional analysis.

### 8.5 Interaction with `truncate=false`

When `truncate=false`, the closing `truncate_recursive!` pass is skipped.
However, the root-core SVD still occurs because it is required to assemble the
MPS.

With the default root parameters, this means:

- the root split is lossless up to exact zeros;
- the closing rank control is disabled;
- the K/L `split_cutoff` remains active;
- `maxdim` is not enforced by the closing pass.

This behavior is intentional in the midpoint first stage. The half-step state
must retain the rank and directions generated for midpoint frame construction.

### Root-truncation audit checks

The post-S-step truncation passes if:

- `S1` is split exactly once into the two centre tensors;
- the root bond indices of `W[c]`, `U`, `Vd`, and `Z[c]` are compatible;
- root truncation is disabled by default unless explicitly requested;
- `root_cutoff` and `root_maxdim` are not confused with `trunc_thresh` and
  `maxdim`;
- the discarded weight includes symmetry-sector degeneracies;
- the closing pass visits every left and right bond once in root-to-leaf order;
- the centre is recanonicalized between the left and right passes;
- `truncate=false` leaves the midpoint half-step uncompressed after assembly;
- the returned diagnostic reflects the maximum discarded weight from the root
  split and closing pass.

The principal risk is over-truncation after the root solve. The root
exponential may produce a valid state in the enlarged Galerkin space, but a
small `root_cutoff`, small `root_maxdim`, aggressive `trunc_thresh`, or small
`maxdim` can immediately remove the directions that made the CBE-BUG step
accurate.

## 9. Midpoint BUG update

The midpoint wrapper has two stages.

### 9.1 Stage 1: construct the midpoint state

First, a copy of the current state is advanced by half a time step:

\[
Y_{1/2} \approx \Phi_{\tau/2}(Y_0).
\]

In code, this is an ordinary CBE-BUG step with `tau/2` and truncation
disabled:

```julia
half = copy(psi)
cbe_bug_step!(half, mpo, tau / 2; truncate=false, ...)
```

The half-step state is intended to retain the additional directions generated
by the half-step BUG expansion. Truncating it immediately could remove the
directions needed to represent the midpoint basis.

The first stage therefore produces a rank-enlarged approximation to the
midpoint state rather than only a compressed rank-`r` state.

### 9.2 Stage 2: build frames from the midpoint

The original state `psi` is then updated using a full step `tau`, but the K
and L frames are built from `half`:

\[
W,Z \leftarrow \text{frames from }Y_{1/2}.
\]

The `frame_state` argument separates:

- the state used to construct the basis;
- the state being evolved in the root Galerkin equation.

The second-stage basis approximates

\[
P_{\mathrm{aug}}(Y_{1/2}),
\]

while the root initial condition remains the original state:

\[
S_0 = \langle W(Y_{1/2}),Z(Y_{1/2})\mid Y_0\rangle.
\]

The root equation is integrated over the full interval:

\[
S_1 = \exp\left(\tau H_0(Y_{1/2})\right)S_0.
\]

This is the essential midpoint construction.

### 9.3 Why the original projection must be rebuilt

During the midpoint sweeps, the carries naturally represent the midpoint
state. If those carries were used directly to form `S_0`, the second stage
would effectively start from `Y_{1/2}` and integrate another full step.

That would not be midpoint BUG. It would double-count the half-step evolution
and change the intended order and method.

Therefore, when `frame_state` is present, the implementation rebuilds the
right projection against the original `psi`. It also rebuilds the left
projection from the original state.

The correct root initial condition is

\[
S_0 = \langle W(Y_{1/2}),Z(Y_{1/2})\mid Y_0\rangle,
\]

not

\[
\langle W(Y_{1/2}),Z(Y_{1/2})\mid Y_{1/2}\rangle.
\]

This is one of the most important correctness conditions in the implementation.

### 9.4 Final midpoint assembly

After the full root evolution,

\[
S_1 = \exp\left(\tau H_0(Y_{1/2})\right)S_0,
\]

the updated root tensor is split and the MPS is assembled exactly as in the
ordinary method. Final truncation is applied only after the full second-stage
update.

## 10. Direct comparison

| Component | Ordinary BUG | Midpoint BUG |
|---|---|---|
| Frame-generating state | `Y_0` | `Y_{1/2}` |
| Initial root state | `Y_0` | `Y_0` |
| Frame construction | One K/L sweep | Half-step BUG, then K/L sweep |
| Root step length | `tau` | `tau` |
| Root Hamiltonian | `H_0(Y_0)` | `H_0(Y_{1/2})` |
| Intermediate truncation | Not applicable | Disabled after `tau/2` |
| Final truncation | After root update | After root update |
| Main benefit | Lower cost | Better midpoint evaluation of the evolving subspace |
| Main extra cost | One root solve | Approximately two root solves |

The midpoint update does not replace the root exponential with a simple
midpoint quadrature formula. It uses the midpoint state to construct the
Galerkin space and projected Hamiltonian, then still performs a full root
`expv` solve.

It improves the time-centering of the basis and projected Hamiltonian. It does
not automatically imply an order higher than second order for the complete
method.

## 11. Overall audit conclusion

The method can be summarized as

\[
\boxed{
\text{CBE/RSVD basis selection}
\;\rightarrow\;
\text{K/L basis augmentation}
\;\rightarrow\;
\text{root Galerkin exponential}
\;\rightarrow\;
\text{SVD reconstruction and truncation}
}
\]

For ordinary BUG, every basis object is generated from the current state.

For midpoint BUG, the basis objects are generated from an untruncated
half-step state, while the root equation starts from and evolves the original
state over the full time step.

The central midpoint invariant is

\[
\boxed{
\text{midpoint frames from }Y_{1/2},
\qquad
\text{root initial tensor from }Y_0,
\qquad
\text{root evolution over }\tau
}
\]

Violating any one of these conditions changes the method being implemented.
