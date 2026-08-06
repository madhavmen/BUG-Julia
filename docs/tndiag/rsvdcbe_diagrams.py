#!/usr/bin/env python3
"""
The RSVD-CBE bond update and its four wrappers, drawn line by line.

Sources diagrammed (BUG-Julia/src/RSVDCBEBondUpdate/):
    cbe_core.jl      cbe_expand, cbe_bond_update, cbe_bond_update_sweep!,
                     cbe_gate_evolve!
    cbe_lubich.jl    cbe_lubich_sweep
    tdvp1_cbe/       tdvp1_cbe_step!
    modes/ground_state.jl   GroundState.solve!, _bond_solve, _half_sweep!

Written against TEMPLATE.md; the worked example is tdvp1_test.py.

Run:  python3 rsvdcbe_diagrams.py  ->  out/rsvdcbe.html
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tndiag import TN, Txt, Figure, Panel, Section, Page, unroll   # noqa: E402
import tnlib as T                                                 # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
L = 6


# ==================================================== part 1: the expansion ==

def sec_expand():
    s = Section("1 — the shared core: cbe_expand",
                "All four algorithms below call this one function. It widens a "
                "bond; it never evolves anything. Everything that differs between "
                "the four is what happens after.")

    # the frame
    f, _ = T.frame_uv()
    s.add(Panel(
        code="f = bond_frame(psi, i)        # Theta = U0 * S0 * V0",
        figure=Figure(T.theta_block(), Txt("="), f,
                      caption="The two-site block, factorised. U₀ and V₀ are the "
                              "FRAMES: orthonormal bases for the r-dimensional "
                              "subspaces the state currently uses."),
        note="Everything CBE does is enlarge those two subspaces; everything the "
             "local solve does is evolve S₀ inside them."))

    # what we are looking for
    t = TN(anchor_row=1, bra=(2,))
    l = T.env_L(t, "lch", 0)
    th = t.node("Θ", "bond", 1.5, 0, cols=2)
    w1 = t.op("W_i", 1, 1); w2 = t.op("W_{i+1}", 2, 1)
    r = T.env_R(t, "rch", 3)
    t.link(l, "E:-1", th, "W"); t.link(l, "E", w1, "W")
    t.link(w1, "E", w2, "W"); t.link(th, "N@-0.5", w1, "S")
    t.link(th, "N@+0.5", w2, "S")
    t.dangle(w1, "N", "σ'_i"); t.dangle(w2, "N", "σ'_{i+1}")
    t.link(th, "E", r, "W:-1"); t.link(w2, "E", r, "W")
    t.dangle(l, "E:+1", "ℓ'", free_dir="E", lab_side="N")
    t.dangle(r, "W:+1", "ℓ'", free_dir="W", lab_side="N")
    t.box(0, 3, 0, 2, label="HΘ  —  formed, then projected")
    s.add(Panel(
        code="# the directions that matter, to first order:\n"
             "#   P_perp (H Theta),   P_perp = I - U0 U0'",
        figure=Figure(t, caption="What we want the leading directions of."),
        note="<b>Project-first (the only path since 2026-08-06):</b> P_perp is applied "
             "to this rank-4 object and Ω is folded in afterwards. Because P_perp "
             "acts on (ℓ_{i-1}, σ_i) and Ω on (σ_{i+1}, ℓ_{i+1}) — disjoint legs — "
             "the two commute, so the result is identical to folding Ω in first. "
             "What differs is cost: HΘ is now formed once per bond, and H is applied "
             "to d·χ_r columns rather than Ω's D_pre."))

    # the sketch
    a = TN(anchor_row=1, bra=(2,))
    l = T.env_L(a, "lch", 0)
    u0 = a.liso("U_0", 1, 0)
    w1 = a.op("W_i", 1, 1)
    om = a.node("Ω", "scalar", 2, 0)
    a.link(l, "E:-1", u0, "W"); a.link(l, "E", w1, "W")
    a.link(u0, "N", w1, "S", label="σ_i")
    a.dangle(w1, "N", "σ'_i")
    a.link(u0, "E", om, "W")
    a.dangle(om, "E", "g")
    a.dangle(l, "E:+1", "ℓ'", free_dir="E", lab_side="N")
    a.dangle(w1, "E", "w_i", lab_side="N")
    s.add(Panel(
        code="OmR = sector_graded_sketch(V0, :right, npre)\n"
             "Y   = sketch_h_left(f, h, i, lch, rch, OmR)   # (P_perp H Theta) OmR'",
        figure=Figure(a, caption="The sketch probes the PROJECTOR: P_perp(HΘ) first, "
                                 "Ω after. Y has legs (ℓ_{i-1}, σ_i, g) with "
                                 "g ≈ 1.2·D_ex columns, and is orthogonal to U₀ by "
                                 "construction."),
        note="The random matrix is <b>sector-graded</b>, not Gaussian: under a "
             "symmetry the columns must be allocated per charge sector, split "
             "between the orthogonal complement and the isometry space in the ratio "
             "<code>comp_ratio</code>. A plain Gaussian silently draws zero weight "
             "in most sectors."))

    # projection and assembly
    a = TN()
    y = a.node("Y", "site", 0, hi=True)
    a.dangle(y, "W", "ℓ_{i-1}"); a.phys(y, "σ_i"); a.dangle(y, "E", "g")
    b = TN(anchor_row=0.5)
    q = b.liso("Q_L", 0, hi=True)
    b.dangle(q, "W", "ℓ_{i-1}"); b.phys(q, "σ_i"); b.dangle(q, "E", "D_pre")
    s.add(Panel(
        code="QL, err = _preselect_left(U0, Y, stol_pre)   # P_perp (idempotent here), then orthonormalise",
        figure=Figure(a, Txt("⟶"), b,
                      caption="Project the frame out, orthonormalise what is left. "
                              "Q_L arrives already weight-ordered."),
        note="Y arrives already projected, so this P_perp is now a no-op — P_perp is "
             "idempotent. The call is kept because it also orthonormalises and "
             "returns the residual the selection ranks on. Within it the order still "
             "matters: project <em>then</em> orthonormalise, never the reverse."))

    def half(name, rlab):
        t = TN()
        n = t.liso(name, 0)
        t.dangle(n, "W", "ℓ_{i-1}"); t.phys(n, "σ_i"); t.dangle(n, "E", rlab)
        return t
    b, _ = T.expanded_frame("L")
    s.add(Panel(
        code="U_ex = oplus([U0, TLC], (3,))     # V_ex mirrors it",
        figure=Figure(half("U_0", "r"), Txt("⊕"), half("C_L", "D_ex"),
                      Txt("="), b,
                      caption="A direct sum: U₀ survives as the first block of "
                              "U_ex, so the expansion is lossless."),
        note="<b>Expansion alone cannot change a Schmidt rank.</b> It is a change "
             "of basis; a <code>canonical!</code> would collapse the widened bond "
             "straight back. The rank grows only because the core is then EVOLVED "
             "in the widened space and comes out dense."))
    return s


def sec_projectors():
    s = Section("2 — the projectors, constructed",
                "Every method below is a choice of projector, so they are built "
                "here once, from the two halves of the local space. Light and "
                "empty is the kept space; shaded is the discarded space.")
    s.add(*T.projector_panels())
    return s


# =============================================== part 3: the Lubich BUG sweep =

def sec_lubich():
    """EVERY line of cbe_lubich_sweep (cbe_lubich.jl:314-487). Nothing elided."""
    c = 3          # centre bond for L = 6
    s = Section("3 — cbe_lubich_sweep, every line",
                "cbe_lubich.jl lines 314–487. Each panel pastes the source "
                "verbatim; the long comments in the file become the notes. Two "
                "basis sweeps that touch nothing, then exactly one time evolution.")

    # ---- 328-345: preamble --------------------------------------------------
    t, _ = T.mps(L, names=lambda k, sh: f"psi[{k}]")
    t.brace(c - 1, c, label="centre bond c")
    s.add(Panel(
        title="lines 328–338 — preamble",
        code="L = length(psi)\n"
             "length(h) == L || throw(DimensionMismatch(...))\n"
             "L >= 2 || throw(ArgumentError(...))\n"
             "c = max(1, min(L - 1, L ÷ 2))                     # the centre bond\n"
             "\n"
             "expanded = zeros(Int, L - 1)\n"
             "n_new    = zeros(Int, L - 1)\n"
             "err_pre  = 0.0\n"
             "err_fnl  = 0.0\n"
             "peak     = 0",
        figure=Figure(t, caption=f"L = {L}, so c = {c}. Everything after this "
                                 f"happens relative to that one bond."),
        note="Only <code>c</code> matters to the mathematics; the rest are "
             "diagnostics."))

    s.add(Panel(
        title="lines 342–356 — captured knobs and diagnostics",
        code="exkw = (dex = dex, growth = growth, dover = dover,\n"
             "        comp_ratio = comp_ratio, sulz_cap = sulz_cap,\n"
             "        rmax = maximum(bond_dims(psi); init = 0),\n"
             "        preselect_only = preselect_only, exact = exact, rng = rng)\n"
             "\n"
             "function record!(ex)\n"
             "    err_pre = max(err_pre, ex.err_pre)\n"
             "    err_fnl = max(err_fnl, ex.err_fnl)\n"
             "    return ex\n"
             "end\n"
             "\n"
             "peak!(transients...) = (peak = max(peak, _state_stored(psi) +\n"
             "                        sum(tensor_elements(t) for t in transients; init = 0)))",
        figure=None,
        note="<code>rmax</code> is read <em>once</em>, before either sweep widens "
             "anything, so the optional Sulz bound is a global statement about the "
             "state rather than a per-bond one. No tensors are touched here."))

    s.add(Panel(
        title="lines 374–375 — the frame stores",
        code="W = Vector{Any}(undef, c)              # augmented LEFT frames,  sites 1 … c\n"
             "Z = Vector{Any}(undef, L - 1)          # augmented RIGHT frames, Z[j] absorbs site j+1",
        figure=None,
        note="Frames are accumulated here rather than written into psi. That "
             "choice is the whole correctness argument of the sweep, and the next "
             "section is where it bites."))

    # ---- 383-387: K-sweep setup ---------------------------------------------
    t, _ = T.mps(L, centre=1, riso=tuple(range(2, L + 1)),
                 names=lambda k, sh: (f"B_{k}" if sh == "riso" else "C_1"))
    s.add(Panel(
        title="lines 383–387 — K-sweep setup",
        code="canonical!(psi, 1)\n"
             "rstack = right_env_stack(psi, h; downto = 3)\n"
             "lch = boundary_channels(h)\n"
             "lenv_c = lch                           # left channels at the expanded link c\n"
             "C = nothing",
        figure=Figure(t, caption="Amplitude at the left end, so psi[i+1] is an "
                                 "isometry and the carry holds the weight."),
        note="The right stack is <b>prebuilt</b> because this sweep never touches "
             "the sites right of the bond it is on. The left channels are "
             "<b>carried</b> instead — they have to be expressed in the new frames "
             "the sweep is in the middle of creating, and a prebuilt left stack "
             "would be written in the old bases and silently wrong."))

    # ---- 388-401: the K-sweep body, every line, unrolled --------------------
    def k_iter(i):
        panels = []
        a = TN()
        if i > 1:
            cc = a.bond("C", 0)
            ps = a.site(f"psi[{i}]", 1)
            a.link(cc, "E", ps, "W", label=f"ℓ_{i-1}")
            a.dangle(cc, "W", f"b_{i}")
        else:
            ps = a.site("psi[1]", 0)
            a.dangle(ps, "W", "ℓ_0")
        a.phys(ps, f"σ_{i}"); a.dangle(ps, "E", f"ℓ_{i}")
        b = TN(); k = b.site("K", 0, hi=True)
        b.dangle(k, "W", f"b_{i}"); b.phys(k, f"σ_{i}"); b.dangle(k, "E", f"ℓ_{i}")
        panels.append(Panel(
            title=f"── K-sweep i = {i} — line 389, the carry",
            code=f"K = C === nothing ? psi[{i}] : to_concrete(contract(C, (2,), psi[{i}], (1,)))",
            figure=Figure(a, Txt("="), b,
                          caption="The left block A_1…A_i, already written in the "
                                  "expanded basis built at the previous bond."),
            note=("At i = 1 the carry is empty and K is just psi[1]." if i == 1 else
                  "This is the line that makes the frames <b>chain</b>. Build the "
                  "frame from psi[i] instead and its left leg is the OLD link "
                  "ℓ_{i-1}; such frames cannot be composed into a state at all.")))

        f, _ = T.frame_uv()
        panels.append(Panel(
            title=f"── K-sweep i = {i} — lines 390–391",
            code=f"f = _frame_from(K, psi[{i+1}])\n"
                 f"{i} == c && (lenv_c = lch)",
            figure=Figure(f, caption="Two SVDs, one from each side, at "
                                     "cutoff = 0.0 — a change of basis, not an "
                                     "approximation."),
            note="<code>_frame_from</code> rather than the library's "
                 "<code>bond_frame</code> precisely because the tensor whose frame "
                 "we want is the carry, not psi[i]. <code>bond_frame</code> insists "
                 "the orthogonality centre sit on site i and reads psi directly."))

        e, _ = T.expanded_frame("L")
        panels.append(Panel(
            title=f"── K-sweep i = {i} — lines 392–398",
            code=f"ex = (centre_expand || {i} != c) ?\n"
                 f"     record!(cbe_expand(f, h, {i}, lch, right_channels(rstack, {i+2}); exkw...)) :\n"
                 f"     CBEExpansion(f.U0, f.V0, 0, 0, 0.0, 0.0)\n"
                 f"W[{i}] = ex.U_ex\n"
                 f"expanded[{i}] = leg_dim(ex.U_ex, 3)\n"
                 f"n_new[{i}] = ex.n_new_l\n"
                 f"peak!(ex.U_ex)",
            figure=Figure(e, caption="Accumulated into W. psi is NOT written."),
            note="<b>The basis sweeps must not touch the state.</b> An earlier "
                 "version wrote each frame straight back with "
                 "<code>_absorb_left!</code>, which leaves the zero-padded "
                 "remainder in psi[i+1]; the next iteration's frame was then "
                 "re-derived from that padded tensor with "
                 "<code>svd(cutoff = 0.0)</code>, collapsing the padding and "
                 "propagating the collapse back into the bond just expanded. "
                 "Measured symptom: L = 18 pinned at maxbond 2 with an error "
                 "<em>independent of dt</em>. That independence is the diagnostic — "
                 "an error that does not respond to the time step is not a "
                 "time-integration error."))

        a2 = TN()
        uu = a2.liso("U_ex", 0, dag=True, mirror="h")
        kk2 = a2.site("K", 1)
        a2.link(uu, "E", kk2, "W", label="ℓ_{i-1}, σ_i")
        a2.dangle(uu, "W", f"b_{i+1}"); a2.dangle(kk2, "E", f"ℓ_{i+1}")
        b2 = TN(); cn = b2.bond("C", 0, hi=True)
        b2.dangle(cn, "W", f"b_{i+1}"); b2.dangle(cn, "E", f"ℓ_{i+1}")
        panels.append(Panel(
            title=f"── K-sweep i = {i} — lines 399–400",
            code=f"C = to_concrete(contract(ex.U_ex', (1, 2), K, (1, 2)))    # (b_{{{i+1}}}, link_{{{i+1}}})\n"
                 f"lch = push_left_channels(lch, h, ex.U_ex, {i})",
            figure=Figure(a2, Txt("="), b2,
                          caption="The new carry, C = U_ex† K — and the "
                                  "environment advanced through the AUGMENTED "
                                  "frame, not through psi[i]."),
            note="Pushing is O(1) per bond against O(L) to rebuild a stack; "
                 "carrying is what makes one sweep O(L) rather than O(L²). The "
                 "price is the ordering constraint: a carried environment must be "
                 "advanced through the same tensors the sweep is using."))
        return panels

    for i in range(1, c + 1):
        s.add(*k_iter(i))
    s.add(Panel(title="line 401 — end of the K-sweep",
                code="end", figure=None,
                note=f"W[1…{c}] now hold augmented left frames, and psi is "
                     f"bit-for-bit what it was at line 383."))

    # ---- 408-428: the L-sweep ----------------------------------------------
    t, _ = T.mps(L, centre=L, liso=tuple(range(1, L)),
                 names=lambda k, sh: (f"A_{k}" if sh == "liso" else f"C_{L}"))
    s.add(Panel(
        title="lines 408–412 — L-sweep setup, the gauge is flipped on purpose",
        code="canonical!(psi, L)\n"
             "lstack = left_env_stack(psi, h; upto = L - 2)\n"
             "rch = boundary_channels(h)\n"
             "renv_c = rch                           # right channels at the expanded link c+2\n"
             "D = nothing",
        figure=Figure(t, caption="Now sites 1…L−1 are LEFT isometries and the "
                                 "amplitude rides in D."),
        note="Re-gauging does not change the state, so the W's stay valid — and "
             "the seed below is a complete contraction, hence gauge independent. "
             "That is why two incompatible gauges can coexist inside one step."))

    def l_iter(j):
        panels = []
        a = TN()
        ps = a.site(f"psi[{j+1}]", 0)
        d = a.bond("D", 1)
        a.link(ps, "E", d, "W", label=f"ℓ_{j+1}")
        a.dangle(ps, "W", f"ℓ_{j}"); a.phys(ps, f"σ_{j+1}"); a.dangle(d, "E", "b")
        b = TN(); bb = b.site("B", 0, hi=True)
        b.dangle(bb, "W", f"ℓ_{j}"); b.phys(bb, f"σ_{j+1}"); b.dangle(bb, "E", "b")
        panels.append(Panel(
            title=f"── L-sweep j = {j} — line 414, the mirror carry",
            code=f"B = D === nothing ? psi[{j+1}] : to_concrete(contract(psi[{j+1}], (3,), D, (1,)))",
            figure=Figure(a, Txt("="), b,
                          caption="The right block, in the expanded basis built at "
                                  "the previous bond.")))
        f, _ = T.frame_uv()
        e, _ = T.expanded_frame("R")
        panels.append(Panel(
            title=f"── L-sweep j = {j} — lines 415–425",
            code=f"f = _frame_from(psi[{j}], B)\n"
                 f"{j} == c && (renv_c = rch)\n"
                 f"ex = (centre_expand || {j} != c) ?\n"
                 f"     record!(cbe_expand(f, h, {j}, left_channels(lstack, {j}), rch; exkw...)) :\n"
                 f"     CBEExpansion(f.U0, f.V0, 0, 0, 0.0, 0.0)\n"
                 f"Z[{j}] = ex.V_ex\n"
                 f"if {j} != c\n"
                 f"    expanded[{j}] = leg_dim(ex.V_ex, 1)\n"
                 f"    n_new[{j}] = ex.n_new_r\n"
                 f"end\n"
                 f"peak!(ex.V_ex)",
            figure=Figure(f, Txt("⟶"), e,
                          caption="The mirror of the K-sweep: V_ex is kept, psi is "
                                  "untouched."),
            note="Note which environment comes from where: <em>prebuilt</em> on the "
                 "left now, <em>carried</em> on the right. Exactly reversed from the "
                 "K-sweep, and for the same reason."))
        panels.append(Panel(
            title=f"── L-sweep j = {j} — lines 426–427",
            code=f"D = to_concrete(contract(B, (2, 3), ex.V_ex', (2, 3)))    # (link_{{{j+1}}}, b)\n"
                 f"rch = push_right_channels(rch, h, ex.V_ex, {j+1})",
            figure=None,
            note="<code>push_right_channels</code> <b>opens</b> with "
                 "<code>term.right</code> and <b>closes</b> with "
                 "<code>term.left</code> — the reverse of the left sweep, because "
                 "the automaton is now scanning right to left. Getting this "
                 "backwards produces a Hermitian-looking but wrong H."))
        return panels

    for j in range(L - 1, c - 1, -1):
        s.add(*l_iter(j))
    s.add(Panel(title="line 428 — end of the L-sweep", code="end", figure=None))

    # ---- 434-445: the seed --------------------------------------------------
    seed = TN()
    w = [seed.liso(f"W_{k+1}", k) for k in range(c)]
    z = [seed.riso(f"Z_{c+k}", c + k) for k in range(L - c)]
    for k in range(len(w) - 1):
        seed.link(w[k], "E", w[k + 1], "W")
    seed.link(w[-1], "E", z[0], "W", label="b")
    for k in range(len(z) - 1):
        seed.link(z[k], "E", z[k + 1], "W")
    for nd in w + z:
        seed.phys(nd, None)
    seed.dangle(w[0], "W", None); seed.dangle(z[-1], "E", None)
    seed.brace(0, c - 1, label="W: K-sweep")
    seed.brace(c, L - 1, label="Z: L-sweep", above=False)
    s.add(Panel(
        title="lines 434–445 — the seed",
        code="AL = nothing\n"
             "for i in 1:c\n"
             "    T = AL === nothing ? psi[i] : to_concrete(contract(AL, (2,), psi[i], (1,)))\n"
             "    AL = to_concrete(contract(W[i]', (1, 2), T, (1, 2)))       # (b_L, link_{i+1})\n"
             "end\n"
             "AR = nothing\n"
             "for j in (L - 1):-1:c\n"
             "    T = AR === nothing ? psi[j + 1] : to_concrete(contract(psi[j + 1], (3,), AR, (1,)))\n"
             "    AR = to_concrete(contract(T, (2, 3), Z[j]', (2, 3)))       # (link_{j+1}, b_R)\n"
             "end\n"
             "S0 = to_concrete(contract(AL, (2,), AR, (1,)))                 # (b_L, b_R)\n"
             "peak!(W[c], Z[c], S0)",
        figure=Figure(seed, caption="S₀ = ⟨W, Z | psi⟩, by full contraction from "
                                    "each end."),
        note="<b>No overlap matrices and no inverse anywhere.</b> Textbook BUG "
             "forms M = Û†U, N = V̂†V and seeds Ŝ = MSN†. Here the seed is a direct "
             "contraction of the state against the frames — which is the "
             "inverse-free property that makes BUG robust to small singular "
             "values, the failure mode that forces TDVP implementations to "
             "regularise."))

    # ---- 451-456: the one Galerkin step -------------------------------------
    s.add(Panel(
        title="lines 451–456 — the only time evolution in the step",
        code="H0 = zero_site_h(h, c, lenv_c, renv_c, W[c], Z[c])\n"
             "nmv = Ref(0)\n"
             "S1 = expv(x -> (nmv[] += 1; apply_zero_site(H0, x)), tau, S0;\n"
             "          hermitian = true, maxiter = maxiter, tol = tol)\n"
             "res = svd(S1, (1,); cutoff = max(trunc_thresh, 1e-14), Nkeep = maxdim, get_lists = true)\n"
             "discarded = _trunc_weight(res)",
        figure=Figure(T.heff_zero(lname="lenv_c", rname="renv_c", core="S_0"),
                      caption="0-site: two bond indices, no physical index — the "
                              "operand is smaller than 2-site TDVP's by d²."),
        note="<b>Exactly one.</b> A Galerkin step at every bond applies the global "
             "H (L−1) times per sweep, over-counting τ; measured, full-rank "
             "exactness degrades to 2.24e−02. In the gauge W…W S Z…Z a chain has "
             "exactly one connection tensor, and Algorithm 7 fires once per "
             "connection tensor. <b>A tree has one per internal node — which is "
             "where this stops being trivial.</b>"))

    # ---- 463-477: reassembly ------------------------------------------------
    a = TN(); s1 = a.bond("S_1", 0, hi=True)
    a.dangle(s1, "W", "b_L"); a.dangle(s1, "E", "b_R")
    b = TN()
    uu = b.liso("U", 0, hi=True); ss = b.bond("S", 1, hi=True)
    vv = b.riso("V†", 2, hi=True)
    b.link(uu, "E", ss, "W"); b.link(ss, "E", vv, "W")
    b.dangle(uu, "W", "b_L"); b.dangle(vv, "E", "b_R")
    s.add(Panel(
        title="lines 463–477 — reassembly",
        code='relink(T, t1, t3) = to_concrete(setitag(setitag(T, 1, t1), 3, t3))\n'
             "\n"
             "for i in 1:(c - 1)\n"
             "    psi[i] = relink(W[i], psi[i].inds[1].itags, psi[i].inds[3].itags)\n"
             "end\n"
             "t1c, t3c = psi[c].inds[1].itags, psi[c].inds[3].itags\n"
             "t3r = psi[c + 1].inds[3].itags\n"
             "psi[c]     = relink(to_concrete(W[c] * res.U), t1c, t3c)\n"
             "psi[c + 1] = relink(to_concrete((res.S * res.Vd) * Z[c]), t3c, t3r)\n"
             "for j in (c + 1):(L - 1)\n"
             "    psi[j + 1] = relink(Z[j], psi[j + 1].inds[1].itags, psi[j + 1].inds[3].itags)\n"
             "end\n"
             "psi.center = c + 1\n"
             "centre_rank = leg_dim(psi[c], 3)",
        figure=Figure(a, Txt("="), b,
                      caption="psi = W₁…W_{c−1} (W_c U) (S V† Z_c) Z_{c+1}…Z_{L−1}."),
        note="<code>relink</code> retags <b>both</b> bond legs of every frame, not "
             "just one. The frames carry <code>_frame_from</code> / "
             "<code>cbe_expand</code> tags, adjacent tensors in a SymMPS must agree "
             "on the shared link tag, and retagging one side only leaves "
             "<code>psi[i].inds[3] != psi[i+1].inds[1]</code>, which "
             "<code>contract</code> rejects on the next sweep."))

    s.add(Panel(
        title="lines 484–486 — compress once, at the end",
        code="truncate_sweep!(psi; maxdim = maxdim, cutoff = max(trunc_thresh, 1e-14))\n"
             "\n"
             "return CBEBugInfo(expanded, n_new, centre_rank, err_pre, err_fnl,\n"
             "                  discarded, nmv[], peak)",
        figure=None,
        note="<b>Safe only AFTER the Galerkin step.</b> The evolved core has now "
             "propagated out through every retained frame, so a direction carrying "
             "no weight is genuinely unwanted rather than merely not-yet-populated. "
             "Run before, this deletes exactly what the expansion just bought — "
             "which is what pinned the rank at 2 in the single-centre-step version."))
    return s


# ================================================ part 3: the gate sweep =====

def _rung(u, v, rank):
    """One rung of the growth ladder: frames of the stated rank around a core."""
    t = TN()
    a = t.liso(u, 0); c = t.bond("S", 1, sub=rank); b = t.riso(v, 2)
    t.link(a, "E", c, "W"); t.link(c, "E", b, "W")
    t.dangle(a, "W", None); t.phys(a, None); t.phys(b, None)
    t.dangle(b, "E", None)
    return t


def sec_gate():
    s = Section("4 — the gate sweep (cbe_bond_update, cbe_gate_evolve!)",
                "The Trotter path. Here the operator at a bond is the bare two-site "
                "gate: the environments are the identity by canonical form, so no "
                "channel machinery appears at all.")

    a, _ = T.term_pair()
    b, _ = T.gate_block()
    s.add(Panel(code="gates = bond_gates(psi)",
                figure=Figure(a, Txt("⟶"), b,
                              caption="The term list, fused per bond."),
                note="This fusion is why the gate path cannot host the ground-state "
                     "search: minimising ⟨h_{i,i+1}⟩ bond by bond does not have the "
                     "global ground state as its fixed point."))

    # one bond update
    f, _ = T.frame_uv()
    s.add(Panel(
        code="canonical!(psi, i)\nf = bond_frame(psi, i)\n"
             "ex = cbe_expand_bond(f, gates[i])",
        figure=Figure(f, caption="Same frame, same expansion — but sketched against "
                                 "the gate instead of the environment-dressed H."),
        note="<code>sketch_bond_left/right</code> replace "
             "<code>sketch_h_left/right</code>. That substitution is the ONLY "
             "difference between this path and the MPO path at the level of the "
             "expansion."))

    g = TN(anchor_row=0.5)
    ua = g.liso("U_aug", 0)
    sc = g.bond("S", 1, hi=True)
    va = g.riso("V_aug", 2)
    g.link(ua, "E", sc, "W"); g.link(sc, "E", va, "W")
    g.dangle(ua, "W", "ℓ_{i-1}"); g.dangle(va, "E", "ℓ_{i+1}")
    gate = g.node("G_i", "op", 1, row=1, cols=2)
    g.link(ua, "N", gate, "S@-0.5", label="σ_i")
    g.link(va, "N", gate, "S@+0.5", label="σ_{i+1}")
    g.dangle(gate, "N@-0.5", "σ'_i"); g.dangle(gate, "N@+0.5", "σ'_{i+1}")
    s.add(Panel(
        code="S_start = U_aug' * theta0 * V_aug'\n"
             "apply_s(x) = U_aug' * apply_gate(gate, (U_aug*x)*V_aug) * V_aug'\n"
             "S_new = expv(apply_s, tau_s, S_start)",
        figure=Figure(g, caption="The Galerkin problem in the expanded basis: "
                                 "expand out, apply the gate, project back."),
        note="<code>S_start</code> is rebuilt from the ORIGINAL theta0 on every "
             "call, never from the previous pass's S_new — which is why running "
             "<code>s_iters</code> passes still applies exp(τH_eff) exactly once."))

    s.add(Panel(
        title="the growth ladder (s_iters > 1)",
        code="for k in 1:s_iters\n"
             "    ex = expand_at(probe);  S_new = galerkin(ex.U_ex, ex.V_ex)\n"
             "    probe = _reframe(f, U_aug, S_new, V_aug)\n"
             "end",
        figure=Figure(_rung("U_0", "V_0", "r"), Txt("⟶"),
                      _rung("U^{(1)}", "V^{(1)}", "g·r"), Txt("⟶"),
                      _rung("U^{(2)}", "V^{(2)}", "g²·r"),
                      caption="Each pass re-sketches around the state the previous "
                              "pass evolved, and recomputes the budget from the "
                              "CURRENT rank — so the rank climbs a ladder."),
        note="It does <b>not</b> build a Krylov space — with τ small every pass "
             "re-derives HΘ₀, never H²Θ₀. What it compounds is the GROWTH "
             "SCHEDULE: the budget is recomputed from the current rank each pass, "
             "so r → g·r → g²·r, each rung probed by a sketch matched to the rank "
             "it is probing. Measured at L=8, growth 1.25: one shot 1.449e−11 at "
             "rank 9; three passes 1.978e−15 at rank 12."))

    # the Strang composition
    st = TN()
    nodes = [st.site(f"M_{k+1}", k) for k in range(L)]
    for k in range(L - 1):
        st.link(nodes[k], "E", nodes[k + 1], "W")
    for nd in nodes:
        st.phys(nd, None)
    st.dangle(nodes[0], "W", None); st.dangle(nodes[-1], "E", None)
    for k in range(0, L - 1, 2):
        st.box(k, k + 1, label="even" if k == 0 else None)
    s.add(Panel(
        code="a = cbe_bond_update_sweep!(psi, gates, :even, tau/2)\n"
             "b = cbe_bond_update_sweep!(psi, gates, :odd,  tau)\n"
             "c = cbe_bond_update_sweep!(psi, gates, :even, tau/2)",
        figure=Figure(st, caption="Bonds of one parity act on disjoint site pairs, "
                                  "so a parity group is an EXACT factor of the "
                                  "Trotter step."),
        note="No re-canonicalisation between bonds: the update returns "
             "<code>left_core</code> already a left isometry with the centre in "
             "<code>right_core</code>, so <code>psi.center = i+1</code> is exact and "
             "the next bond of the group is one move to the right."))
    return s


# ======================================== part 4: 1-site TDVP + CBE ==========

def sec_tdvp1():
    s = Section("5 — tdvp1_cbe_step! (tdvp1_cbe/tdvp1_cbe.jl)",
                "1-site TDVP, where CBE is mandatory rather than optional: a 1-site "
                "sweep cannot change a bond dimension by itself, so without the "
                "expansion the rank is frozen at whatever it started as.")

    a, _ = T.frame_uv()
    b = TN()
    ps = b.site("psi[i]", 0, hi=True, sub="zero-padded")
    vx = b.riso("V_ex", 1, hi=True)
    b.link(ps, "E", vx, "W", label="b_i")
    b.dangle(ps, "W", "ℓ_{i-1}"); b.phys(ps, "σ_i")
    b.phys(vx, "σ_{i+1}"); b.dangle(vx, "E", "ℓ_{i+1}")
    s.add(Panel(
        code="canonical!(psi, i)\nf  = bond_frame(psi, i)\n"
             "ex = cbe_expand(f, h, i, lch, rch)\n"
             "_absorb_right!(psi, i, ex.V_ex)     # psi[i+1] = V_ex",
        figure=Figure(a, Txt("⟶"), b,
                      caption="The neighbour BECOMES the expanded frame; the "
                              "current site takes the zero-padded remainder."),
        note="<b>Here the write-back is correct</b>, and that is the sharp contrast "
             "with the Lubich sweep, where the identical move destroys the "
             "expansion. The difference is what happens next: there, the padded "
             "tensor is re-read to build the NEXT frame and the padding collapses; "
             "here it is immediately EVOLVED, and the one-site update is exactly "
             "what fills the new columns."))

    s.add(Panel(
        code="M = expv(x -> apply_one_site(one_site_h(h, i, lch, rch1), x), tau, psi[i])",
        figure=Figure(T.heff_one(site="psi[i]", lname="lch", rname="rch1"),
                      caption="Forward, +τ, on the WIDENED bond — this is the step "
                              "that puts amplitude into the new columns."),
        note="Compare the Lubich sweep, whose one time evolution is 0-site. This "
             "one carries a physical index, so its operand is larger by d."))

    a = TN(); m = a.site("M", 0, hi=True)
    a.dangle(m, "W", "ℓ_{i-1}"); a.phys(m, "σ_i"); a.dangle(m, "E", "b_i")
    b = TN()
    u = b.liso("psi[i]", 0, hi=True); cc = b.bond("C", 1, hi=True)
    b.link(u, "E", cc, "W"); b.dangle(u, "W", "ℓ_{i-1}"); b.phys(u, "σ_i")
    b.dangle(cc, "E", "b_i")
    s.add(Panel(
        code="res = svd(M, (1, 2); cutoff = ..., Nkeep = maxdim)\n"
             "psi[i] = res.U;  C = res.S * res.Vd",
        figure=Figure(a, Txt("="), b,
                      caption="Split, truncate, and leave a left isometry behind."),
        note="Truncation happens here, AFTER the forward step — the same ordering "
             "rule the Lubich sweep obeys, for the same reason."))

    s.add(Panel(
        code="H0 = zero_site_h(h, i, lch, rch, psi[i], psi[i+1])\n"
             "C  = expv(x -> apply_zero_site(H0, x), -tau, C)\n"
             "psi[i+1] = contract(C, (2,), psi[i+1], (1,))",
        figure=Figure(T.heff_zero(lname="lch", rname="rch"),
                      caption="Backward, −τ, 0-site — the projector-splitting "
                              "correction the BUG sweep does not have."),
        note="This is the structural difference from part 2 in one line: TDVP "
             "splits a projector and therefore needs the backward substep; BUG "
             "enlarges a basis and therefore does not."))

    strip = TN()
    n1 = [strip.liso(f"A_{k+1}", k) for k in range(2)]
    ctr = strip.site("C", 2, hi=True)
    n2 = [strip.riso(f"B_{k+4}", 3 + k) for k in range(L - 3)]
    allo = n1 + [ctr] + n2
    for k in range(len(allo) - 1):
        strip.link(allo[k], "E", allo[k + 1], "W")
    for nd in allo:
        strip.phys(nd, None)
    strip.dangle(allo[0], "W", None); strip.dangle(allo[-1], "E", None)
    s.add(Panel(
        code="for i in 1:(L-1);  lch = _tdvp1_cbe_bond!(psi, h, i, half, :right, ...);  end\n"
             "onesite!(L, lch, boundary_channels(h))\n"
             "for i in (L-1):-1:1;  rch = _tdvp1_cbe_bond!(psi, h, i, half, :left, ...);  end\n"
             "onesite!(1, boundary_channels(h), rch)",
        figure=Figure(strip, caption="A symmetric sweep of τ/2 out and τ/2 back — "
                                     "second order."),
        note="Each half-sweep carries the side it moves away from and reads the "
             "other from one stack built at the start. The stack survives because a "
             "rightward pass only rewrites sites at or behind the bond it is on."))
    return s


# ================================================ part 5: ground state ======

def sec_ground():
    s = Section("6 — GroundState.solve! (modes/ground_state.jl)",
                "Two-site DMRG whose expansion is RSVD-CBE and whose local solve is "
                "the same expanding Krylov iteration the time modes use. Two things "
                "are swapped: the operator, and the reduction.")

    s.add(Panel(
        code="applyH = Theta -> apply_h_two_site(Theta, h, i, lch, rch)",
        figure=Figure(T.heff_two(lname="lch", rname="rch", hi=True),
                      caption="The environment-dressed two-site operator L + h + R "
                              "— NOT a bare gate."),
        note="<b>Why this mode is MPO-path only.</b> The gate path's environments "
             "are the identity by canonical form, so its local operator is the bare "
             "two-site term; minimising that at each bond independently does not "
             "have the global ground state as its fixed point — each bond drives its "
             "own pair toward a local singlet and the next bond undoes it. DMRG "
             "converges precisely because L + h + R makes a LOCAL minimisation lower "
             "the TOTAL energy."))

    def phase_a(k):
        t, (a, c, b) = T.frame_uv(u="U^{(%d)}" % k, s="S", v="V^{(%d)}" % k,
                                  hi=("u", "v"))
        return Panel(
            title=f"── phase A, growth pass {k}",
            code=("U, V, S_g, ga = _expanding_krylov(f, applyH, expand_at,\n"
                  "                                  LowestEigen(), theta0, nmv;\n"
                  "                                  grow_iters = grow_iters)"
                  if k == 1 else "# pass %d: sketch around the %d-th Krylov vector" % (k, k)),
            figure=Figure(t, caption=f"Pass {k} sketches around Krylov vector {k}, "
                                     f"which carries H^{k}."),
            note=None if k > 1 else
                 "This is how it improves on one-shot CBE-DMRG: standard CBE picks "
                 "directions from a single sketch of HΘ, so the basis can only "
                 "contain what ONE application of H reaches. Here pass k sketches "
                 "around the k-th Krylov vector, so the basis accumulates "
                 "directions from successive powers — and the stopping signal is "
                 "the Lanczos β, which the iteration already computes.")

    s.add(*[phase_a(k) for k in (1, 2)])

    t, _ = T.frame_uv(u="U", s="S", v="V", hi=("s",))
    s.add(Panel(
        title="── phase B, the variational solve",
        code="fg = _reframe(f, U, S_g, V)\n"
             "U, V, S, info = _expanding_krylov(fg, applyH, identity, LowestEigen(),\n"
             "                                  theta0, nmv; grow_iters = 0)",
        figure=Figure(t, caption="A clean Lanczos in the now-FIXED frames: an exact "
                                 "Rayleigh–Ritz, so variational again."),
        note="<b>Why the growth and the eigensolve are not one loop.</b> "
             "<code>_expanding_krylov</code>'s tridiagonal is a Galerkin "
             "approximation across a nested sequence of subspaces: α_k is exact but "
             "β_k is measured against the projector in force at step k and misses "
             "whatever part of H·v_k fell outside the then-current basis. For a "
             "short-time exponential that is a small perturbation of an "
             "already-small correction. For an eigensolve it is not acceptable — "
             "the Ritz value IS the answer, and one computed from an inconsistent "
             "tridiagonal is not variational, so it can sit BELOW the true ground "
             "energy and report false convergence."))

    a = TN(anchor_row=0.5)
    ss = a.bond("S", 0, hi=True)
    a.dangle(ss, "W", "b_L"); a.dangle(ss, "E", "b_R")
    b = TN()
    ua = b.liso("psi[i]", 0, hi=True); cb = b.site("psi[i+1]", 1, hi=True)
    b.link(ua, "E", cb, "W", label="ℓ_i")
    b.dangle(ua, "W", "ℓ_{i-1}"); b.phys(ua, "σ_i")
    b.phys(cb, "σ_{i+1}"); b.dangle(cb, "E", "ℓ_{i+1}")
    s.add(Panel(
        code="res = svd(S, (1,); cutoff = ..., Nkeep = maxdim)\n"
             "psi[i]   = U_aug * res.U\n"
             "psi[i+1] = (res.S * res.Vd) * V_aug;   psi.center = i + 1",
        figure=Figure(a, Txt("⟶"), b,
                      caption="Write back with the centre left where the sweep is "
                              "heading."),
        note="Getting the direction backwards silently breaks the next bond's "
             "<code>bond_frame</code>, which assumes a true canonical snapshot."))

    s.add(Panel(
        title="the sweep",
        code="for i in bonds\n"
             "    canonical!(psi, i);  f = bond_frame(psi, i)\n"
             "    U, V, S, e, ng = _bond_solve(f, applyH, expand_at, nmv; ...)\n"
             "    _write_bond!(psi, i, f, U, V, S, moving; ...)\n"
             "    carried = push_left_channels(carried, h, psi[i], i)\n"
             "end",
        figure=None,
        note="<b>grow_iters = 0 freezes the state at bond dimension 1.</b> Measured "
             "at L = 8, Δ = 1, half filling: the frames never widen, the local "
             "eigensolve minimises over a one-dimensional space, and the sweep "
             "reports the Néel product-state energy −1.75 at bond dimension 1 and "
             "“converges” in two sweeps. The CBE growth pass is doing ALL of the "
             "rank generation and the eigensolve none of it — which also means "
             "<code>info.converged</code> is not evidence of anything by itself."))
    return s


COMPARE = """
<h2>7 — the four side by side</h2>
<p class="blurb">Same expansion in every column. What differs is the operator it is
sketched against, where the time evolution (or eigensolve) happens, and whether a
backward substep is needed.</p>
<table>
<tr><th></th><th>Lubich BUG</th><th>gate sweep</th><th>1-site TDVP+CBE</th>
    <th>ground state</th></tr>
<tr><td>operator at a bond</td><td>env-dressed H</td><td>bare two-site gate</td>
    <td>env-dressed H</td><td>env-dressed H</td></tr>
<tr><td>expansion</td><td colspan="4" style="text-align:center">
    <code>cbe_expand</code> — identical in all four</td></tr>
<tr><td>projector used</td><td>none — a Galerkin problem in an enlarged basis</td>
    <td>none — the gate, in an enlarged basis</td>
    <td>P1 forward, P0 backward (the tangent projector, split)</td>
    <td>none — a variational minimisation</td></tr>
<tr><td>reaches the discarded space?</td>
    <td colspan="4" style="text-align:center">yes, and only via
    <code>cbe_expand</code> — the shaded triangles enter nowhere else</td></tr>
<tr><td>local solve</td><td>0-site, at the centre bond only</td>
    <td>2-site Galerkin, every bond</td>
    <td>1-site forward + 0-site backward</td>
    <td>lowest Ritz vector, every bond</td></tr>
<tr><td>time evolutions per step</td><td><b>one</b></td><td>one per bond × 3 parity
    sweeps</td><td>two per bond × 2 half-sweeps</td><td>— (no time)</td></tr>
<tr><td>backward substep</td><td><b>none</b></td><td>none</td><td><b>yes</b></td>
    <td>none</td></tr>
<tr><td>write-back during basis building</td><td><b>forbidden</b> — collapses the
    rank</td><td>n/a</td><td><b>required</b> — then immediately evolved</td>
    <td>n/a</td></tr>
<tr><td>inverses</td><td>none</td><td>none</td><td>none</td><td>none</td></tr>
</table>
"""


def main():
    page = Page("RSVD-CBE bond update — four algorithms, drawn",
                "One expansion, four wrappers. Diagrams generated from "
                "<code>src/RSVDCBEBondUpdate/</code>; conventions after Jan von "
                "Delft (QSpace, arXiv:2405.06632).")
    page.add("<h2>0 — the vocabulary</h2>", T.legend_html())
    page.add(sec_expand(), sec_projectors(), sec_lubich(), sec_gate(),
             sec_tdvp1(), sec_ground(), COMPARE)
    p = page.save(os.path.join(OUT, "rsvdcbe.html"))
    print("wrote", p)


if __name__ == "__main__":
    main()
