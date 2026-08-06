#!/usr/bin/env python3
"""
THE TEMPLATE PAGE. A 1-site TDVP sweep, drawn line by line.

This is the worked example TEMPLATE.md describes: read it alongside that file when
translating a new algorithm. Every line of the Julia below gets a diagram, and the
`for` loop is unrolled over three iterations.

    canonical!(psi, 1)
    R = right_env_stack(psi, H)
    for i in 1:L-1
        Heff1    = one_site_h(H, L[i], R[i+1])
        psi[i]   = expv(Heff1, +tau, psi[i])
        Q, C     = qr(psi[i], (1, 2))
        psi[i]   = Q
        L[i+1]   = push_left(L[i], H, Q, i)
        Heff0    = zero_site_h(H, L[i+1], R[i+1])
        C        = expv(Heff0, -tau, C)
        psi[i+1] = contract(C, (2,), psi[i+1], (1,))
    end

Run:  python3 tdvp1_test.py   ->  out/tdvp1.html + out/*.svg
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from tndiag import TN, Txt, Figure, Panel, Section, Page, unroll   # noqa: E402
import tnlib as T                                                 # noqa: E402

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "out")
L = 5


# =================================================================== panels ==

def sec_construction():
    s = Section("1 — how the state and the Hamiltonian were built",
                "Before any algorithm runs, two objects have to exist. Both "
                "constructions are drawn here because both are what generalises "
                "to trees.")
    s.add(*T.construction_panels(n=L))
    return s


def sec_canonical():
    s = Section("2 — canonical form",
                "The shape rules exist so that the gauge is readable without "
                "labels.")

    a, _ = T.mps(L, names=lambda k, sh: f"M_{k}")
    b, _ = T.mps(L, centre=1, riso=(2, 3, 4, 5),
                 names=lambda k, sh: (f"B_{k}" if sh == "riso" else "C_1"))
    s.add(Panel(code="canonical!(psi, 1)",
                figure=Figure(a, Txt("⟶"), b,
                              caption="Sites 2…L become right isometries; all the "
                                      "amplitude collects at site 1."),
                note="Triangles point <em>away</em> from the orthogonality centre, "
                     "so the square is visibly the odd one out."))
    s.add(Panel(code="# the property a triangle asserts",
                figure=T.ortho_figure(), title="what a triangle means"))
    s.add(Panel(figure=T.conj_figure(), title="conjugation"))
    return s


def sec_projectors():
    s = Section("3 — the projectors, constructed",
                "TDVP is defined by a projector, so it is worth building one "
                "explicitly. Light and empty is the kept space; shaded is the "
                "discarded space.")
    s.add(*T.projector_panels())
    return s


def sec_env():
    s = Section("4 — the environment stack",
                "Built once, right to left, before the sweep starts. The three "
                "channel moves of §1 are what one step of this recursion is made "
                "of.")
    t = TN(bra=(2,))
    r = T.env_R(t, "R_i", 0)
    t.dangle(r, "W:+1", "ℓ'_{i-1}", free_dir="W")
    t.dangle(r, "W", "w_{i-1}", free_dir="W")
    t.dangle(r, "W:-1", "ℓ_{i-1}", free_dir="W")
    t.anchor_row = 1

    u = TN(anchor_row=1)
    k, w, br = T.sandwich(u, 0, ket_shape="riso", ket="B_i")
    rn = T.env_R(u, "R_{i+1}", 1)
    u.link(k, "E", rn, "W:-1")
    u.link(w, "E", rn, "W")
    u.link(br, "E", rn, "W:+1")
    u.dangle(k, "W", "ℓ_{i-1}")
    u.dangle(w, "W", "w_{i-1}")
    u.dangle(br, "W", "ℓ'_{i-1}")

    s.add(Panel(code="R = right_env_stack(psi, H)   # R[i] from R[i+1]",
                figure=Figure(t, Txt("="), u,
                              caption="One step of the recursion: absorb site i "
                                      "into the right environment."),
                note="The bra row is the conjugate row — hollow shapes, arrows "
                     "reversed."))
    return s


# ---------------------------------------------------------------- loop body --

def state_strip(i):
    t, _ = T.mixed(L, i, hi=(i,),
                   names=lambda k, sh: (f"A_{k}" if sh == "liso" else
                                        f"B_{k}" if sh == "riso" else f"C_{k}"))
    return t


def loop_iter(i, full=True):
    out = [Panel(title=f"── iteration i = {i} — state on entry",
                 figure=Figure(state_strip(i),
                               caption=f"Sites 1…{i-1} left isometries, {i+1}…{L} "
                                       f"right isometries, centre at {i}."))]
    if not full:
        return out

    out.append(Panel(
        code=f"Heff1 = one_site_h(H, L[{i}], R[{i+1}])\n"
             f"psi[{i}] = expv(Heff1, +tau, psi[{i}])",
        figure=Figure(T.heff_one(site=f"C_{i}", lname=f"L_{i}",
                                 rname=f"R_{{{i+1}}}"),
                      caption="Contract the centre against both environments and "
                              "the local MPO tensor; the three open legs on top "
                              "are the result's."),
        note="Forward step, +τ. The operand carries one physical index."))

    a = TN(); n = a.site(f"C_{i}", 0, hi=True)
    a.dangle(n, "W", "ℓ_{i-1}"); a.dangle(n, "E", "ℓ_i"); a.phys(n, "σ_i")
    b = TN()
    q = b.liso(f"A_{i}", 0, hi=True); c = b.bond("C", 1, hi=True)
    b.link(q, "E", c, "W", label="b_i")
    b.dangle(q, "W", "ℓ_{i-1}"); b.phys(q, "σ_i"); b.dangle(c, "E", "ℓ_i")
    out.append(Panel(
        code=f"Q, C = qr(psi[{i}], (1, 2));  psi[{i}] = Q",
        figure=Figure(a, Txt("="), b,
                      caption="The QR splits a square into a left triangle and a "
                              "circle: the centre moves off site i onto the bond."),
        note="This is the line where the shape changes, and the diagram says so "
             "without a word of prose."))

    p = TN(anchor_row=1, bra=(2,))
    ln = T.env_L(p, f"L_{{{i+1}}}", 0, hi=True)
    p.dangle(ln, "E:+1", "b'_i", free_dir="E", lab_side="N")
    p.dangle(ln, "E", "w_i", free_dir="E", lab_side="N")
    p.dangle(ln, "E:-1", "b_i", free_dir="E", lab_side="S")
    u = TN(anchor_row=1)
    lo = T.env_L(u, f"L_{i}", 0)
    qq, ww, qd = T.sandwich(u, 1, ket_shape="liso", ket=f"A_{i}")
    u.link(lo, "E:-1", qq, "W")
    u.link(lo, "E", ww, "W")
    u.link(lo, "E:+1", qd, "W")
    u.dangle(qq, "E", "b_i"); u.dangle(ww, "E", "w_i")
    u.dangle(qd, "E", "b'_i", lab_side="N")
    out.append(Panel(
        code=f"L[{i+1}] = push_left(L[{i}], H, Q, {i})",
        figure=Figure(p, Txt("="), u,
                      caption="The freshly made left isometry is absorbed into the "
                              "left environment.")))

    out.append(Panel(
        code=f"Heff0 = zero_site_h(H, L[{i+1}], R[{i+1}])\n"
             f"C = expv(Heff0, -tau, C)",
        figure=Figure(T.heff_zero(lname=f"L_{{{i+1}}}", rname=f"R_{{{i+1}}}"),
                      caption="The 0-site operator: no physical leg anywhere. The "
                              "MPO link runs straight from L to R."),
        note="Backward step, −τ. This is the −Σ P⁰ term of the tangent-space "
             "projector; without it the bond is double counted."))

    a = TN()
    c = a.bond("C", 0, hi=True); bb = a.riso(f"B_{i+1}", 1)
    a.link(c, "E", bb, "W", label="ℓ_i")
    a.dangle(c, "W", "b_i"); a.phys(bb, "σ_{i+1}"); a.dangle(bb, "E", "ℓ_{i+1}")
    b = TN(); nn = b.site(f"C_{i+1}", 0, hi=True)
    b.dangle(nn, "W", "b_i"); b.phys(nn, "σ_{i+1}"); b.dangle(nn, "E", "ℓ_{i+1}")
    out.append(Panel(
        code=f"psi[{i+1}] = contract(C, (2,), psi[{i+1}], (1,))",
        figure=Figure(a, Txt("="), b,
                      caption="Circle × right triangle = square. The centre lands "
                              f"on site {i+1}, and the loop repeats."),
        note="Shape arithmetic: absorbing the bond tensor destroys the right "
             "isometry and recreates the orthogonality centre."))
    return out


def sec_loop():
    s = Section("5 — the sweep, unrolled",
                "The loop body is identical every pass; only the index moves. "
                "Iterations 1 and 2 in full, iteration 3 as the state change alone.")
    s.add(*unroll(2, lambda i: loop_iter(i, full=True), start=1))
    s.add(*loop_iter(3, full=False))
    t, _ = T.mps(L, centre=L, liso=tuple(range(1, L)), hi=(L,),
                 names=lambda k, sh: (f"A_{k}" if sh == "liso" else f"C_{L}"))
    s.add(Panel(title="── after the last iteration",
                figure=Figure(t, caption="The centre has walked to the right end; "
                                         "everything behind it is a left isometry.")))
    return s


CHOICES = """
<h2>6 — the design choices</h2>
<table>
<tr><th>choice</th><th>what was done</th><th>how to change it</th></tr>
<tr><td>Triangle direction</td>
    <td>Left isometry → apex <b>left</b>, right isometry → apex <b>right</b>: they
        point <i>away</i> from the orthogonality centre. The commoner convention in
        the literature is the opposite.</td>
    <td><code>ISO_APEX_TOWARD_CENTER = True</code></td></tr>
<tr><td>Arrows</td>
    <td>Derived from the diagram, never declared: horizontal legs point +x on ket
        rows and −x on bra rows, vertical legs point up.</td>
    <td><code>rev=True</code> per leg</td></tr>
<tr><td>Conjugates</td>
    <td>Mirrored shape + hollow fill + reversed arrows.</td>
    <td><code>dag=True</code></td></tr>
<tr><td>Highlight</td>
    <td>The object a line <i>builds</i> is stroked in red, so a diagram reads as an
        assignment rather than a picture.</td>
    <td><code>hi=True</code></td></tr>
<tr><td>Loop rendering</td>
    <td><code>unroll(n, fn)</code> repeats the body with the index substituted; the
        last iteration is abbreviated to the state change.</td>
    <td>the <code>full=</code> flag</td></tr>
<tr><td>Output</td>
    <td>Self-contained SVG per figure plus this HTML page. Note
        <code>pdflatex</code> cannot include SVG directly — the figures need an
        SVG→PDF conversion first, and this machine has no LaTeX or Inkscape. A TikZ
        emitter would remove that step; nothing above depends on the backend.</td>
    <td><code>Figure.save(path)</code></td></tr>
</table>
"""


def main():
    page = Page("Translating code to tensor diagrams — the template",
                "A 1-site TDVP sweep, one diagram per line of code. Conventions "
                "after Jan von Delft (QSpace, arXiv:2405.06632); shapes per house "
                "rules. Generated by <code>docs/tndiag/</code>.")
    page.add("<h2>0 — the vocabulary</h2>", T.legend_html())
    page.add(sec_construction(), sec_canonical(), sec_projectors(), sec_env(),
             sec_loop(), CHOICES)
    p = page.save(os.path.join(OUT, "tdvp1.html"))

    Figure(T.heff_one()).save(os.path.join(OUT, "fig_heff1.svg"))
    Figure(T.heff_zero()).save(os.path.join(OUT, "fig_heff0.svg"))
    Figure(T.heff_two()).save(os.path.join(OUT, "fig_heff2.svg"))
    print("wrote", p)


if __name__ == "__main__":
    main()
