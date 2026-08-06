"""
tnlib -- the shared vocabulary every diagram page is written against.

`tndiag.py` knows about shapes and legs; this knows about MPS, MPO, environments,
bond frames and CBE expansions. A page (`tdvp1_test.py`, `rsvdcbe_diagrams.py`, ...)
should mostly be calls into here plus the code lines being annotated.

Everything returns a `TN` (or a node tuple), never a `Figure`, so callers stay free
to compose `Figure(lhs, Txt("="), rhs)`.
"""

from tndiag import TN, Txt, Figure, Panel

# ================================================================ the state ===


def mps(n=5, centre=None, liso=(), riso=(), hi=(), col0=0, row=0, tn=None,
        names=None, links=True, phys=True, bond1=False):
    """An n-site MPS.

    centre  site carrying the orthogonality centre -> square
    liso    sites that are left  isometries        -> left  triangles
    riso    sites that are right isometries        -> right triangles
    bond1   label every link "1" (a product state)
    """
    tn = tn or TN()
    ns = []
    for k in range(1, n + 1):
        if k in liso:
            shape, nm = "liso", f"A_{k}"
        elif k in riso:
            shape, nm = "riso", f"B_{k}"
        else:
            shape, nm = "site", (f"C_{k}" if k == centre else f"M_{k}")
        if names:
            nm = names(k, shape)
        ns.append(tn.node(nm, shape, col0 + k - 1, row, hi=(k in hi)))
    if links:
        for k in range(n - 1):
            lab = "1" if bond1 else (f"ℓ_{k+1}" if n <= 6 else None)
            tn.link(ns[k], "E", ns[k + 1], "W", label=lab)
    if phys:
        for k, nd in enumerate(ns, 1):
            tn.phys(nd, f"σ_{k}")
    return tn, ns


def mixed(n, centre, **kw):
    """Mixed-canonical MPS with the centre at `centre`."""
    return mps(n, centre=centre, liso=tuple(range(1, centre)),
               riso=tuple(range(centre + 1, n + 1)), **kw)


# ========================================================== the Hamiltonian ===


def term_pair(tn=None, col=0, row=0, t="t", coeff=True):
    """One nearest-neighbour term  c_t · O^L_t(i) O^R_t(i+1).

    Under U(1) both halves are rank 3: the op-leg between them carries the charge
    that makes the pair symmetry-allowed, which is why the code stores a TERM LIST
    and not a fused gate -- the sketch has to split a term across the bond.
    """
    tn = tn or TN()
    a = tn.op(f"O^L_{t}", col, row)
    b = tn.op(f"O^R_{t}", col + 1, row)
    tn.link(a, "E", b, "W", label="op")          # the charge-carrying op-leg
    for nd, k in ((a, "i"), (b, "i+1")):
        tn.phys(nd, f"σ_{{{k}}}")
        tn.phys(nd, f"σ'_{{{k}}}", down=True)
    return tn, (a, b)


def gate_block(tn=None, col=0, row=0, name="G_i"):
    """A two-site gate: the same term list, fused. What the Trotter path applies."""
    tn = tn or TN()
    g = tn.node(name, "op", col, row, cols=2)
    tn.dangle(g, "N@-0.5", "σ_i")
    tn.dangle(g, "N@+0.5", "σ_{i+1}")
    tn.dangle(g, "S@-0.5", "σ'_i")
    tn.dangle(g, "S@+0.5", "σ'_{i+1}")
    return tn, g


# ============================================================ environments ====

def env_L(tn, name, col, row=1, hi=False, rows=3):
    """Left environment slab: three legs on its east edge (bra / MPO / ket)."""
    return tn.env(name, col, row, rows=rows, hi=hi)


def env_R(tn, name, col, row=1, hi=False, rows=3):
    """Right environment slab: three legs on its west edge."""
    return tn.env(name, col, row, rows=rows, hi=hi)


def env2(tn, name, col, row=0.5, hi=False):
    """A two-leg channel block (`id` or `done`): bra and ket, no op leg."""
    return tn.env(name, col, row, rows=2, hi=hi)


def sandwich(tn, col, ket_shape="site", ket="A_i", op="W_i", row=0, dag_name=None):
    """The ket / MPO / bra column that every environment step pushes through."""
    k = tn.node(ket, ket_shape, col, row)
    w = tn.op(op, col, row + 1)
    b = tn.node(dag_name or ket, ket_shape, col, row + 2, dag=True)
    tn.link(k, "N", w, "S", label="σ_i")
    tn.link(w, "N", b, "S", label="σ'_i")
    return k, w, b


# ==================================================== effective operators =====

def heff_one(site="C_i", lname="L_i", rname="R_{i+1}", ket_shape="site"):
    """1-site effective Hamiltonian applied to a site tensor. Open legs on the bra row."""
    t = TN(anchor_row=1, bra=(2,))
    l = env_L(t, lname, 0)
    k = t.node(site, ket_shape, 1, 0)
    w = t.op("W_i", 1, 1)
    r = env_R(t, rname, 2)
    t.link(l, "E:-1", k, "W")
    t.link(l, "E", w, "W")
    t.dangle(l, "E:+1", "ℓ'_{i-1}", free_dir="E", lab_side="N")
    t.link(k, "N", w, "S", label="σ_i")
    t.dangle(w, "N", "σ'_i")
    t.link(k, "E", r, "W:-1")
    t.link(w, "E", r, "W")
    t.dangle(r, "W:+1", "ℓ'_i", free_dir="W", lab_side="N")
    return t


def heff_zero(lname="L_{i+1}", rname="R_{i+1}", core="C", hi=True):
    """0-site effective Hamiltonian: no physical leg anywhere."""
    t = TN(anchor_row=1, bra=(2,))
    l = env_L(t, lname, 0)
    c = t.bond(core, 1, 0, hi=hi)
    r = env_R(t, rname, 2)
    t.link(l, "E:-1", c, "W")
    t.link(l, "E", r, "W", label="w_i")
    t.dangle(l, "E:+1", "b'_i", free_dir="E", lab_side="N")
    t.link(c, "E", r, "W:-1")
    t.dangle(r, "W:+1", "ℓ'_i", free_dir="W", lab_side="N")
    return t


def heff_two(lname="L_i", rname="R_{i+2}", hi=False):
    """2-site effective Hamiltonian on the block Theta -- `apply_h_two_site`."""
    t = TN(anchor_row=1, bra=(2,))
    l = env_L(t, lname, 0)
    th = t.node("Θ", "bond", 1.5, 0, cols=2, hi=hi)
    w1 = t.op("W_i", 1, 1)
    w2 = t.op("W_{i+1}", 2, 1)
    r = env_R(t, rname, 3)
    t.link(l, "E:-1", th, "W")
    t.link(l, "E", w1, "W")
    t.dangle(l, "E:+1", "ℓ'_{i-1}", free_dir="E", lab_side="N")
    t.link(w1, "E", w2, "W", label="w_i")
    t.link(th, "N@-0.5", w1, "S")
    t.link(th, "N@+0.5", w2, "S")
    t.dangle(w1, "N", "σ'_i")
    t.dangle(w2, "N", "σ'_{i+1}")
    t.link(th, "E", r, "W:-1")
    t.link(w2, "E", r, "W")
    t.dangle(r, "W:+1", "ℓ'_{i+1}", free_dir="W", lab_side="N")
    return t


# =========================================================== bond frames ======

def theta_block(hi=False, name="Θ_i"):
    """The two-site block, as one object."""
    t = TN()
    th = t.node(name, "bond", 0.5, 0, cols=2, hi=hi)
    t.dangle(th, "W", "ℓ_{i-1}")
    t.dangle(th, "N@-0.5", "σ_i")
    t.dangle(th, "N@+0.5", "σ_{i+1}")
    t.dangle(th, "E", "ℓ_{i+1}")
    return t


def frame_uv(u="U_0", s="S_0", v="V_0", hi=(), ushape="liso", vshape="riso"):
    """Theta = U0 S0 V0 -- the BondFrame. U0 a left isometry, V0 a right one."""
    t = TN()
    a = t.node(u, ushape, 0, hi=("u" in hi))
    c = t.bond(s, 1, hi=("s" in hi))
    b = t.node(v, vshape, 2, hi=("v" in hi))
    t.link(a, "E", c, "W", label="r")
    t.link(c, "E", b, "W", label="r")
    t.dangle(a, "W", "ℓ_{i-1}"); t.phys(a, "σ_i")
    t.phys(b, "σ_{i+1}"); t.dangle(b, "E", "ℓ_{i+1}")
    return t, (a, c, b)


def expanded_frame(side="L", hi=True):
    """U_ex = [ U0 | C_L ]: a direct sum, so U0 sits inside U_ex as its first block."""
    t = TN()
    if side == "L":
        n = t.liso("U_ex", 0, hi=hi, sub="r + D_ex")
        t.dangle(n, "W", "ℓ_{i-1}"); t.phys(n, "σ_i"); t.dangle(n, "E", "b_i")
    else:
        n = t.riso("V_ex", 0, hi=hi, sub="r + D_ex")
        t.dangle(n, "W", "b_i"); t.phys(n, "σ_{i+1}"); t.dangle(n, "E", "ℓ_{i+1}")
    return t, n


# ================================================================ legends =====

_LEGEND = [
    ("site", "square — site tensor"),
    ("bond", "circle — bond tensor"),
    ("liso", "left triangle — left isometry, A†A = 1"),
    ("riso", "right triangle — right isometry, BB† = 1"),
    ("op", "wide box — MPO / operator"),
    ("env", "slab — environment block"),
]


def legend_html():
    cells = []
    for shape, text in _LEGEND:
        t = TN(bra=(2,))
        if shape == "env":
            n = t.env("L", 0, 1, rows=3)
            t.dangle(n, "E:+1", "ℓ'", free_dir="E")
            t.dangle(n, "E", "w", free_dir="E")
            t.dangle(n, "E:-1", "ℓ", free_dir="E")
        elif shape == "bond":
            n = t.bond("C", 0)
            t.dangle(n, "W", "a"); t.dangle(n, "E", "b")
        else:
            n = t.node({"site": "A", "liso": "A", "riso": "B", "op": "W"}[shape],
                       shape, 0)
            t.dangle(n, "W", "ℓ_{i-1}"); t.dangle(n, "E", "ℓ_i")
            t.phys(n, "σ_i")
            if shape == "op":
                t.phys(n, "σ'_i", down=True)
        cells.append(f'<div class="lg">{Figure(t).to_svg()}'
                     f'<div class="t">{text}</div></div>')
    return '<div class="legend">' + "".join(cells) + "</div>"


def conj_figure():
    a = TN(); n = a.liso("A", 0)
    a.dangle(n, "W", "ℓ_{i-1}"); a.dangle(n, "E", "ℓ_i"); a.phys(n, "σ_i")
    b = TN(); m = b.liso("A", 0, dag=True)
    b.dangle(m, "W", "ℓ_{i-1}"); b.dangle(m, "E", "ℓ_i"); b.phys(m, "σ_i")
    return Figure(a, Txt("↦"), b,
                  caption="Conjugation: mirror the shape, filled → hollow, reverse "
                          "every arrow. For a shape symmetric under the mirror the "
                          "fill and the arrows carry it alone.")


def ortho_figure():
    a = TN(anchor_row=0.5)
    x = a.liso("A", 0); xd = a.liso("A", 0, row=1, dag=True)
    a.link(x, "N", xd, "S", label="σ_i")
    a.dangle(x, "W", "ℓ_{i-1}"); a.dangle(xd, "W", "ℓ'_{i-1}")
    a.dangle(x, "E", "ℓ_i"); a.dangle(xd, "E", "ℓ'_i")
    b = TN(anchor_row=0.5)
    b.wire(0, label_l="ℓ_i", label_r="ℓ'_i", length=26)
    return Figure(a, Txt("="), b,
                  caption="A†A = 1: the contracted pair collapses to a bare line.")


# ======================================================== how it was built ====

def construction_panels(n=5):
    """How the MPS and the MPO are CONSTRUCTED -- the part that generalises to trees."""
    out = []

    # ---- MPS: the state as one tensor ---------------------------------------
    t = TN()
    psi = t.node("Ψ", "site", (n - 1) / 2.0, 0, cols=n)
    for k in range(n):
        t.dangle(psi, f"N@{k-(n-1)/2.0:+.1f}", f"σ_{k+1}")
    out.append(Panel(
        title="the object being factorised",
        code="# psi :: the full state, 2^L amplitudes",
        figure=Figure(t, caption="One tensor of rank L. Every construction below is "
                                 "a way of never writing this down."),
        note="The MPS is not assembled from parts — it is this object, split."))

    # ---- MPS: one split ------------------------------------------------------
    a = TN()
    psi = a.node("Ψ", "site", (n - 1) / 2.0, 0, cols=n)
    for k in range(n):
        a.dangle(psi, f"N@{k-(n-1)/2.0:+.1f}", f"σ_{k+1}")
    b = TN()
    a1 = b.liso("A_1", 0, hi=True)
    rest = b.node("Ψ'", "site", 1 + (n - 2) / 2.0, 0, cols=n - 1, hi=True)
    b.link(a1, "E", rest, "W", label="ℓ_1")
    b.phys(a1, "σ_1")
    for k in range(n - 1):
        b.dangle(rest, f"N@{k-(n-2)/2.0:+.1f}", f"σ_{k+2}")
    out.append(Panel(
        code="U, S, Vd = svd(psi, (1,))      # split site 1 off the rest\n"
             "A_1 = U;  psi = S * Vd",
        figure=Figure(a, Txt("="), b,
                      caption="One SVD across the cut (σ₁ | σ₂…σ_L). U is a left "
                              "isometry by construction — hence the triangle."),
        note="The rank kept at the cut <em>is</em> the bond dimension ℓ₁."))

    # ---- MPS: the recursion terminates ---------------------------------------
    t, _ = mps(n, liso=tuple(range(1, n)), centre=n,
               names=lambda k, sh: (f"A_{k}" if sh == "liso" else f"C_{n}"))
    out.append(Panel(
        code="for i in 1:L-1;  U, S, Vd = svd(psi, (1, 2));  A_i = U;  psi = S*Vd;  end",
        figure=Figure(t, caption="Repeating the split along the chain leaves left "
                                 "isometries behind and carries the centre right."),
        note="<b>This is the part that generalises.</b> Nothing above says "
             "<em>chain</em>: the recursion is “cut an edge, SVD across it, keep the "
             "isometry, recurse on what is left”. On a tree the same three steps run "
             "over the edges of the tree, leaves inward, and the centre lands on a "
             "node instead of a site. The shapes carry over unchanged — a triangle "
             "still means “isometric toward the centre”, there are just more "
             "directions for it to point away from."))

    # ---- MPS: the product state ---------------------------------------------
    t, _ = mps(n, bond1=True, names=lambda k, sh: f"M_{k}")
    out.append(Panel(
        title="the other construction",
        code="psi = neel_state(L)          # or any product state",
        figure=Figure(t, caption="Every bond is dimension 1. No SVD needed — but "
                                 "also no entanglement, and a 1-site sweep can never "
                                 "leave it."),
        note="This is the start state the ground-state search and the cold-start "
             "time evolutions use, and it is exactly why they need CBE: at bond "
             "dimension 1 there is nothing for a rank-preserving update to work with."))

    # ---- MPO: one term -------------------------------------------------------
    t, _ = term_pair()
    out.append(Panel(
        title="the Hamiltonian, as a term list",
        code="XXZTerm(q.Sp, q.Sp', J)      # one nearest-neighbour term\n"
             "H = Σ_i Σ_t  c_t · O^L_t(i) · O^R_t(i+1)",
        figure=Figure(t, caption="Two local operators on adjacent sites, joined by "
                                 "an op-leg."),
        note="Under U(1) the halves are rank 3 and the op-leg carries the ±2 charge "
             "that makes the pair symmetry-allowed; contracting it <em>is</em> the "
             "charge conservation. Under SU(2) there is a single term S·S, because "
             "no one of S⁺, S<sup>z</sup>, S⁻ is an SU(2) tensor by itself."))

    # ---- MPO: fused gate, and why not -----------------------------------------
    a, _ = term_pair()
    b, _ = gate_block()
    out.append(Panel(
        code="bond_gate(h, i)              # the same term list, fused",
        figure=Figure(a, Txt("⟶"), b,
                      caption="Fusing the two halves gives the two-site gate the "
                              "Trotter path applies."),
        note="The fusion is one-way, and that is the whole reason the Hamiltonian is "
             "stored as a term list: <b>the CBE sketch has to split a bond term "
             "across the two halves of the bond</b>. Once fused there is nothing "
             "left to split."))

    # ---- MPO: the channel automaton ------------------------------------------
    for label, code, mk in _channel_moves():
        out.append(Panel(title=label, code=code, figure=mk()))
    return out


def _channel_moves():
    """The three moves of `push_left_channels`, drawn one at a time."""

    def transport():
        p = TN(anchor_row=0.5, bra=(1,))
        new = env2(p, "id_{i+1}", 0, hi=True)
        p.dangle(new, "E:+0.5", "ℓ'_i", free_dir="E", lab_side="N")
        p.dangle(new, "E:-0.5", "ℓ_i", free_dir="E", lab_side="S")
        u = TN(anchor_row=0.5, bra=(1,))
        old = env2(u, "id_i", 0)
        k = u.liso("A_i", 1, 0)
        b = u.liso("A_i", 1, 1, dag=True)
        u.link(old, "E:-0.5", k, "W")
        u.link(old, "E:+0.5", b, "W")
        u.link(k, "N", b, "S", label="σ_i")
        u.dangle(k, "E", "ℓ_i"); u.dangle(b, "E", "ℓ'_i")
        return Figure(p, Txt("="), u,
                      caption="No operator inserted: the identity is transported "
                              "through the site.")

    def open_():
        p = TN(anchor_row=1, bra=(2,))
        new = env_L(p, "open_{i+1}[t]", 0)
        p.dangle(new, "E:+1", "ℓ'_i", free_dir="E", lab_side="N")
        p.dangle(new, "E", "op", free_dir="E", lab_side="N")
        p.dangle(new, "E:-1", "ℓ_i", free_dir="E", lab_side="S")
        u = TN(anchor_row=1, bra=(2,))
        old = env2(u, "id_i", 0, row=1.0)
        k = u.liso("A_i", 1, 0)
        w = u.op("O^L_t", 1, 1)
        b = u.liso("A_i", 1, 2, dag=True)
        u.link(old, "E:-0.5", k, "W")
        u.link(old, "E:+0.5", b, "W")
        u.link(k, "N", w, "S", label="σ_i")
        u.link(w, "N", b, "S", label="σ'_i")
        u.dangle(w, "E", "op")
        u.dangle(k, "E", "ℓ_i"); u.dangle(b, "E", "ℓ'_i")
        return Figure(p, Txt("="), u,
                      caption="A term's LEFT operator is placed; the op-leg is left "
                              "hanging, waiting for its partner on the next site.")

    def close():
        p = TN(anchor_row=0.5, bra=(1,))
        new = env2(p, "done_{i+1}", 0, hi=True)
        p.dangle(new, "E:+0.5", "ℓ'_i", free_dir="E", lab_side="N")
        p.dangle(new, "E:-0.5", "ℓ_i", free_dir="E", lab_side="S")
        u = TN(anchor_row=0.5, bra=(1,))
        old = env2(u, "done_i", 0)
        k = u.liso("A_i", 1, 0)
        b = u.liso("A_i", 1, 1, dag=True)
        u.link(old, "E:-0.5", k, "W")
        u.link(old, "E:+0.5", b, "W")
        u.link(k, "N", b, "S", label="σ_i")
        u.dangle(k, "E", "ℓ_i"); u.dangle(b, "E", "ℓ'_i")
        v = TN(anchor_row=1, bra=(2,))
        oldo = env_L(v, "open_i[t]", 0)
        k2 = v.liso("A_i", 1, 0)
        w2 = v.op("O^R_t", 1, 1)
        b2 = v.liso("A_i", 1, 2, dag=True)
        v.link(oldo, "E:-1", k2, "W")
        v.link(oldo, "E", w2, "W", label="op")
        v.link(oldo, "E:+1", b2, "W")
        v.link(k2, "N", w2, "S", label="σ_i")
        v.link(w2, "N", b2, "S", label="σ'_i")
        v.dangle(k2, "E", "ℓ_i"); v.dangle(b2, "E", "ℓ'_i")
        return Figure(p, Txt("="), u, Txt("+  Σ_t c_t"), v,
                      caption="A completed term is retired into `done` — and the "
                              "coefficient c_t is applied here, at the moment of "
                              "closing, exactly once per term.")

    return [
        ("channel move 1 of 3 — TRANSPORT",
         "ChannelSet(_left_step(cs.id, A, nothing, i), ...)", transport),
        ("channel move 2 of 3 — OPEN",
         "nxt = [_left_step(cs.id, A, terms[t].left, i) for t in eachindex(terms)]",
         open_),
        ("channel move 3 of 3 — CLOSE",
         "acc = _accum(acc, term.coeff * _left_step(cs.open[t], A, term.right, i))",
         close),
    ]


# ============================================================== projectors ====
#
# von Delft's construction (Gleis / Li / von Delft, controlled bond expansion).
# At site i the fused local space (link_{i-1} x sigma_i) has dimension D*d and
# splits in two:
#
#     A_i   spans the KEPT space       (light, empty)   A' A = 1
#     At_i  spans the DISCARDED space  (shaded)         At' At = 1,  A' At = 0
#     A A' + At At' = 1                                 completeness
#
# Every projector below is assembled from those two pieces and nothing else.

def _proj(name, dagname=None, space="kept", shape="liso", hi=False,
          ket=("ℓ_{i-1}", "σ_i"), bra=("ℓ'_{i-1}", "σ'_i"), bond="D_i"):
    """P = X X' drawn horizontally: the two flat edges face each other.

    The conjugate is mirrored HORIZONTALLY here (`mirror='h'`) because it sits
    beside its ket, not above it.
    """
    t = TN()
    a = t.node(name, shape, 0, space=space, hi=hi)
    b = t.node(dagname or name, shape, 1, dag=True, mirror="h", space=space, hi=hi)
    t.link(a, "E", b, "W", label=bond)
    t.dangle(a, "W", ket[0]); t.phys(a, ket[1])
    t.dangle(b, "E", bra[0]); t.phys(b, bra[1])
    return t


def _fused_identity(labels=(("ℓ_{i-1}", "ℓ'_{i-1}"), ("σ_i", "σ'_i"))):
    """The identity on the fused space: one bare wire per index."""
    t = TN(anchor_row=0.5)
    for k, (a, b) in enumerate(labels):
        t.wire(0, row=1 - k, label_l=a, label_r=b, length=26)
    return t


def projector_panels():
    """How the projectors are built, and which one each method uses."""
    out = []

    # ---- the two halves ------------------------------------------------------
    a = TN(); n = a.liso("A_i", 0, space="kept", sub="kept")
    a.dangle(n, "W", "ℓ_{i-1}"); a.phys(n, "σ_i"); a.dangle(n, "E", "D_i")
    b = TN(); m = b.liso("Ã_i", 0, space="disc", sub="discarded")
    b.dangle(m, "W", "ℓ_{i-1}"); b.phys(m, "σ_i"); b.dangle(m, "E", "D·d − D_i")
    out.append(Panel(
        title="the two halves of the local space",
        code="# fused local space at site i:  (link_{i-1} ⊗ sigma_i),  dim = D·d",
        figure=Figure(a, Txt("⊕"), b,
                      caption="A_i spans the kept space; Ã_i spans everything "
                              "orthogonal to it. Light and empty is kept; shaded "
                              "is discarded."),
        note="Both are isometries — both are triangles. The fill says <em>which "
             "half of the local space</em> the triangle spans, and nothing else. "
             "This is the one distinction every projector below is made of."))

    # ---- the three defining identities ---------------------------------------
    def contracted(x, sp, lab):
        t = TN(anchor_row=0.5)
        k = t.liso(x, 0, space=sp); d = t.liso(x, 0, row=1, dag=True, space=sp)
        t.link(k, "N", d, "S", label="σ_i")
        t.dangle(k, "W", "ℓ_{i-1}"); t.dangle(d, "W", "ℓ_{i-1}")
        t.dangle(k, "E", lab); t.dangle(d, "E", lab + "'")
        return t
    w = TN(anchor_row=0.5); w.wire(0, label_l="D_i", label_r="D_i'", length=26)
    out.append(Panel(
        title="what each triangle asserts",
        code="# A' A = 1        (kept is orthonormal)",
        figure=Figure(contracted("A_i", "kept", "D_i"), Txt("="), w,
                      caption="A†A = 1.")))
    w2 = TN(anchor_row=0.5); w2.wire(0, label_l="D̃", label_r="D̃'", length=26)
    out.append(Panel(
        code="# At' At = 1      (discarded is orthonormal too)",
        figure=Figure(contracted("Ã_i", "disc", "D̃"), Txt("="), w2,
                      caption="Ã†Ã = 1 — the discarded space has its own "
                              "orthonormal basis; it is not a leftover.")))
    mix = TN(anchor_row=0.5)
    k = mix.liso("A_i", 0, space="kept")
    d = mix.liso("Ã_i", 0, row=1, dag=True, space="disc")
    mix.link(k, "N", d, "S", label="σ_i")
    mix.dangle(k, "W", "ℓ_{i-1}"); mix.dangle(d, "W", "ℓ_{i-1}")
    mix.dangle(k, "E", "D_i"); mix.dangle(d, "E", "D̃")
    out.append(Panel(
        code="# A' At = 0       (the two halves are orthogonal)",
        figure=Figure(mix, Txt("=  0"),
                      caption="A†Ã = 0. Light against shaded annihilates.")))

    # ---- the two projectors ---------------------------------------------------
    out.append(Panel(
        title="the kept projector",
        code="P_K = A_i * A_i'",
        figure=Figure(_proj("A_i", space="kept", hi=True),
                      caption="P_K = A A†. The bond leg is contracted; the fused "
                              "legs stay open on both sides."),
        note="This is the projector 1-site TDVP lives on — and the reason it "
             "cannot change a bond dimension. Its image is D_i-dimensional by "
             "construction, whatever the dynamics wants."))
    out.append(Panel(
        title="the discarded projector",
        code="P_D = At_i * At_i'   =   1 - A_i * A_i'",
        figure=Figure(_proj("Ã_i", space="disc", bond="D̃", hi=True),
                      caption="P_D = ÃÃ†. Same construction, shaded — this is the "
                              "space a rank-preserving method never enters."),
        note="In the code this projector is never assembled: "
             "<code>_preselect_left</code> applies it as "
             "<code>P_perp = I − U0 U0'</code>, which is the same operator written "
             "as a subtraction. <b>That subtraction is the only place CBE touches "
             "the discarded space</b>, and everything the expansion returns lives "
             "in its image."))
    out.append(Panel(
        title="completeness",
        code="A_i * A_i'  +  At_i * At_i'  =  1",
        figure=Figure(_proj("A_i", space="kept"), Txt("+"),
                      _proj("Ã_i", space="disc", bond="D̃"), Txt("="),
                      _fused_identity(),
                      caption="Kept plus discarded is the identity on the whole "
                              "fused space — two bare wires, one per index."),
        note="The identity has no triangle in it: once both halves are present "
             "there is nothing left to project onto."))

    # ---- the tangent-space projectors ----------------------------------------
    out.append(Panel(
        title="the 1-site projector",
        code="P1_i = (kept, sites < i)  ⊗  1_i  ⊗  (kept, sites > i)",
        figure=Figure(_block(1, 2, "A"), Txt("⊗"), _site_identity(),
                      Txt("⊗"), _block(4, 5, "B"),
                      caption="Three separate factors — the left block, the bare "
                              "identity on site i, and the right block. Nothing "
                              "runs through site i."),
        note="Left of site i every A is kept, right of it every B is kept, and "
             "site i carries the identity. All light: <b>no shaded triangle "
             "appears anywhere in it</b>, which is the statement that 1-site TDVP "
             "cannot leave the kept space."))
    out.append(Panel(
        title="the 0-site projector",
        code="P0_i = (kept, sites <= i)  ⊗  (kept, sites > i)",
        figure=Figure(_block(1, 3, "A"), Txt("⊗"), _block(4, 5, "B"),
                      caption="Site i has joined the left block; the identity "
                              "factor is gone and only the bond's own legs stay "
                              "open."),
        note="Site i has joined the left block, so the open legs are the bond's "
             "alone. The tangent projector is the alternating sum "
             "P_T = Σ_i P1_i − Σ_{i} P0_i; the subtraction is what the backward "
             "−τ substep implements, and it is there because each bond would "
             "otherwise be counted twice."))

    # ---- the 2-site sectors ---------------------------------------------------
    out.append(Panel(
        title="the two-site space, by sector",
        code="# label each of the two sites by which half of its local space it uses",
        figure=Figure(*_sectors()),
        note="A two-site update reaches all four sectors; that is exactly what it "
             "is paying for. 1-site TDVP reaches KK alone. <b>CBE buys the leading "
             "directions of KD and DK</b> — the sectors where one site steps into "
             "its discarded space — at 1-site cost, which is why the expansion is "
             "sketched against HΘ and then projected with P_D."))
    return out


def _block(lo, hi, kind="A"):
    """A contracted comb of kept isometries over sites lo..hi, ket row + bra row.

    Open legs: the outer link on each row. This is P_K over a block of sites.
    """
    t = TN(anchor_row=0.5)
    ket, bra = [], []
    sh = "liso" if kind == "A" else "riso"
    for k in range(lo, hi + 1):
        a = t.node(f"{kind}_{k}", sh, k - lo, 0, space="kept")
        b = t.node(f"{kind}_{k}", sh, k - lo, 1, dag=True, space="kept")
        t.link(a, "N", b, "S")          # the physical index, contracted
        ket.append(a); bra.append(b)
    for k in range(len(ket) - 1):
        t.link(ket[k], "E", ket[k + 1], "W")
        t.link(bra[k], "E", bra[k + 1], "W")
    if kind == "A":
        t.dangle(ket[0], "W", None, length=18)
        t.dangle(bra[0], "W", None, length=18)
        t.dangle(ket[-1], "E", "ℓ", length=22)
        t.dangle(bra[-1], "E", "ℓ'", length=22)
    else:
        t.dangle(ket[0], "W", "ℓ", length=22)
        t.dangle(bra[0], "W", "ℓ'", length=22)
        t.dangle(ket[-1], "E", None, length=18)
        t.dangle(bra[-1], "E", None, length=18)
    return t


def _site_identity(lab="σ_i", labp="σ'_i"):
    """The identity on one physical index: a bare vertical wire."""
    t = TN(anchor_row=0.5)
    p = t.node("", "none", 0, 0.5)
    t.dangle(p, "N", labp, length=29)
    t.dangle(p, "S", lab, length=29, arrow=False)   # one line, one arrow
    return t


def _sectors():
    """KK / KD / DK / DD as four little two-site pictures."""
    items = []
    for tag, sl, sr in (("KK", "kept", "kept"), ("KD", "kept", "disc"),
                        ("DK", "disc", "kept"), ("DD", "disc", "disc")):
        t = TN()
        a = t.liso("A_i", 0, space=sl, sub=tag)
        b = t.riso("B_{i+1}", 1, space=sr)
        t.link(a, "E", b, "W")
        t.dangle(a, "W", None); t.phys(a, None)
        t.phys(b, None); t.dangle(b, "E", None)
        items.append(t)
        if tag != "DD":
            items.append(Txt("|", size=15))
    return items
