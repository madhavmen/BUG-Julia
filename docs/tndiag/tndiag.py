"""
tndiag -- tensor-network diagrams generated from code, in the von Delft style.

Pure stdlib. Emits standalone SVG (drop into LaTeX with \\includegraphics, or into
HTML). No LaTeX toolchain required.

--------------------------------------------------------------------------------
CONVENTIONS (von Delft / QSpace, arXiv:2405.06632)
--------------------------------------------------------------------------------
  * A tensor is a blob with one leg per index.
  * "Every line in a tensor network carries an arrow: from the point of view of a
    tensor, arrows specify incoming/outgoing legs or, equivalently, raised/lowered
    indices."  Ket-like indices are OUTGOING.
  * "A contracted line has no open ends."  Dangling legs are free indices.
  * Conjugation = mirror the shape, flip filled <-> empty, reverse every arrow.

SHAPES (this project's house rules, on top of the above)
  square         site tensor A_i                     shape="site"
  circle         bond tensor C, S, Theta core        shape="bond"
  left triangle  left  isometry / left-normalized    shape="liso"
  right triangle right isometry / right-normalized   shape="riso"
  wide box       operator / MPO tensor               shape="op"
  rounded slab   environment block (multi-row)       shape="env"
  diamond        scalar                              shape="scalar"

Flip ISO_APEX_TOWARD_CENTER to True to get the other common convention, in which a
triangle's apex points toward the orthogonality centre (left-normalized => apex
right).  One line, everything else is unchanged.
"""

from __future__ import annotations

import html
import math
import os
import re

# ------------------------------------------------------------------ geometry --

COLW = 78.0     # horizontal grid pitch
ROWH = 58.0     # vertical grid pitch
HALF = 17.0     # half-size of a node
LEG = 27.0      # dangling-leg length
GAP = 26.0      # gap between items in a Figure row

ISO_APEX_TOWARD_CENTER = False   # see module docstring

# ---------------------------------------------------------------- math text --

_SUP = "^"
_SUB = "_"


def _mathtext(s, x, y, cls="tn-lab", anchor="middle", size=13.0):
    """Render `A_i^{s_i}`-style labels as <text> with tspan sub/superscripts."""
    if s is None:
        return ""
    parts, i = [], 0
    while i < len(s):
        c = s[i]
        if c in (_SUB, _SUP) and i + 1 < len(s):
            kind = "sub" if c == _SUB else "sup"
            i += 1
            if s[i] == "{":
                j = s.index("}", i)
                body, i = s[i + 1:j], j + 1
            else:
                body, i = s[i], i + 1
            parts.append((kind, body))
        else:
            parts.append(("base", c))
            i += 1
    merged = []
    for kind, body in parts:
        if kind == "base" and merged and merged[-1][0] == "base":
            merged[-1] = ("base", merged[-1][1] + body)
        else:
            merged.append((kind, body))
    out = [f'<text class="{cls}" x="{x:.1f}" y="{y:.1f}" '
           f'text-anchor="{anchor}" font-size="{size:.1f}">']
    for kind, body in merged:
        b = html.escape(body)
        if kind == "base":
            out.append(f"<tspan>{b}</tspan>")
        elif kind == "sub":
            out.append(f'<tspan font-size="{size*0.72:.1f}" dy="{size*0.26:.1f}">'
                       f'{b}</tspan><tspan dy="{-size*0.26:.1f}"></tspan>')
        else:
            out.append(f'<tspan font-size="{size*0.72:.1f}" dy="{-size*0.42:.1f}">'
                       f'{b}</tspan><tspan dy="{size*0.42:.1f}"></tspan>')
    out.append("</text>")
    return "".join(out)


def _textwidth(s, size=13.0):
    return 0.58 * size * len(re.sub(r"[_^{}]", "", s or ""))


# --------------------------------------------------------------------- nodes --

class Node:
    """One tensor. Placed on an integer grid; `rows` > 1 makes a tall slab."""

    def __init__(self, tn, name, shape, col, row, rows=1, cols=1, dag=False,
                 hi=False, label=None, sub=None, space=None, mirror='v'):
        self.tn = tn
        self.name = name
        self.shape = shape
        self.col = col
        self.row = row
        self.rows = rows
        self.cols = cols         # >1 spans several grid columns (a whole-state blob)
        self.dag = dag           # conjugate: mirrored + open fill + arrows reversed
        self.hi = hi             # highlight: this is what the code line builds
        self.label = name if label is None else label
        self.sub = sub           # small caption under the node
        # Which half of the local space this isometry spans. von Delft's gray-white
        # filling: 'kept' -> light/empty, 'disc' -> shaded.
        self.space = space
        # Which way a conjugate is mirrored. 'v' (default) suits a bra drawn ABOVE
        # its ket, as in an environment; 'h' suits a bra drawn BESIDE it, as in a
        # projector A A-dagger, where the two flat edges must face each other.
        self.mirror = mirror

    @property
    def x(self):
        return self.col * COLW

    @property
    def y(self):
        return -self.row * ROWH

    def _apex_left(self):
        left_iso = self.shape == "liso"
        if ISO_APEX_TOWARD_CENTER:
            left_iso = not left_iso
        if self.dag and self.mirror == "h":
            left_iso = not left_iso
        return left_iso

    def half_h(self):
        if self.shape == "none":
            return 0.0
        if self.rows > 1:
            return (self.rows - 1) * ROWH / 2.0 + HALF
        return HALF

    def half_w(self):
        if self.shape == "none":
            return 0.0
        base = HALF * (1.55 if self.shape == "op" else
                       0.95 if self.shape == "env" else 1.0)
        return base + (self.cols - 1) * COLW / 2.0

    def port(self, spec):
        """Port spec: side, optional ':<dr>' row offset, optional '@<dc>' column offset.

        'E'      centre of the east edge
        'E:+1'   one row up   -- the bra leg of a tall environment slab
        'N@-2'   two columns left along the north edge -- a leg of a wide blob
        """
        num = r"[+-]?\d+(?:\.\d+)?"
        m = re.match(rf"^([NSEW])(?::({num}))?(?:@({num}))?$", spec)
        if not m:
            raise ValueError(f"bad port {spec!r}")
        side = m.group(1)
        off = float(m.group(2) or 0)      # fractional offsets place a 2-leg block
        dc = float(m.group(3) or 0)
        dy = -off * ROWH
        dx = dc * COLW
        hw, hh = self.half_w(), self.half_h()
        x, y = self.x + dx, self.y + dy

        if self.shape == "none":
            return (x, y)
        if self.shape in ("liso", "riso") and side in ("N", "S"):
            # exact point where the slanted edge crosses the node's vertical axis
            return (x, self.y + (-HALF / 2 if side == "N" else HALF / 2))
        if self.shape == "scalar" and side in ("N", "S"):
            return (x, self.y + (-HALF if side == "N" else HALF))
        if self.shape == "bond" and self.cols > 1 and side in ("N", "S"):
            # land the leg ON the ellipse, not on its bounding box
            k = max(0.0, 1.0 - (dx / hw) ** 2) ** 0.5
            return (x, self.y + (-HALF * k if side == "N" else HALF * k))

        if side == "W":
            return (self.x - hw, y)
        if side == "E":
            return (self.x + hw, y)
        if side == "N":
            return (x, self.y - hh)
        return (x, self.y + hh)

    def svg(self):
        if self.shape == "none":
            return ""
        cls = {
            "site": "tn-site", "bond": "tn-bond", "liso": "tn-iso",
            "riso": "tn-iso", "op": "tn-op", "env": "tn-env",
            "scalar": "tn-scalar",
        }.get(self.shape, "tn-site")
        if self.dag:
            cls += " tn-dag"
        if self.space:
            cls += " tn-kept" if self.space == "kept" else " tn-disc"
        if self.hi:
            cls += " tn-hi"
        x, y = self.x, self.y
        hw, hh = self.half_w(), self.half_h()
        p = []

        if self.shape == "bond":
            if self.cols > 1:            # a block spanning several sites: an ellipse
                p.append(f'<ellipse class="{cls}" cx="{x:.1f}" cy="{y:.1f}" '
                         f'rx="{hw:.1f}" ry="{HALF:.1f}"/>')
            else:
                p.append(f'<circle class="{cls}" cx="{x:.1f}" cy="{y:.1f}" '
                         f'r="{HALF:.1f}"/>')
        elif self.shape in ("liso", "riso"):
            left = self._apex_left()           # already honours a horizontal mirror
            if self.dag and self.mirror == "v":   # mirror image
                pts = ([(x - HALF, y), (x + HALF, y + HALF), (x + HALF, y - HALF)]
                       if left else
                       [(x + HALF, y), (x - HALF, y + HALF), (x - HALF, y - HALF)])
            else:
                pts = ([(x - HALF, y), (x + HALF, y - HALF), (x + HALF, y + HALF)]
                       if left else
                       [(x + HALF, y), (x - HALF, y - HALF), (x - HALF, y + HALF)])
            d = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
            p.append(f'<polygon class="{cls}" points="{d}"/>')
        elif self.shape == "scalar":
            pts = [(x, y - HALF), (x + HALF, y), (x, y + HALF), (x - HALF, y)]
            d = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
            p.append(f'<polygon class="{cls}" points="{d}"/>')
        else:
            r = 9.0 if self.shape == "env" else (4.0 if self.shape == "op" else 2.5)
            p.append(f'<rect class="{cls}" x="{x-hw:.1f}" y="{y-hh:.1f}" '
                     f'width="{2*hw:.1f}" height="{2*hh:.1f}" rx="{r:.1f}"/>')

        lx = x
        if self.shape in ("liso", "riso"):
            lx = x + (HALF * 0.30 if self._apex_left() else -HALF * 0.30)
        lab = self.label + ("†" if self.dag else "")
        p.append(_mathtext(lab, lx, y + 4.6, cls="tn-name", size=13.5))
        if self.sub:
            p.append(_mathtext(self.sub, x, y + hh + 15, cls="tn-sub", size=10.5))
        return "".join(p)


# --------------------------------------------------------------------- legs ---

class Leg:
    """A line. Either a contraction (two endpoints) or a dangling free index.

    The ARROW is derived from the diagram rather than declared per leg:
      * horizontal legs on a ket row point +x, on a bra row point -x
      * vertical legs point up, i.e. from ket, through the MPO, into the bra
    A row counts as a bra row when it holds a conjugated (dag=True) tensor.
    Pass `rev=True` to flip one leg, or `arrow=False` to suppress the head.
    """

    def __init__(self, tn, a, aport, b=None, bport=None, label=None,
                 rev=False, arrow=True, free_dir=None, length=LEG, hi=False,
                 lab_side=None, elbow=None):
        self.tn, self.a, self.aport = tn, a, aport
        self.b, self.bport = b, bport
        self.label = label
        self.rev = rev               # flip the automatic arrow direction
        self.arrow = arrow
        self.free_dir = free_dir     # 'N','S','E','W' for a dangling leg
        self.length = length
        self.hi = hi
        self.lab_side = lab_side
        self.elbow = elbow           # 'h' -> horizontal first, 'v' -> vertical first

    def points(self):
        p0 = self.a.port(self.aport)
        if self.b is not None:
            p1 = self.b.port(self.bport)
            if abs(p0[0] - p1[0]) < 0.5 or abs(p0[1] - p1[1]) < 0.5:
                return [p0, p1]
            mode = self.elbow or ("h" if self.aport[0] in ("E", "W") else "v")
            mid = (p1[0], p0[1]) if mode == "h" else (p0[0], p1[1])
            return [p0, mid, p1]
        d = self.free_dir or self.aport[0]
        dx, dy = {"N": (0, -1), "S": (0, 1), "E": (1, 0), "W": (-1, 0)}[d]
        return [p0, (p0[0] + dx * self.length, p0[1] + dy * self.length)]

    def svg(self):
        pts = self.points()
        cls = "tn-leg" + (" tn-hi-leg" if self.hi else "")
        d = " ".join(f"{x:.1f},{y:.1f}" for x, y in pts)
        out = [f'<polyline class="{cls}" points="{d}"/>']

        if self.arrow:
            seg = max(zip(pts, pts[1:]),
                      key=lambda s: math.hypot(s[1][0] - s[0][0], s[1][1] - s[0][1]))
            (x0, y0), (x1, y1) = seg
            ax, ay = (x0 + x1) / 2, (y0 + y1) / 2
            if abs(x1 - x0) >= abs(y1 - y0):
                row = round(-((y0 + y1) / 2) / ROWH)
                sx = -1.0 if row in self.tn.bra_rows() else 1.0
                ang = 0.0 if sx > 0 else math.pi
            else:
                ang = -math.pi / 2                     # vertical legs point up
            if self.rev:
                ang += math.pi
            out.append(self._head(ax, ay, ang))

        if self.label:
            if self.b is None and self.lab_side is None:
                ex, ey = pts[-1]
                dirn = self.free_dir or self.aport[0]
                off = {"N": (0, -7), "S": (0, 14), "E": (9, 4.5), "W": (-9, 4.5)}[dirn]
                anc = {"N": "middle", "S": "middle", "E": "start", "W": "end"}[dirn]
                out.append(_mathtext(self.label, ex + off[0], ey + off[1],
                                     anchor=anc, size=11.5))
            elif self.b is None:
                # explicit side: hang the label off the middle of the leg instead
                mx = (pts[0][0] + pts[-1][0]) / 2
                my = (pts[0][1] + pts[-1][1]) / 2
                off = {"N": (0, -7), "S": (0, 15), "E": (9, 4), "W": (-9, 4)}[self.lab_side]
                anc = {"N": "middle", "S": "middle",
                       "E": "start", "W": "end"}[self.lab_side]
                out.append(_mathtext(self.label, mx + off[0], my + off[1],
                                     anchor=anc, size=11.5))
            else:
                mx = (pts[0][0] + pts[-1][0]) / 2
                my = (pts[0][1] + pts[-1][1]) / 2
                horiz = abs(pts[0][1] - pts[-1][1]) < 0.5
                side = self.lab_side or ("N" if horiz else "E")
                off = {"N": (0, -8), "S": (0, 16), "E": (13, 4), "W": (-13, 4)}[side]
                anc = "middle" if side in "NS" else ("start" if side == "E" else "end")
                out.append(_mathtext(self.label, mx + off[0], my + off[1],
                                     anchor=anc, size=11.5))
        return "".join(out)

    @staticmethod
    def _head(x, y, ang, s=5.4):
        pts = [(x + s * math.cos(ang), y + s * math.sin(ang)),
               (x - s * 0.75 * math.cos(ang) + s * 0.62 * math.sin(ang),
                y - s * 0.75 * math.sin(ang) - s * 0.62 * math.cos(ang)),
               (x - s * 0.75 * math.cos(ang) - s * 0.62 * math.sin(ang),
                y - s * 0.75 * math.sin(ang) + s * 0.62 * math.cos(ang))]
        d = " ".join(f"{a:.1f},{b:.1f}" for a, b in pts)
        return f'<polygon class="tn-arrow" points="{d}"/>'


# ----------------------------------------------------------------- diagrams ---

class TN:
    """One tensor network. Nodes live on an integer (col, row) grid."""

    def __init__(self, anchor_row=0, caption=None, bra=()):
        self.nodes, self.legs, self.marks = [], [], []
        self.anchor_row = anchor_row      # may be fractional, to centre a stack
        self.caption = caption
        self.bra = set(bra)               # rows that are bra rows by declaration

    def bra_rows(self):
        """Rows whose arrows run leftward: declared, or holding a conjugate.

        Declaration is needed when a bra row carries only open legs and so has
        no dag tensor to infer from -- e.g. an effective Hamiltonian.
        """
        # Only a VERTICALLY mirrored conjugate marks a bra row. A horizontally
        # mirrored one (a projector's A-dagger, drawn beside its ket) shares the
        # ket's row, and the arrows there flow straight through.
        return self.bra | {n.row for n in self.nodes if n.dag and n.mirror == "v"}

    def wire(self, col, row=0, label_l=None, label_r=None, length=LEG):
        """A bare line -- what a contracted isometry pair collapses to."""
        p = self.node("", "none", col, row)
        self.dangle(p, "W", label_l, length=length, arrow=False)
        self.dangle(p, "E", label_r, length=length)
        return p

    def node(self, name, shape, col, row=0, **kw):
        n = Node(self, name, shape, col, row, **kw)
        self.nodes.append(n)
        return n

    def site(self, name, col, row=0, **kw):
        return self.node(name, "site", col, row, **kw)

    def bond(self, name, col, row=0, **kw):
        return self.node(name, "bond", col, row, **kw)

    def liso(self, name, col, row=0, **kw):
        return self.node(name, "liso", col, row, **kw)

    def riso(self, name, col, row=0, **kw):
        return self.node(name, "riso", col, row, **kw)

    def op(self, name, col, row=0, **kw):
        return self.node(name, "op", col, row, **kw)

    def env(self, name, col, row=0, rows=3, **kw):
        return self.node(name, "env", col, row, rows=rows, **kw)

    def scalar(self, name, col, row=0, **kw):
        return self.node(name, "scalar", col, row, **kw)

    def link(self, a, aport, b, bport, label=None, **kw):
        lg = Leg(self, a, aport, b, bport, label=label, **kw)
        self.legs.append(lg)
        return lg

    def dangle(self, a, aport, label=None, **kw):
        lg = Leg(self, a, aport, label=label, **kw)
        self.legs.append(lg)
        return lg

    def phys(self, a, label=None, down=False, **kw):
        """Physical (site) leg. Ket-like => outgoing, drawn upward by default."""
        return self.dangle(a, "S" if down else "N", label, **kw)

    def brace(self, col0, col1, row=0, label=None, above=True):
        self.marks.append(("brace", col0, col1, row, label, above))

    def box(self, col0, col1, row0=0, row1=0, label=None):
        self.marks.append(("box", col0, col1, row0, row1, label))

    def chain(self, specs, col0=0, row=0, links=None, phys=True, phys_down=False):
        """specs: list of (name, shape) or (name, shape, kwargs)."""
        made = []
        for k, sp in enumerate(specs):
            name, shape = sp[0], sp[1]
            kw = sp[2] if len(sp) > 2 else {}
            made.append(self.node(name, shape, col0 + k, row, **kw))
        for k in range(len(made) - 1):
            lab = links[k] if links else None
            self.link(made[k], "E", made[k + 1], "W", label=lab)
        if phys:
            for n in made:
                self.phys(n, down=phys_down)
        return made

    def bbox(self):
        xs, ys = [], []
        for n in self.nodes:
            xs += [n.x - n.half_w(), n.x + n.half_w()]
            ys += [n.y - n.half_h(), n.y + n.half_h()]
            if n.sub:
                ys.append(n.y + n.half_h() + 20)
        for lg in self.legs:
            for x, y in lg.points():
                xs.append(x); ys.append(y)
            if lg.label and lg.b is None:
                w = _textwidth(lg.label, 11.5)
                pp = lg.points()
                if lg.lab_side:
                    ex = (pp[0][0] + pp[-1][0]) / 2
                    ey = (pp[0][1] + pp[-1][1]) / 2
                    d = lg.lab_side
                else:
                    ex, ey = pp[-1]
                    d = lg.free_dir or lg.aport[0]
                if d == "E":
                    xs.append(ex + 10 + w)
                elif d == "W":
                    xs.append(ex - 10 - w)
                elif d == "N":
                    ys.append(ey - 14); xs += [ex - w / 2, ex + w / 2]
                else:
                    ys.append(ey + 20); xs += [ex - w / 2, ex + w / 2]
            elif lg.label:
                pts = lg.points()
                my = (pts[0][1] + pts[-1][1]) / 2
                ys += [my - 14, my + 18]
        for m in self.marks:
            if m[0] == "brace":
                _, c0, c1, row, label, above = m
                y = -row * ROWH
                ys.append(y + (-HALF - 40 if above else HALF + 40))
                xs += [c0 * COLW - HALF, c1 * COLW + HALF]
            else:
                _, c0, c1, r0, r1, label = m
                xs += [c0 * COLW - HALF - 9, c1 * COLW + HALF + 9]
                ys += [-r1 * ROWH - HALF - 22, -r0 * ROWH + HALF + 9]
        if not xs:
            xs, ys = [0.0], [0.0]
        return min(xs) - 3, min(ys) - 3, max(xs) + 3, max(ys) + 3

    def anchor_y(self):
        return -self.anchor_row * ROWH

    def body(self):
        p = []
        for m in self.marks:                      # group boxes behind everything
            if m[0] == "box":
                _, c0, c1, r0, r1, label = m
                x0, x1 = c0 * COLW - HALF - 9, c1 * COLW + HALF + 9
                y0, y1 = -r1 * ROWH - HALF - 9, -r0 * ROWH + HALF + 9
                p.append(f'<rect class="tn-groupbox" x="{x0:.1f}" y="{y0:.1f}" '
                         f'width="{x1-x0:.1f}" height="{y1-y0:.1f}" rx="8"/>')
                if label:
                    p.append(_mathtext(label, (x0 + x1) / 2, y0 - 6,
                                       cls="tn-grouplab", size=11.5))
        for lg in self.legs:
            p.append(lg.svg())
        for n in self.nodes:
            p.append(n.svg())
        for m in self.marks:
            if m[0] == "brace":
                _, c0, c1, row, label, above = m
                y = -row * ROWH + (-HALF - 14 if above else HALF + 14)
                x0, x1 = c0 * COLW - HALF, c1 * COLW + HALF
                k = -7 if above else 7
                p.append(f'<path class="tn-brace" d="M{x0:.1f},{y:.1f} '
                         f'q0,{k:.1f} 9,{k:.1f} L{(x0+x1)/2-9:.1f},{y+k:.1f} '
                         f'q9,0 9,{k:.1f} q0,{-k:.1f} 9,{-k:.1f} '
                         f'L{x1-9:.1f},{y+k:.1f} q9,0 9,{-k:.1f}"/>')
                if label:
                    p.append(_mathtext(label, (x0 + x1) / 2,
                                       y + (3.0 * k) + (0 if above else 9),
                                       cls="tn-grouplab", size=12))
        return "".join(p)


class Txt:
    """A text token inside a Figure row: '=', '+', 'e^{-i tau H}', ..."""

    def __init__(self, s, size=17.0, cls="tn-tok"):
        self.s, self.size, self.cls = s, size, cls

    def width(self):
        return _textwidth(self.s, self.size) + 6

    def height(self):
        return self.size * 1.4


class Figure:
    """A horizontal row of TNs and text tokens, aligned on their anchor rows."""

    def __init__(self, *items, caption=None):
        self.items = list(items)
        self.caption = caption

    def add(self, *items):
        self.items += list(items)
        return self

    def to_svg(self, pad=10.0, scale=1.0):
        placed, cx = [], 0.0
        tops, bots = [], []
        for it in self.items:
            if isinstance(it, Txt):
                placed.append((it, cx, 0.0))
                tops.append(-it.size * 0.55)
                bots.append(it.size * 0.55)
                cx += it.width() + GAP
            else:
                x0, y0, x1, y1 = it.bbox()
                dx = cx - x0
                dy = -it.anchor_y()
                placed.append((it, dx, dy))
                tops.append(y0 + dy)
                bots.append(y1 + dy)
                cx += (x1 - x0) + GAP
        cx -= GAP
        top, bot = min(tops), max(bots)
        W, H = cx + 2 * pad, (bot - top) + 2 * pad
        oy = pad - top

        out = [f'<svg class="tndiag" xmlns="http://www.w3.org/2000/svg" '
               f'viewBox="0 0 {W:.1f} {H:.1f}" width="{W*scale:.0f}" '
               f'height="{H*scale:.0f}" role="img">']
        for it, dx, dy in placed:
            out.append(f'<g transform="translate({dx+pad:.1f},{oy+dy:.1f})">')
            if isinstance(it, Txt):
                out.append(_mathtext(it.s, 0, it.size * 0.36, cls=it.cls,
                                     anchor="start", size=it.size))
            else:
                out.append(it.body())
            out.append("</g>")
        out.append("</svg>")
        return "".join(out)

    def save(self, path, scale=1.0):
        """Standalone .svg, styles inlined so it works outside the HTML page."""
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        svg = self.to_svg(scale=scale)
        svg = svg.replace("<svg ", "<svg ", 1)
        svg = svg.replace(">", f"><style>{_SVG_STYLE}</style>", 1)
        with open(path, "w") as f:
            f.write('<?xml version="1.0" encoding="UTF-8"?>\n')
            f.write("<!-- generated by tndiag -->\n")
            f.write(svg)
        return path


_SVG_STYLE = """
.tn-leg{fill:none;stroke:#20242a;stroke-width:1.5}
.tn-arrow{fill:#20242a}
.tn-hi-leg{stroke:#b23a2c;stroke-width:2.2}
.tn-site,.tn-bond,.tn-iso,.tn-op,.tn-env,.tn-scalar{stroke:#20242a;stroke-width:1.6}
.tn-site{fill:#dbe7f6}.tn-bond{fill:#fbe3c2}.tn-iso{fill:#d8ecd9}
.tn-op{fill:#e6dcf4}.tn-env{fill:#e9edf2}.tn-scalar{fill:#f6dede}
.tn-dag{fill:#ffffff}.tn-kept{fill:#ffffff}.tn-disc{fill:#9fadbb}.tn-hi{stroke:#b23a2c;stroke-width:2.6}
.tn-name{fill:#1a1d21;font-family:Georgia,serif;font-style:italic}
.tn-lab{fill:#5d646e;font-family:Georgia,serif;font-style:italic}
.tn-sub{fill:#5d646e;font-family:sans-serif}
.tn-tok{fill:#1a1d21;font-family:Georgia,serif}
.tn-grouplab{fill:#b23a2c;font-family:Georgia,serif;font-style:italic}
.tn-brace{fill:none;stroke:#b23a2c;stroke-width:1.3}
.tn-groupbox{fill:none;stroke:#b23a2c;stroke-width:1.2;stroke-dasharray:4 3}
"""


# ------------------------------------------------------------------- panels ---

class Panel:
    """One line of code + the diagram(s) that line produces."""

    def __init__(self, code=None, figure=None, note=None, title=None):
        self.code, self.figure, self.note, self.title = code, figure, note, title


class Section:
    def __init__(self, title, blurb=None):
        self.title, self.blurb, self.panels = title, blurb, []

    def add(self, *panels):
        self.panels += list(panels)
        return self


def unroll(n, fn, start=1):
    """Unroll a for-loop body into panels: fn(i) -> Panel | [Panel]."""
    out = []
    for i in range(start, start + n):
        r = fn(i)
        out += r if isinstance(r, list) else [r]
    return out


# --------------------------------------------------------------------- page ---

_CSS = """
:root{
  --bg:#fcfcfd; --fg:#191c21; --muted:#5a626d; --rule:#e0e4ea; --ink:#20242a;
  --site:#dbe7f6; --bond:#fbe3c2; --iso:#d8ecd9; --op:#e6dcf4;
  --env:#e9edf2; --scalar:#f6dede; --kept:#ffffff; --disc:#9fadbb;
  --hi:#b23a2c; --codebg:#f4f6f8; --card:#ffffff;
}
@media (prefers-color-scheme: dark){
  :root{
    --bg:#131619; --fg:#e6eaef; --muted:#949daa; --rule:#282e35; --ink:#dfe4ea;
    --site:#274563; --bond:#6a4a1e; --iso:#2c5334; --op:#43355f;
    --env:#2a3038; --scalar:#5a2f2f; --kept:#0d1013; --disc:#68767f; --hi:#ff8874; --codebg:#191d22; --card:#171b1f;
  }
}
:root[data-theme="dark"]{
  --bg:#131619; --fg:#e6eaef; --muted:#949daa; --rule:#282e35; --ink:#dfe4ea;
  --site:#274563; --bond:#6a4a1e; --iso:#2c5334; --op:#43355f;
  --env:#2a3038; --scalar:#5a2f2f; --kept:#0d1013; --disc:#68767f; --hi:#ff8874; --codebg:#191d22; --card:#171b1f;
}
:root[data-theme="light"]{
  --bg:#fcfcfd; --fg:#191c21; --muted:#5a626d; --rule:#e0e4ea; --ink:#20242a;
  --site:#dbe7f6; --bond:#fbe3c2; --iso:#d8ecd9; --op:#e6dcf4;
  --env:#e9edf2; --scalar:#f6dede; --kept:#ffffff; --disc:#9fadbb; --hi:#b23a2c; --codebg:#f4f6f8; --card:#ffffff;
}

*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
     font:16px/1.62 Charter,"Bitstream Charter","Iowan Old Style","Source Serif Pro",
          Georgia,serif;}
.wrap{max-width:960px;margin:0 auto;padding:40px 22px 96px}
h1{font-size:30px;line-height:1.22;margin:0 0 8px;letter-spacing:-.012em;
   text-wrap:balance;font-weight:600}
h2{font-size:20px;margin:52px 0 4px;padding-top:18px;text-wrap:balance;
   border-top:1px solid var(--rule);font-weight:600;letter-spacing:-.006em}
h3{font-size:16px;margin:26px 0 2px;font-weight:600}
.sub{color:var(--muted);margin:0 0 30px;max-width:64ch;font-size:15px}
.blurb{color:var(--muted);margin:4px 0 18px;font-size:14.5px;max-width:64ch}
.panel{margin:22px 0 30px}
.ptitle{font-size:12px;margin-bottom:8px;color:var(--muted);
        font-family:ui-sans-serif,-apple-system,"Segoe UI",sans-serif;
        text-transform:uppercase;letter-spacing:.09em;font-weight:600}
pre{background:var(--codebg);border:1px solid var(--rule);border-radius:7px;
    padding:9px 12px;overflow-x:auto;margin:0 0 11px;
    font:12.5px/1.55 ui-monospace,SFMono-Regular,Menlo,Consolas,monospace}
.fig{background:var(--card);border:1px solid var(--rule);border-radius:3px;
     padding:16px 14px;overflow-x:auto;text-align:center}
.note{color:var(--muted);font-size:14.5px;margin:10px 0 0;max-width:64ch}
.cap{color:var(--muted);font-size:13px;margin:9px auto 0;text-align:center;
     max-width:62ch;font-style:italic}
svg.tndiag{max-width:100%;height:auto;display:inline-block;color:var(--ink)}

.tndiag .tn-leg{fill:none;stroke:var(--ink);stroke-width:1.5}
.tndiag .tn-arrow{fill:var(--ink);stroke:none}
.tndiag .tn-hi-leg{stroke:var(--hi);stroke-width:2.2}
.tndiag .tn-site,.tndiag .tn-bond,.tndiag .tn-iso,.tndiag .tn-op,
.tndiag .tn-env,.tndiag .tn-scalar{stroke:var(--ink);stroke-width:1.6}
.tndiag .tn-site{fill:var(--site)}
.tndiag .tn-bond{fill:var(--bond)}
.tndiag .tn-iso{fill:var(--iso)}
.tndiag .tn-op{fill:var(--op)}
.tndiag .tn-env{fill:var(--env)}
.tndiag .tn-scalar{fill:var(--scalar)}
.tndiag .tn-dag{fill:var(--bg)}
.tndiag .tn-kept{fill:var(--kept)}
.tndiag .tn-disc{fill:var(--disc)}
.tndiag .tn-hi{stroke:var(--hi);stroke-width:2.6}
.tndiag .tn-name{fill:var(--fg);font-family:Georgia,"Times New Roman",serif;
                 font-style:italic}
.tndiag .tn-lab{fill:var(--muted);font-family:Georgia,serif;font-style:italic}
.tndiag .tn-sub{fill:var(--muted);font-family:-apple-system,sans-serif;
                font-style:normal}
.tndiag .tn-tok{fill:var(--fg);font-family:Georgia,serif}
.tndiag .tn-grouplab{fill:var(--hi);font-family:Georgia,serif;font-style:italic}
.tndiag .tn-brace{fill:none;stroke:var(--hi);stroke-width:1.3}
.tndiag .tn-groupbox{fill:none;stroke:var(--hi);stroke-width:1.2;
                     stroke-dasharray:4 3;opacity:.75}

.legend{display:grid;grid-template-columns:repeat(auto-fit,minmax(178px,1fr));
        gap:10px;margin:14px 0}
.lg{border:1px solid var(--rule);border-radius:8px;padding:9px;background:var(--card);
    text-align:center}
.lg .t{font-size:12.5px;color:var(--muted);margin-top:5px}
table{border-collapse:collapse;width:100%;font-size:13.5px;margin:12px 0}
th,td{border-bottom:1px solid var(--rule);padding:6px 8px;text-align:left;
      vertical-align:top}
th{color:var(--muted);font-weight:600}
code{font:12.5px ui-monospace,Menlo,monospace;background:var(--codebg);
     padding:1px 4px;border-radius:4px}
"""


class Page:
    def __init__(self, title, subtitle=None):
        self.title, self.subtitle, self.blocks = title, subtitle, []

    def add(self, *blocks):
        self.blocks += list(blocks)
        return self

    def html(self, raw_head=""):
        o = [f"<title>{html.escape(self.title)}</title>",
             f"<style>{_CSS}</style>", '<div class="wrap">',
             f"<h1>{html.escape(self.title)}</h1>"]
        if self.subtitle:
            o.append(f'<p class="sub">{self.subtitle}</p>')
        o.append(raw_head)
        for b in self.blocks:
            o.append(self._render(b))
        o.append("</div>")
        return "\n".join(o)

    def _render(self, b):
        if isinstance(b, str):
            return b
        if isinstance(b, Section):
            o = [f"<h2>{html.escape(b.title)}</h2>"]
            if b.blurb:
                o.append(f'<p class="blurb">{b.blurb}</p>')
            o += [self._render(p) for p in b.panels]
            return "\n".join(o)
        if isinstance(b, Panel):
            o = ['<div class="panel">']
            if b.title:
                o.append(f'<div class="ptitle">{html.escape(b.title)}</div>')
            if b.code:
                o.append(f"<pre>{html.escape(b.code)}</pre>")
            if b.figure is not None:
                o.append(f'<div class="fig">{b.figure.to_svg()}</div>')
                if b.figure.caption:
                    o.append(f'<p class="cap">{b.figure.caption}</p>')
            if b.note:
                o.append(f'<p class="note">{b.note}</p>')
            o.append("</div>")
            return "\n".join(o)
        if isinstance(b, Figure):
            return f'<div class="fig">{b.to_svg()}</div>'
        raise TypeError(type(b))

    def save(self, path, raw_head=""):
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "w") as f:
            f.write(self.html(raw_head))
        return path
