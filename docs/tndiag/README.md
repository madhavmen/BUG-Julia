# tndiag — tensor-network diagrams generated from code

A small pure-stdlib Python library for drawing the tensor networks that a piece of
code actually builds, one diagram per line, in Jan von Delft's diagrammatic style.

```
python3 tdvp1_test.py      # -> out/tdvp1.html  +  out/*.svg
```

No dependencies, no LaTeX toolchain.

## Conventions

Graphical conventions follow von Delft / QSpace ([arXiv:2405.06632](https://arxiv.org/abs/2405.06632)):

- a tensor is a blob with one leg per index;
- **every line carries an arrow** — "arrows specify incoming/outgoing legs or,
  equivalently, raised/lowered indices"; ket-like indices are outgoing;
- a contracted line has no open ends; dangling legs are free indices;
- conjugation = mirror the shape, flip filled ↔ hollow, reverse every arrow.

Shapes are this project's house rules on top of that:

| shape | meaning | constructor |
|---|---|---|
| square | site tensor | `tn.site(...)` |
| circle | bond tensor | `tn.bond(...)` |
| left triangle | left isometry, `A†A = 1` | `tn.liso(...)` |
| right triangle | right isometry, `BB† = 1` | `tn.riso(...)` |
| wide box | MPO / operator | `tn.op(...)` |
| rounded slab | environment block | `tn.env(...)` |
| diamond | scalar | `tn.scalar(...)` |
| light/empty fill | the **kept** space | `space="kept"` |
| shaded fill | the **discarded** space | `space="disc"` |
| (nothing) | bare wire / identity | `tn.wire(...)` |

Triangles point **away** from the orthogonality centre. The other common
convention — apex toward the centre — is one line:
`ISO_APEX_TOWARD_CENTER = True`.

## Arrows are derived, not declared

You never specify arrow directions. They follow from the diagram:

- horizontal legs on a ket row point **+x**, on a bra row point **−x**;
- vertical legs point **up** — from ket, through the MPO, into the bra.

A row counts as a bra row when it holds a conjugated (`dag=True`) tensor, or when
it is declared: `TN(bra=(2,))`, which is needed when a bra row carries only open
legs (an effective Hamiltonian has no bra tensor drawn). Override one leg with
`rev=True`; drop its head with `arrow=False`.

This is deliberate: hand-specified arrows drifted out of agreement across figures
during development, and the derived rule fixed every one of them at once.

## Writing a diagram

Nodes sit on an integer `(col, row)` grid; `row` increases upward. Ports are
`"N" "S" "E" "W"`, with a row offset for tall slabs: `"E:+1"`, `"W:-1"`.

```python
from tndiag import TN, Txt, Figure, Panel, Section, Page, unroll

a = TN()
n = a.site("C_i", 0, hi=True)                 # hi=True: what this line builds
a.dangle(n, "W", "ℓ_{i-1}"); a.dangle(n, "E", "ℓ_i"); a.phys(n, "σ_i")

b = TN()
q = b.liso("A_i", 0); c = b.bond("C", 1)
b.link(q, "E", c, "W", label="b_i")
b.dangle(q, "W", "ℓ_{i-1}"); b.phys(q, "σ_i"); b.dangle(c, "E", "ℓ_i")

Panel(code="Q, C = qr(psi[i], (1, 2))", figure=Figure(a, Txt("="), b))
```

`Figure` lays items out left to right, aligned on each `TN`'s `anchor_row` — use a
fractional `anchor_row=0.5` to centre a two-row stack against a one-row diagram.

Labels accept `A_i^{s_i}`-style sub/superscripts. Annotate groupings with
`tn.brace(...)` and `tn.box(...)`.

## Loops

`unroll(n, fn, start=1)` repeats a loop body with the index substituted, returning
a flat list of panels. `tdvp1_test.py` draws iterations 1 and 2 in full and
abbreviates the third to the state change.

## Output

`Page.save(path)` writes a self-contained HTML page (light and dark themes).
`Figure.save(path)` writes one standalone SVG with styles inlined.

**LaTeX:** `pdflatex` cannot `\includegraphics` an SVG — the figures need an
SVG→PDF conversion first (`rsvg-convert`, Inkscape, or `svg` package with
`--shell-escape`). None of those is installed on this cluster, and there is no
LaTeX either. A TikZ emitter alongside `Figure.to_svg()` would remove the step
entirely; nothing in the node/leg model depends on the backend.

## Files

- `TEMPLATE.md` — **the recipe**: read this before diagramming a new algorithm
- `tndiag.py` — the drawing library (shapes, legs, SVG, HTML pages)
- `tnlib.py` — the shared vocabulary: MPS, MPO, environments, bond frames,
  effective Hamiltonians, and the construction panels
- `tdvp1_test.py` — the worked example: a 1-site TDVP sweep, line by line
- `rsvdcbe_diagrams.py` — the RSVD-CBE bond update and its four wrappers
  (Lubich BUG sweep, gate sweep, 1-site TDVP+CBE, ground-state search)
- `out/` — generated; rebuild with `python3 tdvp1_test.py && python3 rsvdcbe_diagrams.py`

## Adding a page

Write it against `tnlib`, follow the six-part skeleton in `TEMPLATE.md`, and
verify by rasterising before publishing — there is no SVG viewer, LaTeX or
Inkscape on this cluster.
