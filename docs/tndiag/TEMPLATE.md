# The template: translating code into tensor diagrams

Use this when diagramming a new algorithm. The worked example is
[`tdvp1_test.py`](tdvp1_test.py) (1-site TDVP); the four-algorithm example is
[`rsvdcbe_diagrams.py`](rsvdcbe_diagrams.py). Shared builders live in
[`tnlib.py`](tnlib.py) — reach for those before writing a `TN` by hand.

## The rule

**One code line, one diagram.** If a line changes the state, the diagram is an
equation with the line's left-hand side highlighted. If a line only reads, the
diagram is the object it reads.

## The six-part skeleton

Every page so far has the same spine. Follow it unless there's a reason not to.

| § | content | builder |
|---|---|---|
| 0 | the vocabulary — shapes legend | `T.legend_html()` |
| 1 | **how the objects were constructed** | `T.construction_panels(n)` |
| 2 | canonical form + what a triangle asserts | `T.ortho_figure()`, `T.conj_figure()` |
| 3 | **the projectors, constructed** | `T.projector_panels()` |
| 4 | the environments | `T.env_L/env_R/sandwich` |
| 5 | the algorithm, loops unrolled | `unroll(n, fn)` |
| 6 | the design choices, stated as a table | plain HTML |

Section 1 is not optional. An algorithm page that starts at "here is an MPS"
hides the step that actually generalises: the MPS *is* the full state, split by a
sequence of SVDs, and a tree is the same recursion run over the edges of a tree
instead of along a path. Draw the split once and the tree case needs no new
vocabulary.

Section 3 is not optional either. Every method here *is* a choice of projector, so
the projectors get built explicitly from the two halves of the local space before
any algorithm uses one. The fill carries which half:

| fill | meaning |
|---|---|
| light, empty | the **kept** space — what the current bond dimension spans |
| shaded | the **discarded** space — the orthogonal complement |

`space="kept"` / `space="disc"` on any node. A conjugate drawn *beside* its ket (a
projector `A A†`) needs `mirror="h"` so the two flat edges face each other; the
default `mirror="v"` suits a bra drawn *above* its ket, as in an environment. Only
a vertically mirrored conjugate marks its row as a bra row for arrow purposes.

## Show every line

When diagramming a sweep, paste **every line of the source**, in order, with the
source line numbers in the panel title. Bookkeeping lines (`expanded[i] = ...`,
`peak!(...)`) can share a panel with the tensor line they accompany, but they must
appear. A reader has to be able to lay the page beside the file and account for all
of it — an elided line is exactly where a reader assumes the obvious thing and the
code does something else.

Turn the source's long comments into the panel's `note`. That is what they are for,
and it keeps the claim next to the picture that supports it.

## Writing a panel

```python
Panel(
    code="Q, C = qr(psi[i], (1, 2))",     # verbatim from the source
    figure=Figure(lhs, Txt("="), rhs),    # the equation the line asserts
    note="why this line is here, or what breaks without it",
)
```

- `code` is the real line, not pseudocode. Keep the variable names.
- `figure` is an equation whenever the line assigns.
- `note` carries the *reason*. Prefer a measured number over an adjective —
  "L = 18 pinned at maxbond 2 with an error independent of dt" beats "this is
  important".

## Highlight what the line builds

`hi=True` strokes a node red. Apply it to the assignment's left-hand side and to
nothing else. This is what makes a diagram read as a statement rather than a
picture; a panel with no highlight is usually a panel that should have been a
plain figure.

## Loops

```python
s.add(*unroll(2, lambda i: loop_iter(i, full=True), start=1))
s.add(*loop_iter(3, full=False))
```

Two iterations in full, the third abbreviated to the state change. Two is the
minimum that shows the index moving; three without abbreviation is padding.

Open each iteration with the state on entry, so the reader can see the invariant
the loop maintains.

## Arrows: don't declare them

Arrow direction is derived — horizontal legs point +x on ket rows and −x on bra
rows, vertical legs point up. A row is a bra row when it holds a `dag=True`
tensor, or when declared with `TN(bra=(2,))` (needed when a bra row carries only
open legs, as in an effective Hamiltonian).

Specifying arrows by hand was tried first and they drifted out of agreement
across figures within a day. Don't reintroduce it; use `rev=True` for the rare
genuine exception.

## Layout

- Nodes on an integer `(col, row)` grid, `row` increasing upward.
- Ket row 0, MPO row 1, bra row 2. Keep this across pages so figures compare.
- `anchor_row` aligns items in a `Figure`; use a fractional value (`0.5`) to
  centre a two-row stack against a one-row diagram.
- `cols=n` makes a node span columns — a whole-state blob, a two-site Θ, a gate.
- Annotate groupings with `tn.box(...)` (dashed, for "this whole thing is X") and
  `tn.brace(...)` (for spans of a chain).

## Shapes carry the argument

The shapes are chosen so that structural claims are visible without prose:

- `square = left triangle · circle` is the QR moving the centre onto the bond;
- `circle · right triangle = square` is absorbing it into the next site;
- a chain of triangles pointing away from one square *is* the mixed-canonical
  form.

When a line does something the shapes can express, let them express it and keep
the `note` for the reason rather than the mechanics.

## Checklist before publishing

1. Every code line in the function has a panel, or the page says why not.
2. Each panel's highlight matches its assignment's left-hand side.
3. Loops show the state on entry.
4. Bra rows are declared wherever an effective operator appears.
5. Sections 1 (construction) and 3 (projectors) are present.
6. For a sweep: every source line appears, with its line numbers.
7. Kept/discarded fills are used wherever the argument is about which space a
   direction lives in.
8. The design-choices table is current.
