# ── 2D quantics grids (QTT plan, Task 1.1) ──
# A field on a 2^nx x 2^ny grid is an MPS of nx + ny spin-1/2 sites under `:none`, one site per bit;
# bit value b of a site is its local index b + 1. The plan's default ordering is GROUPED (every x bit
# before every y bit) with each coordinate LEAST significant bit first,
#   i_x = sum_k 2^k b_{x,k},   site k + 1 holds b_{x,k},   site nx + k + 1 holds b_{y,k}.
# `:interleaved` (x_1 y_1 x_2 y_2 ...) and `:msb_first` exist only as diagnostics (grouped and
# interleaved are NOT equivalent at a fixed attainable rank). For Heston the coordinate `y` is the
# variance `v`.

"""
    QTTGrid

A uniform 2D grid `x_i = x0 + i dx`, `y_j = y0 + j dy`, `i < 2^nx`, `j < 2^ny`, and its quantics
site map: `sites_x[k + 1]` / `sites_y[k + 1]` is the MPS site holding bit `k` (`k = 0` the least
significant) of `i` / `j`. `periodic_*` records which boundary the grid spacing was chosen for
(`dx = (xmax - xmin) / 2^nx` periodic, `/(2^nx - 1)` with both end points on the grid).
"""
struct QTTGrid
    nx::Int
    ny::Int
    x0::Float64
    dx::Float64
    y0::Float64
    dy::Float64
    periodic_x::Bool
    periodic_y::Bool
    order::Symbol
    endian::Symbol
    sites_x::Vector{Int}
    sites_y::Vector{Int}
end

"""
    qtt_grid(nx, ny; x = (-1, 1), y = (-1, 1), periodic_x = true, periodic_y = true,
             order = :grouped, endian = :lsb_first) -> QTTGrid

The grid on `x[1] <= x < x[2]` (periodic) or `x[1] <= x <= x[2]` (end points on the grid), likewise
in `y`. See the file header for `order` and `endian`.
"""
function qtt_grid(nx::Int, ny::Int; x = (-1.0, 1.0), y = (-1.0, 1.0), periodic_x::Bool = true,
                  periodic_y::Bool = true, order::Symbol = :grouped, endian::Symbol = :lsb_first)
    nx >= 1 && ny >= 1 || throw(ArgumentError("qtt_grid needs at least one bit per coordinate"))
    order in (:grouped, :interleaved) || throw(ArgumentError(
        "order must be :grouped or :interleaved, got :$order"))
    endian in (:lsb_first, :msb_first) || throw(ArgumentError(
        "endian must be :lsb_first or :msb_first, got :$endian"))
    x[2] > x[1] && y[2] > y[1] || throw(ArgumentError("empty domain $x x $y"))
    dx = (x[2] - x[1]) / (periodic_x ? 2^nx : 2^nx - 1)
    dy = (y[2] - y[1]) / (periodic_y ? 2^ny : 2^ny - 1)
    sx, sy = _qtt_site_map(nx, ny, order, endian)
    return QTTGrid(nx, ny, Float64(x[1]), dx, Float64(y[1]), dy, periodic_x, periodic_y, order,
                   endian, sx, sy)
end

"Site of every bit: `slot j` of a coordinate holds bit `j - 1` (`:lsb_first`) or `n - j` (`:msb_first`)."
function _qtt_site_map(nx::Int, ny::Int, order::Symbol, endian::Symbol)
    bit(j, n) = endian === :lsb_first ? j - 1 : n - j
    sx, sy = zeros(Int, nx), zeros(Int, ny)
    s = 0
    if order === :grouped
        for j in 1:nx
            sx[bit(j, nx) + 1] = (s += 1)
        end
        for j in 1:ny
            sy[bit(j, ny) + 1] = (s += 1)
        end
    else
        for j in 1:max(nx, ny)
            j <= nx && (sx[bit(j, nx) + 1] = (s += 1))
            j <= ny && (sy[bit(j, ny) + 1] = (s += 1))
        end
    end
    return sx, sy
end

"Number of quantics sites, `nx + ny`."
qtt_length(g::QTTGrid) = g.nx + g.ny

"Grid coordinates `(x, y)` as two vectors (length `2^nx`, `2^ny`)."
qtt_coords(g::QTTGrid) = (g.x0 .+ g.dx .* (0:(2^g.nx - 1)), g.y0 .+ g.dy .* (0:(2^g.ny - 1)))

"The quadrature weight `dx dy` of the discrete L2 inner product."
qtt_cell(g::QTTGrid) = g.dx * g.dy

"""
    qtt_bits(g, ix, iy) -> Vector{Int}

The bit value (0 or 1) on every site for the grid point `(ix, iy)` (0-based).
"""
function qtt_bits(g::QTTGrid, ix::Integer, iy::Integer)
    0 <= ix < 2^g.nx && 0 <= iy < 2^g.ny || throw(BoundsError((2^g.nx, 2^g.ny), (ix, iy)))
    b = zeros(Int, qtt_length(g))
    for k in 0:(g.nx - 1)
        b[g.sites_x[k + 1]] = (ix >> k) & 1
    end
    for k in 0:(g.ny - 1)
        b[g.sites_y[k + 1]] = (iy >> k) & 1
    end
    return b
end

"""
    qtt_point(g, bits) -> (ix, iy)

Inverse of [`qtt_bits`](@ref).
"""
function qtt_point(g::QTTGrid, bits::AbstractVector{<:Integer})
    length(bits) == qtt_length(g) || throw(DimensionMismatch("need $(qtt_length(g)) bits"))
    ix = sum(bits[g.sites_x[k + 1]] << k for k in 0:(g.nx - 1))
    iy = sum(bits[g.sites_y[k + 1]] << k for k in 0:(g.ny - 1))
    return ix, iy
end

"""
    qtt_site_permutation(g) -> Vector{Int}

`p[c]` is the 1-based grid index `ix + 1 + 2^nx iy` of site configuration `c`, where `c - 1 =
sum_s 2^(s-1) b_s` (site 1 fastest -- the column-major order of a dense `(2, ..., 2)` core
contraction). For dense comparisons on small grids only.
"""
function qtt_site_permutation(g::QTTGrid)
    L = qtt_length(g)
    L <= 24 || throw(ArgumentError("qtt_site_permutation is a small-grid check (L = $L)"))
    p = zeros(Int, 2^L)
    bits = zeros(Int, L)
    for c in 0:(2^L - 1)
        for s in 1:L
            bits[s] = (c >> (s - 1)) & 1
        end
        ix, iy = qtt_point(g, bits)
        p[c + 1] = ix + 1 + 2^g.nx * iy
    end
    return p
end
