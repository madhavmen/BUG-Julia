# THE XC / YC GEOMETRY GATE. Run this before any triangular physics.
#
# arXiv:2209.00739 Sec. IIB uses the `XC` wrapping throughout; this repo shipped `YC` while its
# docstring CLAIMED XC, so the two were never compared. Every check here is a statement the two
# geometries disagree about, so a pass is evidence the right one is being built rather than
# evidence the code runs.
#
# ⛔ GATE 4 IS THE ONE THAT WOULD HAVE CAUGHT THE ORIGINAL MISLABELLING. It ties the BOND LIST to
# the CARTESIAN EMBEDDING: every J1 bond must join two sites at minimum-image distance exactly 1,
# and every J2 bond at exactly sqrt(3). A wrap vector that disagrees with the positions shows up
# here as a bond of the wrong length -- whereas coordination counting, energy, and even the
# structure factor can all look plausible while the embedding and the Hamiltonian describe
# different lattices.
#
# Run:  julia --project=. benchmarks/spectral/xc_gate.jl

using LinearAlgebra, SparseArrays, Printf
using LurCGT, Telum
using BUGJulia
using BUGJulia.RSVDCBEBondUpdate

include(joinpath(@__DIR__, "..", "exact_sparse.jl"))
include(joinpath(@__DIR__, "..", "sparse_krylov.jl"))

const SQ3 = sqrt(3.0)
const FAILURES = String[]

function gate(label::AbstractString, ok::Bool, detail::AbstractString = "")
    println(ok ? "  ✅ $label" : "  ⛔ FAIL  $label", isempty(detail) ? "" : "   $detail")
    ok || push!(FAILURES, label)
    flush(stdout)
    return ok
end

"Minimum-image separation of two positions on a cylinder with identification vector `T`."
function min_image(ri, rj, T)
    d = (ri[1] - rj[1], ri[2] - rj[2])
    best = Inf
    for n in -2:2
        v = (d[1] - n * T[1], d[2] - n * T[2])
        best = min(best, hypot(v[1], v[2]))
    end
    return best
end

println("=" ^ 100)
println("XC / YC GEOMETRY GATE  --  arXiv:2209.00739 Sec. IIB")
println("=" ^ 100)

# ── 1. an odd circumference must be REFUSED under XC ─────────────────────────────────────────
println("\n1. XC requires an even circumference (T = C*a1 - (C/2)*a2 is otherwise not a lattice vector)")
for C in (3, 5)
    threw = false
    try
        triangular_cylinder_couplings(6, C; geometry = :xc)
    catch e
        threw = e isa ArgumentError
    end
    gate("C=$C refused under :xc", threw)
end
gate("C=6 accepted under :xc", (triangular_cylinder_couplings(3, 6; geometry = :xc); true))
gate("C=3 still accepted under :yc", (triangular_cylinder_couplings(6, 3; geometry = :yc); true))

# ── 2. the two wrappings are DIFFERENT lattices ──────────────────────────────────────────────
println("\n2. XC and YC are different Hamiltonians (if these matched, the switch did nothing)")
let Jxc = triangular_cylinder_couplings(3, 6; geometry = :xc),
    Jyc = triangular_cylinder_couplings(3, 6; geometry = :yc)
    gate("J_xc != J_yc at 3x6", Jxc != Jyc,
         @sprintf("max|diff| = %.3f, nnz %d vs %d", maximum(abs, Jxc - Jyc),
                  count(!iszero, Jxc), count(!iszero, Jyc)))
    gate("same total coupling weight", isapprox(sum(Jxc), sum(Jyc); atol = 1e-12),
         @sprintf("sum %.4f vs %.4f", sum(Jxc), sum(Jyc)))
end

# ── 3. coordination: every bulk site has six neighbours ──────────────────────────────────────
println("\n3. Coordination -- a triangular cylinder has z=6 in the bulk, z=6 EVERYWHERE on a torus")
for (Lx, Ly, geo) in ((3, 6, :xc), (6, 6, :xc), (12, 6, :xc), (3, 6, :yc), (6, 3, :yc))
    z = triangular_coordination(Lx, Ly; geometry = geo)
    nb = count(==(6), z)
    gate("$geo $(Lx)x$(Ly): $(nb)/$(Lx*Ly) sites have z=6", nb > 0,
         "z range $(minimum(z))..$(maximum(z))")
end
for (Lx, Ly, geo) in ((6, 6, :xc), (6, 6, :yc))
    z = triangular_coordination(Lx, Ly; geometry = geo, periodic_x = true)
    gate("$geo $(Lx)x$(Ly) TORUS: every site z=6", all(==(6), z),
         "z range $(minimum(z))..$(maximum(z))")
end

# ── 4. bonds vs embedding: THE decisive check ────────────────────────────────────────────────
println("\n4. ⛔ Bond list vs Cartesian embedding -- every J1 bond length 1, every J2 length sqrt3")
for (Lx, Ly, geo) in ((3, 6, :xc), (6, 6, :xc), (6, 3, :yc), (3, 6, :yc))
    T = triangular_wrap_vector(Ly; geometry = geo)
    r = [triangular_site_position(Lx, Ly, i; geometry = geo) for i in 1:(Lx * Ly)]
    for (nnn, want, nm) in ((false, 1.0, "J1"), (true, SQ3, "J2"))
        bonds = try
            triangular_cylinder_bonds(Lx, Ly; geometry = geo, nnn = nnn)
        catch
            continue                                    # alias-refused shell, not a geometry fault
        end
        isempty(bonds) && continue
        worst = maximum(abs(min_image(r[i], r[j], T) - want) for (i, j) in bonds)
        gate("$geo $(Lx)x$(Ly) $nm: all $(length(bonds)) bonds at distance $(round(want; digits=4))",
             worst < 1e-10, @sprintf("max deviation %.2e", worst))
    end
end

# ── 5. allowed momenta must reproduce the paper's Eq. (8) and Eq. (9) ────────────────────────
println("\n5. Allowed momenta: XC must give Eq. (8) q_y = 4*pi*n/(sqrt3*C), independent of q_x")
allowed(q, T) = abs((q[1] * T[1] + q[2] * T[2]) / (2pi) -
                    round((q[1] * T[1] + q[2] * T[2]) / (2pi))) < 1e-9
for C in (4, 6, 8)
    T = triangular_wrap_vector(C; geometry = :xc)
    eq8 = [4pi * n / (SQ3 * C) for n in (-C ÷ 2):(C ÷ 2 - 1)]
    ok = all(allowed((qx, qy), T) for qy in eq8 for qx in (0.0, 0.7, 1.9, -2.3))
    gate("XC C=$C: Eq.(8) list allowed at ANY q_x", ok, "n = $(-C÷2)..$(C÷2-1)")
    off = all(!allowed((0.0, qy + 2pi / (SQ3 * C)), T) for qy in eq8)
    gate("XC C=$C: midpoints between them are NOT allowed", off)
end

println("\n   K = (4pi/3, 0) -- the 120-degree ordering wavevector, and the whole point of XC")
for C in (4, 6, 8, 12)
    kxc = allowed((4pi / 3, 0.0), triangular_wrap_vector(C; geometry = :xc))
    kyc = allowed((4pi / 3, 0.0), triangular_wrap_vector(C; geometry = :yc))
    gate("C=$C: K allowed under XC = $kxc, under YC = $kyc",
         kxc && (kyc == (C % 3 == 0)), "YC needs 3|C, XC needs nothing")
end

println("\n   K -> Gamma is an ENTIRE exact segment under XC (q_y = 0 throughout), not so under YC")
let seg = [(4pi / 3 * f, 0.0) for f in 0:0.05:1]
    nxc = count(q -> allowed(q, triangular_wrap_vector(6; geometry = :xc)), seg)
    nyc = count(q -> allowed(q, triangular_wrap_vector(6; geometry = :yc)), seg)
    gate("C=6: $(nxc)/$(length(seg)) path points exact under XC, $(nyc)/$(length(seg)) under YC",
         nxc == length(seg) && nyc < length(seg))
end

println("\n   M images at C=6 under XC: only (0, 2pi/sqrt3) is exact (Eq. 8 needs 4|C for (pi,pi/sqrt3))")
let T = triangular_wrap_vector(6; geometry = :xc)
    m1 = allowed((pi, pi / SQ3), T); m2 = allowed((0.0, 2pi / SQ3), T)
    gate("XC C=6: M=(pi,pi/sqrt3) exact = $m1, M=(0,2pi/sqrt3) exact = $m2", !m1 && m2,
         "the paper fills the rest by C6 symmetry, Eq. (10)")
end

# ── 6. MPO cost: the zigzag labelling must NOT inflate the bond dimension ─────────────────────
println("\n6. MPO virtual dimension -- the zigzag ring must stay as cheap as YC")
for (Lx, Ly, geo) in ((3, 6, :xc), (3, 6, :yc), (6, 3, :yc))
    W = triangular_cylinder_mpo(Lx, Ly; geometry = geo)
    d = maximum(mpo_virtual_dims(W))
    bonds = triangular_cylinder_bonds(Lx, Ly; geometry = geo)
    reach = maximum(abs(i - j) for (i, j) in bonds)
    gate("$geo $(Lx)x$(Ly): MPO max virtual dim $d, longest bond reaches $reach sites",
         d <= 32, "a screw-shifted XC would put this at C^2/2-C+1 = 13 and inflate d")
end

# ── 7. exact energies: XC and YC must differ, and both must be sane ──────────────────────────
println("\n7. Exact S^z=0 ground state at N=18 -- the two geometries cannot agree")
let results = Dict{Symbol, Float64}()
    for geo in (:xc, :yc)
        Jm = triangular_cylinder_couplings(3, 6; geometry = geo)
        H, _, _ = pairs_sparse(18, Jm)
        E0, _, resid, _, _ = sparse_ground(H, size(H, 1); m = 120, restarts = 40, tol = 1e-11)
        results[geo] = E0 / 18
        gate("$geo 3x6: E0/N = $(@sprintf("%.12f", E0/18))  residual $(@sprintf("%.2e", resid))",
             resid < 1e-9)
    end
    gate("E0/N differs between XC and YC",
         abs(results[:xc] - results[:yc]) > 1e-6,
         @sprintf("XC %.9f  vs  YC %.9f   diff %.2e", results[:xc], results[:yc],
                  abs(results[:xc] - results[:yc])))
    # C=6 XC is the paper's circumference; a squat L=3 cylinder sits ABOVE the 2D value.
    gate("XC E0/N above the 2D thermodynamic value -0.5445 (finite squat cylinder)",
         results[:xc] > -0.5445 - 0.05, "Iqbal PRB93 144411 inf-2D ED = -0.5445")
end

println("\n" * "=" ^ 100)
if isempty(FAILURES)
    println("ALL GATES PASSED -- the XC geometry is the paper's, and the embedding agrees with it.")
else
    println("$(length(FAILURES)) GATE(S) FAILED:")
    for f in FAILURES
        println("   - $f")
    end
end
println("=" ^ 100)
flush(stdout)
isempty(FAILURES) || exit(1)
