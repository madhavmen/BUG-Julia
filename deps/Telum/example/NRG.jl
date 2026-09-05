using LurCGT
using Telum
using Plots
using LinearAlgebra

"""
    _linear_interp_extrap(x, y, xq)

Evaluate the piecewise-linear interpolant defined by sample points `(x, y)` at
the query points `xq`, using linear extrapolation outside the sampled domain.

This helper reproduces the behavior needed for the MATLAB routine
`interp1(..., "linear", "extrap")` used by the logarithmic discretization
code. The abscissae `x` must be sorted in strictly increasing order and contain
at least two points.
"""
function _linear_interp_extrap(x::AbstractVector, y::AbstractVector,
                               xq::AbstractVector)
    length(x) == length(y) ||
        throw(ArgumentError("`x` and `y` must have the same length."))
    length(x) ≥ 2 ||
        throw(ArgumentError("At least two sample points are required for interpolation."))

    T = promote_type(float(eltype(x)), float(eltype(y)), float(eltype(xq)))
    xp = T.(x)
    yp = T.(y)
    xqp = T.(xq)

    issorted(xp) || throw(ArgumentError("`x` must be sorted in increasing order."))
    all(diff(xp) .> zero(T)) ||
        throw(ArgumentError("`x` entries must be strictly increasing."))

    result = similar(xqp, T)

    @inbounds for i in eachindex(xqp)
        xi = xqp[i]
        seg = if xi <= xp[1]
            1
        elseif xi >= xp[end]
            length(xp) - 1
        else
            searchsortedlast(xp, xi)
        end

        x1 = xp[seg]
        x2 = xp[seg + 1]
        y1 = yp[seg]
        y2 = yp[seg + 1]
        result[i] = y1 + (y2 - y1) * (xi - x1) / (x2 - x1)
    end

    return result
end

"""
    doCLD_1side(oz, rho, nstep) -> repE, repT

Compute the representative energies and interval weights for one energy half
of the Campo-Oliveira logarithmic discretization.

Arguments
- `oz`: Positive logarithmic grid points in increasing order.
- `rho`: Hybridization values sampled on `oz`.
- `nstep`: Number of grid samples inside each discretization interval.

Returns
- `repE`: Representative energies for each interval.
- `repT`: Integrated hybridization weights for each interval.

This is a direct Julia translation of the helper routine from `doCLD (1).m`.
The returned quantities correspond to the symbols `\\mathcal{E}` and the
interval integrals used in the star-geometry representation.
"""
function doCLD_1side(oz::AbstractVector, rho::AbstractVector, nstep::Integer)
    length(oz) == length(rho) ||
        throw(ArgumentError("`oz` and `rho` must have the same length."))
    nstep > 0 || throw(ArgumentError("`nstep` must be positive."))

    T = promote_type(float(eltype(oz)), float(eltype(rho)))
    ozp = T.(oz)
    rhop = T.(rho)

    ids = length(rhop):-nstep:1
    repT = zeros(T, length(ids) - 1)
    repE = similar(repT)

    @inbounds for itx in eachindex(repT)
        lo = ids[itx + 1]
        hi = ids[itx]
        ozseg = @view ozp[lo:hi]
        rhoseg = @view rhop[lo:hi]

        dω = ozseg[2:end] .- ozseg[1:end-1]
        repT[itx] = sum((rhoseg[2:end] .+ rhoseg[1:end-1]) .* dω) / 2

        repE_denom = (rhoseg[end] - rhoseg[1]) +
                     sum(((ozseg[2:end] .* rhoseg[1:end-1] .-
                           ozseg[1:end-1] .* rhoseg[2:end]) ./ dω) .*
                         log.(abs.(ozseg[2:end] ./ ozseg[1:end-1])))
        repE[itx] = repT[itx] / repE_denom
    end

    return repE, repT
end

"""
    doCLD(ozin, rhoV2in, Lambda, N; estep=10, emax=maximum(abs, ozin), emin=100eps(Float64)) -> ff, gg

Perform Campo-Oliveira logarithmic discretization of a bath hybridization
function and map the resulting star geometry to a Wilson chain.

Arguments
- `ozin`: Frequency grid used to define the hybridization function
  `Δ(ω) ≡ rhoV2in(ω)`. The first and last entries determine the sampled
  bandwidth.
- `rhoV2in`: Hybridization values evaluated on `ozin`.
- `Lambda`: Logarithmic discretization parameter. Typical NRG applications use
  `Lambda > 1`.
- `N`: Requested Wilson-chain length.

Keyword arguments
- `estep`: Number of grid points used to resolve each logarithmic interval.
  This matches the MATLAB option `"estep"` and defaults to `10`.
- `emax`: Largest positive energy used for the logarithmic grid. The default is
  `maximum(abs, ozin)`.
- `emin`: Smallest positive energy used for the logarithmic grid. The default
  is `100eps(Float64)`, mirroring the MATLAB code's `100*eps`.

Returns
- `ff`: Wilson-chain hopping amplitudes obtained from the Lanczos
  tridiagonalization of the discretized star Hamiltonian.
- `gg`: Wilson-chain on-site energies.

Notes
- Negative values produced by interpolation are clipped to zero, following the
  MATLAB implementation and ensuring a positive hybridization density.
- The routine preserves the MATLAB indexing convention exactly: `ff` and `gg`
  both have length `min(N, nintervals)`, and the final entry of `gg` remains
  zero because the original code only writes `gg[it]` when the next Lanczos
  vector exists.
- If the number of discretization intervals is smaller than `N`, the returned
  chain is truncated and a warning is emitted.

This function is a faithful Julia port of `doCLD (1).m`, including the helper
interval formulas and the final Lanczos step used to build the Wilson chain.
"""
function doCLD(ozin::AbstractVector, rhoV2in::AbstractVector,
               Lambda::Real, N::Integer;
               estep::Integer = 10,
               emax = maximum(abs, ozin),
               emin = 100eps(Float64))
    isempty(ozin) && throw(ArgumentError("Empty frequency input `ozin`."))
    isempty(rhoV2in) && throw(ArgumentError("Empty hybridization input `rhoV2in`."))
    length(ozin) == length(rhoV2in) ||
        throw(ArgumentError("`ozin` and `rhoV2in` must have the same length."))
    length(ozin) ≥ 2 ||
        throw(ArgumentError("At least two bath samples are required."))
    estep > 0 || throw(ArgumentError("`estep` must be positive."))
    N > 0 || throw(ArgumentError("`N` must be positive."))
    Lambda > 1 || throw(ArgumentError("`Lambda` must be larger than 1."))
    emax > 0 || throw(ArgumentError("`emax` must be positive."))
    emin > 0 || throw(ArgumentError("`emin` must be positive."))
    emin < emax || throw(ArgumentError("`emin` must be smaller than `emax`."))

    T = promote_type(float(eltype(ozin)), float(eltype(rhoV2in)),
                     typeof(float(Lambda)), typeof(float(emax)), typeof(float(emin)))
    ozinp = T.(ozin)
    rhoinp = T.(rhoV2in)

    perm = sortperm(ozinp)
    ozsorted = ozinp[perm]
    rhosorted = rhoinp[perm]
    all(diff(ozsorted) .> zero(T)) ||
        throw(ArgumentError("`ozin` must contain strictly increasing frequencies after sorting."))

    xs_hi = log(T(emax)) / log(T(Lambda)) * estep
    xs_lo = log(T(emin)) / log(T(Lambda)) * estep
    xs = reverse(collect(xs_hi:-1:xs_lo)) ./ estep
    oz = T(Lambda) .^ xs

    rho1 = _linear_interp_extrap(ozsorted, rhosorted, oz)
    rho1 .= max.(rho1, zero(T))
    repE1, repT1 = doCLD_1side(oz, rho1, estep)

    rho2 = _linear_interp_extrap(ozsorted, rhosorted, .-oz)
    rho2 .= max.(rho2, zero(T))
    repE2, repT2 = doCLD_1side(oz, rho2, estep)

    nintervals = length(repE1) + length(repE2)
    N2 = if nintervals < N
        @warn "Number of discretization intervals is smaller than the requested chain length." nintervals N
        nintervals
    else
        N
    end

    ff = zeros(T, N2)
    gg = zeros(T, N2)

    Xis = vcat(repE1, .-repE2)
    Gammas = vcat(sqrt.(repT1), sqrt.(repT2))
    H = zeros(T, length(Xis) + 1, length(Xis) + 1)
    H[1, 2:end] .= Gammas
    H[2:end, 1] .= Gammas
    H[2:end, 2:end] .= Diagonal(Xis)

    U = zeros(T, size(H, 1), N2)
    U[1, 1] = one(T)

    for itN in 1:N2
        v = H * view(U, :, itN)
        v .-= U * (U' * v)
        v .-= U * (U' * v)  # repeat for numerical stability, as in MATLAB
        ff[itN] = LinearAlgebra.norm(v)

        if itN < N2 && ff[itN] > zero(T)
            U[:, itN + 1] .= v ./ ff[itN]
            gg[itN] = dot(view(U, :, itN + 1), H * view(U, :, itN + 1))
        end
    end

    return ff, gg
end


function NRG_test()
    U = 4e-3 # Coulomb interaction at the impurity
    epsd = -U/2 #  impurity on-site energy
    Γ = 8e-5*pi # hybridization strength

    λ = 2.5 # logarithmic discretization parameter
    N = 55 # Wilson chain length
    Nkeep = 300

    ff, gg = doCLD([-1, 1], [1, 1].*(Γ/pi), λ, N)

    option = FermionSOptions(1, :U1, :SU2, nothing)
    q0 = getLocalSpace(option, ("s,0", "s,0", "op"))
    n0 = lock(q0.F', 2) * q0.F

    H0 = U/2*lock(n0, 1) * (n0 - q0.I) + epsd * n0 
    v = getvac(q0.I, ("K,vac", "K,vac"))
    A0 = permutedims(getIdentity((v, 2), (H0, 2); itag="K,0"), (1, 3, 2))

    H0 = A0' * lock(A0 * H0, 2)

    return NRG_IterDiag(H0, A0, λ, ff, q0.F, gg, n0, q0.Z, Nkeep)
end

function NRG_IterDiag(H0::TLArray{T, 2, N}, 
    A0::TLArray{T, 3, N}, 
    λ::Float64, 
    ff::Vector{Float64}, 
    q0F::TLArray{T, 3, N}, 
    gg::Vector{Float64}, 
    n0::TLArray{T, 2, N}, 
    q0Z::TLArray{T, 2, N}, 
    Nkeep::Int) where {T, N}
    # Placeholder for the actual NRG iteration and diagonalization routine.
    # This function would perform the iterative diagonalization of the Wilson
    # chain Hamiltonian constructed from H0 and A0, using the hopping amplitudes
    # ff and on-site energies gg. The local space is defined by q0F and q0Z.
    # The function would return the final effective Hamiltonian or other relevant
    # data structures after NRG iterations with a truncation to Nkeep states.
    println("NRG start")

    Nsite = length(ff) + 1
    EScale = [1, (λ.^(((Nsite-2):-1:0)./2).*ff[end])...]
    AK = Vector{TLArray{T, 3, N}}(undef, Nsite)
    AD = Vector{TLArray{T, 3, N}}(undef, Nsite)
    EK = []
    ED = []
    E0 = Vector{Float64}(undef, Nsite)

    Nfac = 0.1
    Hprev = H0
    Fprev = q0F
    for itN=1:Nsite
        si = itN - 1
        Z = TLArray(q0Z, ("s,$si", "s,$si"))
        n0 = TLArray(n0, ("s,$si", "s,$si"))
        F = TLArray(q0F, ("s,$si", "s,$si", "op"))
        if itN == 1
            AK[itN] = A0
            # Do not update AD since there is no discarded states
            e = eigen(H0)
            ek, ed = discard_eigen(e, Nkeep, "K,$si", "D,$si")
            push!(EK, ek.eig_list)
            push!(ED, ed.eig_list)

        else
            l = findleg(Hprev; dir='-')
            Anow = getIdentity((Hprev, l), (Z, 2); itag="L,$si")
            
            Hnow = lock(Anow'; itag="L") * Hprev * Anow
            Hnow = Hnow * (EScale[itN-1] / EScale[itN])

            Fnow = F' * lock(Z, 2)
            Hhop = lock(Anow'; itag="L") * (Fprev * Anow * Fnow)
            Hhop = Hhop * (ff[itN-1] / EScale[itN])
            Hhop = Hhop + Hhop'

            Hon = lock(Anow'; itag="L") * n0 * Anow
            Hon = (gg[itN-1] / EScale[itN]) * Hon

            Hnow = Hnow + Hhop + Hon
            Hnow = (Hnow + Hnow') / 2 # Ensure Hermiticity

            # Eigendecomposition and discard spaces
            e = eigen(Hnow)
            e0 = e.eig_list[1][1] # Ground state energy
            E0[itN] = e0

            ek, ed = discard_eigen(e, Nkeep, "K,$si", "D,$si"; tol=0.1)

            # Shift energies to make the lowest energy value be 0
            Hprev = ek.D - e0
            klst = [(x-e0, k...) for (x, k...) in ek.eig_list]
            dlst = [(x-e0, d...) for (x, d...) in ed.eig_list]
            push!(EK, klst); push!(ED, dlst)
            AK[itN] = Anow * ek.V; AD[itN] = Anow * ed.V
        end

        if itN < Nsite
            ak = AK[itN]
            Fprev = ak' * lock(F * ak; itag="K,$si")
        end
    end


    return EK, AK, ED, AD, E0  
end


_nrg_sector_label(sector_qlabels) =
    join(["(" * join(ql, ",") * ")" for ql in sector_qlabels], " ")

function _nrg_plot_series(EK::AbstractVector, parity::Int, ymax::Real)
    series = Dict{Tuple{Any, Int}, Vector{Tuple{Int, Float64}}}()

    for (it, eig_entries) in enumerate(EK)
        isodd(it) == (parity == 1) || continue
        for (ev, _, sector_qlabels, sector_index) in eig_entries
            energy = Float64(real(ev))
            energy < ymax || continue

            key = (sector_qlabels, sector_index)
            push!(get!(series, key, Tuple{Int, Float64}[]), (it, energy))
        end
    end

    return series
end

function _nrg_sector_colors(EK::AbstractVector, ymax::Real)
    sector_order = Any[]
    seen_sectors = Set{Any}()

    for eig_entries in EK
        for (ev, _, sector_qlabels, _) in eig_entries
            Float64(real(ev)) < ymax || continue
            if sector_qlabels ∉ seen_sectors
                push!(sector_order, sector_qlabels)
                push!(seen_sectors, sector_qlabels)
            end
        end
    end

    palette = Plots.palette(:tab20, max(length(sector_order), 1))
    return Dict(sector => palette[i] for (i, sector) in enumerate(sector_order))
end

function _plot_nrg_parity(EK::AbstractVector, parity::Int;
                          sector_colors,
                          ymax::Real,
                          linewidth::Real,
                          show_legend::Bool)
    parity_name = parity == 1 ? "Odd iterations" : "Even iterations"
    series = _nrg_plot_series(EK, parity, ymax)

    p = Plots.plot(
        xlabel = "Iteration",
        ylabel = "Eigenvalue",
        title = parity_name,
        ylim = (0, ymax),
        legend = show_legend ? :outerright : false,
    )

    shown_labels = Set{Any}()
    for key in sort!(collect(keys(series)); by = x -> (_nrg_sector_label(x[1]), x[2]))
        points = sort!(series[key]; by = first)
        sector_qlabels, sector_index = key
        label = if show_legend && sector_qlabels ∉ shown_labels
            push!(shown_labels, sector_qlabels)
            _nrg_sector_label(sector_qlabels)
        else
            ""
        end

        Plots.plot!(
            p,
            first.(points),
            last.(points);
            color = sector_colors[sector_qlabels],
            linewidth = linewidth,
            label = label,
        )
    end

    return p
end

"""
    plot_NRG_EK(EK; ymax=3.0, linewidth=1.5, show_legend=true)
    plot_NRG_EK(result::Tuple; ymax=3.0, linewidth=1.5, show_legend=true)

Plot the NRG eigenvalue flow stored in `EK`, splitting odd and even iterations
into separate subplots. Only eigenvalues strictly smaller than `ymax` are
drawn, and all lines with the same symmetry-sector qlabels share the same
color.

This helper uses `Plots.jl` as an optional dependency. If `Plots` is not
installed in the active Julia environment, the function throws an error with a
short installation hint.
"""
function plot_NRG_EK(EK::AbstractVector;
                     ymax::Real = 3.0,
                     linewidth::Real = 1.5,
                     show_legend::Bool = true)
    Base.find_package("Plots") === nothing &&
        throw(ArgumentError("`plot_NRG_EK` requires Plots.jl. Install it with `import Pkg; Pkg.add(\"Plots\")`."))
    @eval import Plots

    sector_colors = _nrg_sector_colors(EK, ymax)
    odd_plot = _plot_nrg_parity(EK, 1;
        sector_colors,
        ymax, linewidth, show_legend)
    even_plot = _plot_nrg_parity(EK, 2;
        sector_colors,
        ymax, linewidth, show_legend)

    fig = Plots.plot(odd_plot, even_plot; layout = (2, 1), size = (950, 900))
    display(fig)
    return fig
end

plot_NRG_EK(result::Tuple; kwargs...) = plot_NRG_EK(result[1]; kwargs...)

EK, AK, ED, AD, E0 = NRG_test()
plot_NRG_EK(EK; ymax=3.0, linewidth=1.5, show_legend=true)
