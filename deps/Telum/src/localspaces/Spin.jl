# s2: spin * 2
function get_SU2_symmops(s2::Int)
    S = s2 / 2
    CGT = LurCGT.getNsave_cg3(SU{2}, BigInt, ((2,), (s2,)), [(s2,)])[(s2,)]
    arr = LurCGT.to_float(CGT, Float64)[1] * sqrt(S * (S + 1))
    if arr[2, 1, 1, 1] < 0 arr = -arr end

    sp = sparse(transpose(arr[1, :, :, 1]))
    sz = sparse(transpose(arr[2, :, :, 1]))
    sm = sparse(transpose(arr[3, :, :, 1]))
    I = spdiagm(0 => ones(Float64, s2 + 1))
    lowering = sm * sqrt(2)

    return sp, sz, sm, I, lowering
end

function getSymmetryInfo(opts::SpinOptions)
    @assert opts.symmetry === nothing || opts.symmetry == :SU2 || opts.symmetry == :U1 
    "SpinOptions currently only supports :SU2, :U1, or no symmetry"
    @assert opts.spin > 0 

    sp, sz, sm, I, lowering = get_SU2_symmops(opts.spin)

    # Basis: |up>, |down>
    weights = opts.symmetry === nothing ? () : ([(s,) for s in opts.spin:-2:-opts.spin],)
    
    if opts.symmetry == :SU2
        lowering_ops = ([lowering],)
    elseif opts.symmetry == :U1
        lowering_ops = (Matrix{Float64}[],)
    elseif opts.symmetry === nothing
        lowering_ops = ()
    else
        error("SpinOptions currently only supports :SU2, :U1, or no symmetry")
    end

    mwirops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
    if opts.symmetry == :SU2
        mwirops[:S] = sp
        symm = (SU{2},)
    else 
        mwirops[:Sp] = sp
        mwirops[:Sz] = sz
        mwirops[:Sm] = sm
        if opts.symmetry == :U1
            symm = (U1,)
        elseif opts.symmetry === nothing
            symm = ()
        else
            error("SpinOptions currently only supports :SU2, :U1, or no symmetry")
        end
    end
    mwirops[:I] = I

    return symm, weights, lowering_ops, mwirops
end
