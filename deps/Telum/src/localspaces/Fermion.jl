function _covers_all_channels(channs::AbstractVector{<:Integer}, N::Int)
    return length(channs) == N && Set(channs) == Set(1:N)
end

function _channel_suffix(channs::AbstractVector{<:Integer}, N::Int)
    return _covers_all_channels(channs, N) ? "" : join(channs)
end

_spin_symbol(channs::AbstractVector{<:Integer}, N::Int, component::AbstractString="") =
    Symbol("S", _channel_suffix(channs, N), component)

_fermion_symbol(channs::AbstractVector{<:Integer}, N::Int, component::AbstractString="") =
    Symbol("F", _channel_suffix(channs, N), component)

function _check_channels(channs::AbstractVector{<:Integer}, N::Int, label)
    isempty(channs) && error("$label must contain at least one channel")
    length(unique(channs)) == length(channs) ||
        error("$label contains duplicate channels: $channs")
    all(1 .<= channs .<= N) ||
        error("$label contains channels outside 1:$N: $channs")
end

function _check_disjoint_channel_groups(groups, N::Int, label::AbstractString)
    seen = Set{Int}()
    for (ssymbol, channs) in groups
        _check_channels(channs, N, "$label $ssymbol")
        overlap = intersect(seen, channs)
        isempty(overlap) ||
            error("$label groups overlap on channels $(sort!(collect(overlap)))")
        union!(seen, channs)
    end
    return seen
end

function _parse_su_channel_symbol(ssymbol::Symbol, channs::AbstractVector{<:Integer})
    str = String(ssymbol)
    startswith(str, "SU") || error("Unsupported channel symmetry: $ssymbol")

    num_part = str[3:end]
    nchannels = isempty(num_part) ? length(channs) : parse(Int, num_part)
    @assert nchannels >= 2 && nchannels == length(channs)
    return nchannels
end

# N: The number of spinless fermion channels
function Fermion_basicops(N::Int)
    @assert N > 0 "Number of channels must be positive"

    FF = Vector{SparseMatrixCSC{Int}}(undef, N)
    NN = Vector{SparseMatrixCSC{Int}}(undef, N)

    for i in 1:N
        mats = vcat([z2 for _ in 1:i-1], [f2], [I2 for _ in i+1:N])
        FF[i] = sparse(reduce(⊗, mats))
        NN[i] = FF[i]' * FF[i]
    end

    chan_l = Vector{SparseMatrixCSC{Int}}(undef, max(N - 1, 0))
    chan_z = Vector{Vector{Int}}(undef, max(N - 1, 0))
    for i in 1:(N - 1)
        chan_l[i] = FF[i+1]' * FF[i]
        chan_z[i] = Int.(diag(sum([NN[j] for j in 1:i]) - i * NN[i+1]))
    end

    for i in 1:(N - 1)
        @assert comm(chan_l[i], diagm(chan_z[i])) == (i + 1) * chan_l[i]
        if i > 1
            @assert comm(chan_l[i-1], diagm(chan_z[i])) == 0 * chan_l[i-1]
            @assert comm(chan_l[i], diagm(chan_z[i-1])) == -(i - 1) * chan_l[i]
        end
        for j in 1:(N - 1)
            if abs(i - j) > 1
                @assert comm(chan_l[j], diagm(chan_z[i])) == 0 * chan_l[j]
            end
        end
    end

    return (; FF, NN, chan_l, chan_z)
end

"""
    charge_symmops(opts::FermionOptions, basic_ops)

Build Abelian charge-symmetry data for each charge group in
`opts.charge_symm`. The current spinless implementation supports `:U1` only.
The weight of a basis state is the occupation number on the selected channels.
"""
function charge_symmops(opts::FermionOptions, basic_ops)
    N = opts.nchannels
    weights, symms, lops = [], [], []
    for (ssymbol, channs) in opts.charge_symm
        ssymbol == :U1 || error("Unsupported spinless charge symmetry: $ssymbol")
        _check_channels(channs, N, "Charge symmetry $ssymbol")

        charge = Int.(diag(sum(basic_ops.NN[i] for i in channs)))
        push!(weights, [(Int(i),) for i in charge])
        push!(symms, U1)
        push!(lops, Matrix{Float64}[])
    end
    return weights, symms, lops
end

"""
    chan_symmops(opts::FermionOptions, basic_ops)

Build SU(N) channel-symmetry data for each channel group in
`opts.channel_symm`.
"""
function chan_symmops(opts::FermionOptions, basic_ops)
    Ntot = opts.nchannels
    weights, symms, lops = [], [], []
    for (ssymbol, channs) in opts.channel_symm
        _check_channels(channs, Ntot, "Channel symmetry $ssymbol")
        N = _parse_su_channel_symbol(ssymbol, channs)
        push!(symms, SU{N})

        lop = Vector{SparseMatrixCSC{Float64, Int}}(undef, N - 1)
        zvals = Vector{Vector{Int}}(undef, N - 1)
        for i in 1:(N - 1)
            lop[i] = Float64.(basic_ops.FF[channs[i+1]]' * basic_ops.FF[channs[i]])
            zvals[i] = Int.(diag(sum([basic_ops.NN[channs[j]] for j in 1:i]) -
                                  i * basic_ops.NN[channs[i+1]]))
        end
        push!(lops, lop)
        push!(weights, collect(zip(zvals...)))
    end
    return weights, symms, lops
end

function _fermion_annihilation_primary_blocks(opts::FermionOptions, N::Int)
    charge_groups = opts.charge_symm === nothing ? Tuple{Symbol, Vector{Int}}[] : opts.charge_symm

    blocks = NamedTuple[]
    for (ssymbol, channs) in charge_groups
        ssymbol == :U1 || error("Unsupported spinless charge symmetry: $ssymbol")
        _check_channels(channs, N, "Charge symmetry $ssymbol")
        push!(blocks, (; channs = collect(channs), channel_symm = false))
    end

    _check_disjoint_channel_groups(charge_groups, N, "Charge symmetry")
    return blocks
end

function _split_fermion_annihilation_blocks_by_channel_symmetry(blocks, opts::FermionOptions, N::Int)
    channel_groups = opts.channel_symm === nothing ? Tuple{Symbol, Vector{Int}}[] : opts.channel_symm
    channel_selected = Set{Int}()

    for (ssymbol, channs) in channel_groups
        _parse_su_channel_symbol(ssymbol, channs)
        _check_channels(channs, N, "Channel symmetry $ssymbol")

        overlap_with_channel = intersect(channel_selected, channs)
        isempty(overlap_with_channel) ||
            error("Channel symmetry groups overlap on channels " *
                  "$(sort!(collect(overlap_with_channel)))")

        chann_set = Set(channs)
        containing_idxs = [
            idx for (idx, block) in pairs(blocks)
            if issubset(chann_set, Set(block.channs))
        ]
        overlapping_idxs = [
            idx for (idx, block) in pairs(blocks)
            if !isempty(intersect(chann_set, block.channs))
        ]

        if isempty(overlapping_idxs)
            push!(blocks, (; channs = collect(channs), channel_symm = true))
        elseif length(containing_idxs) == 1 && length(blocks[only(containing_idxs)].channs) > 1
            block_idx = only(containing_idxs)
            block = blocks[block_idx]
            remainder = [ch for ch in block.channs if !(ch in chann_set)]

            blocks[block_idx] = (; channs = collect(channs), channel_symm = true)
            if !isempty(remainder)
                push!(blocks, (; channs = remainder, channel_symm = false))
            end
        else
            error("Channel symmetry $ssymbol on channels $channs must contain " *
                  "only channels with no charge symmetry or be contained in " *
                  "one already generated multi-channel charge set")
        end

        union!(channel_selected, channs)
    end

    return blocks
end

function _add_fermion_annihilation_block!(
    mwirops::Dict{Symbol, SparseMatrixCSC{Float64, Int}},
    block,
    N::Int,
    basic_ops)

    channs = block.channs
    op_channs = block.channel_symm ? [last(channs)] : channs
    mwirops[_fermion_symbol(channs, N)] =
        sum(basic_ops.FF[i] for i in op_channs)
end

"""
    add_annihilation_irop!(mwirops, opts::FermionOptions, basic_ops)

Add spinless fermion annihilation IROPs to the maximal-weight IROP dictionary.
When a generated operator covers all channels, the channel suffix is omitted.
"""
function add_annihilation_irop!(
    mwirops::Dict{Symbol, SparseMatrixCSC{Float64, Int}},
    opts::FermionOptions,
    basic_ops)

    N = opts.nchannels
    blocks = _fermion_annihilation_primary_blocks(opts, N)
    _split_fermion_annihilation_blocks_by_channel_symmetry(blocks, opts, N)

    selected = Set{Int}()
    for block in blocks
        _add_fermion_annihilation_block!(mwirops, block, N, basic_ops)
        union!(selected, block.channs)
    end

    for i in 1:N
        if !(i in selected)
            _add_fermion_annihilation_block!(mwirops, (; channs = [i], channel_symm = false), N, basic_ops)
        end
    end
end

function getSymmetryInfo(opts::FermionOptions)
    N = opts.nchannels
    basic_ops = Fermion_basicops(N)
    symms = Vector{Type{<:Symmetry}}()
    weights = Vector{<:Tuple{Vararg{Int}}}[]
    lowering_ops = Vector{<:AbstractMatrix{<:Real}}[]

    if opts.charge_symm !== nothing
        ws, ss, ls = charge_symmops(opts, basic_ops)
        append!(symms, ss)
        append!(weights, ws)
        append!(lowering_ops, ls)
    end

    if opts.channel_symm !== nothing
        ws, ss, ls = chan_symmops(opts, basic_ops)
        append!(symms, ss)
        append!(weights, ws)
        append!(lowering_ops, ls)
    end

    totalN = diag(sum(basic_ops.NN[i] for i in 1:N))
    mwirops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
    add_annihilation_irop!(mwirops, opts, basic_ops)
    mwirops[:Z] = spdiagm(0 => Float64[isodd(i) ? -1 : 1 for i in totalN])
    mwirops[:I] = spdiagm(0 => ones(Float64, 2^N))

    return Tuple(symms), Tuple(weights), Tuple(lowering_ops), mwirops
end
