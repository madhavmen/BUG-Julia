# N: The number of channels
function FermionS_basicops(N::Int)
    @assert N > 0 "Number of channels must be positive"

    # Spin lowering operator for single channel
    ss = f4down' * f4up

    # Annihilation operations, dim: (channel, spin)
    # FF[1, n]: spin-up annihilation operator for channel n
    # FF[2, n]: spin-down annihilation operator for channel n
    FF = Matrix{SparseMatrixCSC{Int}}(undef, 2, N)
    NN = Matrix{SparseMatrixCSC{Int}}(undef, 2, N)
    # SS[n]: spin lowering operator for channel n
    SS = Vector{SparseMatrixCSC{Int}}(undef, N)
    # CC[n]: charge lowering operator for channel n
    CC = Vector{SparseMatrixCSC{Int}}(undef, N)
    
    # Total number of fermionic modes: 2 spins × N channels
    total_modes = 2 * N

    for i in 1:total_modes
        mats = vcat([z2 for _=1:i-1], [f2], [I2 for _=i+1:total_modes])
        FF[i] = reduce(⊗, mats)
        NN[i] = FF[i]' * FF[i]
    end
    
    for i in 1:N
        mats = vcat([I2 for _=1:2*(i-1)], [ss], [I2 for _=1:2*(N-i)])
        SS[i] = reduce(⊗, mats)
        @assert SS[i] == FF[2, i]' * FF[1, i]
    end

    for i in 1:N
        mats = vcat([I2 for _=1:2*(i-1)], [f4down * f4up], [I2 for _=1:2*(N-i)])
        CC[i] = reduce(⊗, mats)
        @assert CC[i] == FF[2, i] * FF[1, i]
    end

    SZ = Vector{SparseMatrixCSC{Int}}(undef, N)
    for i in 1:N
        mats = vcat([I2 for _=1:2*(i-1)], [sz4], [I2 for _=1:2*(N-i)])
        SZ[i] = reduce(⊗, mats)
    end 

    # Channel lowering operators for SU{N}: N-1 operators
    # chan_l[i] connects channel i to channel i+1
    chan_l = Vector{SparseMatrixCSC{Int}}(undef, N-1)
    for i in 1:(N-1)
        chan_l[i] = FF[1, i+1]'*FF[1, i] + FF[2, i+1]'*FF[2, i]
    end
    
    # Channel z-operators for SU{N}: N-1 operators following Cartan subalgebra
    # Using standard Dynkin basis: H_i = E_{i,i} - E_{i+1,i+1}
    chan_z = Vector{Vector{Int}}(undef, N-1)
    for i in 1:(N-1)
        # H_i: occupation of channel i minus occupation of channel i+1
        chan_z[i] = Int.(diag(sum([NN[1, j] + NN[2, j] for j=1:i]) 
        - i * NN[1, i+1] - i * NN[2, i+1]))
    end

    spin_z = Int.(diag(sum(SZ)))
    spin_lowering = sum(SS)

    # Verify commutation relations for SU{N} algebra
    @assert comm(spin_lowering, diagm(spin_z)) == 2 * spin_lowering
    for i in 1:(N-1)
        # [H_i, E_i] = 2*E_i (lowering operator i raised by H_i by 2)
        @assert comm(chan_l[i], diagm(chan_z[i])) == (i+1) * chan_l[i]
        
        # [H_i, E_{i-1}] = -E_{i-1} (lowering operator i-1 lowered by H_i by 1)
        if i > 1
            @assert comm(chan_l[i-1], diagm(chan_z[i])) == 0 * chan_l[i-1]
            @assert comm(chan_l[i], diagm(chan_z[i-1])) == -(i-1) * chan_l[i]
        end

        
        # [H_i, E_j] = 0 for |i-j| > 1
        for j in 1:(N-1)
            if abs(i - j) > 1
                @assert comm(chan_l[j], diagm(chan_z[i])) == 0 * chan_l[j]
            end
        end
    end

    return (; FF, NN, SS, CC, SZ, chan_l, chan_z)
end

const sym_dict = Dict(:U1 => U1, :SU2 => SU{2})

"""
    charge_symmops(opts::FermionSOptions, basic_ops)

Build charge-symmetry data for each charge group in `opts.charge_symm`.

Each group is a `(symmetry_symbol, channel_indices)` tuple.  The weight of a
state is the total occupation on those channels measured relative to
half-filling.  `:U1` charge symmetry has no lowering operators.  `:SU2` charge
symmetry uses the channel-summed pair-annihilation operator as the lowering
operator.
"""

# For now, we only implement U1 charge relative to half-filling
# Charge symmetry is not nothing here
# TODO: Z_N charge symmetry, relative to other fillings, etc.
function charge_symmops(opts::FermionSOptions, basic_ops)
    N = opts.nchannels
    weights, symms, lops = [], [], []
    for (ssymbol, channs) in opts.charge_symm
        @assert ssymbol in [:U1, :SU2] "Unsupported charge symmetry: $ssymbol"
        sym = sym_dict[ssymbol]
        nchan = length(channs)
        charge = Int.(diag(sum(basic_ops.NN[1, i] + basic_ops.NN[2, i] for i in channs)) .- nchan)

        lop = if sym == SU{2}
            [Float64.(sum(basic_ops.CC[i] for i in channs))]
        elseif sym == U1
            Matrix{Float64}[]
        else error("Unsupported charge symmetry") end

        push!(weights, [(Int(i),) for i in charge])
        push!(symms, sym)
        push!(lops, lop)
    end
    return weights, symms, lops
end

"""
    spin_symmops(opts::FermionSOptions, basic_ops)

Build spin-symmetry data for each spin group in `opts.spin_symm`.

Each group is a `(symmetry_symbol, channel_indices)` tuple.  The weights are
the total spin-z values on those channels.  `:U1` spin symmetry has no lowering
operators.  `:SU2` spin symmetry uses the channel-summed spin-lowering
operator.
"""

# This covers U1 and SU(2) spin symmetries since the weights are 
# just the total spin-z component
function spin_symmops(opts::FermionSOptions, basic_ops)
    N = opts.nchannels
    weights, symms, lops = [], [], []
    for (ssymbol, channs) in opts.spin_symm
        @assert ssymbol in [:U1, :SU2] "Unsupported spin symmetry: $ssymbol"
        sym = sym_dict[ssymbol]
        nchan = length(channs)
        spin_z = Int.(diag(sum(basic_ops.SZ[i] for i in channs)))

        lop = if sym == SU{2}
            [Float64.(-sum(basic_ops.SS[i] for i in channs))]
        elseif sym == U1
            Matrix{Float64}[]
        else error("Unsupported spin symmetry") end

        push!(weights, [(Int(i),) for i in spin_z])
        push!(symms, sym)
        push!(lops, lop)
    end
    return weights, symms, lops
end

"""
    chan_symmops(opts::FermionSOptions, basic_ops)

Build channel-symmetry data for each channel group in `opts.channel_symm`.

The current implementation supports symbols of the form `:SU2`, `:SU3`, ...
where the SU(N) rank must match the number of listed channels.  It returns the
Dynkin-basis channel weights and adjacent channel-lowering operators.
"""

# For now, we only implement SU(N) channel symmetry
# TODO: generalize to another channel symmetry
function chan_symmops(opts::FermionSOptions, basic_ops)
    N = opts.nchannels; FF = basic_ops.FF
    weights, symms, lops = [], [], []
    for (ssymbol, channs) in opts.channel_symm

        # Get symmetry type and number of channels involved
        str = String(ssymbol)
        if startswith(str, "SU")
            num_part = str[3:end]
            if isempty(num_part) N = length(channs)
            else N = parse(Int, num_part) end
            @assert N >= 2 && N == length(channs)
            push!(symms, SU{N})
        end

        lop = Vector{SparseMatrixCSC{Float64, Int}}(undef, N-1)
        zvals = Vector{Vector{Int}}(undef, N-1)
        for i in 1:(N-1)
            lop[i] = Float64.(FF[1, channs[i+1]]' * FF[1, channs[i]] +
                              FF[2, channs[i+1]]' * FF[2, channs[i]])

            zvals[i] = Int.(diag(sum([basic_ops.NN[1, channs[j]] + 
                                      basic_ops.NN[2, channs[j]] for j=1:i])
                               - i * (basic_ops.NN[1, channs[i+1]] + 
                                      basic_ops.NN[2, channs[i+1]])))
        end
        push!(lops, lop)
        push!(weights, collect(zip(zvals...)))
    end
    return weights, symms, lops
end

function _add_spin_components!(
    mwirops::Dict{Symbol, SparseMatrixCSC{Float64, Int}},
    channs::AbstractVector{<:Integer},
    N::Int,
    basic_ops)

    suffix_channs = collect(channs)
    mwirops[_spin_symbol(suffix_channs, N, "p")] =
        (-1 / sqrt(2)) * sum(basic_ops.SS[i]' for i in suffix_channs)
    mwirops[_spin_symbol(suffix_channs, N, "z")] =
        (1 / 2) * sum(basic_ops.SZ[i] for i in suffix_channs)
    mwirops[_spin_symbol(suffix_channs, N, "m")] =
        (1 / sqrt(2)) * sum(basic_ops.SS[i] for i in suffix_channs)
end

function _add_spin_irop!(
    mwirops::Dict{Symbol, SparseMatrixCSC{Float64, Int}},
    channs::AbstractVector{<:Integer},
    N::Int,
    basic_ops)

    suffix_channs = collect(channs)
    mwirops[_spin_symbol(suffix_channs, N)] =
        (-1 / sqrt(2)) * sum(basic_ops.SS[i]' for i in suffix_channs)
end

"""
    add_spin_irop!(mwirops, opts::FermionSOptions, basic_ops)

Add spin operators to the maximal-weight IROP dictionary.

Channels that are not selected by any spin or channel symmetry group get
explicit component operators named `S<idx>p`, `S<idx>z`, and `S<idx>m`, except
when the operator covers all channels where the channel suffix is omitted.
Grouped `:U1` spin symmetries and SU(N) channel symmetries add channel-summed
component operators named `S<indices>p`, `S<indices>z`, and `S<indices>m`,
where `<indices>` is the juxtaposed channel list.  Grouped `:SU2` spin
symmetries add one fused IROP named `S<indices>` using the raising operator as
the maximal-weight component; channel symmetries contained in an SU(2)-spin
group do not add separate spin components for those channels.

Channel-symmetry groups must either be disjoint from all spin-symmetry groups,
or be contained in exactly one already selected multi-channel spin group.
Crossing multiple spin groups is ambiguous and raises an error.
"""
function add_spin_irop!(mwirops::Dict{Symbol, SparseMatrixCSC{Float64, Int}}, 
                   opts::FermionSOptions, basic_ops)
    N = opts.nchannels

    spin_groups = opts.spin_symm === nothing ? Tuple{Symbol, Vector{Int}}[] : opts.spin_symm
    channel_groups = opts.channel_symm === nothing ? Tuple{Symbol, Vector{Int}}[] : opts.channel_symm

    spin_selected = _check_disjoint_channel_groups(spin_groups, N, "Spin symmetry")
    channel_selected = Set{Int}()

    for (ssymbol, channs) in spin_groups
        if ssymbol == :SU2
            _add_spin_irop!(mwirops, channs, N, basic_ops)
        elseif ssymbol == :U1
            _add_spin_components!(mwirops, channs, N, basic_ops)
        else
            error("Unsupported spin symmetry: $ssymbol")
        end
    end

    for (ssymbol, channs) in channel_groups
        _check_channels(channs, N, "Channel symmetry $ssymbol")
        startswith(String(ssymbol), "SU") ||
            error("Unsupported channel symmetry for spin IROPs: $ssymbol")

        overlap_with_channel = intersect(channel_selected, channs)
        isempty(overlap_with_channel) ||
            error("Channel symmetry groups overlap on channels " *
                  "$(sort!(collect(overlap_with_channel)))")

        chann_set = Set(channs)
        overlap_with_spin = intersect(chann_set, spin_selected)
        contained_in_su2_spin = false
        if !isempty(overlap_with_spin)
            containing = [
                (spin_symm, spin_channs) for (spin_symm, spin_channs) in spin_groups
                if issubset(chann_set, Set(spin_channs)) && length(spin_channs) > 1
            ]
            length(containing) == 1 || error(
                "Channel symmetry $ssymbol on channels $channs must be disjoint " *
                "from spin-symmetry groups or contained in one multi-channel " *
                "spin-symmetry group")
            contained_in_su2_spin = first(only(containing)) == :SU2
        end

        if !contained_in_su2_spin
            _add_spin_components!(mwirops, channs, N, basic_ops)
        end
        union!(channel_selected, channs)
    end

    selected = union(spin_selected, channel_selected)
    for i in 1:N
        if !(i in selected)
            _add_spin_components!(mwirops, [i], N, basic_ops)
        end
    end
end

function _annihilation_primary_blocks(opts::FermionSOptions, N::Int)
    charge_groups = opts.charge_symm === nothing ? Tuple{Symbol, Vector{Int}}[] : opts.charge_symm
    spin_groups = opts.spin_symm === nothing ? Tuple{Symbol, Vector{Int}}[] : opts.spin_symm

    charge_group_by_channel = fill(0, N)
    charge_su2_by_group = Bool[]
    for (group_idx, (ssymbol, channs)) in enumerate(charge_groups)
        ssymbol in [:U1, :SU2] || error("Unsupported charge symmetry: $ssymbol")
        _check_channels(channs, N, "Charge symmetry $ssymbol")
        push!(charge_su2_by_group, ssymbol == :SU2)
        for ch in channs
            charge_group_by_channel[ch] == 0 ||
                error("Charge symmetry groups overlap on channel $ch")
            charge_group_by_channel[ch] = group_idx
        end
    end

    spin_su2_group_by_channel = fill(0, N)
    spin_su2_group_idx = 0
    for (ssymbol, channs) in spin_groups
        ssymbol in [:U1, :SU2] || error("Unsupported spin symmetry: $ssymbol")
        _check_channels(channs, N, "Spin symmetry $ssymbol")
        ssymbol == :SU2 || continue
        spin_su2_group_idx += 1
        for ch in channs
            spin_su2_group_by_channel[ch] == 0 ||
                error("SU(2) spin symmetry groups overlap on channel $ch")
            spin_su2_group_by_channel[ch] = spin_su2_group_idx
        end
    end

    block_by_key = Dict{Tuple{Int, Int}, Int}()
    blocks = NamedTuple[]
    for ch in 1:N
        charge_group = charge_group_by_channel[ch]
        spin_group = spin_su2_group_by_channel[ch]
        charge_group == 0 && spin_group == 0 && continue

        key = (charge_group, spin_group)
        charge_su2 = charge_group == 0 ? false : charge_su2_by_group[charge_group]
        spin_su2 = spin_group != 0
        if haskey(block_by_key, key)
            push!(blocks[block_by_key[key]].channs, ch)
        else
            push!(blocks, (;
                channs = [ch],
                charge_su2,
                spin_su2,
                channel_symm = false,
            ))
            block_by_key[key] = length(blocks)
        end
    end

    return blocks
end

function _split_annihilation_blocks_by_channel_symmetry(blocks, opts::FermionSOptions, N::Int)
    channel_groups = opts.channel_symm === nothing ? Tuple{Symbol, Vector{Int}}[] : opts.channel_symm
    channel_selected = Set{Int}()

    for (ssymbol, channs) in channel_groups
        startswith(String(ssymbol), "SU") ||
            error("Unsupported channel symmetry for annihilation IROPs: $ssymbol")
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
            push!(blocks, (;
                channs = collect(channs),
                charge_su2 = false,
                spin_su2 = false,
                channel_symm = true,
            ))
        elseif length(containing_idxs) == 1 && length(blocks[only(containing_idxs)].channs) > 1
            block_idx = only(containing_idxs)
            block = blocks[block_idx]
            remainder = [ch for ch in block.channs if !(ch in chann_set)]

            blocks[block_idx] = (;
                channs = collect(channs),
                charge_su2 = block.charge_su2,
                spin_su2 = block.spin_su2,
                channel_symm = true,
            )
            if !isempty(remainder)
                push!(blocks, (;
                    channs = remainder,
                    charge_su2 = block.charge_su2,
                    spin_su2 = block.spin_su2,
                    channel_symm = false,
                ))
            end
        else
            error("Channel symmetry $ssymbol on channels $channs must contain " *
                  "only channels with no charge/SU(2)-spin symmetry or be " *
                  "contained in one already generated multi-channel set")
        end

        union!(channel_selected, channs)
    end

    return blocks
end

function _add_annihilation_block!(
    mwirops::Dict{Symbol, SparseMatrixCSC{Float64, Int}},
    block,
    N::Int,
    basic_ops)

    channs = block.channs
    op_channs = block.channel_symm ? [last(channs)] : channs

    if !block.charge_su2 && !block.spin_su2
        mwirops[_fermion_symbol(channs, N, "u")] =
            sum(basic_ops.FF[1, i] for i in op_channs)
        mwirops[_fermion_symbol(channs, N, "d")] =
            sum(basic_ops.FF[2, i] for i in op_channs)
    elseif block.charge_su2 && !block.spin_su2
        mwirops[_fermion_symbol(channs, N, "u")] =
            sum(basic_ops.FF[2, i]' for i in op_channs)
        mwirops[_fermion_symbol(channs, N, "d")] =
            sum(basic_ops.FF[1, i]' for i in op_channs)
    elseif !block.charge_su2 && block.spin_su2
        mwirops[_fermion_symbol(channs, N)] =
            sum(basic_ops.FF[2, i] for i in op_channs)
    else
        mwirops[_fermion_symbol(channs, N)] =
            sum(basic_ops.FF[1, i]' for i in op_channs)
    end
end

"""
    add_annihilation_irop!(mwirops, opts::FermionSOptions, basic_ops)

Add fermion annihilation operators to the maximal-weight IROP dictionary.

Charge symmetries and SU(2) spin symmetries first generate channel sets.  SU(N)
channel symmetries may either select only otherwise unselected channels, or be
contained in one existing multi-channel set, in which case that set is split.
Unselected channels get individual default `F<idx>u` and `F<idx>d` operators.
When a generated operator covers all channels, the channel suffix is omitted.

For each generated set, `F<indices>u`/`F<indices>d` are emitted when charge and
spin are not both SU(2)-fused.  `F<indices>` is emitted when SU(2) spin fuses
the spin-up and spin-down annihilation components.  If the set also has channel
symmetry, the maximal-weight operator is taken only from `last(indices)`.
"""
function add_annihilation_irop!(mwirops::Dict{Symbol, SparseMatrixCSC{Float64, Int}}, 
                            opts::FermionSOptions, basic_ops)
    N = opts.nchannels

    blocks = _annihilation_primary_blocks(opts, N)
    _split_annihilation_blocks_by_channel_symmetry(blocks, opts, N)

    selected = Set{Int}()
    for block in blocks
        _add_annihilation_block!(mwirops, block, N, basic_ops)
        union!(selected, block.channs)
    end

    for i in 1:N
        if !(i in selected)
            _add_annihilation_block!(
                mwirops,
                (; channs = [i], charge_su2 = false, spin_su2 = false, channel_symm = false),
                N,
                basic_ops)
        end
    end
end


function getSymmetryInfo(opts::FermionSOptions)
    # For N-channel SU(2)-spin case, local Hilbert space is:
    #   |0>, |↑>, |↓>, |↑↓>
    # This should be changed for more general options

    N = opts.nchannels

    basic_ops = FermionS_basicops(N)
    symms = Vector{Type{<:Symmetry}}()
    weights = Vector{<:Tuple{Vararg{Int}}}[]
    lowering_ops = Vector{<:AbstractMatrix{<:Real}}[]

    # If charge symmetry is present, add it 
    if opts.charge_symm !== nothing
        ws, ss, ls = charge_symmops(opts, basic_ops)
        append!(symms, ss)
        append!(weights, ws)
        append!(lowering_ops, ls)
    end

    # If spin symmetry is present, add it
    if opts.spin_symm !== nothing
        ws, ss, ls = spin_symmops(opts, basic_ops)
        append!(symms, ss)
        append!(weights, ws)
        append!(lowering_ops, ls)
    end

    # If channel symmetry is present, add it 
    if opts.channel_symm !== nothing
        ws, ss, ls = chan_symmops(opts, basic_ops)
        append!(symms, ss)
        append!(weights, ws)
        append!(lowering_ops, ls)
    end

    totalN = diag(sum(basic_ops.NN[i] for i in 1:2*N))
    mwirops = Dict{Symbol, SparseMatrixCSC{Float64, Int}}()
    add_spin_irop!(mwirops, opts, basic_ops)
    add_annihilation_irop!(mwirops, opts, basic_ops)
    mwirops[:Z] = spdiagm(0 => Float64[i%2==0 ? 1 : -1 for i in totalN])
    mwirops[:I] = spdiagm(0 => ones(Float64, 4^N))

    return Tuple(symms), Tuple(weights), Tuple(lowering_ops), mwirops
end
