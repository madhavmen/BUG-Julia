"""
	svd_leg(arr::AbstractArray, leg::Integer; cutoff::Real=1e-12, maxdim::Union{Nothing, Integer}=nothing)

Perform SVD on an N-dimensional array by isolating one leg and merging every
other leg into a single right index.

Algorithm:
1. Move `leg` to the first axis and merge the remaining axes.
2. Compute SVD and truncate singular values smaller than `cutoff`.
3. Contract `Diagonal(S)` with `Vt`, then split merged right legs back to
   their original dimensions.

Returns `(U, SV, S)` where:
- `U` has shape `(size(arr, leg), χ)`
- `SV` has shape `(size(arr,1), ..., size(arr,leg-1), χ, size(arr,leg+1), ..., size(arr,N))`
  i.e. same leg order as the original array with the `leg`-th dimension replaced by `χ`
- `S` is the kept singular value vector of length `χ`
"""
function svd_leg(arr::AbstractArray{T, N}, 
    leg::Integer;
	cutoff::Real=1e-12,
	maxdim::Union{Nothing, Integer}=nothing) where {T<:Real, N}

	@assert 1 <= leg <= N "leg index must be in 1:$N"
	@assert cutoff >= 0 "cutoff must be non-negative"
	if maxdim !== nothing
		@assert maxdim >= 0 "maxdim must be non-negative"
	end

	other_legs = Tuple(i for i in 1:N if i != leg)
	perm = (leg, other_legs...)
	arr_perm = permutedims(arr, perm)

	left_dim = size(arr, leg)
	right_dims = size(arr_perm)[2:end]
	mat = reshape(arr_perm, left_dim, :)

	F = svd(mat)
	singular_vals = F.S

	keep = singular_vals .> cutoff
	if maxdim !== nothing
		keep_ids = findall(keep)
		if length(keep_ids) > maxdim
			keep .= false
			keep[keep_ids[1:maxdim]] .= true
		end
	end

	U = F.U[:, keep]
	S = singular_vals[keep]
	Vt = F.Vt[keep, :]

	SV_mat = Diagonal(S) * Vt
	SV_flat = reshape(SV_mat, (length(S), right_dims...))

	# Permute SV so its leg order matches the original array:
	# currently axis 1 is χ, axes 2..N are other_legs in order;
	# move χ back to position `leg`.
	perm_back = ntuple(i -> i < leg ? i + 1 : i == leg ? 1 : i, N)
	SV = permutedims(SV_flat, perm_back)

	# Sign convention: If the first nonzero element of U is negative, 
	# flip signs of U and SV
	# The first element of U can be zero
	if !isempty(S) && U[findfirst(!iszero, U), 1] < 0
		U .= -U; 
		SV .= -SV; 
	end
	return U, SV, S
end
