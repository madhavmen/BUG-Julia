# Build hook: make Telum's SVD work with ZERO symmetries (`set_symmetry!(:none)`).
#
# Telum's SVD was only ever exercised with >=1 symmetry. At N=0 it hits empty
# `prod(... for n in 1:N)` reductions (which throw) and indexes `cores[1]` on an
# empty tuple, so `bond_update_bug!` under `:none` crashes inside `svd.jl` -- not
# in BUG-Julia code. The fix is four lines; the canonical diff is committed at
# `docs/telum_nosym_svd.patch`.
#
# This runs automatically on `] build BUGJulia` (and on `] add`). It edits the
# *resolved* Telum source in place, is idempotent (a no-op once patched), and
# backs the file up once. Re-run `] build BUGJulia` if Telum is ever re-extracted.
# The real fix belongs upstream in Telum; this keeps `:none` working until then.

function telum_svd_path()
    p = Base.find_package("Telum")            # .../Telum/src/Telum.jl
    p === nothing && return nothing
    svd = joinpath(dirname(p), "svd.jl")
    isfile(svd) ? svd : nothing
end

# Each entry: (needle, replacement). `needle` must be present verbatim in the
# stock file; `replacement` is its patched form. Applied only if not already done.
const _EDITS = [
    # 3x empty-product reductions -> add `init = 1`
    ("om_dim = prod(rmt_size[QD + n] for n in 1:N)",
     "om_dim = prod(rmt_size[QD + n] for n in 1:N; init = 1)   # init=1: N=0 (no symmetry)"),
    ("prod(size(_svd_left_iso(left_payload, PS, Val(n)), 2) for n in 1:N)",
     "prod(size(_svd_left_iso(left_payload, PS, Val(n)), 2) for n in 1:N; init = 1)"),
    ("prod(size(_svd_right_iso(right_payload, PS, Val(n)), 2) for n in 1:N)",
     "prod(size(_svd_right_iso(right_payload, PS, Val(n)), 2) for n in 1:N; init = 1)"),
    # N=0 core-kron method: add just before the NTuple{N} method
    ("function _svd_core_kron_matrix(cores::NTuple{N, Array{Float64, 3}}) where {N}",
     "_svd_core_kron_matrix(::Tuple{}) = ones(Float64, 1, 1)   # N=0 (no symmetry): trivial 1x1 core\nfunction _svd_core_kron_matrix(cores::NTuple{N, Array{Float64, 3}}) where {N}"),
]

const _MARKER = "_svd_core_kron_matrix(::Tuple{})"   # present iff already patched

function patch_telum!()
    svd = telum_svd_path()
    if svd === nothing
        @warn "BUGJulia/build: could not locate Telum's svd.jl; :none mode will \
               error until Telum is patched (see docs/telum_nosym_svd.patch)."
        return
    end
    src = read(svd, String)
    if occursin(_MARKER, src)
        @info "BUGJulia/build: Telum already patched for no-symmetry SVD ($svd)."
        return
    end
    # Every needle must be present, else Telum changed shape and a blind edit is
    # unsafe -- bail loudly rather than half-patch.
    for (needle, _) in _EDITS
        if !occursin(needle, src)
            @warn "BUGJulia/build: Telum svd.jl does not match the expected 0.2.0 \
                   shape (missing: $(first(split(needle, '(')))). Apply \
                   docs/telum_nosym_svd.patch by hand for :none mode."
            return
        end
    end
    backup = svd * ".orig_nosym_backup"
    isfile(backup) || cp(svd, backup)
    for (needle, repl) in _EDITS
        src = replace(src, needle => repl)
    end
    write(svd, src)
    @info "BUGJulia/build: patched Telum for no-symmetry (:none) SVD ($svd). \
           Backup at $backup."
end

try
    patch_telum!()
catch err
    @warn "BUGJulia/build: no-symmetry Telum patch failed; :none mode may error." exception=err
end
