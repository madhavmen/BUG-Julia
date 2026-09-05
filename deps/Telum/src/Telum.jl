module Telum

using MKL
using HPTT_jll
using LinearAlgebra
using SparseArrays
using HDF5

import Base: lock, unlock
import LurCGT
import LurCGT: AbelianSymm, NonabelianSymm, Symmetry, SO, SU, Sp, U1, Z
import LurCGT: add_qn, decompose_irop, decompose_space, detect_1j, dimension
import LurCGT: get_CGTom, get_IROP, get_conj_perm, get_dualq
import LurCGT: getNsave_CGTperm, getNsave_CGTSVD, getNsave_Xsymbol, getNsave_Conjperm
import LurCGT: getNsave_omlist, getNsave_validout, isabelian
import LurCGT: nzops, remove_zeros, totxt, transf_basis!

⊗(a::AbstractMatrix, b::AbstractMatrix) = kron(b, a)
⊗(a::AbstractVector, b::AbstractVector) = kron(b, a)
comm(A, B) = A * B - B * A

include("DiagRMT.jl")

const contraction_cost = Ref{Int}(0)
const svd_cost = Ref{Int}(0)
const accumul_costs = Ref(false)

@inline function _add_contraction_cost!(cost::Integer)
    accumul_costs[] || return nothing
    contraction_cost[] += Int(cost)
    return nothing
end

@inline function _add_svd_cost!(cost::Integer)
    accumul_costs[] || return nothing
    svd_cost[] += Int(cost)
    return nothing
end

function set_accumul_costs!(enabled::Bool)
    accumul_costs[] = enabled
    return enabled
end

function read_reset_costs!()
    costs = (contraction_cost[], svd_cost[])
    contraction_cost[] = 0
    svd_cost[] = 0
    return costs
end

include("TLArray.jl")
include("io_hdf5.jl")

export AbstractTLArray, TLArray, TLArrayView, TLIndex, Itag, DiagRMT
export ProductSymm, productsymm, product_symms, symm, nsymms
export LocalSpaceOptions, SpinOptions, FermionOptions, FermionSOptions
export Z, U1, SU, SO, Sp
export getLocalSpace, getSymmetryInfo, getIdentity, get1jtensor, legflip, contract
export TELUM_CONTRACT_THREADS
export discard_eigen
export svd_leg, svd_cgtsvd, get_new_cgp
export empty_tlarray, qlabeltype, zero_qlabels, getvac
export addSingleton, deleteSingleton, ⊗
export oplus, getsub
export complete_oplus_matrix
export printmeta, to_concrete
export findlegs, findleg, matchings, matching, unmatchings, unmatching
export contractables, contractable, uncontractables, uncontractable
export prime, setprime, noprime
export lock, lockp, unlock
export additag, removeitag, replaceitag, setitag
export accumul_costs, read_reset_costs!, set_accumul_costs!
export save_tlarray, load_tlarray

end
