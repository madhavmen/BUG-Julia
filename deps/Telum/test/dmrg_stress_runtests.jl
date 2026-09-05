using Test

include("support/dmrg.jl")

function _dmrg_stress_fixture(; nsites::Int=20, nkeep::Int=30, nsweep::Int=1,
                              opt::FermionSOptions=FermionSOptions(1, :U1, :SU2, nothing))
    mpo = HubbardMPO(4.0, 1.5, 1.0, nsites, opt)
    mps, energy = do_dmrg_result(mpo, nkeep, nkeep, nsweep; time_blocks=false)
    return (; mpo, mps, energy, bond=nsites ÷ 2, nkeep)
end

function _required_leg(q::AbstractTLArray; itag::AbstractString)
    leg = findleg(q; itag)
    @test !isnothing(leg)
    return leg
end

function _largest_rmt_length(q::TLArray)
    lengths = Int[]
    for sector in Telum.sector_slots(q)
        q.iszero[sector] && continue
        push!(lengths, length(Telum.sector_rmt(q, sector)))
    end
    return isempty(lengths) ? 0 : maximum(lengths)
end

function _single_contractable_leg(a::AbstractTLArray, b::AbstractTLArray; itag::AbstractString)
    legs = contractables(a, b; itag)
    @test length(legs) == 1
    return only(legs)
end

function _svd_left_leg_cases(rank::Int)
    if rank == 6
        return [[3], [6], [1, 2], [4, 5], [1, 2, 3], [4, 5, 6], [3, 4, 5, 6]]
    end

    cases = Vector{Vector{Int}}()
    push!(cases, [rank])
    rank >= 2 && push!(cases, collect(1:2))
    rank >= 3 && push!(cases, collect(1:3))
    rank >= 4 && push!(cases, collect((rank - 2):rank))
    return filter(legs -> 0 < length(legs) < rank, cases)
end

function _test_contract_stress(a::TLArray, legs_a::Tuple, b::TLArray, legs_b::Tuple;
                               check_sparse::Bool, atol=1e-9, rtol=1e-9)
    if check_sparse
        return _test_contract_matches_sparse_and_preserves_inputs(
            a, legs_a, b, legs_b; atol, rtol)
    end

    result = contract(a, legs_a, b, legs_b)
    @test result isa TLArray
    @test norm(result) > 0
    return result
end

function _test_two_site_mpo_contract_and_svd(mpo::AbstractVector, bond::Int;
                                             check_sparse::Bool, check_svd::Bool)
    mpo_left = mpo[bond]
    mpo_right = mpo[bond + 1]
    mpo_left_bond_leg = _required_leg(mpo_left; itag="OB,$bond")
    mpo_right_bond_leg = _required_leg(mpo_right; itag="OB,$bond")
    two_site_mpo = _test_contract_stress(
        mpo_left, (mpo_left_bond_leg,), mpo_right, (mpo_right_bond_leg,);
        check_sparse, atol=1e-9, rtol=1e-9)
    @test ndims(two_site_mpo) == ndims(mpo_left) + ndims(mpo_right) - 2
    @test _largest_rmt_length(two_site_mpo) > 0

    if check_svd
        for left_legs in _svd_left_leg_cases(ndims(two_site_mpo))
            @testset "two-site MPO SVD left_legs=$(left_legs)" begin
                test_svdQS(two_site_mpo, left_legs; cutoff=1e-12, tol=1e-8, verbose=false)
            end
        end
    end
    return two_site_mpo
end

function _test_dmrg_fixture_contractions_and_svd(fixture; check_large_rmt::Bool,
                                                check_sparse::Bool, check_svd::Bool)
    mpo = fixture.mpo
    mps = fixture.mps
    bond = fixture.bond

    dimensions, _ = central_bond_rmt_dimensions(mps; bond)
    @test sum(dimensions) <= fixture.nkeep
    @test !isempty(dimensions)

    left = mps[bond]
    right = mps[bond + 1]
    left_bond_leg = _required_leg(left; itag="SB,$bond")
    right_bond_leg = _required_leg(right; itag="SB,$bond")

    two_site = _test_contract_stress(
        left, (left_bond_leg,), right, (right_bond_leg,);
        check_sparse, atol=1e-9, rtol=1e-9)
    @test ndims(two_site) == 4
    if check_large_rmt
        @test _largest_rmt_length(two_site) >= fixture.nkeep
    else
        @test _largest_rmt_length(two_site) > 0
    end

    next_site = mps[bond + 2]
    two_site_next_leg = _required_leg(two_site; itag="SB,$(bond + 1)")
    next_site_leg = _required_leg(next_site; itag="SB,$(bond + 1)")
    three_site = _test_contract_stress(
        two_site, (two_site_next_leg,), next_site, (next_site_leg,);
        check_sparse, atol=1e-9, rtol=1e-9)
    @test ndims(three_site) == 5
    if check_large_rmt
        @test _largest_rmt_length(three_site) >= fixture.nkeep
    else
        @test _largest_rmt_length(three_site) > 0
    end

    if check_svd
        test_svdQS(two_site, [1, 2]; cutoff=1e-12, tol=1e-8, verbose=false)
        test_svdQS(three_site, [1, 2, 3]; cutoff=1e-12, tol=1e-8, verbose=false)
    end

    two_site_mpo = _test_two_site_mpo_contract_and_svd(mpo, bond; check_sparse, check_svd)

    mps_mpo_mps_legs = (
        _single_contractable_leg(two_site, two_site_mpo; itag="S,$bond"),
        _single_contractable_leg(two_site, two_site_mpo; itag="S,$(bond + 1)"),
    )
    mps_mpo_mpo_legs = (
        _single_contractable_leg(two_site_mpo, two_site; itag="S,$bond"),
        _single_contractable_leg(two_site_mpo, two_site; itag="S,$(bond + 1)"),
    )
    mps_mpo = _test_contract_stress(
        two_site, mps_mpo_mps_legs, two_site_mpo, mps_mpo_mpo_legs;
        check_sparse, atol=1e-9, rtol=1e-9)
    @test ndims(mps_mpo) == ndims(two_site) + ndims(two_site_mpo) - 4
    if check_large_rmt
        @test _largest_rmt_length(mps_mpo) >= fixture.nkeep
    else
        @test _largest_rmt_length(mps_mpo) > 0
    end
end

@testset "DMRG large-RMT contraction and SVD stress" begin
    fixture = _dmrg_stress_fixture()
    _test_dmrg_fixture_contractions_and_svd(
        fixture; check_large_rmt=true, check_sparse=true, check_svd=true)

    option_fixture = _dmrg_stress_fixture(; 
    nsites=4, nkeep=4, nsweep=1, opt=FermionSOptions(2, :U1, :SU2, :SU2))

    _test_dmrg_fixture_contractions_and_svd(
        option_fixture; check_large_rmt=true, check_sparse=true, check_svd=true)
end
