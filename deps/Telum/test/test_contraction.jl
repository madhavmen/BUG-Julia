@testset "lock reduction in contract" begin
    test_lock_reduce(FermionSOptions(1, :U1, :SU2, nothing))
end

@testset "auto contract requires matching spaces" begin
    test_contract_requires_matching_spaces_in_star(
        FermionSOptions(1, :U1, :SU2, nothing))
end

@testset "contract X-symbol cache shares duplicated non-Abelian symmetry" begin
    option = FermionSOptions(1, :SU2, :SU2, nothing)
    q = getLocalSpace(option)
    PS = Telum.productsymm(q.I)
    QT = qlabeltype(q.I)
    caches = Telum._new_xsym_caches(PS, QT, Val(1), Val(1), Val(1))

    @test length(caches) == 1
    @test @inferred(Telum._xsym_cache_slot(PS, Val(1))) == 1
    @test @inferred(Telum._xsym_cache_slot(PS, Val(2))) == 1
    @test Telum._xsym_cache_for(SU{2}, caches, PS, Val(1)) ===
          Telum._xsym_cache_for(SU{2}, caches, PS, Val(2))

    qi1 = TLArray(q.I, ("dup1", "dup1"))
    qi2 = TLArray(q.I, ("dup2", "dup2"))
    a = getIdentity((qi1, 2), (qi2, 2))
    @test qi1 * a isa TLArray
end

@testset "getIdentity direct contraction" begin
    test_getIdentity_direct_contract(FermionSOptions(1, :U1, :SU2, nothing))
    test_getIdentity_direct_contract(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "contract sparse equivalence edge cases" begin
    test_contract_sparse_equivalence_diag_rmt()
end

@testset "sum sparse equivalence mixed RMT and eltypes" begin
    test_sum_sparse_equivalence_mixed_rmt_eltypes()
end

test_contract_tlarrayview_inputs()

@testset "Generating 1jtensor of TLArray test" begin
    test_1jpair(FermionSOptions(1, :U1, :SU2, nothing))
    test_1jpair(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "Keyword get1jtensor and legflip" begin
    test_get1jtensor_and_legflip_keywords(FermionSOptions(1, :U1, :SU2, nothing))
    test_get1jtensor_and_legflip_keywords(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "contract verify_legs checks dual" begin
    test_contract_verify_legs_checks_dual(FermionSOptions(1, :U1, :SU2, nothing))
    test_contract_verify_legs_checks_dual(FermionSOptions(3, :U1, :SU2, :SU3))
end

@testset "contraction of TLArray test" begin
    test_FAcont(FermionSOptions(1, :U1, :SU2, nothing))
    test_FAcont(FermionSOptions(3, :U1, :SU2, :SU3))
end
