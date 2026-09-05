function _make_index_metadata_tlarray()
    option = FermionSOptions(3, :U1, :SU2, :SU3)
    q0 = getLocalSpace(option)
    return TLArray(q0.F, ("site1", "site2", "op"))
end

@testset "index metadata selection and modifiers" begin
    q = _make_index_metadata_tlarray()

    @testset "findlegs" begin
        @test findlegs(q; dir='+') == [1]
        @test findlegs(q; dir='-') == [2, 3]
        @test findlegs(q; itag="site1") == [1]
        @test findlegs(q; itag="site2") == [2]
        @test findlegs(q; itag="op") == [3]
        @test findlegs(q; plev=0) == [1, 2, 3]
        @test findlegs(q; lock=0) == [1, 2, 3]

        q_p = prime(q, 2)
        @test findlegs(q_p; plev=1) == [2]
        @test findlegs(q_p; plev=0) == [1, 3]

        q_k = lock(q, 3)
        @test findlegs(q_k; lock=0) == [1, 2]
        @test findlegs(q_k; lock=1) == [3]

        @test findlegs(q; dir='+', rev=true) == [2, 3]
        @test findlegs(q; dir='-', rev=true) == [1]
        @test findlegs(q; itag="site1", rev=true) == [2, 3]
        @test findlegs(q; plev=0, rev=true) == []
        @test findlegs(q; dir='-', itag="op") == [3]
        @test findlegs(q; dir='-', itag="op", rev=true) == [1, 2]

        q_tagsets = TLArray(q, ("aaa,bbb", "aaa,ccc", "bbb,ccc"))
        @test findlegs(q_tagsets; itag="aaa,bbb") == [1]
        @test findlegs(q_tagsets; itag=("aaa,bbb", "aaa,ccc")) == [1, 2]
        @test findlegs(q_tagsets; itag=["aaa,ccc", "bbb,ccc"]) == [2, 3]
    end

    @testset "findleg" begin
        @test findleg(q; dir='+') == 1
        @test findleg(q; dir='-') == 2
        @test findleg(q; itag="op") == 3
        @test findleg(q; plev=0) == 1
        @test findleg(q; dir='+', rev=true) == 2
        @test findleg(q; plev=0, rev=true) === nothing
        @test findleg(q; itag="nope") === nothing

        q_tagsets = TLArray(q, ("aaa,bbb", "aaa,ccc", "bbb,ccc"))
        @test findleg(q_tagsets; itag=("aaa,ccc", "bbb,ccc")) == 2
        @test findleg(q_tagsets; itag=["missing", "bbb,ccc"]) == 3
        @test findleg(q_tagsets; itag=["aaa,bbb,ccc", "aaa,ccc"]) == 2
    end

    @testset "matching and unmatching" begin
        q_adj = q'
        @test q_adj isa TLArrayView

        @test matching(q, q_adj) == 1
        @test matchings(q, q_adj) == [1, 2, 3]
        @test unmatching(q, q_adj) === nothing
        @test unmatchings(q, q_adj) == Int[]
        @test matching(q_adj, q) == 1
        @test matchings(q_adj, q) == [1, 2, 3]

        q_selective = TLArray(q_adj, (
            TLIndex("site1", '-', 1, 0, false),
            TLIndex("site2", '+', 0, 0, true),
            TLIndex("op", '+', 0, 7, false),
        ))

        @test matching(q, q_selective) == 3
        @test matchings(q, q_selective) == [3]
        @test unmatching(q, q_selective) == 1
        @test unmatchings(q, q_selective) == [1, 2]

        @test matching(q, q_selective; dir='+') === nothing
        @test matchings(q, q_selective; itag="op") == [3]
        @test matchings(q, q_selective; dir='+', rev=true) == [3]
        @test unmatching(q, q_selective; itag="op") === nothing
        @test unmatchings(q, q_selective; dir='-', rev=true) == [1]

        q_match_locked = lock(q, 3)
        q_unmatch_locked = lock(q, 1)
        @test matchings(q_match_locked, q_selective; lock=1) == [3]
        @test matchings(q_match_locked, q_selective; lock=0) == Int[]
        @test unmatchings(q_unmatch_locked, q_selective; lock=1) == [1]

        q_tagsets = TLArray(q, ("aaa,bbb", "aaa,ccc", "bbb,ccc"))
        q_tagsets_adj = q_tagsets'
        @test q_tagsets_adj isa TLArrayView
        @test matchings(q_tagsets, q_tagsets_adj; itag=("aaa,bbb", "bbb,ccc")) == [1, 3]
        @test matchings(q_tagsets, q_tagsets_adj; itag=["aaa,ccc", "bbb,ccc"], rev=true) == [1]
    end

    @testset "contractable and uncontractable" begin
        q_adj = q'
        @test q_adj isa TLArrayView

        @test contractable(q, q_adj) == 1
        @test contractables(q, q_adj) == [1, 2, 3]
        @test uncontractable(q, q_adj) === nothing
        @test uncontractables(q, q_adj) == Int[]
        @test contractable(q_adj, q) == 1
        @test contractables(q_adj, q) == [1, 2, 3]

        q_selective = TLArray(q_adj, (
            TLIndex("site1", '-', 1, 0, false),
            TLIndex("site2", '+', 0, 0, true),
            TLIndex("op", '+', 0, 7, false),
        ))

        @test contractable(q, q_selective) === nothing
        @test contractables(q, q_selective) == Int[]
        @test uncontractable(q, q_selective) == 1
        @test uncontractables(q, q_selective) == [1, 2, 3]

        q_a_locked = lock(q, 3)
        q_b_locked = lock(q_adj, 2)
        @test contractables(q_a_locked, q_adj) == [1, 2]
        @test uncontractables(q_a_locked, q_adj) == [3]
        @test contractables(q, q_b_locked) == [1, 3]
        @test uncontractables(q, q_b_locked) == [2]

        @test contractables(q, q_adj; dir='-') == [2, 3]
        @test contractables(q, q_adj; itag="op") == [3]
        @test contractables(q_a_locked, q_adj; lock=1) == Int[]
        @test uncontractables(q_a_locked, q_adj; lock=1) == [3]
        @test uncontractables(q, q_b_locked; dir='+', rev=true) == [2]

        q_tagsets = TLArray(q, ("aaa,bbb", "aaa,ccc", "bbb,ccc"))
        q_tagsets_adj = q_tagsets'
        @test q_tagsets_adj isa TLArrayView
        @test contractables(q_tagsets, q_tagsets_adj; itag=("aaa,bbb", "aaa,ccc")) == [1, 2]
        @test uncontractables(lock(q_tagsets, 2), q_tagsets_adj; itag=["aaa,ccc", "bbb,ccc"]) == [2]
    end

    @testset "Itag predicate equality" begin
        q_unsorted = TLArray(q, ("beta,alpha", "site2", "op"))

        @test q_unsorted.inds[1].itags isa Itag
        @test q_unsorted.inds[1].itags == "alpha,beta"
        @test q_unsorted.inds[1].itags == "beta,alpha"
        @test "beta,alpha" == q_unsorted.inds[1].itags
        @test findleg(q_unsorted, idx -> idx.itags == "beta,alpha") == 1
        @test findlegs(q_unsorted, idx -> idx.itags == "beta,alpha") == [1]
    end

    @testset "lock" begin
        q2 = lock(q, 1)
        @test q2.inds[1].lock == 1
        @test q2.inds[2].lock == 0
        @test q2.inds[3].lock == 0

        q2 = lock(q, 3; inc=2)
        @test q2.inds[3].lock == 2
        @test q2.inds[1].lock == 0

        q2 = lock(q, [1, 3])
        @test q2.inds[1].lock == 1
        @test q2.inds[3].lock == 1
        @test q2.inds[2].lock == 0

        q2 = lock(q, (2, 3))
        @test q2.inds[2].lock == 1
        @test q2.inds[3].lock == 1
        @test q2.inds[1].lock == 0

        q2 = lock(q; dir='+')
        @test q2.inds[1].lock == 1
        @test q2.inds[2].lock == 0
        @test q2.inds[3].lock == 0

        q2 = lock(q; inc=3, itag="op")
        @test q2.inds[3].lock == 3
        @test q2.inds[1].lock == 0

        q2 = lock(q; dir='+', rev=true)
        @test q2.inds[1].lock == 0
        @test q2.inds[2].lock == 1
        @test q2.inds[3].lock == 1

        q_perm = lockp(q, 1)
        q2 = lock(q_perm, 1; inc=5)
        @test q2.inds[1].lock == -1
    end

    @testset "lockp and unlock" begin
        q2 = lockp(q, 2)
        @test q2.inds[2].lock == -1
        @test q2.inds[1].lock == 0

        q2 = lockp(q, [1, 3])
        @test q2.inds[1].lock == -1
        @test q2.inds[3].lock == -1
        @test q2.inds[2].lock == 0

        q2 = lockp(q; itag="site2")
        @test q2.inds[2].lock == -1
        @test q2.inds[1].lock == 0

        q2 = lockp(q; dir='+', rev=true)
        @test q2.inds[2].lock == -1
        @test q2.inds[3].lock == -1
        @test q2.inds[1].lock == 0

        q_locked = lock(q, [1, 2])
        q2 = unlock(q_locked, 1)
        @test q2.inds[1].lock == 0
        @test q2.inds[2].lock == 1

        q_perm = lockp(q, 3)
        q2 = unlock(q_perm, 3)
        @test q2.inds[3].lock == 0

        q_locked = lock(q, [2, 3])
        q2 = unlock(q_locked; dir='-')
        @test q2.inds[2].lock == 0
        @test q2.inds[3].lock == 0
        @test q2.inds[1].lock == 0

        q2 = unlock(q_locked; dir='-', rev=true)
        @test q2.inds[1].lock == 0
        @test q2.inds[2].lock == 1
        @test q2.inds[3].lock == 1
    end

    @testset "prime and noprime" begin
        q2 = prime(q, 1)
        @test q2.inds[1].plev == 1
        @test q2.inds[2].plev == 0

        q2 = prime(q, 2; inc=3)
        @test q2.inds[2].plev == 3
        @test q2.inds[1].plev == 0

        q2 = prime(q, [1, 3])
        @test q2.inds[1].plev == 1
        @test q2.inds[3].plev == 1
        @test q2.inds[2].plev == 0

        q2 = prime(q, (1, 2); inc=2)
        @test q2.inds[1].plev == 2
        @test q2.inds[2].plev == 2
        @test q2.inds[3].plev == 0

        q2 = prime(q)
        @test all(q2.inds[i].plev == 1 for i in 1:3)

        q2 = prime(q; inc=2)
        @test all(q2.inds[i].plev == 2 for i in 1:3)

        q2 = prime(q; inc=-5)
        @test all(q2.inds[i].plev == 0 for i in 1:3)

        q2 = prime(q; dir='+')
        @test q2.inds[1].plev == 1
        @test q2.inds[2].plev == 0
        @test q2.inds[3].plev == 0

        q2 = prime(q; dir='+', rev=true)
        @test q2.inds[1].plev == 0
        @test q2.inds[2].plev == 1
        @test q2.inds[3].plev == 1

        q_primed = prime(q)
        q2 = noprime(q_primed)
        @test all(q2.inds[i].plev == 0 for i in 1:3)

        q2 = noprime(q_primed, 2)
        @test q2.inds[2].plev == 0
        @test q2.inds[1].plev == 1
        @test q2.inds[3].plev == 1

        q2 = noprime(q_primed, [1, 3])
        @test q2.inds[1].plev == 0
        @test q2.inds[3].plev == 0
        @test q2.inds[2].plev == 1

        q2 = noprime(q_primed; dir='+')
        @test q2.inds[1].plev == 0
        @test q2.inds[2].plev == 1
        @test q2.inds[3].plev == 1

        q2 = noprime(q_primed; dir='+', rev=true)
        @test q2.inds[1].plev == 1
        @test q2.inds[2].plev == 0
        @test q2.inds[3].plev == 0
    end

    @testset "setprime" begin
        q2 = setprime(q, [1, 2], 7)
        @test q2.inds[1].plev == 7
        @test q2.inds[2].plev == 7
        @test q2.inds[3].plev == 0

        q2 = setprime(q, (1, 3), 5)
        @test q2.inds[1].plev == 5
        @test q2.inds[3].plev == 5
        @test q2.inds[2].plev == 0

        q2 = setprime(q, 3; dir='-')
        @test q2.inds[2].plev == 3
        @test q2.inds[3].plev == 3
        @test q2.inds[1].plev == 0

        q2 = setprime(q, 3; dir='-', rev=true)
        @test q2.inds[1].plev == 3
        @test q2.inds[2].plev == 0
        @test q2.inds[3].plev == 0

        @test_throws ArgumentError setprime(q, -1)
    end

    @testset "additag and removeitag" begin
        q2 = additag(q, "new")
        @test q2.inds[1].itags == "new,site1"
        @test q2.inds[2].itags == "new,site2"
        @test q2.inds[3].itags == "new,op"

        q2 = additag(q, 1, "u1")
        @test q2.inds[1].itags == "site1,u1"
        @test q2.inds[2].itags == "site2"

        q2 = additag(q, [2, 3], "phys")
        @test q2.inds[2].itags == "phys,site2"
        @test q2.inds[3].itags == "op,phys"
        @test q2.inds[1].itags == "site1"

        q2 = additag(q, (1, 3), "x")
        @test q2.inds[1].itags == "site1,x"
        @test q2.inds[3].itags == "op,x"
        @test q2.inds[2].itags == "site2"

        q2 = additag(q, "u1"; dir='+')
        @test q2.inds[1].itags == "site1,u1"
        @test q2.inds[2].itags == "site2"

        q2 = additag(q, "u1"; dir='+', rev=true)
        @test q2.inds[1].itags == "site1"
        @test q2.inds[2].itags == "site2,u1"
        @test q2.inds[3].itags == "op,u1"

        q2 = removeitag(q, "site1")
        @test q2.inds[1].itags == ""
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"

        q2 = removeitag(q, 2, "site2")
        @test q2.inds[2].itags == ""
        @test q2.inds[1].itags == "site1"

        q_extra = additag(q, "extra")
        q2 = removeitag(q_extra, [1, 3], "extra")
        @test q2.inds[1].itags == "site1"
        @test q2.inds[3].itags == "op"
        @test q2.inds[2].itags == "extra,site2"

        q2 = removeitag(q_extra, (2, 3), "extra")
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"
        @test q2.inds[1].itags == "extra,site1"

        q_grouped = TLArray(q, ("aaa,bbb", "aaa,bbb,ccc", "bbb,ccc"))
        q2 = removeitag(q_grouped, ("aaa,bbb", "ccc"))
        @test q2.inds[1].itags == ""
        @test q2.inds[2].itags == ""
        @test q2.inds[3].itags == "bbb"

        q2 = removeitag(q_grouped, 2, ["aaa,bbb", "ccc"])
        @test q2.inds[1].itags == "aaa,bbb"
        @test q2.inds[2].itags == ""
        @test q2.inds[3].itags == "bbb,ccc"

        q2 = removeitag(q_grouped, ("aaa,bbb", "ccc"); dir='-')
        @test q2.inds[1].itags == "aaa,bbb"
        @test q2.inds[2].itags == ""
        @test q2.inds[3].itags == "bbb"

        q2 = removeitag(q_grouped, ["aaa,bbb", "ccc"]; dir='-', rev=true)
        @test q2.inds[1].itags == ""
        @test q2.inds[2].itags == "aaa,bbb,ccc"
        @test q2.inds[3].itags == "bbb,ccc"

        q2 = removeitag(q_extra, "extra"; dir='+')
        @test q2.inds[1].itags == "site1"
        @test q2.inds[2].itags == "extra,site2"

        q2 = removeitag(q_extra, "extra"; dir='+', rev=true)
        @test q2.inds[1].itags == "extra,site1"
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"
    end

    @testset "replaceitag and setitag" begin
        q2 = replaceitag(q, "site1"=>"phys")
        @test q2.inds[1].itags == "phys"
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"

        q2 = replaceitag(q, "site1"=>"left", "site2"=>"right")
        @test q2.inds[1].itags == "left"
        @test q2.inds[2].itags == "right"
        @test q2.inds[3].itags == "op"

        q_chain = TLArray(q, ("aaa,bbb", "aaa", "bbb"))
        q2 = replaceitag(q_chain, "aaa"=>"bbb", "bbb"=>"ccc")
        @test q2.inds[1].itags == "bbb,ccc"
        @test q2.inds[2].itags == "bbb"
        @test q2.inds[3].itags == "ccc"

        q_grouped = TLArray(q, ("aaa,bbb", "aaa,bbb,ddd", "bbb,ddd"))
        q2 = replaceitag(q_grouped, "aaa,bbb"=>"ccc")
        @test q2.inds[1].itags == "ccc"
        @test q2.inds[2].itags == "ccc,ddd"
        @test q2.inds[3].itags == "bbb,ddd"

        q2 = replaceitag(q, Dict("site1"=>"left", "op"=>"operator"))
        @test q2.inds[1].itags == "left"
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "operator"

        q2 = replaceitag(q, 2, "site2"=>"phys")
        @test q2.inds[1].itags == "site1"
        @test q2.inds[2].itags == "phys"

        q2 = replaceitag(q, "site2"=>"phys"; dir='+', rev=true)
        @test q2.inds[1].itags == "site1"
        @test q2.inds[2].itags == "phys"
        @test q2.inds[3].itags == "op"

        q2 = setitag(q, 3, "phys")
        @test q2.inds[3].itags == "phys"
        @test q2.inds[1].itags == "site1"

        q2 = setitag(q, [1, 2], "lur")
        @test q2.inds[1].itags == "lur"
        @test q2.inds[2].itags == "lur"
        @test q2.inds[3].itags == "op"

        q2 = setitag(q, (1, 3), "x")
        @test q2.inds[1].itags == "x"
        @test q2.inds[3].itags == "x"
        @test q2.inds[2].itags == "site2"

        q2 = setitag(q, "phys"; dir='+')
        @test q2.inds[1].itags == "phys"
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"

        q2 = setitag(q, "phys"; dir='-', rev=true)
        @test q2.inds[1].itags == "phys"
        @test q2.inds[2].itags == "site2"
        @test q2.inds[3].itags == "op"
    end

    @testset "TLArrayView leg property modifiers use visible legs" begin
        qv = permutedims(q, (3, 1, 2))
        @test qv isa TLArrayView
        @test qv.inds == (q.inds[3], q.inds[1], q.inds[2])

        q2 = lock(qv, 1)
        @test q2 isa TLArrayView
        @test getfield(q2, :arr).inds[3].lock == 1
        @test q2.inds[1].lock == 1
        @test q2.inds[2].lock == 0
        @test q2.inds[3].lock == 0

        q2 = lockp(qv, 2)
        @test getfield(q2, :arr).inds[1].lock == -1
        @test q2.inds[2].lock == -1

        q2 = unlock(lock(qv, (1, 3)), 3)
        @test getfield(q2, :arr).inds[3].lock == 1
        @test getfield(q2, :arr).inds[2].lock == 0
        @test q2.inds[1].lock == 1
        @test q2.inds[3].lock == 0

        q2 = prime(qv, 1; inc=2)
        @test getfield(q2, :arr).inds[3].plev == 2
        @test q2.inds[1].plev == 2
        @test q2.inds[2].plev == 0
        @test q2.inds[3].plev == 0

        q2 = setprime(qv, 2, 4)
        @test getfield(q2, :arr).inds[1].plev == 4
        @test q2.inds[2].plev == 4

        q2 = noprime(prime(qv), 1)
        @test getfield(q2, :arr).inds[3].plev == 0
        @test q2.inds[1].plev == 0
        @test q2.inds[2].plev == 1
        @test q2.inds[3].plev == 1

        q2 = additag(qv, 1, "visible")
        @test getfield(q2, :arr).inds[3].itags == "op,visible"
        @test q2.inds[1].itags == "op,visible"

        q2 = removeitag(additag(qv, (1, 3), "extra"), 3, "extra")
        @test getfield(q2, :arr).inds[2].itags == "site2"
        @test q2.inds[1].itags == "extra,op"
        @test q2.inds[3].itags == "site2"

        q2 = replaceitag(qv, 1, "op"=>"operator")
        @test getfield(q2, :arr).inds[3].itags == "operator"
        @test q2.inds[1].itags == "operator"

        q2 = setitag(qv, 2, "visible-site1")
        @test getfield(q2, :arr).inds[1].itags == "visible-site1"
        @test q2.inds[2].itags == "visible-site1"

        q2 = additag(qv, "selected"; itag="site1")
        @test getfield(q2, :arr).inds[1].itags == "selected,site1"
        @test q2.inds[2].itags == "selected,site1"

        qc = conj(q)
        @test qc isa TLArrayView
        @test findlegs(qc; dir='-') == [1]
        q2 = prime(qc; dir='-')
        @test getfield(q2, :arr).inds[1].plev == 1
        @test q2.inds[1].plev == 1
        @test q2.inds[2].plev == 0
        @test q2.inds[3].plev == 0
    end

    @testset "non-targeted legs are unmodified" begin
        q2 = prime(q, 1)
        for i in 2:3
            @test q2.inds[i].plev == q.inds[i].plev
            @test q2.inds[i].lock == q.inds[i].lock
            @test q2.inds[i].itags == q.inds[i].itags
            @test q2.inds[i].dir == q.inds[i].dir
        end

        q2 = lock(q, 1)
        for i in 2:3
            @test q2.inds[i].lock == q.inds[i].lock
            @test q2.inds[i].plev == q.inds[i].plev
            @test q2.inds[i].itags == q.inds[i].itags
        end

        q2 = additag(q, 1, "extra")
        for i in 2:3
            @test q2.inds[i].itags == q.inds[i].itags
        end
    end
end
