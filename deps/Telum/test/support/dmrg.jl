using LinearAlgebra
using Telum

# DMRG validation support for integration/stress tests.
# These helpers intentionally live outside src/: they exercise public Telum
# operations together without becoming part of the package API.

macro time_block(enabled, msg, ex)
    @info "$msg started"
    quote
        if $(esc(enabled))
            println($(esc(msg)))
            @time $(esc(ex))
        else
            $(esc(ex))
        end
    end
end

# MPO builders used by DMRG tests and contraction/SVD stress fixtures.
function build_MPO(qss::Matrix{<:AbstractTLArray}, N::Int)
    println("Building MPO...")
    t = oplus(qss, (3, 4))
    zq = zero_qlabels(t)
    MPO = Vector{TLArray{Float64, 4}}(undef, N)
    for i in 1:N
        tags = ("S,$i", "S,$i", i > 1 ? "OB,$(i - 1)" : "OLeft",
                i < N ? "OB,$i" : "ORight")

        if i == 1
            MPO[i] = TLArray(getsub(t, 3, x -> x == zq ? -1 : nothing), tags)
        elseif i == N
            MPO[i] = TLArray(getsub(t, 4, x -> x == zq ? 1 : nothing), tags)
        else
            MPO[i] = TLArray(t, tags)
        end
    end
    return MPO
end

function MajumdarGhoshMPO(J, N)
    option = SpinOptions(SU{2}, 1 // 2)
    q = getLocalSpace(option, ("site", "site", "op"))

    qss = Matrix{AbstractTLArray{Float64, 4, 1, 5}}(undef, 4, 4)

    i4d = addSingleton(q.I; nlegs=2, itag=("left", "right"), dir=('+', '-'))
    s4d = addSingleton(q.S, 3; itag="left", dir='+')
    s4d = setitag(s4d, 4, "right")
    sc4d = addSingleton(q.S'; itag="right", dir='-')
    sc4d = permutedims(setitag(sc4d, 3, "left"), (2, 1, 3, 4))
    opid = TLArray(getIdentity((q.S, 3)), ("left", "right"))

    qss[1, 1] = qss[4, 4] = i4d
    qss[2, 1] = sc4d
    qss[3, 2] = q.I ⊗ opid
    qss[4, 2] = J * s4d
    qss[4, 3] = J / 2 * s4d

    return build_MPO(qss, N)
end

function HubbardMPO(U, μ, t, N,
                    opt::FermionSOptions=FermionSOptions(1, :U1, :SU2, nothing))
    q = getLocalSpace(opt, ("site", "site", "op"))
    nloc = lock(q.F', 2) * q.F

    qss = Matrix{AbstractTLArray{Float64, 4}}(undef, 4, 4)
    i4d = addSingleton(q.I; nlegs=2, itag=("left", "right"), dir=('+', '-'))
    ZF_flip = setitag(legflip(lock(q.Z, 1) * q.F, 3), 3, "left")
    FcZ = permutedims(q.F' * lock(q.Z, 2), (1, 3, 2))

    f4d = addSingleton(q.F, 3; itag="left", dir='+')
    f4d = setitag(f4d, 4, "right")
    fc_flip = addSingleton(legflip(q.F', 3), 3; itag="left", dir='+')
    fc_flip = setitag(permutedims(fc_flip, (2, 1, 3, 4)), 4, "right")

    qss[1, 1] = qss[4, 4] = i4d
    qss[2, 1] = setitag(addSingleton(FcZ; itag="right", dir='-'), 3, "left")
    qss[3, 1] = addSingleton(ZF_flip; itag="right", dir='-')
    qss[4, 1] = addSingleton(lock(nloc - 1, 1) * nloc * U / 2 - μ * nloc;
        nlegs=2, itag=("left", "right"), dir=('+', '-'))
    qss[4, 2] = -t * f4d
    qss[4, 3] = -t * fc_flip

    return build_MPO(qss, N)
end

function XYMPO(J, N)
    option = SpinOptions(:U1, 1)
    q = getLocalSpace(option, ("site", "site", "op"))

    i4d = addSingleton(q.I, (3, 4); itag=("left", "right"), dir=('+', '-'))

    sm4d = addSingleton(q.Sm, 3; itag="left", dir='+')
    sm4d = setitag(sm4d, 4, "right")
    sp4d = addSingleton(q.Sp, 3; itag="left", dir='+')
    sp4d = setitag(sp4d, 4, "right")

    smc4d = addSingleton(q.Sm', 4; itag="right", dir='-')
    smc4d = permutedims(setitag(smc4d, 3, "left"), (2, 1, 3, 4))
    spc4d = addSingleton(q.Sp', 4; itag="right", dir='-')
    spc4d = permutedims(setitag(spc4d, 3, "left"), (2, 1, 3, 4))

    qss = Matrix{AbstractTLArray{Float64, 4, 1, 5}}(undef, 4, 4)

    qss[1, 1] = qss[4, 4] = i4d
    qss[2, 1] = spc4d
    qss[3, 1] = smc4d
    qss[4, 2] = J * sp4d
    qss[4, 3] = J * sm4d
    return build_MPO(qss, N)
end

function XXZMPO(δ, h, N)
    option = SpinOptions(:U1, 1)
    q = getLocalSpace(option, ("site", "site", "op"))

    i4d = addSingleton(q.I, (3, 4); itag=("left", "right"), dir=('+', '-'))

    sm4d = addSingleton(q.Sm, 3; itag="left", dir='+')
    sm4d = setitag(sm4d, 4, "right")
    sz4d = addSingleton(q.Sz, (3, 4); itag=("left", "right"), dir=('+', '-'))
    sp4d = addSingleton(q.Sp, 3; itag="left", dir='+')
    sp4d = setitag(sp4d, 4, "right")

    smc4d = addSingleton(q.Sm', 4; itag="right", dir='-')
    smc4d = permutedims(setitag(smc4d, 3, "left"), (2, 1, 3, 4))
    spc4d = addSingleton(q.Sp', 4; itag="right", dir='-')
    spc4d = permutedims(setitag(spc4d, 3, "left"), (2, 1, 3, 4))

    qss = Matrix{AbstractTLArray{Float64, 4, 1, 5}}(undef, 5, 5)

    qss[1, 1] = qss[5, 5] = i4d
    qss[2, 1] = spc4d
    qss[3, 1] = smc4d
    qss[4, 1] = sz4d
    qss[5, 1] = h * sz4d
    qss[5, 2] = sp4d
    qss[5, 3] = sm4d
    qss[5, 4] = δ * sz4d
    return build_MPO(qss, N)
end

# Lanczos algorithm for the local DMRG effective problem.
function eigs_GS(Hl, Hs, Hr, M; tol, nKrylov, time_blocks=true)
    As = Vector{AbstractTLArray}(undef, nKrylov)
    As[1] = M / norm(M)
    αs = zeros(nKrylov)
    βs = zeros(nKrylov - 1)
    cnt = 0

    @time_block time_blocks "Lanczos iteration:" begin
        for i in 1:nKrylov
            Amul = Hl * As[i]
            for H in Hs
                Amul = Amul * H
            end
            Amul = Amul * Hr
            αs[i] = (As[i]' * Amul)[]
            cnt += 1

            if i < nKrylov
                for _ in 1:2
                    coeffs = [(As[j]' * Amul)[] for j in 1:i]
                    Amul = sum((Amul, (-coeffs[j] * As[j] for j in 1:i)...))
                end
                Anorm = norm(Amul)
                Anorm < tol && break
                As[i + 1] = Amul / Anorm
                βs[i] = Anorm
            end
        end
    end

    @time_block time_blocks "Diagonalizing the tridiagonal matrix..." begin
        Hkrylov = SymTridiagonal(αs[1:cnt], βs[1:cnt - 1])
        _, V = eigen(Hkrylov)
        vec = V[:, 1]
        Anew = sum((vec[i] * As[i] for i in 1:cnt))
        Enew = Hl * Anew
        for H in Hs
            Enew = Enew * H
        end
        Enew = ((Enew * Hr) * Anew')[]
        @assert imag(Enew) < tol
    end
    return Anew, real(Enew)
end

function getHrl(MPO, MPS)
    N = length(MPS)
    Hrl = Vector{AbstractTLArray}(undef, N + 2)

    li = findleg(MPS[1]; itag="SLeft")
    left_id = getIdentity((MPS[1]', li); itag="SLeft")
    Hrl[1] = addSingleton(left_id; itag="OLeft", dir='-')

    li = findleg(MPS[end]; itag="SRight")
    right_id = getIdentity((MPS[end], li); itag="SRight")

    Hrl[end] = addSingleton(right_id; itag="ORight", dir='+')
    for i in 1:N
        Hrl[i + 1] = MPS[i]' * lock(Hrl[i] * MPO[i] * MPS[i]; itag="SB,$i")
    end
    return Hrl
end

# One-site sweep is kept for compatibility with older experiments; the required
# validation runner below uses the two-site sweep.
function DMRG_GS_1site!(MPS::Vector{<:TLArray{T1, 3}},
                        MPO::Vector{<:TLArray{T2, 4}},
                        Nkeep::Int,
                        Nsweep::Int; time_blocks=true) where {T1, T2}

    println("One-site DMRG...")
    tol = 1e-8
    nKrylov = 5
    N = length(MPO)
    @assert length(MPS) == N
    Hrl = getHrl(MPO, MPS)
    MPS[N] = legflip(MPS[N]; itag="SRight")
    Hrl[end] = legflip(Hrl[end]; itag="SRight")
    E = 0.0

    for si in 1:Nsweep
        println("DMRG right->left sweep $si")
        for i in N:-1:1
            println("Site $i")
            M, E = eigs_GS(Hrl[i], [MPO[i]], Hrl[i + 2], MPS[i]; tol, nKrylov, time_blocks)
            target_tag = i == 1 ? "SLeft" : "SB,$(i - 1)"
            if i > 1
                @time_block time_blocks "Time for svd: " begin
                    result = svd(M, "temp", target_tag; itag=target_tag)
                    U, S, Vd = result.U, result.S, result.Vd
                    MPS[i] = Vd
                    MPS[i - 1] = (MPS[i - 1] * U) * S
                end
            else
                MPS[i] = M
            end
            Hrl[i + 1] = (Hrl[i + 2] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag=target_tag)
        end
        println("Energy: $E")

        println("DMRG left->right sweep $si")
        for i in 1:N
            println("Site $i")
            M, E = eigs_GS(Hrl[i], [MPO[i]], Hrl[i + 2], MPS[i]; tol, nKrylov, time_blocks)
            target_tag = i == N ? "SRight" : "SB,$i"
            if i < N
                @time_block time_blocks "Time for svd: " begin
                    result = svd(M, target_tag, "temp"; itag=target_tag, rev=true)
                    U, S, Vd = result.U, result.S, result.Vd
                    MPS[i] = U
                    MPS[i + 1] = (MPS[i + 1] * Vd) * S
                end
            else
                MPS[i] = M
            end
            Hrl[i + 1] = (Hrl[i] * MPS[i]) * MPO[i] * lock(MPS[i]'; itag=target_tag)
        end
        println("Energy: $E")
    end
    return MPS, E
end

function DMRG_GS_2site!(MPS::Vector{<:TLArray{T1, 3}},
                        MPO::Vector{<:TLArray{T2, 4}},
                        Nkeep::Int,
                        Nsweep::Int; time_blocks=true) where {T1, T2}

    println("Two-site DMRG...")
    tol = 1e-8
    nKrylov = 5
    N = length(MPO)
    @assert length(MPS) == N
    Hrl = getHrl(MPO, MPS)
    E = 0.0

    for si in 1:Nsweep
        println("DMRG right->left sweep $si")
        for i in N - 1:-1:1
            time_blocks && println("Sites $i and $(i + 1)")
            M, E = eigs_GS(Hrl[i], [MPO[i], MPO[i + 1]], Hrl[i + 3], MPS[i] * MPS[i + 1];
                           tol, nKrylov, time_blocks)

            @time_block time_blocks "Time for svd: " begin
                tags = i == 1 ? ("SLeft", "S,$i") : ("SB,$(i - 1)", "S,$i")
                lids = [findleg(M; itag=t) for t in tags]
                result = svd(M, lids, "SB,$i,left", "SB,$i,right"; Nkeep=Nkeep)
                U, S, Vd = result.U, result.S, result.Vd

                MPS[i] = removeitag(U * S, "right")
                MPS[i + 1] = removeitag(Vd, "right")
                Hrl[i + 2] = (Hrl[i + 3] * MPS[i + 1]) * MPO[i + 1] *
                             lock(MPS[i + 1]'; itag="SB,$i")
            end
        end
        println("Energy: $E")

        println("DMRG left->right sweep $si")
        for i in 1:N - 1
            time_blocks && println("Sites $i and $(i + 1)")
            M, E = eigs_GS(Hrl[i], [MPO[i], MPO[i + 1]], Hrl[i + 3], MPS[i] * MPS[i + 1];
                           tol, nKrylov, time_blocks)

            @time_block time_blocks "Time for svd: " begin
                tags = i == 1 ? ("SLeft", "S,$i") : ("SB,$(i - 1)", "S,$i")
                lids = [findleg(M; itag=t) for t in tags]
                result = svd(M, lids, "SB,$i,left", "SB,$i,right"; Nkeep=Nkeep)
                U, S, Vd = result.U, result.S, result.Vd

                MPS[i] = removeitag(U, "left")
                MPS[i + 1] = removeitag(S * Vd, "left")
                Hrl[i + 1] = (Hrl[i] * MPS[i]) * MPO[i] *
                             lock(MPS[i]'; itag="SB,$i")
            end
        end
        println("Energy: $E")
    end
    return MPS, E
end

# Initial MPS construction by iterative diagonalization of left blocks.
function init_MPS(MPO::Vector{<:TLArray}, Nkeep::Int, Nkeep_last::Int=1; tol=0.0)
    N = length(MPO)
    MPS = Vector{TLArray{Float64, 3}}(undef, N)
    zq = zero_qlabels(MPO[1])
    Aprev = getvac(MPO[1], ("SLeft", "SLeft"))
    Hprev = addSingleton(Aprev; itag="OLeft", dir='-')
    E, sp = nothing, nothing

    for i in 1:N
        li = findleg(Aprev; itag=i == 1 ? "SLeft" : "SB", dir='-')
        Anow = getIdentity((Aprev, li), (MPO[i], 2); itag="SL,$i")
        Hnow = Anow' * lock(Anow * Hprev * MPO[i]; itag="SL,$i")

        bli = findleg(Hnow; itag=i == N ? "ORight" : "OB,$i")
        Hmat = deleteSingleton(getsub(Hnow, bli, x -> x == zq ? 1 : nothing), bli)
        e = eigen((Hmat + Hmat') / 2; hermitian=true)

        ek, _ = discard_eigen(e, i == N ? Nkeep_last : Nkeep,
                              i == N ? "SRight" : "SB,$i", "SD,$i"; tol)
        MPS[i] = Aprev = Anow * ek.V
        if i < N
            Hprev = lock(ek.V * Hnow; itag="SB") * ek.V'
        else
            E, sp = ek.eig_list[1][1], ek.eig_list[1][3]
        end
    end
    return MPS, E, sp
end

function do_dmrg_result(MPO, Nkeep_init=50, Nkeep_DMRG=50, Nsweep=4; time_blocks=true)
    MPS, Einit, sp = init_MPS(MPO, Nkeep_init; tol=0.0)
    println("Initial energy: $Einit")
    MPS, E = DMRG_GS_2site!(MPS, MPO, Nkeep_DMRG, Nsweep; time_blocks)
    return MPS, E
end

do_dmrg(MPO, Nkeep_init=50, Nkeep_DMRG=50, Nsweep=4; time_blocks=true) =
    first(do_dmrg_result(MPO, Nkeep_init, Nkeep_DMRG, Nsweep; time_blocks))

function central_bond_rmt_dimensions(MPS::AbstractVector{<:TLArray}; bond::Int=length(MPS) ÷ 2)
    1 <= bond < length(MPS) || throw(ArgumentError(
        "central bond index must satisfy 1 <= bond < $(length(MPS)), got $bond"))
    leg = findleg(MPS[bond]; itag="SB,$bond")
    isnothing(leg) && throw(ArgumentError("MPS[$bond] has no leg tagged SB,$bond"))
    bond_space = MPS[bond].spaces[leg]
    qlabels = first.(bond_space)
    dimensions = last.(bond_space)
    perm = sortperm(dimensions; rev=true)
    return dimensions[perm], qlabels[perm]
end

# Structured test entry point. Each result is a NamedTuple so tests can assert
# energies while retaining central-bond and cost diagnostics for debugging.
function run_dmrg_validation(MPO=HubbardMPO(4.0, 1.5, 1.0, 40);
                             time_blocks=false, nkeeps=[20, 50],
                             nsweep=4)
    read_reset_costs!()
    return map(nkeeps) do Nkeep_DMRG
        println("Nkeep_DMRG = $Nkeep_DMRG")
        set_accumul_costs!(true)
        Nkeep_init = max(50, Nkeep_DMRG)
        MPS, energy = try
            @time do_dmrg_result(MPO, Nkeep_init, Nkeep_DMRG, nsweep; time_blocks)
        finally
            set_accumul_costs!(false)
        end
        dimensions, qlabels = central_bond_rmt_dimensions(MPS)
        contraction_cost, svd_cost = read_reset_costs!()
        GC.gc()
        return (; nkeep=Nkeep_DMRG, energy, dimensions, qlabels,
                contraction_cost, svd_cost)
    end
end

function run(MPO=HubbardMPO(4.0, 1.5, 1.0, 40); time_blocks=false, nkeeps=[20, 50])
    results = run_dmrg_validation(MPO; time_blocks, nkeeps)
    return getproperty.(results, :dimensions),
           getproperty.(results, :qlabels),
           getproperty.(results, :contraction_cost),
           getproperty.(results, :svd_cost)
end
